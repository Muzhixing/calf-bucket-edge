#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""MPU-6050 I2C reader.

This module talks to the sensor through Linux /dev/i2c-* directly, so it does
not require smbus/smbus2. On the ELF 2 board, the 40-pin I2C header is exposed
as /dev/i2c-4 and the MPU-6050 default address is 0x68.
"""

import argparse
import fcntl
import math
import os
import struct
import time
from dataclasses import dataclass
from typing import Generator, Optional, Tuple


I2C_SLAVE = 0x0703

REG_SMPLRT_DIV = 0x19
REG_CONFIG = 0x1A
REG_GYRO_CONFIG = 0x1B
REG_ACCEL_CONFIG = 0x1C
REG_ACCEL_XOUT_H = 0x3B
REG_PWR_MGMT_1 = 0x6B
REG_WHO_AM_I = 0x75

GRAVITY_MPS2 = 9.80665


@dataclass(frozen=True)
class ImuSample:
    """One converted MPU-6050 sample."""

    timestamp_s: float
    accel_mps2: Tuple[float, float, float]
    gyro_rad_s: Tuple[float, float, float]
    temperature_c: float
    raw_accel: Tuple[int, int, int]
    raw_gyro: Tuple[int, int, int]
    raw_temp: int


@dataclass(frozen=True)
class ImuCalibration:
    """Static IMU calibration result."""

    gyro_bias_rad_s: Tuple[float, float, float]
    accel_rest_mps2: Tuple[float, float, float]


class Mpu6050:
    """Small Linux I2C driver for MPU-6050."""

    ACCEL_SENS = {
        2: 16384.0,
        4: 8192.0,
        8: 4096.0,
        16: 2048.0,
    }
    GYRO_SENS = {
        250: 131.0,
        500: 65.5,
        1000: 32.8,
        2000: 16.4,
    }

    def __init__(
        self,
        bus: int = 4,
        address: int = 0x68,
        accel_range_g: int = 2,
        gyro_range_dps: int = 250,
        dlpf_config: int = 3,
        sample_rate_hz: float = 50.0,
    ):
        if accel_range_g not in self.ACCEL_SENS:
            raise ValueError(f"unsupported accel range: {accel_range_g}g")
        if gyro_range_dps not in self.GYRO_SENS:
            raise ValueError(f"unsupported gyro range: {gyro_range_dps} dps")
        if sample_rate_hz <= 0:
            raise ValueError("sample_rate_hz must be > 0")

        self.bus = bus
        self.address = address
        self.path = f"/dev/i2c-{bus}"
        self.accel_range_g = accel_range_g
        self.gyro_range_dps = gyro_range_dps
        self.dlpf_config = max(0, min(6, dlpf_config))
        self.sample_rate_hz = sample_rate_hz
        self._fd: Optional[int] = None

    def __enter__(self) -> "Mpu6050":
        self.open()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def open(self) -> None:
        if self._fd is not None:
            return
        self._fd = os.open(self.path, os.O_RDWR)
        fcntl.ioctl(self._fd, I2C_SLAVE, self.address)

    def close(self) -> None:
        if self._fd is not None:
            os.close(self._fd)
            self._fd = None

    def initialize(self) -> None:
        """Wake the sensor and apply range/filter/sample-rate settings."""
        who = self.read_u8(REG_WHO_AM_I)
        if who != self.address:
            raise RuntimeError(f"unexpected WHO_AM_I=0x{who:02x}, expected 0x{self.address:02x}")

        # Use X gyro PLL clock, then configure low-pass filter and output rate.
        self.write_u8(REG_PWR_MGMT_1, 0x01)
        time.sleep(0.05)
        self.write_u8(REG_CONFIG, self.dlpf_config)
        base_hz = 1000.0 if self.dlpf_config else 8000.0
        divider = max(0, min(255, round(base_hz / self.sample_rate_hz - 1)))
        self.write_u8(REG_SMPLRT_DIV, int(divider))

        accel_bits = {2: 0, 4: 1, 8: 2, 16: 3}[self.accel_range_g] << 3
        gyro_bits = {250: 0, 500: 1, 1000: 2, 2000: 3}[self.gyro_range_dps] << 3
        self.write_u8(REG_ACCEL_CONFIG, accel_bits)
        self.write_u8(REG_GYRO_CONFIG, gyro_bits)

    def read_sample(self) -> ImuSample:
        data = self.read_block(REG_ACCEL_XOUT_H, 14)
        ax, ay, az, temp_raw, gx, gy, gz = struct.unpack(">7h", data)
        accel_scale = GRAVITY_MPS2 / self.ACCEL_SENS[self.accel_range_g]
        gyro_scale = math.pi / 180.0 / self.GYRO_SENS[self.gyro_range_dps]
        return ImuSample(
            timestamp_s=time.monotonic(),
            accel_mps2=(ax * accel_scale, ay * accel_scale, az * accel_scale),
            gyro_rad_s=(gx * gyro_scale, gy * gyro_scale, gz * gyro_scale),
            temperature_c=temp_raw / 340.0 + 36.53,
            raw_accel=(ax, ay, az),
            raw_gyro=(gx, gy, gz),
            raw_temp=temp_raw,
        )

    def iter_samples(self, rate_hz: Optional[float] = None) -> Generator[ImuSample, None, None]:
        period_s = 1.0 / (rate_hz or self.sample_rate_hz)
        next_ts = time.monotonic()
        while True:
            yield self.read_sample()
            next_ts += period_s
            delay_s = next_ts - time.monotonic()
            if delay_s > 0:
                time.sleep(delay_s)
            else:
                next_ts = time.monotonic()

    def calibrate_static(self, samples: int = 300, rate_hz: float = 50.0) -> ImuCalibration:
        """Estimate gyro bias and the resting acceleration vector.

        Keep the vehicle still while this runs. The returned acceleration vector
        includes gravity and mounting tilt; subtracting it gives a practical
        zero-acceleration reference for short-term ground-vehicle dead reckoning.
        """
        if samples <= 0:
            raise ValueError("samples must be > 0")
        period_s = 1.0 / rate_hz
        gyro_sum = [0.0, 0.0, 0.0]
        accel_sum = [0.0, 0.0, 0.0]
        for _ in range(samples):
            s = self.read_sample()
            for i in range(3):
                gyro_sum[i] += s.gyro_rad_s[i]
                accel_sum[i] += s.accel_mps2[i]
            time.sleep(period_s)
        inv_n = 1.0 / samples
        return ImuCalibration(
            gyro_bias_rad_s=tuple(v * inv_n for v in gyro_sum),
            accel_rest_mps2=tuple(v * inv_n for v in accel_sum),
        )

    def read_u8(self, reg: int) -> int:
        return self.read_block(reg, 1)[0]

    def write_u8(self, reg: int, value: int) -> None:
        self._ensure_open()
        assert self._fd is not None
        os.write(self._fd, bytes([reg & 0xFF, value & 0xFF]))

    def read_block(self, reg: int, length: int) -> bytes:
        self._ensure_open()
        assert self._fd is not None
        os.write(self._fd, bytes([reg & 0xFF]))
        data = os.read(self._fd, length)
        if len(data) != length:
            raise OSError(f"short i2c read: expected {length}, got {len(data)}")
        return data

    def _ensure_open(self) -> None:
        if self._fd is None:
            self.open()


def _fmt_triplet(values: Tuple[float, float, float], unit: str) -> str:
    return f"({values[0]:+.4f},{values[1]:+.4f},{values[2]:+.4f}){unit}"


def main() -> None:
    parser = argparse.ArgumentParser(description="Read MPU-6050 samples over Linux I2C")
    parser.add_argument("--bus", type=int, default=4, help="I2C bus number, default 4")
    parser.add_argument("--address", type=lambda x: int(x, 0), default=0x68, help="I2C address, default 0x68")
    parser.add_argument("--rate", type=float, default=50.0, help="read/output rate in Hz, default 50")
    parser.add_argument("--samples", type=int, default=0, help="number of samples to print, 0 means forever")
    parser.add_argument("--calibrate", type=int, default=0, help="static calibration sample count before output")
    parser.add_argument("--accel-range", type=int, default=2, choices=(2, 4, 8, 16), help="accelerometer range in g")
    parser.add_argument("--gyro-range", type=int, default=250, choices=(250, 500, 1000, 2000), help="gyro range in dps")
    parser.add_argument("--dlpf", type=int, default=3, help="MPU DLPF config 0-6, default 3")
    args = parser.parse_args()

    with Mpu6050(
        bus=args.bus,
        address=args.address,
        accel_range_g=args.accel_range,
        gyro_range_dps=args.gyro_range,
        dlpf_config=args.dlpf,
        sample_rate_hz=args.rate,
    ) as imu:
        imu.initialize()
        print(f"[OK] MPU-6050 ready on /dev/i2c-{args.bus} addr=0x{args.address:02x}")
        calibration = None
        if args.calibrate > 0:
            print(f"[INFO] Keep sensor still, calibrating {args.calibrate} samples...")
            calibration = imu.calibrate_static(samples=args.calibrate, rate_hz=args.rate)
            print(
                "[OK] calibration "
                f"gyro_bias={_fmt_triplet(calibration.gyro_bias_rad_s, 'rad/s')} "
                f"accel_rest={_fmt_triplet(calibration.accel_rest_mps2, 'm/s^2')}"
            )

        printed = 0
        for sample in imu.iter_samples(args.rate):
            accel = sample.accel_mps2
            gyro = sample.gyro_rad_s
            if calibration is not None:
                accel = tuple(accel[i] - calibration.accel_rest_mps2[i] for i in range(3))
                gyro = tuple(gyro[i] - calibration.gyro_bias_rad_s[i] for i in range(3))
            print(
                f"t={sample.timestamp_s:.3f} "
                f"accel={_fmt_triplet(accel, 'm/s^2')} "
                f"gyro={_fmt_triplet(gyro, 'rad/s')} "
                f"temp={sample.temperature_c:.1f}C"
            )
            printed += 1
            if args.samples and printed >= args.samples:
                break


if __name__ == "__main__":
    main()
