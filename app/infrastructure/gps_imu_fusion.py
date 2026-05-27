#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""GPS + MPU-6050 fusion for ground-vehicle positioning.

The implementation is intentionally lightweight and dependency-friendly:

* MPU-6050 provides short-term acceleration and yaw-rate prediction.
* GPS provides absolute position, speed and course corrections.
* A 2D Kalman filter tracks east/north position and velocity in a local ENU
  frame. GPS HDOP is converted into measurement covariance, so weak GPS fixes
  are trusted less than good fixes.

This is not RTK and cannot create centimeter-level absolute GPS accuracy from a
standard receiver. It does reduce jitter, bridge short GPS gaps, and produce a
more stable vehicle state for navigation logic.
"""

import argparse
import copy
import math
import time
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import serial

from app.infrastructure.gps import (
    FixState,
    enu_from_latlon,
    has_valid_fix,
    knots_to_mps,
    latlon_from_enu,
    nmea_checksum_ok,
    parse_gga,
    parse_rmc,
)
from app.infrastructure.imu import ImuCalibration, Mpu6050


@dataclass(frozen=True)
class GpsMeasurement:
    timestamp_s: float
    fix: FixState
    east_m: float
    north_m: float
    speed_mps: Optional[float]
    course_deg: Optional[float]


@dataclass
class FusionConfig:
    accel_noise_mps2: float = 1.2
    gps_sigma_per_hdop_m: float = 2.5
    gps_min_sigma_m: float = 2.5
    gps_speed_sigma_mps: float = 0.7
    gps_velocity_min_speed_mps: float = 0.5
    gps_course_min_speed_mps: float = 0.8
    heading_correction_alpha: float = 0.12
    zero_velocity_speed_mps: float = 0.12
    zero_velocity_sigma_mps: float = 0.08
    max_gps_mahalanobis: float = 25.0
    max_prediction_dt_s: float = 0.25


@dataclass(frozen=True)
class FusedState:
    timestamp_s: float
    east_m: float
    north_m: float
    lat_deg: float
    lon_deg: float
    velocity_east_mps: float
    velocity_north_mps: float
    speed_mps: float
    heading_deg: float
    position_sigma_m: float
    last_gps_age_s: Optional[float]


class GpsAccumulator:
    """Accumulates GGA/RMC lines into complete usable GPS measurements."""

    def __init__(self, origin: Optional[Tuple[float, float]], check_checksum: bool = True):
        self.origin = origin
        self.check_checksum = check_checksum
        self.state = FixState()
        self._last_key = None

    def process_line(self, line: str) -> Optional[GpsMeasurement]:
        line = line.strip()
        if not line.startswith("$"):
            return None
        if self.check_checksum and "*" in line and not nmea_checksum_ok(line):
            return None

        body = line[1:].split("*", 1)[0]
        fields = body.split(",")
        msg = fields[0]
        if msg.endswith("GGA"):
            parse_gga(fields, self.state)
        elif msg.endswith("RMC"):
            parse_rmc(fields, self.state)
        else:
            return None

        if not has_valid_fix(self.state):
            return None

        key = (
            self.state.time_hhmmss,
            self.state.date_ddmmyy,
            self.state.lat_deg,
            self.state.lon_deg,
            self.state.speed_knots,
            self.state.course_deg,
        )
        if key == self._last_key:
            return None
        self._last_key = key

        assert self.state.lat_deg is not None
        assert self.state.lon_deg is not None
        if self.origin is None:
            self.origin = (self.state.lat_deg, self.state.lon_deg)
            print(f"[OK] fusion origin set lat0={self.origin[0]:.7f}, lon0={self.origin[1]:.7f}")

        en = enu_from_latlon(self.state.lat_deg, self.state.lon_deg, self.origin[0], self.origin[1])
        return GpsMeasurement(
            timestamp_s=time.monotonic(),
            fix=copy.copy(self.state),
            east_m=en["east_m"],
            north_m=en["north_m"],
            speed_mps=knots_to_mps(self.state.speed_knots),
            course_deg=self.state.course_deg,
        )


class HeadingEstimator:
    """Yaw estimator using gyro integration corrected by GPS course when moving."""

    def __init__(self, correction_alpha: float, gps_course_min_speed_mps: float):
        self.correction_alpha = max(0.0, min(1.0, correction_alpha))
        self.gps_course_min_speed_mps = gps_course_min_speed_mps
        self.heading_rad = 0.0
        self.initialized = False

    def predict(self, yaw_rate_rad_s: float, dt_s: float) -> None:
        if not self.initialized:
            return
        self.heading_rad = _wrap_rad(self.heading_rad + yaw_rate_rad_s * dt_s)

    def correct_with_gps(self, speed_mps: Optional[float], course_deg: Optional[float]) -> None:
        if speed_mps is None or course_deg is None or speed_mps < self.gps_course_min_speed_mps:
            return
        gps_heading = math.radians(course_deg)
        if not self.initialized:
            self.heading_rad = gps_heading
            self.initialized = True
            return
        error = _wrap_rad(gps_heading - self.heading_rad)
        self.heading_rad = _wrap_rad(self.heading_rad + self.correction_alpha * error)

    @property
    def heading_deg(self) -> float:
        return math.degrees(self.heading_rad) % 360.0


class PositionVelocityKalman:
    """2D constant-acceleration Kalman filter with GPS position/velocity updates."""

    def __init__(self, config: FusionConfig):
        self.config = config
        self.x = np.zeros((4, 1), dtype=float)
        self.p = np.diag([100.0, 100.0, 5.0, 5.0]).astype(float)
        self.initialized = False
        self.last_ts: Optional[float] = None
        self.last_gps_ts: Optional[float] = None
        self.rejected_gps_updates = 0

    def initialize(self, gps: GpsMeasurement) -> None:
        ve, vn = _gps_velocity_components(gps.speed_mps, gps.course_deg)
        self.x[:, 0] = [gps.east_m, gps.north_m, ve, vn]
        sigma = _gps_position_sigma(gps.fix.hdop, self.config)
        self.p = np.diag([sigma * sigma, sigma * sigma, 4.0, 4.0]).astype(float)
        self.initialized = True
        self.last_ts = gps.timestamp_s
        self.last_gps_ts = gps.timestamp_s

    def predict(self, timestamp_s: float, accel_east_mps2: float, accel_north_mps2: float) -> None:
        if not self.initialized:
            return
        if self.last_ts is None:
            self.last_ts = timestamp_s
            return
        dt = timestamp_s - self.last_ts
        if dt <= 0:
            return
        dt = min(dt, self.config.max_prediction_dt_s)
        self.last_ts = timestamp_s

        f = np.array(
            [
                [1.0, 0.0, dt, 0.0],
                [0.0, 1.0, 0.0, dt],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
            dtype=float,
        )
        b = np.array(
            [
                [0.5 * dt * dt, 0.0],
                [0.0, 0.5 * dt * dt],
                [dt, 0.0],
                [0.0, dt],
            ],
            dtype=float,
        )
        u = np.array([[accel_east_mps2], [accel_north_mps2]], dtype=float)
        accel_var = self.config.accel_noise_mps2 * self.config.accel_noise_mps2
        q = b @ (np.eye(2) * accel_var) @ b.T
        q += np.diag([0.005, 0.005, 0.02, 0.02])

        self.x = f @ self.x + b @ u
        self.p = f @ self.p @ f.T + q

    def correct_gps(self, gps: GpsMeasurement) -> bool:
        if not self.initialized:
            self.initialize(gps)
            return True

        sigma = _gps_position_sigma(gps.fix.hdop, self.config)
        accepted = self._update(
            z=np.array([[gps.east_m], [gps.north_m]], dtype=float),
            h=np.array([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]], dtype=float),
            r=np.eye(2) * sigma * sigma,
            mahalanobis_gate=self.config.max_gps_mahalanobis,
        )
        if accepted:
            self.last_gps_ts = gps.timestamp_s
        else:
            self.rejected_gps_updates += 1

        if gps.speed_mps is not None and gps.course_deg is not None:
            if gps.speed_mps >= self.config.gps_velocity_min_speed_mps:
                ve, vn = _gps_velocity_components(gps.speed_mps, gps.course_deg)
                self._update(
                    z=np.array([[ve], [vn]], dtype=float),
                    h=np.array([[0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]], dtype=float),
                    r=np.eye(2) * self.config.gps_speed_sigma_mps * self.config.gps_speed_sigma_mps,
                    mahalanobis_gate=None,
                )
            elif gps.speed_mps <= self.config.zero_velocity_speed_mps:
                self._update(
                    z=np.array([[0.0], [0.0]], dtype=float),
                    h=np.array([[0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]], dtype=float),
                    r=np.eye(2) * self.config.zero_velocity_sigma_mps * self.config.zero_velocity_sigma_mps,
                    mahalanobis_gate=None,
                )
        return accepted

    def _update(
        self,
        z: np.ndarray,
        h: np.ndarray,
        r: np.ndarray,
        mahalanobis_gate: Optional[float],
    ) -> bool:
        innovation = z - h @ self.x
        s = h @ self.p @ h.T + r
        try:
            s_inv_y = np.linalg.solve(s, innovation)
            distance = float((innovation.T @ s_inv_y).item())
            if mahalanobis_gate is not None and distance > mahalanobis_gate:
                return False
            k = self.p @ h.T @ np.linalg.inv(s)
        except np.linalg.LinAlgError:
            return False

        ident = np.eye(self.p.shape[0])
        self.x = self.x + k @ innovation
        self.p = (ident - k @ h) @ self.p @ (ident - k @ h).T + k @ r @ k.T
        return True

    def fused_state(self, timestamp_s: float, origin: Tuple[float, float], heading_deg: float) -> FusedState:
        east = float(self.x[0, 0])
        north = float(self.x[1, 0])
        ve = float(self.x[2, 0])
        vn = float(self.x[3, 0])
        lat, lon = latlon_from_enu(east, north, origin[0], origin[1])
        gps_age = None if self.last_gps_ts is None else max(0.0, timestamp_s - self.last_gps_ts)
        return FusedState(
            timestamp_s=timestamp_s,
            east_m=east,
            north_m=north,
            lat_deg=lat,
            lon_deg=lon,
            velocity_east_mps=ve,
            velocity_north_mps=vn,
            speed_mps=math.hypot(ve, vn),
            heading_deg=heading_deg,
            position_sigma_m=math.sqrt(max(0.0, float(self.p[0, 0] + self.p[1, 1]))),
            last_gps_age_s=gps_age,
        )


def _gps_position_sigma(hdop: Optional[float], config: FusionConfig) -> float:
    if hdop is None or hdop <= 0:
        return config.gps_min_sigma_m * 2.0
    return max(config.gps_min_sigma_m, hdop * config.gps_sigma_per_hdop_m)


def _gps_velocity_components(speed_mps: Optional[float], course_deg: Optional[float]) -> Tuple[float, float]:
    if speed_mps is None or course_deg is None:
        return 0.0, 0.0
    course = math.radians(course_deg)
    return speed_mps * math.sin(course), speed_mps * math.cos(course)


def _wrap_rad(angle: float) -> float:
    while angle > math.pi:
        angle -= 2.0 * math.pi
    while angle < -math.pi:
        angle += 2.0 * math.pi
    return angle


def _axis_value(values: Tuple[float, float, float], axis_spec: str) -> float:
    spec = axis_spec.strip().lower().replace("neg-", "-")
    sign = -1.0 if spec.startswith("-") else 1.0
    axis = spec[1:] if spec.startswith("-") else spec
    index = {"x": 0, "y": 1, "z": 2}.get(axis)
    if index is None:
        raise ValueError(f"invalid axis spec: {axis_spec}")
    return sign * values[index]


def _body_to_enu(accel_forward_mps2: float, accel_right_mps2: float, heading_rad: float) -> Tuple[float, float]:
    east = accel_forward_mps2 * math.sin(heading_rad) + accel_right_mps2 * math.cos(heading_rad)
    north = accel_forward_mps2 * math.cos(heading_rad) - accel_right_mps2 * math.sin(heading_rad)
    return east, north


def _read_serial_lines(ser: serial.Serial, buffer: bytearray) -> list:
    waiting = ser.in_waiting
    if waiting <= 0:
        return []
    buffer.extend(ser.read(waiting))
    lines = []
    while b"\n" in buffer:
        raw, _, rest = buffer.partition(b"\n")
        buffer[:] = rest
        lines.append(raw.decode("ascii", errors="ignore").strip())
    if len(buffer) > 4096:
        buffer.clear()
    return lines


def _format_state(state: FusedState, gps: Optional[GpsMeasurement], rejected_gps: int) -> str:
    parts = [
        f"fused_lat={state.lat_deg:.7f} fused_lon={state.lon_deg:.7f}",
        f"E={state.east_m:+.2f}m N={state.north_m:+.2f}m",
        f"speed={state.speed_mps:.2f}m/s",
        f"heading={state.heading_deg:.1f}deg",
        f"sigma={state.position_sigma_m:.2f}m",
    ]
    if state.last_gps_age_s is not None:
        parts.append(f"gps_age={state.last_gps_age_s:.1f}s")
    if gps is not None:
        parts.append(
            f"gps_lat={gps.fix.lat_deg:.7f} gps_lon={gps.fix.lon_deg:.7f} "
            f"hdop={gps.fix.hdop} sats={gps.fix.sats}"
        )
    if rejected_gps:
        parts.append(f"gps_rejected={rejected_gps}")
    return " | ".join(parts)


def run(args: argparse.Namespace) -> None:
    origin = tuple(args.origin) if args.origin else None
    gps_acc = GpsAccumulator(origin=origin, check_checksum=not args.no_checksum)
    config = FusionConfig(
        accel_noise_mps2=args.accel_noise,
        gps_sigma_per_hdop_m=args.gps_sigma_per_hdop,
        gps_min_sigma_m=args.gps_min_sigma,
        gps_speed_sigma_mps=args.gps_speed_sigma,
        gps_velocity_min_speed_mps=args.gps_velocity_min_speed,
        gps_course_min_speed_mps=args.gps_course_min_speed,
        heading_correction_alpha=args.heading_alpha,
        zero_velocity_speed_mps=args.zero_velocity_speed,
        max_gps_mahalanobis=args.max_gps_mahalanobis,
    )
    heading = HeadingEstimator(
        correction_alpha=config.heading_correction_alpha,
        gps_course_min_speed_mps=config.gps_course_min_speed_mps,
    )
    kf = PositionVelocityKalman(config)

    ser = serial.Serial(
        port=args.gps_port,
        baudrate=args.gps_baud,
        bytesize=serial.EIGHTBITS,
        parity=serial.PARITY_NONE,
        stopbits=serial.STOPBITS_ONE,
        timeout=0.0,
        xonxoff=False,
        rtscts=False,
        dsrdtr=False,
    )

    with Mpu6050(
        bus=args.imu_bus,
        address=args.imu_address,
        accel_range_g=args.accel_range,
        gyro_range_dps=args.gyro_range,
        dlpf_config=args.dlpf,
        sample_rate_hz=args.imu_rate,
    ) as imu:
        imu.initialize()
        print(f"[OK] GPS opened {args.gps_port} @ {args.gps_baud}")
        print(f"[OK] MPU-6050 opened /dev/i2c-{args.imu_bus} addr=0x{args.imu_address:02x}")
        print(
            "[INFO] axis mapping "
            f"forward={args.forward_axis} right={args.right_axis} yaw_gyro={args.yaw_gyro_axis}"
        )

        calibration = ImuCalibration((0.0, 0.0, 0.0), (0.0, 0.0, 0.0))
        if args.calibrate_samples > 0:
            print(f"[INFO] Keep vehicle still, calibrating IMU {args.calibrate_samples} samples...")
            calibration = imu.calibrate_static(samples=args.calibrate_samples, rate_hz=args.imu_rate)
            print(
                "[OK] calibration "
                f"gyro_bias=({calibration.gyro_bias_rad_s[0]:+.5f},"
                f"{calibration.gyro_bias_rad_s[1]:+.5f},"
                f"{calibration.gyro_bias_rad_s[2]:+.5f})rad/s "
                f"accel_rest=({calibration.accel_rest_mps2[0]:+.3f},"
                f"{calibration.accel_rest_mps2[1]:+.3f},"
                f"{calibration.accel_rest_mps2[2]:+.3f})m/s^2"
            )

        start_ts = time.monotonic()
        next_sample_ts = start_ts
        last_imu_ts: Optional[float] = None
        last_print_ts = 0.0
        last_gps: Optional[GpsMeasurement] = None
        serial_buffer = bytearray()

        try:
            while True:
                now = time.monotonic()
                if args.duration > 0 and now - start_ts >= args.duration:
                    break

                for line in _read_serial_lines(ser, serial_buffer):
                    gps = gps_acc.process_line(line)
                    if gps is None:
                        continue
                    if gps_acc.origin is None:
                        continue
                    heading.correct_with_gps(gps.speed_mps, gps.course_deg)
                    accepted = kf.correct_gps(gps)
                    last_gps = gps
                    if accepted and not heading.initialized and gps.course_deg is not None:
                        heading.correct_with_gps(gps.speed_mps, gps.course_deg)

                if now < next_sample_ts:
                    time.sleep(min(0.005, next_sample_ts - now))
                    continue

                sample = imu.read_sample()
                next_sample_ts += 1.0 / args.imu_rate
                if next_sample_ts < sample.timestamp_s:
                    next_sample_ts = sample.timestamp_s + 1.0 / args.imu_rate

                dt = 0.0 if last_imu_ts is None else sample.timestamp_s - last_imu_ts
                last_imu_ts = sample.timestamp_s

                accel_zeroed = tuple(
                    sample.accel_mps2[i] - calibration.accel_rest_mps2[i] for i in range(3)
                )
                gyro_zeroed = tuple(
                    sample.gyro_rad_s[i] - calibration.gyro_bias_rad_s[i] for i in range(3)
                )
                yaw_rate = _axis_value(gyro_zeroed, args.yaw_gyro_axis)
                if dt > 0:
                    heading.predict(yaw_rate, min(dt, config.max_prediction_dt_s))

                accel_forward = _axis_value(accel_zeroed, args.forward_axis)
                accel_right = _axis_value(accel_zeroed, args.right_axis)
                accel_east, accel_north = _body_to_enu(accel_forward, accel_right, heading.heading_rad)
                kf.predict(sample.timestamp_s, accel_east, accel_north)

                if (
                    kf.initialized
                    and gps_acc.origin is not None
                    and sample.timestamp_s - last_print_ts >= 1.0 / args.print_rate
                ):
                    print(
                        _format_state(
                            kf.fused_state(sample.timestamp_s, gps_acc.origin, heading.heading_deg),
                            last_gps,
                            kf.rejected_gps_updates,
                        )
                    )
                    last_print_ts = sample.timestamp_s
        except KeyboardInterrupt:
            print("\n[EXIT] fusion stopped")
        finally:
            ser.close()


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Fuse GPS NMEA and MPU-6050 IMU for vehicle positioning")
    parser.add_argument("--gps-port", default="/dev/ttyS9", help="GPS serial port, default /dev/ttyS9")
    parser.add_argument("--gps-baud", type=int, default=9600, help="GPS baud rate, default 9600")
    parser.add_argument("--imu-bus", type=int, default=4, help="MPU-6050 I2C bus number, default 4")
    parser.add_argument("--imu-address", type=lambda x: int(x, 0), default=0x68, help="MPU-6050 I2C address")
    parser.add_argument("--imu-rate", type=float, default=50.0, help="IMU read rate in Hz, default 50")
    parser.add_argument("--print-rate", type=float, default=2.0, help="output rate in Hz, default 2")
    parser.add_argument("--duration", type=float, default=0.0, help="run seconds, 0 means forever")
    parser.add_argument("--origin", nargs=2, type=float, metavar=("LAT0", "LON0"), help="fixed origin lat lon")
    parser.add_argument("--no-checksum", action="store_true", help="skip NMEA checksum validation")

    parser.add_argument("--calibrate-samples", type=int, default=200, help="static IMU calibration samples")
    parser.add_argument("--accel-range", type=int, default=2, choices=(2, 4, 8, 16), help="accelerometer range")
    parser.add_argument("--gyro-range", type=int, default=250, choices=(250, 500, 1000, 2000), help="gyro range")
    parser.add_argument("--dlpf", type=int, default=3, help="MPU DLPF config 0-6, default 3")
    parser.add_argument("--forward-axis", default="x", help="IMU axis used as vehicle forward: x,y,z,-x,-y,-z")
    parser.add_argument("--right-axis", default="y", help="IMU axis used as vehicle right: x,y,z,-x,-y,-z")
    parser.add_argument("--yaw-gyro-axis", default="z", help="gyro axis for positive heading-rate: z or -z")

    parser.add_argument("--accel-noise", type=float, default=1.2, help="IMU acceleration process noise m/s^2")
    parser.add_argument("--gps-sigma-per-hdop", type=float, default=2.5, help="GPS position sigma = HDOP * this")
    parser.add_argument("--gps-min-sigma", type=float, default=2.5, help="minimum GPS position sigma in meters")
    parser.add_argument("--gps-speed-sigma", type=float, default=0.7, help="GPS velocity sigma in m/s")
    parser.add_argument("--gps-velocity-min-speed", type=float, default=0.5, help="min GPS speed for velocity update")
    parser.add_argument("--gps-course-min-speed", type=float, default=0.8, help="min GPS speed for heading correction")
    parser.add_argument("--heading-alpha", type=float, default=0.12, help="GPS course heading correction alpha")
    parser.add_argument("--zero-velocity-speed", type=float, default=0.12, help="GPS speed threshold for zero-velocity update")
    parser.add_argument("--max-gps-mahalanobis", type=float, default=25.0, help="GPS position outlier gate")
    return parser


def main() -> None:
    run(build_arg_parser().parse_args())


if __name__ == "__main__":
    main()
