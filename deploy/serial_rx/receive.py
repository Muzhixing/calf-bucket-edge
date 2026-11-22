#!/usr/bin/env python3
"""Receive-only F407 v1 telemetry. No command encoder or serial write path.

Protocol source: MCU-slave-device 9db673c9103d3da79434c08d7217c2c402871855,
new_project_f407/F407_HAL_Template/{docs/protocol.md,Hardware/SystemControl.c}.
"""
import argparse
from collections import Counter
from datetime import datetime, timezone
from functools import reduce
import hashlib
import json
import math
import operator
import os
from pathlib import Path
import select
import statistics
import sys
import time

PROTOCOL_COMMIT = "9db673c9103d3da79434c08d7217c2c402871855"
DEFAULT_PORT = "/dev/serial/by-path/platform-fc880000.usb-usb-0:1.2:1.0-port0"
STATES = {0: "disabled", 1: "ready_holding", 2: "axes_moving", 3: "chassis_moving", 4: "fault", 5: "waiting_stop"}
RESULTS = {0: "accepted", 1: "format_error", 2: "origins_unconfirmed", 3: "heading_not_ready", 4: "fault", 5: "zero_timing_rejected", 6: "uart_error", 7: "driver_home_not_verified"}
FAULTS = {0: "none", 1: "communication", 2: "motor", 3: "HWT_data", 4: "axis_start", 5: "driver_origin_lost"}
MOTOR_FAULTS = {0: "none", 1: "no_feedback", 2: "reverse_feedback", 3: "scheduler_timeout", 4: "driver_initialization"}


def utc_now():
    return datetime.now(timezone.utc).isoformat(timespec="microseconds")


def checksum(frame):
    return reduce(operator.xor, frame[1:18], 0)


def decode(frame):
    if len(frame) != 20 or frame[0] != 0xB5 or frame[19] != 0xB6:
        raise ValueError("frame length/header/tail")
    if frame[1] != 1:
        raise ValueError("unsupported version")
    if checksum(frame) != frame[18]:
        raise ValueError("checksum mismatch")
    u16 = lambda i: int.from_bytes(frame[i:i + 2], "little")
    flags, yaw, distance = frame[5], u16(10), u16(12)
    return {
        "hex": frame.hex(" ").upper(), "version": frame[1],
        "command_count_mod256": frame[2], "result": frame[3],
        "result_name": RESULTS.get(frame[3], "unknown"),
        "state": frame[4], "state_name": STATES.get(frame[4], "unknown"),
        "flags": flags, "enabled": bool(flags & 1),
        "lift_zero_confirmed": bool(flags & 2), "slide_zero_confirmed": bool(flags & 4),
        "angle_valid": bool(flags & 8), "distance_valid": bool(flags & 16),
        "heading_reference_established": bool(flags & 32), "unknown_flag_bits": 0,
        "lift_driver_feedback_fresh": bool(flags & 64), "slide_driver_feedback_fresh": bool(flags & 128),
        "lift_position_tenths_mm": u16(6), "slide_position_tenths_mm": u16(8),
        "lift_position_mm": u16(6) / 10, "slide_position_mm": u16(8) / 10,
        "yaw_raw_tenths_deg": yaw,
        "yaw_deg": yaw / 10 if yaw != 0xFFFF and flags & 8 and flags & 32 else None,
        "ultrasound_raw_mm": distance,
        "ultrasound_mm": distance if distance != 0xFFFF and flags & 16 else None,
        "servo_software_deg": frame[14], "system_fault": frame[15],
        "system_fault_name": FAULTS.get(frame[15], "unknown"),
        "motor_fault": frame[16], "motor_fault_name": MOTOR_FAULTS.get(frame[16], "unknown"),
        "uart_error_count_mod256": frame[17],
    }


class Parser:
    """Bounded residual buffer; slide one byte after a rejected candidate."""
    def __init__(self):
        self.buffer = bytearray()
        self.stats = Counter()

    def feed(self, data):
        self.buffer.extend(data)
        frames = []
        while self.buffer:
            try:
                start = self.buffer.index(0xB5)
            except ValueError:
                self.stats["discarded_bytes"] += len(self.buffer)
                self.buffer.clear()
                break
            self.stats["discarded_bytes"] += start
            del self.buffer[:start]
            if len(self.buffer) < 20:
                break
            frame = bytes(self.buffer[:20])
            reason = ("bad_tail" if frame[19] != 0xB6 else
                      "bad_version" if frame[1] != 1 else
                      "bad_checksum" if checksum(frame) != frame[18] else None)
            if reason:
                self.stats[reason] += 1
                self.stats["discarded_bytes"] += 1
                del self.buffer[0]
                continue
            frames.append(decode(frame))
            self.stats["valid_frames"] += 1
            del self.buffer[:20]
        return frames


def summary(records, parser):
    # Timestamps are host read-completion times, not MCU timestamps. Frames
    # coalesced in one read cannot provide individual inter-frame timing.
    intervals = [(b["monotonic_ns"] - a["monotonic_ns"]) / 1e6
                 for a, b in zip(records, records[1:])]
    result = {"valid_frames": len(records), "parser": dict(parser.stats),
              "trailing_partial_hex": parser.buffer.hex(" ").upper(),
              "timing_basis": "host read completion; coalesced frames share timestamp",
              "interval_count": len(intervals),
              "interval_ms_min": min(intervals) if intervals else None,
              "interval_ms_median": statistics.median(intervals) if intervals else None,
              "interval_ms_mean": statistics.mean(intervals) if intervals else None,
              "interval_ms_max": max(intervals) if intervals else None,
              "coalesced_intervals": sum(v == 0 for v in intervals),
              "intervals_over_375ms": sum(v > 375 for v in intervals),
              "enabled_frames": sum(r["enabled"] for r in records),
              "system_fault_codes": sorted(set(r["system_fault"] for r in records)),
              "motor_fault_codes": sorted(set(r["motor_fault"] for r in records)),
              "states": dict(Counter(r["state_name"] for r in records)),
              "command_counts": sorted(set(r["command_count_mod256"] for r in records)),
              "first_frame": records[0] if records else None,
              "last_frame": records[-1] if records else None}
    for field in ("lift_position_mm", "slide_position_mm", "yaw_deg", "uart_error_count_mod256"):
        values = [r[field] for r in records if r[field] is not None]
        result[field + "_range"] = [min(values), max(values)] if values else None
    return result


def dump_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def replay(args):
    parser, records = Parser(), []
    with Path(args.chunks).open() as source:
        for line in source:
            chunk = json.loads(line)
            for frame in parser.feed(bytes.fromhex(chunk["hex"])):
                frame.update(utc=chunk["utc"], monotonic_ns=chunk["monotonic_ns"])
                records.append(frame)
    result = summary(records, parser)
    result["source"] = str(Path(args.chunks).resolve())
    result["mode"] = "offline_replay"
    if args.out:
        dump_json(Path(args.out), result)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if records else 2


def holders(device, allow_partial=False):
    """Reject visible holders and record the scope of process visibility."""
    device_stat, found, inaccessible = os.stat(device), [], []
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit() or int(entry.name) == os.getpid():
            continue
        try:
            descriptors = list((entry / "fd").iterdir())
        except FileNotFoundError:
            continue
        except PermissionError:
            inaccessible.append(entry.name)
            continue
        for descriptor in descriptors:
            try:
                info = descriptor.stat()
                if info.st_rdev == device_stat.st_rdev and info.st_mode == device_stat.st_mode:
                    found.append({"pid": int(entry.name), "fd": descriptor.name})
            except FileNotFoundError:
                pass
            except PermissionError:
                inaccessible.append(entry.name)
    if found:
        raise RuntimeError("Port already open: " + json.dumps(found))
    if inaccessible and not allow_partial:
        raise RuntimeError("Cannot inspect all /proc FDs; use sudo or explicitly accept partial visibility")
    return {"visible_holders": found, "inaccessible_pids": sorted(set(inaccessible)),
            "complete": not inaccessible, "partial_visibility_accepted": allow_partial}


def capture(args):
    # Linux ch341 open itself may assert modem lines even with O_RDONLY.
    # Record on-site conditions; do not claim software controls this side effect.
    import fcntl
    import stat
    import termios
    if sys.platform != "linux":
        raise RuntimeError("Live capture is Linux-only; use replay on other hosts")
    if not args.site_confirmation:
        raise RuntimeError("Actual site conditions must be recorded before capture")
    if not math.isfinite(args.seconds) or not 0 < args.seconds <= 300:
        raise ValueError("Capture seconds must be in (0,300]")
    port = Path(args.port).resolve(strict=True)
    if not stat.S_ISCHR(port.stat().st_mode):
        raise RuntimeError("Port is not a character device")
    occupancy = holders(port, allow_partial=True) if getattr(args, "allow_partial_occupancy", False) else holders(port)
    output = Path(args.out)
    output.mkdir(parents=True, exist_ok=False)
    manifest = {"mode": "physical_receive_only", "start_utc": utc_now(),
                "requested_port": args.port, "resolved_port": str(port),
                "protocol_source_commit": PROTOCOL_COMMIT,
                "firmware_identity_basis": "user-reported; telemetry does not encode firmware SHA",
                "site_confirmation": args.site_confirmation,
                "occupancy_check": occupancy,
                "seconds_requested": args.seconds, "baud": 115200, "format": "8N1",
                "serial_open": "O_RDONLY | O_NOCTTY | O_NONBLOCK",
                "serial_write_calls": 0, "modem_line_ioctls": 0,
                "software_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    dump_json(output / "manifest.json", manifest)
    parser, records, received, fd = Parser(), [], 0, None
    error, reason = None, "duration_elapsed"
    start = time.monotonic_ns()
    try:
        fd = os.open(port, os.O_RDONLY | os.O_NOCTTY | os.O_NONBLOCK)
        fcntl.ioctl(fd, termios.TIOCEXCL)
        # No tcflush, tcdrain, tcflow, DTR/RTS or break operations.
        attrs = termios.tcgetattr(fd)
        attrs[0] = attrs[1] = attrs[3] = 0
        attrs[2] = termios.CS8 | termios.CREAD | termios.CLOCAL  # also clears HUPCL/CRTSCTS
        attrs[4] = attrs[5] = termios.B115200
        attrs[6][termios.VMIN] = attrs[6][termios.VTIME] = 0
        termios.tcsetattr(fd, termios.TCSANOW, attrs)
        start = time.monotonic_ns()
        deadline = start + int(args.seconds * 1e9)
        with (output / "raw.bin").open("xb") as raw, (output / "chunks.jsonl").open("x") as chunks, (output / "frames.jsonl").open("x") as frames:
            while time.monotonic_ns() < deadline:
                wait = max(0, min(0.1, (deadline - time.monotonic_ns()) / 1e9))
                if not select.select([fd], [], [], wait)[0]:
                    continue
                try:
                    data = os.read(fd, 4096)
                except BlockingIOError:
                    continue
                if not data:
                    reason = "device_eof"
                    break
                stamp, utc = time.monotonic_ns(), utc_now()
                raw.write(data)
                chunks.write(json.dumps({"utc": utc, "monotonic_ns": stamp, "offset": received, "hex": data.hex(" ").upper()}) + "\n")
                received += len(data)
                current_frames = parser.feed(data)
                for frame in current_frames:
                    frame.update(utc=utc, monotonic_ns=stamp)
                    records.append(frame)
                    frames.write(json.dumps(frame) + "\n")
                raw.flush(); chunks.flush(); frames.flush()
                if any(r["enabled"] or r["state"] in (2, 3, 5) for r in current_frames):
                    # Never send a stop command. End observation and require on-site action.
                    reason = "enabled_or_motion_reported_site_check_required"
                    break
    except KeyboardInterrupt:
        reason = "operator_interrupted"
    except Exception as exc:
        error, reason = repr(exc), "capture_error"
    finally:
        if fd is not None:
            os.close(fd)  # keep raw/no-echo/HUPCL-off settings; no restore ioctl
    result = summary(records, parser)
    result.update(bytes_received=received, elapsed_s=(time.monotonic_ns() - start) / 1e9,
                  end_utc=utc_now(), stop_reason=reason, error=error,
                  serial_write_calls=0, device_open_attempted=True)
    dump_json(output / "summary.json", result)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if records and not error and reason == "duration_elapsed" else 2


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="mode", required=True)
    live = sub.add_parser("capture", help="Receive only, after on-site confirmation")
    live.add_argument("--port", default=DEFAULT_PORT)
    live.add_argument("--seconds", type=float, default=30)
    live.add_argument("--out", required=True)
    live.add_argument("--site-confirmation", required=True, help="Record the actual on-site confirmation")
    live.add_argument("--allow-partial-occupancy", action="store_true",
                      help="Accept limited /proc visibility; still reject known holders and use TIOCEXCL")
    offline = sub.add_parser("replay", help="Parse timestamped chunks without opening a device")
    offline.add_argument("chunks")
    offline.add_argument("--out")
    args = parser.parse_args()
    return capture(args) if args.mode == "capture" else replay(args)


if __name__ == "__main__":
    raise SystemExit(main())
