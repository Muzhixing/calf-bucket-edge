#!/usr/bin/env python3
"""Explicitly authorized one-shot F407 disable probe; no other command available."""

import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import select
import termios
import time

from bucket_robot.transport.telemetry import (
    DEFAULT_PORT,
    PROTOCOL_COMMIT,
    Parser,
    dump_json,
    holders,
    summary,
    utc_now,
)

DISABLE = bytes.fromhex("B3 00 00 00 5A 00 00 00 00 B4")


def probe(port, output, authorization, before=3.0, after=12.0, occupancy_check=holders):
    occupancy = occupancy_check(port, allow_partial=True)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    dump_json(
        output / "manifest.json",
        {
            "start_utc": utc_now(),
            "requested_port": port,
            "resolved_port": os.path.realpath(port),
            "protocol_source_commit": PROTOCOL_COMMIT,
            "authorization": authorization,
            "command": "disable_once",
            "tx_hex": DISABLE.hex(" ").upper(),
            "baud": 115200,
            "format": "8N1",
            "before_s": before,
            "after_s": after,
            "occupancy": occupancy,
            "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "firmware_identity_basis": "user-reported; no flash readback",
        },
    )
    parser, records, fd, count = Parser(), [], None, 0
    tx, error = {"attempted": False, "write_calls": 0}, None
    start = time.monotonic_ns()
    try:
        fd = os.open(port, os.O_RDWR | os.O_NOCTTY | os.O_NONBLOCK)
        fcntl.ioctl(fd, termios.TIOCEXCL)
        attrs = termios.tcgetattr(fd)
        attrs[0] = attrs[1] = attrs[3] = 0
        attrs[2] = termios.CS8 | termios.CREAD | termios.CLOCAL
        attrs[4] = attrs[5] = termios.B115200
        attrs[6][termios.VMIN] = attrs[6][termios.VTIME] = 0
        termios.tcsetattr(fd, termios.TCSANOW, attrs)
        # No flush, break, flow control, reset, DTR/RTS operation or exit command.
        with (
            (output / "raw.bin").open("xb") as raw,
            (output / "chunks.jsonl").open("x") as chunks,
            (output / "frames.jsonl").open("x") as frames,
        ):
            for phase, duration in (("before", before), ("after", after)):
                if phase == "after":
                    tx = {
                        "attempted": True,
                        "write_calls": 1,
                        "utc": utc_now(),
                        "monotonic_ns": time.monotonic_ns(),
                        "hex": DISABLE.hex(" ").upper(),
                    }
                    dump_json(output / "tx.json", tx)
                    # Exactly one write. Partial write or error is never retried.
                    tx["bytes_accepted_by_host"] = os.write(fd, DISABLE)
                    tx["write_return_utc"] = utc_now()
                    dump_json(output / "tx.json", tx)
                deadline = time.monotonic() + duration
                while time.monotonic() < deadline:
                    if not select.select(
                        [fd], [], [], max(0, min(0.1, deadline - time.monotonic()))
                    )[0]:
                        continue
                    try:
                        data = os.read(fd, 4096)
                    except BlockingIOError:
                        continue
                    if not data:
                        raise RuntimeError("Serial EOF/disconnection")
                    stamp, utc = time.monotonic_ns(), utc_now()
                    raw.write(data)
                    chunks.write(
                        json.dumps(
                            {
                                "utc": utc,
                                "monotonic_ns": stamp,
                                "phase": phase,
                                "offset": count,
                                "hex": data.hex(" ").upper(),
                            }
                        )
                        + "\n"
                    )
                    count += len(data)
                    for item in parser.feed(data):
                        item.update(utc=utc, monotonic_ns=stamp, phase=phase)
                        records.append(item)
                        frames.write(json.dumps(item) + "\n")
                    raw.flush()
                    chunks.flush()
                    frames.flush()
    except (Exception, KeyboardInterrupt) as exc:
        error = repr(exc)
    finally:
        if fd is not None:
            os.close(fd)
    result = summary(records, parser)
    result.update(
        end_utc=utc_now(),
        elapsed_s=(time.monotonic_ns() - start) / 1e9,
        bytes_received=count,
        tx=tx,
        error=error,
        valid_frames_before=sum(r["phase"] == "before" for r in records),
        valid_frames_after=sum(r["phase"] == "after" for r in records),
        interpretation="Host write success alone does not prove MCU receipt or disable; telemetry is not a per-packet ACK",
    )
    dump_json(output / "summary.json", result)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return result


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--port", default=DEFAULT_PORT)
    ap.add_argument("--out", required=True)
    ap.add_argument("--authorization", required=True)
    args = ap.parse_args()
    result = probe(args.port, args.out, args.authorization)
    raise SystemExit(0 if result["valid_frames_after"] and not result["error"] else 2)
