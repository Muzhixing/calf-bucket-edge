#!/usr/bin/env python3
"""Start the accepted field profile in a fresh run directory on RK3588.

--check is read-only. Starting a new run closes only a completed, acknowledged
disabled instance of this project's mission console. It never re-registers zeros.
"""

import argparse
from datetime import datetime, timedelta, timezone
import fcntl
import json
import os
from pathlib import Path
import signal
import socket
import sys
import time
from urllib.request import urlopen


DEFAULT_RELEASE = Path("/home/elf/bucket/releases/0.3.3-postgeometry-rc3-20261126")
RUN_ROOT = Path("/home/elf/bucket/runs")


def mission_processes():
    found = []
    for proc in Path("/proc").iterdir():
        if not proc.name.isdigit():
            continue
        try:
            args = (proc / "cmdline").read_bytes().decode().split("\0")
            if not any(
                args[i : i + 3] == ["-m", "bucket_robot", "mission"] for i in range(len(args) - 2)
            ):
                continue
            root = args[args.index("--release-root") + 1]
            if not Path(root).is_relative_to("/home/elf/bucket/releases"):
                continue
            found.append(int(proc.name))
        except (OSError, ValueError, IndexError, UnicodeError):
            continue
    return found


def check_previous(pids):
    if len(pids) > 1:
        raise RuntimeError("发现多个任务实例，请先处理占用，不自动结束它们。")
    if not pids:
        return None
    with urlopen("http://127.0.0.1:8788/status", timeout=3) as response:
        state = json.load(response)
    task = state.get("task") or {}
    latch = state.get("health", {}).get("motion_latch") or {}
    if (
        state.get("tracking", {}).get("active")
        or task.get("phase") not in ("complete", "stopped", "fault_paused")
        or task.get("disabled_confirmed") is not True
        or latch.get("code") == "STOP_UNCONFIRMED"
    ):
        raise RuntimeError("上一任务未确认停止。请先在控制台停止，并等待禁用确认。")
    return task["phase"]


def process_alive(pid):
    try:
        return (Path("/proc") / str(pid) / "stat").read_text().split(") ", 1)[1][0] != "Z"
    except FileNotFoundError:
        return False


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--check", action="store_true", help="只读检查入口与旧任务，不启动运动")
    args = ap.parse_args()
    release = DEFAULT_RELEASE
    sys.path.insert(0, str(release / "python"))
    from bucket_robot.runtime.release import ReleaseLayout

    manifest = ReleaseLayout(release).verify()
    pids = mission_processes()
    previous = check_previous(pids)
    print("验收版本：" + manifest["version"], flush=True)
    print("旧任务状态：" + (previous or "无旧任务"), flush=True)
    if args.check:
        print("只读检查通过，未停止进程、未打开控制串口、未启动运动。", flush=True)
        return 0

    RUN_ROOT.mkdir(parents=True, exist_ok=True)
    with (RUN_ROOT / ".manual-launch.lock").open("a") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        # Check again after taking the launch lock; another operator may have started.
        pids = mission_processes()
        check_previous(pids)
        for pid in pids:
            os.kill(pid, signal.SIGTERM)
            until = time.monotonic() + 10
            while process_alive(pid) and time.monotonic() < until:
                time.sleep(0.1)
            if process_alive(pid):
                raise RuntimeError("旧控制台尚未正常退出，未启动新任务。")
        with socket.socket() as probe:
            probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            probe.bind(("127.0.0.1", 8788))

        from bucket_robot.motion.arm import ArmPort, Limits, pose
        from bucket_robot.resources import default_reference

        # Read-only preflight; a disabled, uncommanded ArmPort closes without TX.
        port = ArmPort(Limits(default_reference()), lambda *event: None)
        try:
            initial = dict(port.state)
            if pose(initial) != [73, 0, 0]:
                raise RuntimeError("机构尚未复位到A73/B0/C0，当前为" + str(pose(initial)))
        finally:
            error = port.shutdown()
            if error:
                raise RuntimeError(error)

        stamp = datetime.now(timezone(timedelta(hours=8))).strftime("%Y%m%d_%H%M%S")
        out = RUN_ROOT / ("manual_" + stamp + "_" + str(os.getpid()))
        out.mkdir(exist_ok=False)
        command = [
            str(release / "run_mission.sh"),
            "--out",
            str(out / "automatic"),
            "--execute-post-geometry",
            "--wall-budget",
            "420",
            "--seconds",
            "1800",
        ]
        (out / "launch.json").write_text(
            json.dumps(
                dict(pid=os.getpid(), release=str(release), command=command, initial_state=initial),
                indent=2,
            )
            + "\n"
        )
        print("本轮日志：" + str(out), flush=True)
        print("控制台：http://127.0.0.1:8788/；远程侧栏刷新现有页面。", flush=True)
        print("开始自动任务；终端按Ctrl+C执行停车并退出。", flush=True)
        os.execv(command[0], command)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print("未启动任务：" + str(exc), file=sys.stderr)
        raise SystemExit(2)
