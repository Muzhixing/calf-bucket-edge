#!/usr/bin/env python3
"""Start a fresh mission, initializing a newly booted STM32 under the user policy.

--check is read-only. Existing valid origins are reused; new registration uses
disabled-only commands and a persistent at-most-once ledger.
"""

import argparse
from datetime import datetime, timedelta, timezone
import fcntl
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time
from urllib.request import urlopen


_BUNDLED_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_RELEASE = (
    _BUNDLED_ROOT
    if (Path(__file__).parent.name == "field" and (_BUNDLED_ROOT / "manifest.json").is_file())
    else Path("/home/elf/bucket/releases/0.3.3")
)
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
    if latch.get("code") == "STOP_UNCONFIRMED":
        raise RuntimeError("上一任务停止状态未确认，禁止自动接管。")
    readonly = b"--observe-only" in (Path("/proc") / str(pids[0]) / "cmdline").read_bytes().split(
        b"\0"
    )
    if (
        readonly
        and state.get("health", {}).get("motion_enabled") is False
        and not state.get("tracking", {}).get("active")
    ):
        return "只读预览"
    if (
        state.get("tracking", {}).get("active")
        or task.get("phase") not in ("complete", "stopped", "fault_paused")
        or task.get("disabled_confirmed") is not True
        or latch.get("code") == "STOP_UNCONFIRMED"
    ):
        raise RuntimeError("上一任务未确认停止。请先在控制台停止，并等待禁用确认。")
    return task["phase"]


def restore_preview(release, out):
    command = [
        str(release / "run_mission.sh"),
        "--out",
        str(out / "preview"),
        "--observe-only",
        "--startup-result",
        str(out / "initialization_result.json"),
        "--seconds",
        "1800",
    ]
    with (out / "preview.log").open("xb") as stream:
        process = subprocess.Popen(
            command,
            stdin=subprocess.DEVNULL,
            stdout=stream,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    (out / "preview_launch.json").write_text(
        json.dumps(dict(pid=process.pid, command=command), indent=2)
    )
    return process.pid


def process_alive(pid):
    try:
        return (Path("/proc") / str(pid) / "stat").read_text().split(") ", 1)[1][0] != "Z"
    except FileNotFoundError:
        return False


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--check", action="store_true", help="只读检查入口与旧任务，不启动运动")
    ap.add_argument(
        "--initialize-only", action="store_true", help="仅恢复STM32零点，随后保持只读预览"
    )
    ap.add_argument(
        "--prepare-only", action="store_true", help="登记或复用零点并收回机构，不启动底盘任务"
    )
    args = ap.parse_args()
    if args.initialize_only and args.prepare_only:
        ap.error("--initialize-only与--prepare-only不能同时使用")
    release = DEFAULT_RELEASE
    sys.path.insert(0, str(release / "python"))
    from bucket_robot.runtime.release import ReleaseLayout

    manifest = ReleaseLayout(release).verify()
    from bucket_robot.runtime.preparation import initialize, read_policy, StartupResult
    from bucket_robot.runtime.startup import default_startup_state

    read_policy(Path(__file__).with_name("startup_policy.json"))
    pids = mission_processes()
    previous = check_previous(pids)
    print("验收版本：" + manifest["version"], flush=True)
    print("旧任务状态：" + (previous or "无旧任务"), flush=True)
    if args.check:
        print("只读检查通过，未停止进程、未打开控制串口、未启动运动。", flush=True)
        return 0

    def interrupted(signum, frame):
        raise KeyboardInterrupt("signal_" + str(signum))

    for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
        signal.signal(sig, interrupted)

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

        from bucket_robot.motion.arm import Limits
        from bucket_robot.resources import default_reference

        stamp = datetime.now(timezone(timedelta(hours=8))).strftime("%Y%m%d_%H%M%S")
        out = RUN_ROOT / ("manual_" + stamp + "_" + str(os.getpid()))
        out.mkdir(exist_ok=False)
        with (out / "initialization.jsonl").open("x", buffering=1) as stream:

            def log(kind, data):
                stream.write(
                    json.dumps(dict(monotonic=time.monotonic(), kind=kind, data=data)) + "\n"
                )
                if kind == "registered_home_return_started":
                    print("正在自动归位：收回C、B回顶、A回73°；底盘保持停车。", flush=True)
                elif kind == "registered_home_return_completed":
                    print("机构归位完成，已确认输出禁用。", flush=True)

            try:
                initialized = initialize(
                    Limits(default_reference()),
                    default_startup_state(),
                    log,
                    return_home=not args.initialize_only,
                )
            except Exception as exc:
                (out / "initialization_result.json").write_text(
                    json.dumps(
                        getattr(
                            exc, "result", StartupResult(False, False, error=str(exc))
                        ).as_dict(),
                        indent=2,
                    )
                )
                restore_preview(release, out)
                raise
        (out / "initialization_result.json").write_text(json.dumps(initialized, indent=2))
        initial = initialized["final_state"]
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
        if args.initialize_only or args.prepare_only:
            restore_preview(release, out)
            print("准备完成，输出禁用；已恢复只读预览，不启动底盘任务。", flush=True)
            return 0
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
