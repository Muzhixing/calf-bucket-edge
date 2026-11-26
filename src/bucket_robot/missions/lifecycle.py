"""Shared operator pause and latched-fault presentation."""

import time, json


class TaskPause(Exception):
    """Operator connection loss pauses an active mission instead of exiting it."""


def park_fault(folder, result):
    """Leave the task observable after a hard fault; never auto-enable/restart."""
    post = result.get("post_execution") or {}
    final = (
        post.get("final_state")
        or (
            result["approach"].get("final_state")
            if "approach" in result
            else result.get("final_state")
        )
        or {}
    )
    stop_error = (
        post.get("stop_error")
        or result.get("stop_error")
        or result.get("approach", {}).get("stop_error")
    )
    status = dict(
        phase="fault_paused",
        reason=result.get("error"),
        disabled_confirmed=bool(final)
        and not final.get("enabled", True)
        and final.get("state") == 0
        and not stop_error,
        stop_error=stop_error,
        automatic_resume=False,
    )
    (folder / "task_status.json").write_text(json.dumps(status, indent=2))
    print("TASK_PAUSED " + json.dumps(status), flush=True)
    while not (folder / "STOP").exists():
        try:
            time.sleep(0.1)
        except TaskPause:
            continue
        except KeyboardInterrupt:
            break


def stop_and_drain(port, workers, timeout=8.0):
    """Disable first; keep UART telemetry live while model/vision workers finish."""
    error = None
    try:
        if port is not None:
            port.stop()
            until = time.monotonic() + 1.7
            while port.phase == "stopping" and time.monotonic() < until:
                port.tick()
                time.sleep(0.005)
            if port.phase == "stopping" or port.state["enabled"]:
                raise RuntimeError("STOP_UNCONFIRMED_before_worker_drain")
        for worker in workers:
            event = getattr(worker, "end", None)
            if event is not None:
                event.set()
            elif hasattr(worker, "close"):
                worker.close()
        threads = [w.thread for w in workers if hasattr(w.thread, "is_alive")]
        until = time.monotonic() + timeout
        while any(t.is_alive() for t in threads):
            if port is not None:
                frames = port.poll()
                if any(f["enabled"] for f in frames):
                    raise RuntimeError("unexpected_enable_during_cleanup")
                if any(f["system_fault"] or f["motor_fault"] for f in frames):
                    raise RuntimeError("firmware_fault_during_cleanup")
            if time.monotonic() > until:
                raise RuntimeError("worker_cleanup_timeout")
            for thread in threads:
                thread.join(timeout=0.01)
            time.sleep(0.005)
    except BaseException as exc:
        error = repr(exc)
    return error
