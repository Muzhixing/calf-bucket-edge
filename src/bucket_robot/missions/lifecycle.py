"""Shared operator pause and latched-fault presentation."""

import time, json


class TaskPause(Exception):
    """Operator connection loss pauses an active mission instead of exiting it."""


def park_fault(folder, result):
    """Leave the task observable after a hard fault; never auto-enable/restart."""
    final = (
        result["approach"].get("final_state") if "approach" in result else result.get("final_state")
    ) or {}
    stop_error = result.get("stop_error") or result.get("approach", {}).get("stop_error")
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
