"""Explicit cleanup results shared by perception workers and task owners."""

from dataclasses import dataclass


@dataclass(frozen=True)
class CleanupResult:
    stopped: bool
    pending: tuple[str, ...] = ()


def worker_threads(worker):
    """Include nested identity work, even after its outer observer has exited."""
    own = getattr(worker, "thread", None)
    threads = [own] if own is not None and hasattr(own, "is_alive") else []
    for child in getattr(worker, "cleanup_children", lambda: ())():
        threads.extend(worker_threads(child))
    return threads
