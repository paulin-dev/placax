"""Finds the largest batch size that actually fits on this hardware by
trying values for real. Two-phase: exponential growth to bracket, then
binary search for the exact boundary (doubling alone would leave
capacity on the table).

OOM handling: CPU system-RAM exhaustion can SIGKILL the process before
Python sees anything - memory_limit_bytes sets a temporary soft RLIMIT_AS
(Unix only) to make it a catchable error. On GPU, JAX preallocates and
doesn't return memory between candidates, so cleanup_fn is called after
every attempt to release caches/garbage between trials.

Each attempt is timed and its outcome printed as soon as it's known.
On GPU a single candidate can legitimately take a long time to compile
- XLA searches several kernel algorithms internally and discards the
ones that don't fit, which is also what the noisy allocator warnings
during a "Trying n=..." line are (see placax._device for silencing
those). Without per-attempt feedback, that compile time is
indistinguishable from a hang."""
import contextlib
import sys
import time
from collections.abc import Callable, Iterator


def _is_oom(e: Exception) -> bool:
    """True if e is a genuine out-of-memory error."""
    message = str(e)
    return (
        isinstance(e, MemoryError)
        or "RESOURCE_EXHAUSTED" in message
        or "out of memory" in message.lower()
        or (isinstance(e, OSError) and "Cannot allocate memory" in message)
    )


@contextlib.contextmanager
def _rlimit_as(memory_limit_bytes: int | None) -> Iterator[None]:
    """Temporarily caps virtual address space (Unix only) so a real OOM
    raises a catchable MemoryError instead of the OS SIGKILLing the
    process outright. Restores whatever limit was in place before,
    even if the search raises."""
    if memory_limit_bytes is None or sys.platform == "win32":
        yield
        return

    import resource

    soft, hard = resource.getrlimit(resource.RLIMIT_AS)
    new_soft = memory_limit_bytes if hard == resource.RLIM_INFINITY else min(memory_limit_bytes, hard)
    resource.setrlimit(resource.RLIMIT_AS, (new_soft, hard))
    try:
        yield
    finally:
        resource.setrlimit(resource.RLIMIT_AS, (soft, hard))


def _attempt(try_fn: Callable[[int], None], n: int, cleanup_fn: Callable[[], None] | None) -> bool:
    """Runs try_fn(n) once. True on success, False on a genuine OOM;
    any other exception propagates - a real bug must not be mistaken
    for a memory limit. Prints how long it took and how it went, so a
    slow (but working) candidate doesn't read as a stall."""
    print(f"Trying n={n}...", flush=True)
    start = time.monotonic()
    try:
        try_fn(n)
        print(f"  n={n}: ok ({time.monotonic() - start:.1f}s)", flush=True)
        return True
    except Exception as e:
        if _is_oom(e):
            print(f"  n={n}: OOM ({time.monotonic() - start:.1f}s)", flush=True)
            return False
        raise
    finally:
        if cleanup_fn is not None:
            cleanup_fn()


def _grow_to_bracket(
    try_fn: Callable[[int], None],
    start: int,
    max_candidate: int,
    cleanup_fn: Callable[[], None] | None,
) -> tuple[int, int | None]:
    """Doubles n until an attempt fails or max_candidate is exceeded.
    Returns (last_good, first_bad); first_bad is None if nothing failed
    within max_candidate, meaning last_good is already the answer."""
    last_good, n = 0, start
    while n <= max_candidate:
        if _attempt(try_fn, n, cleanup_fn):
            last_good = n
            n *= 2
        else:
            return last_good, n
    return last_good, None


def _binary_search(
    try_fn: Callable[[int], None], last_good: int, first_bad: int, cleanup_fn: Callable[[], None] | None
) -> int:
    """Narrows (last_good, first_bad) down to the exact boundary."""
    while first_bad - last_good > 1:
        mid = (last_good + first_bad) // 2
        if _attempt(try_fn, mid, cleanup_fn):
            last_good = mid
        else:
            first_bad = mid
    return last_good


def find_max_batch_size(
    try_fn: Callable[[int], None],
    max_candidate: int = 1024,
    start: int = 1,
    memory_limit_bytes: int | None = None,
    cleanup_fn: Callable[[], None] | None = None,
) -> int:
    """Largest n for which try_fn(n) succeeds. Real OOM errors stop the
    search; any other exception propagates. memory_limit_bytes applies
    only during the search (Unix only, restored afterward). cleanup_fn
    runs after every attempt (see module docstring)."""
    with _rlimit_as(memory_limit_bytes):
        last_good, first_bad = _grow_to_bracket(try_fn, start, max_candidate, cleanup_fn)
        if first_bad is None:
            return last_good
        return _binary_search(try_fn, last_good, first_bad, cleanup_fn)
