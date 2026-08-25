"""Binary-searches the largest n for which try_fn(n) succeeds, by
trying values for real (double to bracket, then binary search)."""
import contextlib
import subprocess
import sys
import time
from collections.abc import Callable, Iterator

from placax.log import Log


def is_oom(e: Exception) -> bool:
    """True if e looks like a genuine out-of-memory error."""
    message = str(e)
    # Check known OOM signatures across plain Python, JAX/XLA, and the OS.
    return (
        isinstance(e, MemoryError)
        or "RESOURCE_EXHAUSTED" in message
        or "out of memory" in message.lower()
        or (isinstance(e, OSError) and "Cannot allocate memory" in message)
    )


@contextlib.contextmanager
def _rlimit_as(memory_limit_bytes: int | None) -> Iterator[None]:
    """Caps virtual address space so OOM raises MemoryError instead of an uncatchable SIGKILL (Unix only)."""
    if memory_limit_bytes is None or sys.platform == "win32":
        yield
        return

    import resource

    # Remember the current limit so we can put it back - RLIMIT_AS is process-wide, so we must
    # restore it even if the caller's code raises.
    soft, hard = resource.getrlimit(resource.RLIMIT_AS)
    new_soft = memory_limit_bytes if hard == resource.RLIM_INFINITY else min(memory_limit_bytes, hard)
    resource.setrlimit(resource.RLIMIT_AS, (new_soft, hard))
    try:
        yield
    finally:
        resource.setrlimit(resource.RLIMIT_AS, (soft, hard))


def _attempt(
    try_fn: Callable[[int], None], n: int, cleanup_fn: Callable[[], None] | None, verbose: bool
) -> bool:
    """Runs try_fn(n) once, returning True on success and False on OOM (other exceptions propagate)."""
    if verbose:
        Log.info(f"Trying n={n}...")
    start = time.monotonic()
    try:
        try_fn(n)
        if verbose:
            Log.info(f"  n={n}: ok ({time.monotonic() - start:.1f}s)")
        return True
    except Exception as e:
        # Only treat genuine OOM as "doesn't fit" - anything else is a real bug, so let it propagate.
        if is_oom(e):
            if verbose:
                Log.info(f"  n={n}: OOM ({time.monotonic() - start:.1f}s)")
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
    verbose: bool,
) -> tuple[int, int | None]:
    """Doubles n until it fails or exceeds max_candidate, to cheaply bracket the true boundary."""
    last_good, n = 0, start
    while n <= max_candidate:
        if _attempt(try_fn, n, cleanup_fn, verbose):
            # n still fits - remember it and try something bigger.
            last_good = n
            n *= 2
        else:
            # n was the first failure - we now have a (last_good, first_bad) bracket to search.
            return last_good, n
    # Doubled all the way past max_candidate without ever failing.
    return last_good, None


def _binary_search(
    try_fn: Callable[[int], None],
    last_good: int,
    first_bad: int,
    cleanup_fn: Callable[[], None] | None,
    verbose: bool,
) -> int:
    """Binary-searches (last_good, first_bad) down to the exact pass/fail boundary."""
    while first_bad - last_good > 1:
        mid = (last_good + first_bad) // 2
        if _attempt(try_fn, mid, cleanup_fn, verbose):
            last_good = mid  # mid fits - the boundary is somewhere above it
        else:
            first_bad = mid  # mid doesn't fit - the boundary is somewhere below it
    return last_good


def probe_subprocess(
    argv: list[str], timeout_s: float, oom_marker: str = "PLACAX_PROBE_OOM", env: dict | None = None
) -> None:
    """A try_fn for find_max_batch_size that runs argv as a subprocess instead of in-process, so a
    hung or crashing candidate can be killed without losing the calling process. argv must print
    oom_marker and exit nonzero on OOM, exit 0 on success."""
    try:
        result = subprocess.run(argv, capture_output=True, text=True, timeout=timeout_s, env=env)
    except subprocess.TimeoutExpired as e:
        # A hang is treated the same as "doesn't fit" - it's very unlikely to ever finish.
        raise MemoryError(f"probe exceeded {timeout_s:.0f}s, treating as infeasible") from e

    if result.returncode == 0:
        return
    if oom_marker in result.stdout:
        raise MemoryError("probe reported OOM")
    # Any other nonzero exit is a real bug in the probed code, not a capacity limit - surface it.
    raise RuntimeError(f"probe crashed (exit {result.returncode}):\n{result.stderr[-4000:]}")


def find_max_batch_size(
    try_fn: Callable[[int], None],
    max_candidate: int = 1024,
    start: int = 1,
    memory_limit_bytes: int | None = None,
    cleanup_fn: Callable[[], None] | None = None,
    verbose: bool = True,
) -> int:
    """Finds the largest n for which try_fn(n) succeeds, up to max_candidate."""
    with _rlimit_as(memory_limit_bytes):
        # 1. Grow exponentially to cheaply find a (last_good, first_bad) bracket around the boundary.
        last_good, first_bad = _grow_to_bracket(try_fn, start, max_candidate, cleanup_fn, verbose)
        if first_bad is None:  # never failed within max_candidate - nothing left to narrow
            return last_good
        # 2. Binary-search inside that bracket to pin down the exact boundary.
        return _binary_search(try_fn, last_good, first_bad, cleanup_fn, verbose)
