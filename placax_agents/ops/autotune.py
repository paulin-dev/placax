"""Binary-searches the largest n for which try_fn(n) succeeds, by
trying values for real (double to bracket, then binary search)."""
import contextlib
import subprocess
import sys
import time
from collections.abc import Callable, Iterator


def is_oom(e: Exception) -> bool:
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
    """Caps virtual address space so OOM raises MemoryError instead of
    an uncatchable SIGKILL (Unix only; restored on exit)."""
    if memory_limit_bytes is None or sys.platform == "win32":
        yield
        return

    import resource

    # RLIMIT_AS is process-wide - always restore it, even on error.
    soft, hard = resource.getrlimit(resource.RLIMIT_AS)
    new_soft = memory_limit_bytes if hard == resource.RLIM_INFINITY else min(memory_limit_bytes, hard)
    resource.setrlimit(resource.RLIMIT_AS, (new_soft, hard))
    try:
        yield
    finally:
        resource.setrlimit(resource.RLIMIT_AS, (soft, hard))


def _attempt(try_fn: Callable[[int], None], n: int, cleanup_fn: Callable[[], None] | None) -> bool:
    """Runs try_fn(n) once; True on success, False on OOM. Non-OOM
    exceptions propagate - a real bug isn't a capacity limit."""
    print(f"Trying n={n}...", flush=True)
    start = time.monotonic()
    try:
        try_fn(n)
        print(f"  n={n}: ok ({time.monotonic() - start:.1f}s)", flush=True)
        return True
    except Exception as e:
        if is_oom(e):
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
    """Doubles n until it fails or exceeds max_candidate. Returns
    (last_good, first_bad); first_bad is None if nothing failed."""
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
    """Narrows (last_good, first_bad) to the exact boundary."""
    while first_bad - last_good > 1:
        mid = (last_good + first_bad) // 2
        if _attempt(try_fn, mid, cleanup_fn):
            last_good = mid
        else:
            first_bad = mid
    return last_good


def probe_subprocess(
    argv: list[str], timeout_s: float, oom_marker: str = "PLACAX_PROBE_OOM", env: dict | None = None
) -> None:
    """A try_fn for find_max_batch_size that runs argv as a subprocess
    instead of in-process - for candidates that might hang or corrupt
    process-wide state (e.g. GPU memory) if tried directly. argv must
    print oom_marker and exit nonzero on OOM, exit 0 on success. Raises
    MemoryError on OOM or timeout (both mean "doesn't fit"); RuntimeError
    on any other crash."""
    try:
        result = subprocess.run(argv, capture_output=True, text=True, timeout=timeout_s, env=env)
    except subprocess.TimeoutExpired as e:
        raise MemoryError(f"probe exceeded {timeout_s:.0f}s, treating as infeasible") from e

    if result.returncode == 0:
        return
    if oom_marker in result.stdout:
        raise MemoryError("probe reported OOM")
    raise RuntimeError(f"probe crashed (exit {result.returncode}):\n{result.stderr[-4000:]}")


def find_max_batch_size(
    try_fn: Callable[[int], None],
    max_candidate: int = 1024,
    start: int = 1,
    memory_limit_bytes: int | None = None,
    cleanup_fn: Callable[[], None] | None = None,
) -> int:
    """Largest n for which try_fn(n) succeeds, up to max_candidate.
    cleanup_fn, if given, runs after every attempt."""
    with _rlimit_as(memory_limit_bytes):
        last_good, first_bad = _grow_to_bracket(try_fn, start, max_candidate, cleanup_fn)
        if first_bad is None:
            return last_good
        return _binary_search(try_fn, last_good, first_bad, cleanup_fn)
