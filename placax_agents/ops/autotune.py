"""Finds the largest batch size that actually fits on this hardware by
trying values for real. Two-phase: exponential growth to bracket, then
binary search for the exact boundary (doubling alone would leave
capacity on the table).

OOM handling: CPU system-RAM exhaustion can SIGKILL the process before
Python sees anything - memory_limit_bytes sets a temporary soft RLIMIT_AS
(Unix only) to make it a catchable error. On GPU, JAX preallocates and
doesn't return memory between candidates, so cleanup_fn is called after
every attempt to release caches/garbage between trials."""
import sys
from collections.abc import Callable


def _is_oom(e: Exception) -> bool:
    """True if e is a genuine out-of-memory error."""
    message = str(e)
    return (
        isinstance(e, MemoryError)
        or "RESOURCE_EXHAUSTED" in message
        or "out of memory" in message.lower()
        or (isinstance(e, OSError) and "Cannot allocate memory" in message)
    )


def find_max_batch_size(
    try_fn: Callable[[int], None],
    max_candidate: int = 1024,
    start: int = 1,
    memory_limit_bytes: int | None = None,
    cleanup_fn: Callable[[], None] | None = None,
) -> int:
    """Largest n for which try_fn(n) succeeds. Real OOM errors stop the
    search; any other exception propagates - a genuine bug must not be
    mistaken for a memory limit. memory_limit_bytes applies only during
    the search (Unix only, restored afterward). cleanup_fn runs after
    every attempt (see module docstring)."""
    original_limits = None
    if memory_limit_bytes is not None and sys.platform != "win32":
        import resource

        original_limits = resource.getrlimit(resource.RLIMIT_AS)
        soft, hard = original_limits
        new_soft = memory_limit_bytes if hard == resource.RLIM_INFINITY else min(memory_limit_bytes, hard)
        resource.setrlimit(resource.RLIMIT_AS, (new_soft, hard))

    def attempt(n: int) -> bool:
        """True if n succeeded; False on OOM; anything else propagates."""
        print(f"Trying n={n}...")
        try:
            try_fn(n)
            return True
        except Exception as e:
            if _is_oom(e):
                return False
            raise
        finally:
            if cleanup_fn is not None:
                cleanup_fn()

    try:
        # Phase 1: exponential growth to find a (last_good, first_bad) bracket.
        last_good, first_bad, n = 0, None, start
        while n <= max_candidate:
            if attempt(n):
                last_good = n
                n *= 2
            else:
                first_bad = n
                break
        else:
            return last_good  # never failed within max_candidate

        # Phase 2: binary search inside the bracket.
        while first_bad - last_good > 1:
            mid = (last_good + first_bad) // 2
            if attempt(mid):
                last_good = mid
            else:
                first_bad = mid
        return last_good
    finally:
        if original_limits is not None:
            import resource

            resource.setrlimit(resource.RLIMIT_AS, original_limits)
