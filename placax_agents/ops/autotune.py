"""Finds the largest batch size that actually fits on the current
hardware, by trying values for real - not estimating from theoretical
memory math (the kind of estimate that already proved wrong once this
session, for wiremask). General: takes any callable to test, not tied
to a specific placax operation.

Two-phase search: exponential growth (1, 2, 4, 8, ...) to find a rough
bracket, then binary search between the last success and first failure
to find the exact boundary - plain doubling alone would stop at the
last power-of-two success and leave real capacity on the table (e.g.
if 64 works and 128 fails, the true max could be 100).

On CPU, exhausting *system* RAM can trigger the OS's own OOM killer,
which SIGKILLs the whole process before Python ever gets a chance to
catch anything - confirmed directly: an early version of this search
was killed outright (exit 137), not caught. memory_limit mitigates this
with resource.setrlimit (Unix only): a soft per-process ceiling turns
an OOM into a normal, catchable Python error.

On GPU, an earlier version of this docstring claimed exhausting VRAM
"raises a clean, catchable XLA RESOURCE_EXHAUSTED error" - confirmed
FALSE by a real run: JAX preallocates a large, fixed fraction of VRAM
on first use and does not release memory back between candidates
within one process by default (each candidate is a different array
shape, triggering a fresh JIT compilation whose buffers linger) - so
candidates weren't tested independently, and XLA's own internal retry-
with-backoff hung for 6+ minutes with no exception ever raised.
cleanup_fn addresses this directly: called after every attempt, it
lets the caller release whatever needs releasing between candidates
(e.g. jax.clear_caches() + deleting live arrays + gc.collect() - the
real, documented combination from jax-ml/jax#19429, not a guess) -
kept optional and generic rather than hardcoded, since this function
isn't JAX-specific."""
import sys
from collections.abc import Callable


def _is_oom(e: Exception) -> bool:
    message = str(e)
    if "RESOURCE_EXHAUSTED" in message or "out of memory" in message.lower():
        return True
    if isinstance(e, MemoryError):
        return True
    if isinstance(e, OSError) and "Cannot allocate memory" in message:
        return True
    return False


def find_max_batch_size(
    try_fn: Callable[[int], None],
    max_candidate: int = 1024,
    start: int = 1,
    memory_limit_bytes: int | None = None,
    cleanup_fn: Callable[[], None] | None = None,
) -> int:
    """Returns the largest n for which try_fn(n) succeeds, found via
    exponential search (doubling from `start`) then binary-search
    refinement between the last success and first failure. Real out-of-
    memory errors stop the search cleanly; any other error propagates -
    a genuine bug shouldn't be silently treated as a memory limit.

    memory_limit_bytes, if given, sets a soft address-space ceiling for
    the duration of the search only (Unix only - no-op elsewhere),
    restored afterward: leaving it in place permanently would be a
    surprising side effect on the caller's process - confirmed directly,
    a version of this function without the restore leaked a 500MB limit
    into every later test in the same process, crashing the whole suite.

    cleanup_fn, if given, is called after every single attempt
    (success or failure) - see the module docstring for why this
    matters on GPU."""
    original_limits = None
    if memory_limit_bytes is not None and sys.platform != "win32":
        import resource

        original_limits = resource.getrlimit(resource.RLIMIT_AS)
        soft, hard = original_limits
        new_soft = memory_limit_bytes if hard == resource.RLIM_INFINITY else min(memory_limit_bytes, hard)
        resource.setrlimit(resource.RLIMIT_AS, (new_soft, hard))

    def attempt(n: int) -> bool:
        """True if n succeeded, False if it failed with a real OOM.
        Any other exception propagates - never silently swallowed."""
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
        # Phase 1: exponential growth to find a rough (last_good, first_bad) bracket.
        last_good = 0
        first_bad = None
        n = start
        while n <= max_candidate:
            if attempt(n):
                last_good = n
                n *= 2
            else:
                first_bad = n
                break
        else:
            return last_good  # never failed within max_candidate

        if first_bad is None:
            return last_good

        # Phase 2: binary search between last_good (known to work) and
        # first_bad (known to fail) to find the exact boundary.
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
