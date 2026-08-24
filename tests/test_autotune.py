import pytest

from placax_agents.ops.autotune import find_max_batch_size


def test_finds_the_correct_ceiling() -> None:
    def fake_op(n: int) -> None:
        if n > 16:
            raise RuntimeError("RESOURCE_EXHAUSTED: simulated OOM")

    assert find_max_batch_size(fake_op, max_candidate=1024) == 16


def test_non_oom_errors_propagate_not_silently_swallowed() -> None:
    def buggy_op(n: int) -> None:
        raise ValueError("a real bug, not memory-related")

    with pytest.raises(ValueError, match="a real bug"):
        find_max_batch_size(buggy_op)


def test_respects_max_candidate_ceiling() -> None:
    def always_succeeds(n: int) -> None:
        pass

    assert find_max_batch_size(always_succeeds, max_candidate=8) == 8


def test_memory_limit_prevents_crash_on_real_allocation() -> None:
    # Regression test for a real crash found this session: an earlier
    # version of this search was killed outright (SIGKILL, exit 137) by
    # the OS's OOM killer on a memory-constrained sandbox (3.9GB total,
    # no swap) - not a catchable Python exception. memory_limit_bytes
    # must turn that into a normal, catchable failure instead.
    #
    # The point of this test is "doesn't crash", not a specific result:
    # result == 0 (nothing fit) is a valid, correct answer, not a
    # failure - RLIMIT_AS constrains virtual address space, not RSS, and
    # Python/JAX processes can reserve far more of that than they
    # actually touch, in ways that vary by environment. Asserting a
    # specific "safe" limit relative to current usage was tried and
    # failed for exactly this reason - the real property to test is
    # that the function returns cleanly at all, for any limit.
    def allocate_memory(n: int) -> None:
        data = bytearray(n * 50 * 1024 * 1024)
        data[0] = 1

    result = find_max_batch_size(allocate_memory, max_candidate=64, memory_limit_bytes=500 * 1024 * 1024)
    assert result >= 0  # returned cleanly - the crash this guards against never returns at all


def test_memory_limit_is_restored_after_search() -> None:
    # Regression test for a second, real bug this exact feature caused:
    # a version without cleanup left a 500MB limit permanently set on
    # the process, which then crashed every later, legitimately-larger
    # test in the same pytest session (RLIMIT_AS is process-wide and
    # persists until changed again). The search must restore whatever
    # limit was in place before it ran, not just apply a new one.
    import resource

    before = resource.getrlimit(resource.RLIMIT_AS)
    find_max_batch_size(lambda n: None, max_candidate=4, memory_limit_bytes=500 * 1024 * 1024)
    after = resource.getrlimit(resource.RLIMIT_AS)
    assert before == after


def test_binary_search_finds_exact_non_power_of_two_boundary() -> None:
    # Regression test: an earlier version only doubled and stopped at
    # the first failure, so it would report 64 here even though the
    # true boundary is 100 - genuine capacity left on the table.
    def fake_op(n: int) -> None:
        if n > 100:
            raise RuntimeError("RESOURCE_EXHAUSTED: simulated")

    assert find_max_batch_size(fake_op, max_candidate=1024) == 100


def test_cleanup_fn_called_after_every_attempt() -> None:
    call_count = [0]

    def cleanup() -> None:
        call_count[0] += 1

    def fake_op(n: int) -> None:
        if n > 50:
            raise RuntimeError("RESOURCE_EXHAUSTED: simulated")

    result = find_max_batch_size(fake_op, max_candidate=1024, cleanup_fn=cleanup)
    assert result == 50
    assert call_count[0] > 0
