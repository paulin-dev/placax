import subprocess

import pytest

from scripts.subprocess_search import _looks_like_oom, _parse_argv, sweep, try_value


def test_looks_like_oom_matches_known_signatures() -> None:
    assert _looks_like_oom("RESOURCE_EXHAUSTED: simulated")
    assert _looks_like_oom("some prefix\nOut Of Memory\nsome suffix")
    assert _looks_like_oom("MemoryError: could not allocate")
    assert _looks_like_oom("OSError: Cannot allocate memory")


def test_looks_like_oom_does_not_match_unrelated_errors() -> None:
    assert not _looks_like_oom("ValueError: a real bug, not memory-related")
    assert not _looks_like_oom("")


def test_parse_argv_extracts_the_sweep_flag_and_passes_the_rest_through() -> None:
    module, name, values, other_args, timeout_s = _parse_argv(
        ["scripts.run_maskplace", "--n_episodes=[1,2,4,8,10]", "--benchmark_dir=benchmarks/adaptec1", "--macro_budget=128"]
    )
    assert module == "scripts.run_maskplace"
    assert name == "n_episodes"
    assert values == ["1", "2", "4", "8", "10"]
    assert other_args == ["--benchmark_dir=benchmarks/adaptec1", "--macro_budget=128"]
    assert timeout_s == 900.0  # default


def test_parse_argv_accepts_an_explicit_timeout_anywhere() -> None:
    module, name, values, other_args, timeout_s = _parse_argv(
        ["some.module", "--timeout=45", "--n=[1,2]", "--extra=flag"]
    )
    assert timeout_s == 45.0
    assert other_args == ["--extra=flag"]


def test_parse_argv_requires_exactly_one_sweep_flag() -> None:
    with pytest.raises(SystemExit):
        _parse_argv(["some.module", "--benchmark_dir=x"])


def test_parse_argv_requires_a_module() -> None:
    with pytest.raises(SystemExit):
        _parse_argv([])


def test_try_value_succeeds_on_exit_zero(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_run(argv, **kwargs):
        assert argv[-2:] == ["--n=4", "--extra"]
        return subprocess.CompletedProcess(argv, returncode=0, stdout="", stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)
    assert try_value("some.module", "n", "4", ["--extra"], timeout_s=10.0) is True


def test_try_value_treats_oom_output_as_failure_not_error(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_run(argv, **kwargs):
        return subprocess.CompletedProcess(argv, returncode=1, stdout="", stderr="RESOURCE_EXHAUSTED: simulated")

    monkeypatch.setattr(subprocess, "run", fake_run)
    assert try_value("some.module", "n", "8", [], timeout_s=10.0) is False


def test_try_value_treats_signal_kill_as_infeasible(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_run(argv, **kwargs):
        return subprocess.CompletedProcess(argv, returncode=-6, stdout="", stderr="Aborted")

    monkeypatch.setattr(subprocess, "run", fake_run)
    assert try_value("some.module", "n", "8", [], timeout_s=10.0) is False


def test_try_value_propagates_a_real_bug_instead_of_swallowing_it(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_run(argv, **kwargs):
        return subprocess.CompletedProcess(argv, returncode=1, stdout="", stderr="ValueError: a real bug")

    monkeypatch.setattr(subprocess, "run", fake_run)
    with pytest.raises(RuntimeError, match="a real bug"):
        try_value("some.module", "n", "8", [], timeout_s=10.0)


def test_try_value_treats_a_timeout_as_infeasible(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_run(argv, **kwargs):
        raise subprocess.TimeoutExpired(cmd=argv, timeout=kwargs["timeout"])

    monkeypatch.setattr(subprocess, "run", fake_run)
    assert try_value("some.module", "n", "8", [], timeout_s=0.01) is False


def test_sweep_stops_at_the_first_failure_and_returns_the_largest_value_that_worked(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    attempted = []

    def fake_run(argv, **kwargs):
        flag = next(a for a in argv if a.startswith("--n="))
        attempted.append(flag)
        ok = int(flag.removeprefix("--n=")) <= 4
        return subprocess.CompletedProcess(
            argv, returncode=0 if ok else 1, stdout="", stderr="" if ok else "RESOURCE_EXHAUSTED"
        )

    monkeypatch.setattr(subprocess, "run", fake_run)
    result = sweep("some.module", "n", ["1", "2", "4", "8", "16"], [], timeout_s=10.0)
    assert result == "4"
    assert attempted == ["--n=1", "--n=2", "--n=4", "--n=8"]  # never tries 16, stopped at the first failure


def test_sweep_returns_none_if_even_the_first_value_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_run(argv, **kwargs):
        return subprocess.CompletedProcess(argv, returncode=1, stdout="", stderr="RESOURCE_EXHAUSTED")

    monkeypatch.setattr(subprocess, "run", fake_run)
    assert sweep("some.module", "n", ["1", "2"], [], timeout_s=10.0) is None
