"""Auto-detects how many parallel envs fit on this hardware, by probing
candidates for real in disposable subprocesses (see NEnvsDetector)."""
import importlib
import os
import pathlib
import sys

from placax._device import recommended_parallelism_mode  # must precede jax imports
from placax_agents.benchmark import Benchmark
from placax_agents.ops.autotune import find_max_via_subprocess, is_oom
from placax_agents.training.loops.parallel_train import train_parallel

from jax import random

_MODULE = "placax_agents.ops.n_envs"
_OOM_MARKER = "PLACAX_PROBE_OOM"
_DEFAULT_POLICY_PATH = "placax_agents.policy.architectures.cnn.CNNActorCritic"


def _import(dotted_path: str):
    """Imports "module.submodule.Name" and returns Name."""
    module_path, _, name = dotted_path.rpartition(".")
    return getattr(importlib.import_module(module_path), name)


class NEnvsDetector:
    """Auto-detects (mode, n_envs) for parallel training on this hardware by probing candidates in subprocesses.

    policy_path/make_reward_fn_path are dotted import paths ("module.Name") so each disposable
    subprocess can rebuild the exact same policy/reward from nothing but strings."""

    def __init__(
        self,
        benchmark_dir: pathlib.Path,
        policy_path: str = _DEFAULT_POLICY_PATH,
        make_reward_fn_path: str | None = None,
        max_candidate: int = 64,
        timeout_s: float = 90.0,
        verbose: bool = True,
    ):
        self.benchmark_dir = benchmark_dir
        self.policy_path = policy_path
        self.make_reward_fn_path = make_reward_fn_path
        self.max_candidate = max_candidate
        self.timeout_s = timeout_s
        self.verbose = verbose

    def detect(self) -> tuple[str, int]:
        """Auto-detects (mode, n_envs), only searching if parallel training is worth trying on this hardware."""
        mode = recommended_parallelism_mode()
        if mode == "sequential":
            return "sequential", 1
        # preallocate=false: each fresh probe process must report real usage, not one big arena grab.
        probe_env = {**os.environ, "XLA_PYTHON_CLIENT_PREALLOCATE": "false"}
        n_envs = find_max_via_subprocess(
            _MODULE, [str(self.benchmark_dir), self.policy_path, self.make_reward_fn_path or ""],
            max_candidate=self.max_candidate, timeout_s=self.timeout_s, oom_marker=_OOM_MARKER,
            env=probe_env, verbose=self.verbose,
        )
        return "parallel", max(n_envs, 1)

    def resolve(self, override: int | None = None) -> tuple[str, int]:
        """Returns (mode, n_envs): the override if given, otherwise auto-detected via detect()."""
        if override is not None:
            return ("sequential" if override <= 1 else "parallel"), override
        return self.detect()


def _rebuild_benchmark_and_policy(benchmark_dir: str, policy_path: str, make_reward_fn_path: str):
    """Rebuilds (benchmark, policy, variables) from scratch - nothing crosses the process boundary but strings."""
    # Only pass make_reward_fn through if the caller actually specified one - Benchmark.load has its own default.
    load_kwargs = {"make_reward_fn": _import(make_reward_fn_path)} if make_reward_fn_path else {}
    benchmark = Benchmark.load(pathlib.Path(benchmark_dir), **load_kwargs)
    policy = _import(policy_path)()
    variables = benchmark.init_policy(policy, random.PRNGKey(0))
    return benchmark, policy, variables


def _probe_entrypoint(benchmark_dir: str, policy_path: str, make_reward_fn_path: str, n: int) -> None:
    """Subprocess entry point: attempts one n-env training step, reporting the outcome via exit code."""
    benchmark, policy, variables = _rebuild_benchmark_and_policy(benchmark_dir, policy_path, make_reward_fn_path)
    try:
        train_parallel(
            random.PRNGKey(0), variables, policy.apply, benchmark.params, benchmark.reward_fn,
            benchmark.sizes_array, benchmark.cell_size, n_envs=n, n_iterations=1,
        )
    except Exception as e:
        if is_oom(e):  # doesn't fit - report it and let the parent see the exit code
            print(_OOM_MARKER)
            sys.exit(1)
        raise  # anything else is a real bug - propagate with its traceback


if __name__ == "__main__":
    _probe_entrypoint(sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4]))
