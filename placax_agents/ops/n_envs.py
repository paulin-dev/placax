"""Auto-detects how many parallel envs fit on this hardware, by probing
candidates for real in disposable subprocesses (see NEnvsDetector)."""
import importlib
import os
import pathlib
import sys

from placax._device import recommended_parallelism_mode  # must precede jax imports
from placax_agents.benchmark import Benchmark
from placax_agents.ops.autotune import find_max_batch_size, is_oom, probe_subprocess
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
    """Auto-detects (mode, n_envs) for parallel training on this
    hardware. policy_path/make_reward_fn_path are dotted import paths
    ("module.Name") so a fresh subprocess can rebuild the exact same
    policy/reward without any Python object crossing a process
    boundary - both default to this library's own CNNActorCritic /
    make_scaled_hpwl_reward, but any importable alternative works, with
    two constraints the subprocess relies on:
    policy_path must resolve to a zero-argument-constructible flax
    nn.Module implementing AlgorithmFn (.apply) via .init()/.apply() -
    it's instantiated as policy_path() with no arguments.
    make_reward_fn_path, if given, must resolve to a RewardFnFactory
    (see placax_agents.benchmark.RewardFnFactory).

    Each candidate n_envs is tried for real, but in a disposable
    subprocess (see probe_subprocess): JAX's own memory accounting is
    unusable after the first real GPU allocation (preallocation), and a
    hung or crashing candidate can be killed outright without losing
    the calling process."""

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
        """Returns (mode, n_envs). Only searches if parallel is worth
        trying on this hardware at all."""
        mode = recommended_parallelism_mode()
        if mode == "sequential":
            return "sequential", 1
        n_envs = find_max_batch_size(self._try_n_envs, max_candidate=self.max_candidate, verbose=self.verbose)
        return "parallel", max(n_envs, 1)

    def resolve(self, override: int | None = None) -> tuple[str, int]:
        """(mode, n_envs): override if given ("sequential" if <=1, else
        "parallel"), otherwise auto-detected via detect()."""
        if override is not None:
            return ("sequential" if override <= 1 else "parallel"), override
        return self.detect()

    def _try_n_envs(self, n: int) -> None:
        # preallocate=false: this fresh process must report real usage, not one big arena grab.
        env = {**os.environ, "XLA_PYTHON_CLIENT_PREALLOCATE": "false"}
        # Re-invoke this module as `python -m ...` - see the __main__ block below.
        argv = [
            sys.executable, "-m", _MODULE, str(self.benchmark_dir), self.policy_path,
            self.make_reward_fn_path or "", str(n),
        ]
        probe_subprocess(argv, timeout_s=self.timeout_s, oom_marker=_OOM_MARKER, env=env)


def _probe_entrypoint(benchmark_dir: str, policy_path: str, make_reward_fn_path: str, n: int) -> None:
    """Subprocess entry point (see NEnvsDetector._try_n_envs): attempts
    one n-env training step, reporting the outcome via exit code."""
    # Rebuild the benchmark/policy from scratch - nothing crosses the process boundary but strings.
    load_kwargs = {"make_reward_fn": _import(make_reward_fn_path)} if make_reward_fn_path else {}
    benchmark = Benchmark.load(pathlib.Path(benchmark_dir), **load_kwargs)
    policy = _import(policy_path)()
    variables = benchmark.init_policy(policy, random.PRNGKey(0))
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
