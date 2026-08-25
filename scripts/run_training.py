"""Runs (or resumes) extended training on a real benchmark, checkpointing
and evaluating real HPWL along the way.

Usage: python scripts/run_training.py [benchmark_dir] [n_iterations] [n_envs]"""
import pathlib
import sys

from placax import _device  # noqa: F401  must precede jax imports
from placax.log import Log
from placax_agents.benchmark import Benchmark
from placax_agents.ops.n_envs import NEnvsDetector
from placax_agents.ops.resumable_train import resumable_train
from placax_agents.policy.architectures.cnn import CNNActorCritic

from jax import random


def _parse_args(argv: list[str]) -> tuple[pathlib.Path, int, int | None]:
    """(benchmark_dir, n_iterations, n_envs_override) from sys.argv."""
    benchmark_dir = pathlib.Path(argv[1] if len(argv) > 1 else "benchmarks/adaptec1")
    n_iterations = int(argv[2]) if len(argv) > 2 else 100
    n_envs_override = int(argv[3]) if len(argv) > 3 else None
    return benchmark_dir, n_iterations, n_envs_override


def _resolve_n_envs(benchmark_dir: pathlib.Path, n_envs_override: int | None) -> tuple[str, int]:
    """(mode, n_envs) via NEnvsDetector.resolve(); prints progress since
    the auto-detect path can take a while."""
    if n_envs_override is None:
        Log.info("auto-detecting mode and n_envs (probing candidates in disposable subprocesses) ...")
    mode, n_envs = NEnvsDetector(benchmark_dir).resolve(n_envs_override)
    source = "given explicitly" if n_envs_override is not None else "auto-detected"
    Log.info(f"  -> mode={mode}, n_envs={n_envs} ({source})")
    return mode, n_envs


def _print_summary(checkpoint_path: pathlib.Path, snapshot_dir: pathlib.Path, log_path: pathlib.Path) -> None:
    # Closing report, not log messages - each iteration already streamed live
    # via resumable_train's Log.info(), so this isn't a duplicate recap.
    print()
    print(f"checkpoint saved to {checkpoint_path} - re-run this script to continue training.")
    print(f"snapshots (never overwritten) saved to {snapshot_dir}/")
    print(f"full history saved to {log_path}")


def _load_benchmark(benchmark_dir: pathlib.Path) -> Benchmark:
    Log.info(f"loading {benchmark_dir} ...")
    benchmark = Benchmark.load(benchmark_dir)
    Log.info(f"  {len(benchmark.macro_sizes)} macros, {len(benchmark.nets)} nets, cell_size={benchmark.cell_size:.2f}")
    return benchmark


def _output_paths(benchmark_dir: pathlib.Path) -> tuple[pathlib.Path, pathlib.Path, pathlib.Path]:
    """(checkpoint_path, snapshot_dir, log_path), all under their own
    output subdirectory - benchmark_dir itself holds the input netlist
    and shouldn't get mixed up with run artifacts."""
    output_dir = benchmark_dir / "output"
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir / "checkpoint.bin", output_dir / "snapshots", output_dir / "training_log.jsonl"


def main() -> None:
    Log.configure()

    benchmark_dir, n_iterations, n_envs_override = _parse_args(sys.argv)
    if not benchmark_dir.exists():
        Log.error(f"'{benchmark_dir}' not found - run scripts/download_benchmarks.py first.")
        sys.exit(1)

    benchmark = _load_benchmark(benchmark_dir)

    policy = CNNActorCritic()
    key = random.PRNGKey(0)
    key, init_key = random.split(key)
    variables = benchmark.init_policy(policy, init_key)

    checkpoint_path, snapshot_dir, log_path = _output_paths(benchmark_dir)
    mode, n_envs = _resolve_n_envs(benchmark_dir, n_envs_override)

    resuming = checkpoint_path.exists()
    Log.info(f"{'resuming from' if resuming else 'starting fresh, will save to'} {checkpoint_path}")
    Log.info(f"running {n_iterations} more iterations ...")

    _final_variables, _log = resumable_train(
        checkpoint_path, variables, key, policy.apply, benchmark.params, benchmark.reward_fn,
        benchmark.sizes_array, benchmark.cell_size, n_iterations,
        benchmark.padded_pin_idx, benchmark.padded_pin_offset, benchmark.valid_mask,
        checkpoint_every=10, eval_every=10, log_path=log_path,
        n_envs=n_envs, mode=mode, snapshot_dir=snapshot_dir, snapshot_every=50,
    )
    _print_summary(checkpoint_path, snapshot_dir, log_path)


if __name__ == "__main__":
    main()
