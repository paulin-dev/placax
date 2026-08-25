"""Runs (or resumes) extended training on a real benchmark, checkpointing
and evaluating real HPWL along the way.

Usage: python scripts/run_training.py [benchmark_dir] [n_iterations] [n_envs]"""
import pathlib
import sys

from placax import _device  # noqa: F401  must precede jax imports
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
        print("auto-detecting mode and n_envs (probing candidates in disposable subprocesses) ...")
    mode, n_envs = NEnvsDetector(benchmark_dir).resolve(n_envs_override)
    source = "given explicitly" if n_envs_override is not None else "auto-detected"
    print(f"  -> mode={mode}, n_envs={n_envs} ({source})")
    return mode, n_envs


def _print_results(
    log: list[dict], checkpoint_path: pathlib.Path, snapshot_dir: pathlib.Path, log_path: pathlib.Path
) -> None:
    print()
    print("iteration    loss           real_hpwl")
    for entry in log:
        hpwl_str = f"{entry['real_hpwl']:.1f}" if entry["real_hpwl"] is not None else "-"
        print(f"{entry['iteration']:>9}    {entry['loss']:>10.4f}    {hpwl_str}")

    print()
    print(f"checkpoint saved to {checkpoint_path} - re-run this script to continue training.")
    print(f"snapshots (never overwritten) saved to {snapshot_dir}/")
    print(f"full history saved to {log_path}")


def main() -> None:
    # 1. args + benchmark
    benchmark_dir, n_iterations, n_envs_override = _parse_args(sys.argv)
    if not benchmark_dir.exists():
        print(f"'{benchmark_dir}' not found - run scripts/download_benchmarks.py first.")
        sys.exit(1)

    print(f"loading {benchmark_dir} ...")
    benchmark = Benchmark.load(benchmark_dir)
    print(f"  {len(benchmark.macro_sizes)} macros, {len(benchmark.nets)} nets, cell_size={benchmark.cell_size:.2f}")

    # 2. policy
    policy = CNNActorCritic()
    key = random.PRNGKey(0)
    key, init_key = random.split(key)
    variables = benchmark.init_policy(policy, init_key)

    # 3. n_envs (explicit or auto-detected)
    checkpoint_path = benchmark_dir / "checkpoint.bin"
    snapshot_dir = benchmark_dir / "snapshots"
    log_path = benchmark_dir / "training_log.jsonl"
    mode, n_envs = _resolve_n_envs(benchmark_dir, n_envs_override)

    resuming = checkpoint_path.exists()
    print(f"{'resuming from' if resuming else 'starting fresh, will save to'} {checkpoint_path}")
    print(f"running {n_iterations} more iterations ...")

    # 4. train, resuming from checkpoint_path if it exists
    _final_variables, log = resumable_train(
        checkpoint_path, variables, key, policy.apply, benchmark.params, benchmark.reward_fn,
        benchmark.sizes_array, benchmark.cell_size, n_iterations,
        benchmark.padded_pin_idx, benchmark.padded_pin_offset, benchmark.valid_mask,
        checkpoint_every=10, eval_every=10, log_path=log_path,
        n_envs=n_envs, mode=mode, snapshot_dir=snapshot_dir, snapshot_every=50,
    )
    _print_results(log, checkpoint_path, snapshot_dir, log_path)


if __name__ == "__main__":
    main()
