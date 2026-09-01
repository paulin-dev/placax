"""Runs (or resumes) extended training on a real benchmark, checkpointing and evaluating real HPWL along the way."""
import argparse
import functools
import pathlib
import sys

from placax import _device  # noqa: F401  must precede jax imports
from placax.log import Log
from placax_agents.benchmark import Benchmark
from placax_agents.ops.resumable_train import resumable_train
from placax_agents.policy.architectures.cnn import CNNActorCritic
from placax_agents.policy.observation import observation

from jax import random


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run (or resume) training on a real benchmark.")
    parser.add_argument("--benchmark_dir", type=pathlib.Path, default=pathlib.Path("benchmarks/adaptec1"))
    parser.add_argument("--n_iterations", type=int, default=100)
    parser.add_argument(
        "--n_envs", type=int, default=1,
        help="Parallel envs per training step (default: 1). See this module's docstring for how "
             "to find the largest value your hardware supports.",
    )
    parser.add_argument(
        "--mode", choices=["sequential", "parallel"], default=None,
        help="Force sequential/parallel training-step implementation (default: auto-detected "
             "from the JAX backend - CPU picks sequential, GPU/TPU picks parallel).",
    )
    parser.add_argument(
        "--eval_every", type=int, default=10,
        help="Compute real HPWL (a full extra greedy rollout) every this many iterations (default: 10).",
    )
    parser.add_argument(
        "--no_checkpoint", action="store_true",
        help="Don't read or write checkpoint.bin - useful for a quick, disposable run (e.g. "
             "when probing via scripts/subprocess_search.py) that shouldn't resume from or "
             "leave behind any state.",
    )
    parser.add_argument(
        "--placement_images", action="store_true",
        help="Also write a placement snapshot PNG on every --eval_every iteration (default "
             "location: <output_dir>/placements/<iteration>.png) - reuses that iteration's "
             "already-scheduled eval rollout, so this adds no extra rollout, just one image write "
             "per eval.",
    )
    parser.add_argument(
        "--placement_images_dir", type=pathlib.Path, default=None,
        help="Where to write placement snapshots (implies --placement_images; default: "
             "<output_dir>/placements).",
    )
    return parser.parse_args(argv[1:])


def _output_paths(benchmark_dir: pathlib.Path) -> tuple[pathlib.Path | None, pathlib.Path, pathlib.Path]:
    """Returns (checkpoint_path, snapshot_dir, log_path); checkpoint_path is None if --no_checkpoint."""
    output_dir = benchmark_dir / "output"
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir / "checkpoint.bin", output_dir / "snapshots", output_dir / "training_log.jsonl"


def main() -> None:
    """CLI entry point: loads a benchmark, builds a policy, and runs/resumes training on it."""
    Log.configure()
    args = _parse_args(sys.argv)
    if not args.benchmark_dir.exists():
        Log.error(f"'{args.benchmark_dir}' not found - run scripts/download_benchmarks.py first.")
        sys.exit(1)

    Log.info(f"loading {args.benchmark_dir} ...")
    benchmark = Benchmark.load(args.benchmark_dir)
    Log.info(f"  {len(benchmark.macro_sizes)} macros, {len(benchmark.nets)} nets, cell_size={benchmark.cell_size:.2f}")

    # A fresh policy; any resuming from a checkpoint happens later, inside resumable_train itself.
    policy = CNNActorCritic()
    key = random.PRNGKey(0)
    key, init_key = random.split(key)
    variables = benchmark.init_policy(policy, init_key)

    checkpoint_path, snapshot_dir, log_path = _output_paths(args.benchmark_dir)
    output_dir = checkpoint_path.parent
    if args.no_checkpoint:
        checkpoint_path = None
        placement_images_dir = args.placement_images_dir  # only if explicit - no output_dir to default into
    else:
        resuming = checkpoint_path.exists()
        Log.info(f"{'resuming from' if resuming else 'starting fresh, will save to'} {checkpoint_path}")
        placement_images_dir = args.placement_images_dir or (
            output_dir / "placements" if args.placement_images else None
        )
    if placement_images_dir is not None:
        Log.info(f"writing a placement snapshot every {args.eval_every} iterations to {placement_images_dir}")
    Log.info(f"running {args.n_iterations} more iterations (n_envs={args.n_envs}, mode={args.mode or 'auto'}) ...")

    # resumable_train's default state_fn would silently fall back to cell_size=1.0; bind the real one here.
    state_fn = functools.partial(observation, cell_size=benchmark.cell_size)

    _final_variables, _log = resumable_train(
        checkpoint_path, variables, key, policy.apply, benchmark.params, benchmark.reward_fn,
        benchmark.sizes_array, benchmark.cell_size, args.n_iterations,
        benchmark.padded_pin_idx, benchmark.padded_pin_offset, benchmark.valid_mask,
        state_fn=state_fn, checkpoint_every=10, eval_every=args.eval_every, log_path=log_path,
        n_envs=args.n_envs, mode=args.mode, snapshot_dir=snapshot_dir, snapshot_every=50,
        placement_images_dir=placement_images_dir,
    )

    print()
    if checkpoint_path is not None:
        print(f"checkpoint saved to {checkpoint_path} - re-run this script to continue training.")
        print(f"snapshots (never overwritten) saved to {snapshot_dir}/")
    print(f"full history saved to {log_path}")


if __name__ == "__main__":
    main()
