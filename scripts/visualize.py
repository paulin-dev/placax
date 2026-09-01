"""Generates training-progress and placement visualizations from an already-trained run."""
import argparse
import functools
import pathlib
import sys

from placax import _device  # noqa: F401  must precede jax imports
from placax.core import reset
from placax.log import Log
from placax.types import EnvState
from placax_agents.benchmark import Benchmark
from placax_agents.policy.architectures.cnn import CNNActorCritic
from placax_agents.policy.observation import observation
from placax_agents.policy.scale import to_grid_units
from placax_agents.training.loops.common import open_train_state
from placax_agents.types import ExtraIllegalFn, StateFn
from placax_viz.animation import save_placement_gif
from placax_viz.curves import plot_training_curves
from placax_viz.masks import plot_observation_channels
from placax_viz.placement import save_placement_image
from placax_viz.rollout import collect_placement_history

import jax.numpy as jnp
import optax
from jax import random


def _training_setup(benchmark_dir: pathlib.Path, _macro_budget: int | None):
    """scripts/run_training.py's own setup: full netlist, plain CNN policy/observation."""
    benchmark = Benchmark.load(benchmark_dir)
    policy = CNNActorCritic()
    # Bare `observation` defaults to cell_size=1.0 - bind the benchmark's real cell_size instead.
    state_fn: StateFn = functools.partial(observation, cell_size=benchmark.cell_size)
    extra_illegal_fn: ExtraIllegalFn | None = None
    optimizer = optax.adam(3e-4)
    return benchmark, policy, state_fn, extra_illegal_fn, optimizer


def _maskplace_setup(benchmark_dir: pathlib.Path, macro_budget: int | None):
    """scripts/run_maskplace.py's own setup, imported lazily so --preset=training skips its extra deps."""
    from scripts.run_maskplace import (
        WIREMASK_MARGIN,
        _build_policy,
        _build_state_fn,
        _load_benchmark,
        maskplace_optimizer,
        maskplace_ppo_config,
    )
    from placax_agents.policy.action import make_wiremask_quality_illegal

    benchmark = _load_benchmark(benchmark_dir, macro_budget)
    policy = _build_policy(benchmark)
    state_fn = _build_state_fn(benchmark)
    extra_illegal_fn = make_wiremask_quality_illegal(margin=WIREMASK_MARGIN, cell_size=benchmark.cell_size)
    optimizer = maskplace_optimizer(value_coef=maskplace_ppo_config().value_coef)
    return benchmark, policy, state_fn, extra_illegal_fn, optimizer


_PRESETS = {"training": ("output", _training_setup), "maskplace": ("output_maskplace", _maskplace_setup)}


def _parse_args(argv: list[str]):
    parser = argparse.ArgumentParser(description="Visualize a training run's progress and placements.")
    parser.add_argument("--benchmark_dir", type=pathlib.Path, default=pathlib.Path("benchmarks/adaptec1"))
    parser.add_argument("--preset", choices=sorted(_PRESETS), default="training")
    parser.add_argument(
        "--macro_budget", type=str, default=None,
        help='--preset=maskplace only: macro budget it was trained with (default: that script\'s own '
             'default); pass "all" if it was trained with --macro_budget=all.',
    )
    parser.add_argument(
        "--run_dir", type=pathlib.Path, default=None,
        help="Directory holding checkpoint.bin/training_log.jsonl (default: <benchmark_dir>/<preset default>).",
    )
    parser.add_argument(
        "--output_dir", type=pathlib.Path, default=None, help="Where to write images (default: --run_dir).",
    )
    parser.add_argument("--gif", action="store_true", help="Also render a placement-progress GIF (one extra rollout).")
    args = parser.parse_args(argv[1:])

    default_subdir, setup_fn = _PRESETS[args.preset]
    run_dir = args.run_dir or args.benchmark_dir / default_subdir
    output_dir = args.output_dir or run_dir
    macro_budget = None if args.macro_budget is None or args.macro_budget.lower() == "all" else int(args.macro_budget)
    return args.benchmark_dir, run_dir, output_dir, args.gif, setup_fn, macro_budget


def main() -> None:
    Log.configure()
    benchmark_dir, run_dir, output_dir, want_gif, setup_fn, macro_budget = _parse_args(sys.argv)
    checkpoint_path = run_dir / "checkpoint.bin"
    log_path = run_dir / "training_log.jsonl"
    if not checkpoint_path.exists() or not log_path.exists():
        Log.error(f"'{run_dir}' has no checkpoint.bin/training_log.jsonl - run the matching training script first.")
        sys.exit(1)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Training curves need only the JSONL log, not the checkpoint.
    curves_path = output_dir / "training_curves.png"
    plot_training_curves(log_path, save_path=curves_path)
    Log.info(f"wrote {curves_path}")

    # 2. Everything else needs the trained policy: rebuild the matching setup (--preset), then load weights.
    benchmark, policy, state_fn, extra_illegal_fn, optimizer = setup_fn(benchmark_dir, macro_budget)
    obs0 = state_fn(reset(benchmark.params), benchmark.params, benchmark.sizes_array)
    variables = policy.init(random.PRNGKey(0), obs0)
    variables, _opt_state, _running_stats, _key, iteration = open_train_state(
        variables, random.PRNGKey(0), optimizer, checkpoint_path
    )
    Log.info(f"loaded checkpoint at iteration {iteration}")

    # 3. A greedy rollout, recording every intermediate step.
    history = collect_placement_history(
        variables, policy.apply, benchmark.params, benchmark.sizes_array, benchmark.cell_size,
        state_fn=state_fn, extra_illegal_fn=extra_illegal_fn,
    )
    grid_sizes = to_grid_units(benchmark.sizes_array, benchmark.cell_size)

    placement_path = output_dir / "final_placement.png"
    save_placement_image(
        history[-1], grid_sizes, benchmark.params.grid_x, benchmark.params.effective_grid_y, placement_path,
    )
    Log.info(f"wrote {placement_path}")

    # obs0 makes for a flat, uninformative wiremask panel - use a partway-through-the-rollout state instead.
    masks_path = output_dir / "observation_channels.png"
    mid_step = len(history) // 2
    mid_state = EnvState(positions=jnp.asarray(history[mid_step]), step=mid_step)
    obs_mid = state_fn(mid_state, benchmark.params, benchmark.sizes_array)
    plot_observation_channels(obs_mid, save_path=masks_path)
    Log.info(f"wrote {masks_path}")

    if want_gif:
        gif_path = output_dir / "placement_progress.gif"
        save_placement_gif(history, grid_sizes, benchmark.params.grid_x, benchmark.params.effective_grid_y, gif_path)
        Log.info(f"wrote {gif_path}")


if __name__ == "__main__":
    main()
