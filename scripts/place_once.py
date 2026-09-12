"""Loads a trained checkpoint and runs one greedy placement pass - no training, just inference.

Rebuilds the environment from the run's own config rather than from a hardcoded MaskPlace setup.
The previous version reached into `scripts.run_maskplace`'s private helpers for its benchmark,
observation, mask and optimizer, which meant it could only ever load a MaskPlace-preset checkpoint
and would silently mis-load anything else - the same hand-matching problem `ExperimentConfig`
removed from training, surviving in an inference script.
"""
import argparse
import pathlib
import sys

from placax import _device  # noqa: F401  must precede jax imports
from placax.core import reset
from placax.log import Log
from placax_agents.experiment.build import build
from placax_agents.experiment.run import score
from placax_agents.ops.evaluate import evaluate
from placax_agents.training.loops.common import open_train_state
from placax_agents.policy.scale import to_real_centers
from placax_agents.experiment.presets import find_run_dir
from scripts.presets import config_for

from jax import random


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run one greedy placement pass from a trained checkpoint."
    )
    parser.add_argument("--benchmark_dir", type=pathlib.Path,
                        default=pathlib.Path("benchmarks/adaptec1"))
    parser.add_argument("--config", type=pathlib.Path, default=None,
                        help="A run's manifest.json (or a bare ExperimentConfig JSON). STRONGLY "
                             "PREFERRED: it rebuilds the exact environment the checkpoint was "
                             "trained in, including its warm start. --preset and --macro_budget "
                             "are ignored when this is given.")
    parser.add_argument("--preset", default="maskplace",
                        help="Which preset to rebuild from when no --config is given "
                             "(default: %(default)s).")
    parser.add_argument("--checkpoint", type=pathlib.Path, default=None,
                        help="Defaults to checkpoint.bin in the preset's own newest run "
                             "directory under <benchmark_dir>.")
    parser.add_argument("--macro_budget", type=str, default=None,
                        help='--preset only: budget the checkpoint was trained with; "all" for '
                             "the whole netlist.")
    args = parser.parse_args(argv[1:])
    args.macro_budget = (
        None if args.macro_budget is None or args.macro_budget.lower() == "all"
        else int(args.macro_budget)
    )
    # Both run-directory layouts, newest first - see presets.find_run_dir.
    args.checkpoint = args.checkpoint or (
        find_run_dir(args.preset, args.benchmark_dir) / "checkpoint.bin"
    )
    return args


def main() -> None:
    Log.configure()
    args = _parse_args(sys.argv)
    if not args.checkpoint.exists():
        Log.error(f"'{args.checkpoint}' not found - train first with scripts/run_maskplace.py.")
        sys.exit(1)

    # 1. Rebuild the environment the checkpoint was trained in - one build() call, the same one
    #    every training run goes through, so nothing here can drift from what produced the weights.
    config = config_for(args.config, args.preset, args.benchmark_dir, args.macro_budget)
    Log.info(f"loading {config.environment.benchmark.benchmark_dir} ...")
    built = build(config)
    benchmark = built.benchmark

    # 2. Build a template pytree matching what was saved, then deserialize the checkpoint into it.
    key = random.PRNGKey(config.seed)
    key, init_key = random.split(key)
    obs0 = built.state_fn(reset(benchmark.params, built.initial_positions), benchmark.params,
                          benchmark.sizes_array)
    variables = built.policy.init(init_key, obs0)
    variables, _opt_state, _running_stats, _key, iteration = open_train_state(
        variables, key, built.optimizer, args.checkpoint
    )
    Log.info(f"loaded checkpoint at iteration {iteration}")

    # 3. One greedy (no training, no sampling) rollout over the macros the environment left to the
    #    agent - the warm-start prefix comes from the config, not from an assumption of none.
    positions, _orientations, _hpwl = evaluate(
        variables, built.policy.apply, benchmark.params, benchmark.sizes_array, benchmark.cell_size,
        benchmark.padded_pin_idx, benchmark.padded_pin_offset, benchmark.valid_mask,
        built.state_fn, built.extra_illegal_fn, built.initial_positions, built.n_placed,
    )
    real_centers = to_real_centers(positions, benchmark.sizes_array, benchmark.cell_size)
    # Scored by the same `score` every run and every agent goes through, so this number is the
    # one a training log or a comparison table would have reported for the same placement.
    measured = score(benchmark, positions, built.n_placed)

    print()
    print(f"real_hpwl      = {measured['real_hpwl']:,.2f}")
    print(f"reward_return  = {measured['reward_return']:,.3f}")
    print(f"legal          = {measured['is_legal']}"
          f"  (overlap {measured['overlap_ratio']:.2%}, "
          f"out of bounds {measured['out_of_bounds_ratio']:.2%})")
    print(f"run            = {built.config.full_hash()}")
    print(f"placed {positions.shape[0]} macros ({built.n_placed} pre-placed by the environment) - "
          f"grid positions in `positions`, real-unit centers in `real_centers`.")
    return positions, real_centers


if __name__ == "__main__":
    main()
