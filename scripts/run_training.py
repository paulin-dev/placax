"""Trains the plain-CNN baseline setup - a thin CLI over the same shared ExperimentConfig.

Structurally identical to scripts/run_maskplace.py on purpose: both build a config from
placax_agents.experiment.presets and hand it to the same run_experiment. That is what makes the
two comparable, and what makes the differences between them readable as data (compare the two
presets side by side) rather than by diffing two scripts.
"""
import argparse
import pathlib
import sys

from placax import _device  # noqa: F401  must precede jax imports
from placax.log import Log
from placax_agents.experiment.budget import Budget
from placax_agents.experiment.presets import default_output_dir, training
from placax_agents.experiment.run import run_experiment
from scripts.presets import build_setup


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train (or resume) the plain-CNN baseline setup.")
    parser.add_argument("--benchmark_dir", type=pathlib.Path, default=pathlib.Path("benchmarks/adaptec1"))
    parser.add_argument(
        "--seed", type=int, default=0,
        help="RNG seed for policy init and rollout sampling (default: %(default)s). This used to "
             "be hardcoded to PRNGKey(0) with no way to vary it, which made the multi-seed "
             "reporting this project needs impossible for the baseline.",
    )
    parser.add_argument("--n_iterations", type=int, default=None,
                        help="Budget in TOTAL training iterations (default: 100 if no other budget "
                             "flag is given). Note one iteration here is ONE episode, unlike "
                             "run_maskplace's buffered loop - use --env_steps to compare the two.")
    parser.add_argument("--env_steps", type=int, default=None,
                        help="Budget in env steps (macro placements) - the unit comparable across "
                             "agents and loop shapes. Prefer this when comparing two agents.")
    parser.add_argument("--wall_clock_s", type=float, default=None,
                        help="Budget in seconds of training, accumulated across resumes.")
    parser.add_argument("--n_envs", type=int, default=1,
                        help="Episodes vmapped per update (default: %(default)s). >1 selects the "
                             "parallel loop; see this project's subprocess_search.py for sizing it.")
    parser.add_argument("--eval_every", type=int, default=10,
                        help="Compute real HPWL (a full extra greedy rollout) every this many "
                             "iterations (default: %(default)s).")
    parser.add_argument("--log_every", type=int, default=1,
                        help="Console progress line every this many iterations (default: %(default)s).")
    parser.add_argument("--patience", type=int, default=0,
                        help="Stop early once real_hpwl hasn't beaten its best for this many "
                             "consecutive evals (default: %(default)s, disabled).")
    parser.add_argument("--no_checkpoint", action="store_true",
                        help="Run entirely in memory: no manifest, checkpoint or log written.")
    parser.add_argument("--placement_images", action="store_true",
                        help="Also write a placement snapshot PNG on every --eval_every iteration.")
    parser.add_argument("--placement_images_dir", type=pathlib.Path, default=None,
                        help="Where to write placement snapshots (implies --placement_images).")
    parser.add_argument("--config", type=pathlib.Path, default=None,
                        help="Load an ExperimentConfig JSON instead of building one from the flags.")
    parser.add_argument("--output_dir", type=pathlib.Path, default=None,
                        help="Where the manifest, log and checkpoints go (default: "
                             "<benchmark_dir>/output).")
    return parser.parse_args(argv[1:])


def _budget_from_args(args: argparse.Namespace) -> Budget:
    """Whichever caps were given; iterations=100 only if none were, preserving the old default."""
    if args.n_iterations is None and args.env_steps is None and args.wall_clock_s is None:
        return Budget(iterations=100)
    return Budget(
        iterations=args.n_iterations, env_steps=args.env_steps, wall_clock_s=args.wall_clock_s
    )


def main() -> None:
    Log.configure()
    args = _parse_args(sys.argv)
    if not args.benchmark_dir.exists():
        Log.error(f"'{args.benchmark_dir}' not found - run scripts/download_benchmarks.py first.")
        sys.exit(1)

    if args.config is not None:
        from placax_agents.experiment.config import ExperimentConfig

        config = ExperimentConfig.read(args.config)
        Log.info(f"loaded config from {args.config} (flags describing the setup are ignored)")
    else:
        config = training(
            args.benchmark_dir, seed=args.seed, budget=_budget_from_args(args), n_envs=args.n_envs,
        )

    Log.info(f"loading {config.environment.benchmark.benchmark_dir} ...")
    built = build_setup(config)

    if args.no_checkpoint:
        output_dir = None
        placement_images_dir = args.placement_images_dir
    else:
        # Per RUN, not per preset - see presets.default_output_dir.
        output_dir = args.output_dir or default_output_dir(
            "training", args.benchmark_dir, config.seed
        )
        placement_images_dir = args.placement_images_dir or (
            output_dir / "placements" if args.placement_images else None
        )

    run_experiment(
        config, output_dir, built=built, eval_every=args.eval_every, log_every=args.log_every,
        patience=args.patience, placement_images_dir=placement_images_dir,
    )

    if output_dir is not None:
        print()
        print(f"config + machine fingerprint: {output_dir / 'manifest.json'}")
        print(f"per-iteration history:        {output_dir / 'training_log.jsonl'}")
        print(f"resume with the same flags to continue from {output_dir / 'checkpoint.bin'}")


if __name__ == "__main__":
    main()
