"""Trains the MaskPlace-equivalent setup - a thin CLI over one shared ExperimentConfig.

Everything this script used to hardcode (grid, macro order, reward, observation, action mask,
policy, optimizer, PPO hyperparameters, loop shape) now lives in
`placax_agents.experiment.presets.maskplace` as data, and the loop itself is the shared
`run_experiment`. The point is that this script and scripts/run_training.py can no longer drift
apart silently, and that every run writes down the configuration that produced it - see
`manifest.json` in the output directory.
"""
import argparse
import pathlib
import sys

from placax import _device  # noqa: F401  must precede jax imports
from placax.log import Log
from placax_agents.experiment.budget import Budget
from placax_agents.experiment.presets import (
    MASKPLACE_GRID,
    MASKPLACE_MACRO_BUDGET,
    MASKPLACE_N_EPISODES,
    WIREMASK_MARGIN,
    default_output_dir,
    maskplace,
)
from placax_agents.experiment.registry import (
    MASKPLACE_LEARNING_RATE,
    MASKPLACE_MAX_GRAD_NORM,
    MASKPLACE_REWARD_DIVISOR,
)
from placax_agents.experiment.run import run_experiment
from scripts.presets import build_setup

# Re-exported so the handful of scripts and tests that reach for MaskPlace's own constants and
# builders keep one import site, while the definitions themselves live in the registry/presets
# alongside every other component. Listed explicitly rather than left as incidental imports.
__all__ = [
    "MASKPLACE_GRID", "MASKPLACE_LEARNING_RATE", "MASKPLACE_MACRO_BUDGET",
    "MASKPLACE_MAX_GRAD_NORM", "MASKPLACE_N_EPISODES", "MASKPLACE_REWARD_DIVISOR",
    "WIREMASK_MARGIN", "maskplace_optimizer", "maskplace_ppo_config",
    "_build_policy", "_build_state_fn", "_load_benchmark",
]


def maskplace_ppo_config(entropy_coef: float = 0.0):
    """The PPOConfig the maskplace preset resolves to - see that preset for the reference values."""
    from placax_agents.experiment.build import build_ppo_config

    return build_ppo_config(maskplace(".", entropy_coef=entropy_coef).agent.algorithm)


def maskplace_optimizer(
    learning_rate: float = MASKPLACE_LEARNING_RATE,
    max_grad_norm: float = MASKPLACE_MAX_GRAD_NORM,
    critic_param_prefix: str = "critic_",
    value_coef: float = 0.5,
):
    """Separately-clipped actor/critic Adam - the `maskplace_split` optimizer, by its old name."""
    from placax_agents.experiment.registry import OPTIMIZERS

    return OPTIMIZERS["maskplace_split"](
        learning_rate=learning_rate, max_grad_norm=max_grad_norm,
        critic_param_prefix=critic_param_prefix, value_coef=value_coef,
    )


def _load_benchmark(
    benchmark_dir: pathlib.Path, macro_budget: int | None,
    regularity_weight: float = 0.0, regularity_mode: str = "corner",
):
    """The benchmark the maskplace preset resolves to: connectivity order, budget, dense reward."""
    from placax_agents.experiment.build import build_benchmark

    return build_benchmark(maskplace(
        benchmark_dir, macro_budget=macro_budget,
        regularity_weight=regularity_weight, regularity_mode=regularity_mode,
    ))


def _build_state_fn(benchmark):
    """The wiremask + 2-macro-lookahead observation, MaskPlace's own channel set."""
    from placax_agents.experiment.registry import STATES

    return STATES["wiremask"](benchmark, lookahead=2)


def _build_policy(benchmark):
    """MaskPlace's own network shape: fine + coarse-ResNet branches, step-embedding critic."""
    from placax_agents.experiment.registry import POLICIES

    return POLICIES["resnet_coarse_fine"](benchmark, critic_style="step_embedding")


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the MaskPlace-equivalent setup end to end.")
    parser.add_argument(
        "--benchmark_dir", type=pathlib.Path, default=pathlib.Path("benchmarks/adaptec1"),
        help="Path to a downloaded benchmark directory (default: %(default)s).",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="RNG seed for policy init and rollout sampling (default: %(default)s, MaskPlace's own "
             "--seed default). MaskPlace reports mean+-std across several seeds per benchmark, not "
             "a single run - vary this to reproduce that spread. On a GPU backend a single run "
             "isn't reproducible even at a fixed seed (see placax.reproducibility), which makes "
             "reporting across seeds the only defensible option there.",
    )
    parser.add_argument(
        "--n_iterations", type=int, default=None,
        help="Budget in TOTAL training iterations (default: 100 if no other budget flag is given) "
             "- not an additional count: resuming at or past it runs zero further iterations.",
    )
    parser.add_argument(
        "--env_steps", type=int, default=None,
        help="Budget in env steps (macro placements), the unit that is comparable ACROSS agents "
             "and loop shapes - unlike iterations, which mean different amounts of work in "
             "different loops. Prefer this when comparing two agents.",
    )
    parser.add_argument(
        "--wall_clock_s", type=float, default=None,
        help="Budget in seconds of training, accumulated across resumes.",
    )
    parser.add_argument(
        "--macro_budget", type=str, default="all",
        help='Place only the N most important macros, MaskPlace\'s --pnm (default: %(default)s, '
             'matching the paper, which places every macro by RL); pass an integer to cap it '
             f'instead - PPO2.py\'s own --pnm default is {MASKPLACE_MACRO_BUDGET}.',
    )
    parser.add_argument(
        "--n_episodes", type=str, default=str(MASKPLACE_N_EPISODES),
        help="Episodes collected per PPO update (default: %(default)s, MaskPlace's own value). To "
             "find the largest value your hardware supports, run scripts/subprocess_search.py "
             "separately first and pass its result here - auto-detection deliberately isn't a flag "
             "of this script, since this process reserving GPU memory just by starting would "
             "compete with the very subprocesses being probed.",
    )
    parser.add_argument("--log_every", type=int, default=1,
                        help="Console progress line every this many iterations (default: %(default)s).")
    parser.add_argument("--eval_every", type=int, default=10,
                        help="Compute real HPWL (a full extra greedy rollout) every this many "
                             "iterations (default: %(default)s).")
    parser.add_argument("--no_checkpoint", action="store_true",
                        help="Run entirely in memory: no manifest, checkpoint or log written.")
    parser.add_argument("--placement_images", action="store_true",
                        help="Also write a placement snapshot PNG on every --eval_every iteration, "
                             "reusing that iteration's already-scheduled eval rollout.")
    parser.add_argument("--placement_images_dir", type=pathlib.Path, default=None,
                        help="Where to write placement snapshots (implies --placement_images).")
    parser.add_argument("--init_from", type=pathlib.Path, default=None,
                        help="Warm-start weights from a bare-variables checkpoint (typically a "
                             "previous run's best_checkpoint.bin) instead of random init. Training "
                             "starts fresh at iteration 0 with a new optimizer state and budget.")
    parser.add_argument("--patience", type=int, default=0,
                        help="Stop early once real_hpwl hasn't beaten its best for this many "
                             "consecutive evals (default: %(default)s, disabled).")
    parser.add_argument("--entropy_coef", type=float, default=0.0,
                        help="Entropy bonus coefficient (default: %(default)s, MaskPlace's own value).")
    parser.add_argument("--regularity_weight", type=float, default=0.0,
                        help="Weight on EXPlace's regularity (periphery) reward term, normalized to "
                             "[0, 1] per macro (default: %(default)s, off - pure MaskPlace). Run "
                             "scripts/measure_reward_terms.py on the benchmark to size it.")
    parser.add_argument("--regularity_mode", choices=("corner", "edge"), default="corner",
                        help="'corner' sums both axis costs so only the corners are free; 'edge' "
                             "takes their min so the whole border is (default: %(default)s).")
    parser.add_argument("--config", type=pathlib.Path, default=None,
                        help="Load an ExperimentConfig JSON (e.g. a previous run's manifest config) "
                             "instead of building one from the flags above. Every flag except the "
                             "run-control ones (--eval_every, --log_every, --patience, output "
                             "paths) is then ignored, so the run is exactly the recorded one.")
    parser.add_argument("--output_dir", type=pathlib.Path, default=None,
                        help="Where the manifest, log and checkpoints go (default: "
                             "<benchmark_dir>/output_maskplace).")
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

    try:
        n_episodes = int(args.n_episodes)
    except ValueError:
        Log.error(
            f"--n_episodes must be an integer, got {args.n_episodes!r}. To find the largest value "
            "your hardware supports, run this first (separately, not as part of this script):\n\n"
            f"  python -m scripts.subprocess_search scripts.run_maskplace "
            f"'--n_episodes=[1,2,4,8,{MASKPLACE_N_EPISODES}]' "
            f"--benchmark_dir={args.benchmark_dir} --macro_budget={args.macro_budget} "
            "--eval_every=1 --n_iterations=4 --no_checkpoint\n\n"
            "then pass the RESULT= value it prints as --n_episodes here."
        )
        sys.exit(1)

    if args.config is not None:
        from placax_agents.experiment.config import ExperimentConfig

        config = ExperimentConfig.read(args.config)
        Log.info(f"loaded config from {args.config} (flags describing the setup are ignored)")
    else:
        macro_budget = None if args.macro_budget.lower() == "all" else int(args.macro_budget)
        config = maskplace(
            args.benchmark_dir, seed=args.seed, macro_budget=macro_budget,
            budget=_budget_from_args(args), n_episodes=n_episodes,
            entropy_coef=args.entropy_coef, regularity_weight=args.regularity_weight,
            regularity_mode=args.regularity_mode,
        )

    Log.info(f"loading {config.environment.benchmark.benchmark_dir} ...")
    built = build_setup(config)

    if args.no_checkpoint:
        output_dir = None
        placement_images_dir = args.placement_images_dir
    else:
        # Per RUN, not per preset: the seed is part of the path, so two seeds are two
        # directories rather than one run resuming the other. See presets.default_output_dir.
        output_dir = args.output_dir or default_output_dir(
            "maskplace", args.benchmark_dir, config.seed
        )
        placement_images_dir = args.placement_images_dir or (
            output_dir / "placements" if args.placement_images else None
        )

    variables, log = run_experiment(
        config, output_dir, built=built, eval_every=args.eval_every, log_every=args.log_every,
        patience=args.patience, placement_images_dir=placement_images_dir,
        init_from=args.init_from,
    )

    if output_dir is not None:
        print()
        print(f"config + machine fingerprint: {output_dir / 'manifest.json'}")
        print(f"per-iteration history:        {output_dir / 'training_log.jsonl'}")
        print(f"resume with the same flags to continue from {output_dir / 'checkpoint.bin'}")
    return variables, log


if __name__ == "__main__":
    main()
