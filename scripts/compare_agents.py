"""Runs several agents against ONE environment and ONE compute budget, and prints the comparison.

This is the thing the project exists to make possible, so it is worth being explicit about what
it guarantees, because the macro-placement literature routinely reports numbers that have none of
these properties:

  * one netlist, loaded once and shared, so every agent places exactly the same macros in exactly
    the same order under exactly the same legality rules;
  * one environment hash, asserted before anything runs - if a single axis differs, this refuses
    to produce a table rather than producing a misleading one;
  * one compute budget in env steps, which every agent pays in identically regardless of whether
    it uses gradients, a population, or nothing at all;
  * one scoring path - each agent hands over a placement and THIS script computes the HPWL, so a
    difference in the table can never be a difference in how two agents scored themselves;
  * one manifest per run, recording the config and the machine, so the table can be regenerated.

Seeds are the one thing you should vary: on a GPU backend a single run is not reproducible (see
placax.reproducibility), so --seeds runs each agent several times and reports the spread.

    python -m scripts.compare_agents --benchmark_dir=benchmarks/adaptec1 \\
        --env_steps=500000 --agents=greedy_wiremask,random_search,ppo --seeds=3
"""
import argparse
import pathlib
import statistics
import sys

from placax import _device  # noqa: F401  must precede jax imports
from placax.log import Log
from placax.reproducibility import describe_determinism
from placax_agents.experiment.budget import Budget
from placax_agents.experiment.build import build, build_benchmark
from placax_agents.experiment.config import AgentSpec, ExperimentConfig, Spec, assert_comparable
from placax_agents.experiment.presets import OUTPUT_SUBDIRS, build_preset
from placax_agents.experiment.run import run_experiment, score_placement

DEFAULT_AGENTS = ("greedy_wiremask", "random_search", "ppo")


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--benchmark_dir", type=pathlib.Path,
                        default=pathlib.Path("benchmarks/adaptec1"))
    parser.add_argument("--preset", default="training", choices=sorted(OUTPUT_SUBDIRS),
                        help="Which preset supplies the shared ENVIRONMENT - benchmark, grid, "
                             "order, reward, observation and mask (default: %(default)s). Its own "
                             "agent is one of the competitors when 'ppo' is in --agents.")
    parser.add_argument("--agents", default=",".join(DEFAULT_AGENTS),
                        help="Comma-separated agents to compare (default: %(default)s).")
    parser.add_argument("--env_steps", type=int, default=None,
                        help="Compute budget per run, in macro placements. The unit that makes "
                             "these agents comparable; strongly preferred over --n_iterations.")
    parser.add_argument("--n_iterations", type=int, default=None,
                        help="Budget in iterations instead. Note an iteration means different "
                             "amounts of work to different agents, so a table built on it is not "
                             "a fair comparison - use --env_steps unless you know why not.")
    parser.add_argument("--seeds", type=int, default=1,
                        help="Run each agent this many times, with seeds 0..n-1, and report the "
                             "spread (default: %(default)s). More than one is strongly advised on "
                             "a nondeterministic backend.")
    parser.add_argument("--population", type=int, default=16,
                        help="Random-search population per iteration (default: %(default)s).")
    parser.add_argument("--output_dir", type=pathlib.Path, default=None,
                        help="Where each run's manifest, log and checkpoints go; one subdirectory "
                             "per agent and seed. Default: <benchmark_dir>/comparison.")
    parser.add_argument("--eval_every", type=int, default=0,
                        help="Periodic mid-run eval (default: %(default)s, off - the final "
                             "placement is scored regardless).")
    return parser.parse_args(argv[1:])


def _budget(args: argparse.Namespace) -> Budget:
    if args.env_steps is None and args.n_iterations is None:
        raise SystemExit("pass --env_steps (preferred) or --n_iterations to set the compute budget")
    return Budget(env_steps=args.env_steps, iterations=args.n_iterations)


def _agent_spec(name: str, reference: ExperimentConfig, population: int) -> AgentSpec:
    """The agent half of a config; everything else is inherited from the reference unchanged."""
    if name == "ppo":
        return reference.agent
    if name == "random_search":
        return AgentSpec(algorithm=Spec("random_search", {"population": population}))
    return AgentSpec(algorithm=Spec(name))


def build_comparison(
    reference: ExperimentConfig, agents: list[str], seeds: int, population: int
) -> list[ExperimentConfig]:
    """One config per (agent, seed), all sharing the reference's environment exactly."""
    configs = []
    for name in agents:
        for seed in range(seeds):
            configs.append(ExperimentConfig(
                name=f"{name}-seed{seed}",
                seed=seed,
                environment=reference.environment,
                agent=_agent_spec(name, reference, population),
            ))
    # The whole point. If a future edit lets an agent perturb the environment, this stops the run
    # rather than letting an incomparable table reach a paper.
    assert_comparable(*configs)
    return configs


def _format_table(results: dict[str, list[float]], env_steps: int) -> str:
    """Agents ranked by mean HPWL, with the spread across seeds."""
    header = f"{'agent':<24s}{'mean HPWL':>16s}{'best':>16s}{'std':>12s}{'seeds':>7s}"
    lines = [header, "-" * len(header)]
    for name, scores in sorted(results.items(), key=lambda kv: statistics.fmean(kv[1])):
        std = statistics.stdev(scores) if len(scores) > 1 else 0.0
        lines.append(
            f"{name:<24s}{statistics.fmean(scores):>16,.0f}{min(scores):>16,.0f}"
            f"{std:>12,.0f}{len(scores):>7d}"
        )
    lines.append("")
    lines.append(f"all runs: {env_steps:,} env steps, identical environment, HPWL scored by this script")
    return "\n".join(lines)


def main() -> None:
    Log.configure()
    args = _parse_args(sys.argv)
    if not args.benchmark_dir.exists():
        Log.error(f"'{args.benchmark_dir}' not found - run scripts/download_benchmarks.py first.")
        sys.exit(1)

    budget = _budget(args)
    reference = build_preset(args.preset, args.benchmark_dir, budget=budget)
    agents = [name.strip() for name in args.agents.split(",") if name.strip()]
    configs = build_comparison(reference, agents, args.seeds, args.population)

    output_root = args.output_dir or (args.benchmark_dir / "comparison")
    Log.info(f"comparing {len(agents)} agents x {args.seeds} seed(s) on {args.benchmark_dir}")
    Log.info(f"  shared environment: {reference.environment_hash()}")
    Log.info(f"  {describe_determinism()}")

    # Loaded once and shared: two agents parsing the same netlist twice would still be comparable,
    # but sharing it removes the possibility entirely and saves the parse.
    benchmark = build_benchmark(reference)

    results: dict[str, list[float]] = {}
    for config in configs:
        agent_name = config.name.rsplit("-seed", 1)[0]
        built = build(config, benchmark=benchmark)
        state, _log = run_experiment(
            config, output_root / config.name, built=built,
            eval_every=args.eval_every, log_every=max(1, args.eval_every or 1),
        )
        score = score_placement(benchmark, built.agent.best_positions(state))
        results.setdefault(agent_name, []).append(score)
        Log.info(f"  {config.name}: real_hpwl={score:,.0f}")

    spent = budget.env_steps or (budget.iterations * built.env_steps_per_iteration)
    print()
    print(_format_table(results, spent))
    print()
    print(f"per-run manifests and logs: {output_root}")


if __name__ == "__main__":
    main()
