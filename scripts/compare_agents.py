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
  * one scoring path - each agent hands over a placement and THIS script scores it, so a
    difference in the table can never be a difference in how two agents scored themselves;
  * legality reported beside the score, because overlapping macros have shorter wires: a
    wirelength number for an unrealizable placement would top the table on merit it doesn't have;
  * one manifest per run, recording the config and the machine, plus a results.json holding every
    number the table shows beside the run hash that produced it - so the table can be regenerated
    and checked without re-running a single agent;
  * each row's ACTUAL spend printed beside its score, because the budget is what a run was
    offered and a converged agent stops long before it.

Seeds are the one thing you should vary: on a GPU backend a single run is not reproducible (see
placax.reproducibility), so --seeds runs each agent several times and reports the spread.

    python -m scripts.compare_agents --benchmark_dir=benchmarks/adaptec1 \\
        --env_steps=500000 --agents=greedy_wiremask,random_search,ppo --seeds=3
"""
import argparse
import json
import pathlib
import statistics
import sys

from placax import _device  # noqa: F401  must precede jax imports
from placax.log import Log
from placax.reproducibility import describe_determinism, fingerprint
from placax_agents.experiment.budget import Budget
from placax_agents.experiment.build import build, build_benchmark
from placax_agents.experiment.config import AgentSpec, ExperimentConfig, Spec, assert_comparable
from placax_agents.experiment.presets import OUTPUT_SUBDIRS, build_preset
from placax_agents.experiment.run import METRICS, run_experiment, score

DEFAULT_AGENTS = ("greedy_wiremask", "random_search", "ppo")

RESULTS_NAME = "results.json"
"""The table as data, written beside the per-run manifests."""


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
                             "placement is scored regardless). Note an eval rollout is charged "
                             "to the budget like any other episode, so setting this trades "
                             "training work for measurement; leave it at 0 unless you want the "
                             "curve, and use the same value for every agent if you do.")
    parser.add_argument("--level", default="environment",
                        choices=("benchmark", "task", "environment"),
                        help="Which invariant this comparison claims (default: %(default)s - "
                             "everything the agent did not choose). Drop to 'task' only for a "
                             "study that varies the observation on purpose.")
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
    reference: ExperimentConfig, agents: list[str], seeds: int, population: int,
    level: str = "environment",
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
    assert_comparable(*configs, level=level)
    return configs


def _format_table(results: dict[str, list[dict]], budget: Budget, level: str) -> str:
    """Agents ranked by mean HPWL, with the spread across seeds, their legality and their spend.

    Legality is a column rather than a footnote: overlapping macros shorten wires, so an illegal
    placement outranks a legal one on HPWL alone. A row that is not 100% legal has not produced a
    result, whatever its wirelength says.

    So is spend. The budget is what every agent was OFFERED; what a row actually used is a
    different number, and for a deterministic agent that reports `converged` it is dramatically
    different - `greedy_wiremask` stops after one iteration and spends a single episode of a
    budget that may run to millions of steps. Printing one budget line under every row implied a
    compute match that the run itself had already refused, in the one table this script exists to
    make trustworthy.
    """
    header = (f"{'agent':<22s}{'mean HPWL':>15s}{'best':>15s}{'std':>11s}"
              f"{'legal':>8s}{'overlap':>9s}{'env steps':>13s}{'grad steps':>12s}{'seeds':>7s}")
    lines = [header, "-" * len(header)]
    for name, runs in sorted(results.items(),
                             key=lambda kv: statistics.fmean(r["real_hpwl"] for r in kv[1])):
        hpwls = [run["real_hpwl"] for run in runs]
        legal = sum(1 for run in runs if run["is_legal"])
        worst_overlap = max(run["overlap_ratio"] for run in runs)
        std = statistics.stdev(hpwls) if len(hpwls) > 1 else 0.0
        spent = statistics.fmean(run["env_steps"] for run in runs)
        gradients = statistics.fmean(run["gradient_steps"] for run in runs)
        lines.append(
            f"{name:<22s}{statistics.fmean(hpwls):>15,.0f}{min(hpwls):>15,.0f}"
            f"{std:>11,.0f}{legal:>4d}/{len(runs):<3d}{worst_overlap:>8.2%}"
            f"{spent:>13,.0f}{gradients:>12,.0f}{len(runs):>7d}"
        )
    lines.append("")
    offered = ", ".join(
        f"{value:,} {unit}" for unit, value in
        (("env steps", budget.env_steps), ("iterations", budget.iterations),
         ("s wall clock", budget.wall_clock_s))
        if value is not None
    )
    lines.append(f"budget OFFERED to every run: {offered}, identical {level}, scored by this script.")
    lines.append("'env steps' is what each agent actually SPENT (mean over seeds) - a deterministic "
                 "agent that reports convergence stops early and spends a fraction of the budget.")
    lines.append("Rows are sample-matched where their env steps agree, never compute-matched: "
                 "'grad steps' is what env steps deliberately does not price.")
    lines.append("'legal' counts seeds whose placement had no overlap, nothing out of bounds and "
                 "every macro placed; 'overlap' is the worst seed's overlapping macro area.")
    return "\n".join(lines)


def _write_results(path: pathlib.Path, results: dict[str, list[dict]], reference: ExperimentConfig,
                   budget: Budget, level: str) -> pathlib.Path:
    """The table as data, so it can be regenerated and checked without re-running anything.

    The script's own docstring has always promised this ("so the table can be regenerated"), and
    a printed table plus a directory of manifests did not deliver it: reconstructing a row meant
    re-running the agent. Every number the table shows is written here beside the hash of the
    environment it was produced under and the full hash of the run that produced it.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({
        "level": level,
        "shared_hash": reference.hash_at(level),
        "budget": budget.to_dict(),
        "metrics": METRICS,
        "fingerprint": fingerprint(),
        "runs": {name: sorted(runs, key=lambda run: run["seed"]) for name, runs in results.items()},
    }, indent=2, sort_keys=True) + "\n")
    return path


def main() -> None:
    Log.configure()
    args = _parse_args(sys.argv)
    if not args.benchmark_dir.exists():
        Log.error(f"'{args.benchmark_dir}' not found - run scripts/download_benchmarks.py first.")
        sys.exit(1)

    budget = _budget(args)
    reference = build_preset(args.preset, args.benchmark_dir, budget=budget)
    agents = [name.strip() for name in args.agents.split(",") if name.strip()]
    if not agents:
        raise SystemExit("--agents is empty; name at least one agent to compare")
    configs = build_comparison(reference, agents, args.seeds, args.population, args.level)

    output_root = args.output_dir or (args.benchmark_dir / "comparison")
    Log.info(f"comparing {len(agents)} agents x {args.seeds} seed(s) on {args.benchmark_dir}")
    Log.info(f"  shared {args.level}: {reference.hash_at(args.level)}")
    Log.info(f"  {describe_determinism()}")

    # Loaded once and shared: two agents parsing the same netlist twice would still be comparable,
    # but sharing it removes the possibility entirely and saves the parse.
    benchmark = build_benchmark(reference)

    results: dict[str, list[dict]] = {}
    for config in configs:
        agent_name = config.name.rsplit("-seed", 1)[0]
        built = build(config, benchmark=benchmark)
        state, log = run_experiment(
            config, output_root / config.name, built=built,
            eval_every=args.eval_every, log_every=max(1, args.eval_every or 1),
        )
        measured = score(benchmark, built.agent.best_positions(state), built.n_placed)
        # What this run actually spent, from its own last log line - not what the budget offered.
        # An agent that reports convergence stops early, and a table that prints the budget over
        # such a row claims a compute match the run itself declined.
        final = log[-1] if log else {}
        results.setdefault(agent_name, []).append({
            **measured,
            "seed": config.seed,
            "env_steps": final.get("env_steps", 0),
            "eval_env_steps": final.get("eval_env_steps", 0),
            "gradient_steps": final.get("gradient_steps", 0),
            "iterations": final.get("iteration", 0),
            # build() resolves the netlist digest, so hash the config it produced, not the one
            # handed in - that is the one identifying the design by its contents.
            "full_hash": built.config.full_hash(),
        })
        flag = "" if measured["is_legal"] else "  ILLEGAL"
        Log.info(f"  {config.name}: real_hpwl={measured['real_hpwl']:,.0f} "
                 f"return={measured['reward_return']:,.3f} "
                 f"spent={final.get('env_steps', 0):,} env steps{flag}")

    print()
    print(_format_table(results, budget, args.level))
    print()
    results_path = _write_results(
        output_root / RESULTS_NAME, results, built.config, budget, args.level
    )
    print(f"per-run manifests and logs: {output_root}")
    print(f"the table as data:          {results_path}")


if __name__ == "__main__":
    main()
