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
import dataclasses
import json
import pathlib
import statistics
import sys

from placax import _device  # noqa: F401  must precede jax imports
from placax.log import Log
from placax.reproducibility import describe_determinism, fingerprint
from placax_agents.experiment.budget import Budget
from placax_agents.experiment.build import build, build_benchmark
from placax_agents.experiment.config import (
    AgentSpec, ExperimentConfig, Spec, assert_comparable,
)
from placax_agents.experiment.presets import OUTPUT_SUBDIRS, build_preset
from placax_agents.experiment.run import (
    METRICS, best_orientations, run_experiment, score,
)

DEFAULT_AGENTS = ("greedy_wiremask", "random_search", "ppo")

RESULTS_NAME = "results.json"
"""The table as data, written beside the per-run manifests."""


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--benchmark_dir", type=pathlib.Path,
                        default=pathlib.Path("benchmarks/adaptec1"),
                        help="A single design to compare on (default: %(default)s).")
    parser.add_argument("--benchmark_dirs", default=None,
                        help="Comma-separated designs, to run the same protocol as a SUITE. Every "
                             "design's environment is asserted separately, and the protocol - "
                             "everything except which netlist - is asserted across them. Results "
                             "are aggregated by mean rank, since HPWL is not comparable between "
                             "designs.")
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
    parser.add_argument("--grid", type=int, default=None,
                        help="Override the preset's canvas resolution. Worth knowing WHY this "
                             "matters: adaptec1's 543 macros occupy 77.6%% of a 64-grid canvas "
                             "and 57%% of a 224-grid one, and no sampling agent places legally at "
                             "the former - every row comes back overlapping and the table then "
                             "reports a property of the canvas rather than of the agents.")
    parser.add_argument("--macro_budget", type=int, default=None,
                        help="Override the preset's macro budget - place only the first N macros "
                             "by the configured order (MaskPlace's --pnm).")
    parser.add_argument("--action_space", default=None,
                        help="Override the preset's action space, as 'name' or "
                             "'name:key=value,...' - e.g. 'perturbation:n_moves=128' or "
                             "'oriented_grid'. Part of the ENVIRONMENT, so it is applied to the "
                             "reference and every agent runs under it. Some agents require a "
                             "particular space: local_search needs 'perturbation', which also "
                             "needs an --initial_placement that fills the canvas.")
    parser.add_argument("--initial_placement", default=None,
                        help="Override the preset's warm start, same 'name:key=value' form - e.g. "
                             "'greedy_wiremask_prefix:n_macros=null' to pre-place every macro, "
                             "which is what a perturbation space has to start from.")
    parser.add_argument("--agent_environment", default=None,
                        help='JSON mapping an agent name to the environment axes its own '
                             'PARADIGM forces, e.g. \'{"local_search": {"action_space": '
                             '"perturbation:n_moves=128", "initial_placement": '
                             '"greedy_wiremask_prefix:n_macros=null"}}\'. Only "action_space" '
                             'and "initial_placement" may be set here, and only with '
                             '--level=paradigm: a constructive agent and a perturbation one move '
                             'macros differently by definition, which is what that level exists '
                             'to say. Everything else stays shared.')
    parser.add_argument("--agent_kwargs", default=None,
                        help='JSON mapping an agent name to its own kwargs, e.g. '
                             '\'{"genetic": {"population": 64}, "local_search": '
                             '{"temperature": 0.5}}\'. The agent is the thing under test, so '
                             "these differ per row by design; everything else is shared.")
    parser.add_argument("--canvas", default=None, choices=("die", "core"),
                        help="Override the preset's canvas anchor. 'core' scales and anchors the "
                             "grid to the design's placement rows; 'die' (the preset default) is "
                             "the die extent, which puts ~15%% of cells outside the placeable area.")
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
                        choices=("benchmark", "task", "paradigm", "environment"),
                        help="Which invariant this comparison claims (default: %(default)s - "
                             "everything the agent did not choose). Drop to 'task' for a study "
                             "that varies the observation on purpose, or to 'paradigm' to put a "
                             "constructive agent and a perturbation one in one table - they move "
                             "macros differently, so their action space and warm start differ "
                             "and the budget stops being work-matched. The table says so.")
    parsed = parser.parse_args(argv[1:])
    parsed.agent_kwargs = json.loads(parsed.agent_kwargs) if parsed.agent_kwargs else {}
    parsed.agent_environment = (
        json.loads(parsed.agent_environment) if parsed.agent_environment else {}
    )
    unknown = {axis for axes in parsed.agent_environment.values() for axis in axes} - {
        "action_space", "initial_placement"
    }
    if unknown:
        raise SystemExit(
            f"--agent_environment may set 'action_space' and 'initial_placement' and nothing "
            f"else; got {', '.join(sorted(unknown))}. Every other environment axis is shared by "
            f"definition - that is what makes the rows comparable at all."
        )
    if parsed.agent_environment and parsed.level != "paradigm":
        raise SystemExit(
            f"--agent_environment gives agents different action spaces or warm starts, which "
            f"--level={parsed.level} asserts are identical. Use --level=paradigm, which is the "
            f"claim such a table actually makes: same design, reward, observation, legality and "
            f"budget; different ways of moving a macro. Note what that level does NOT claim - "
            f"see the note printed under the table."
        )
    parsed.benchmark_dirs = (
        [pathlib.Path(part.strip()) for part in parsed.benchmark_dirs.split(",") if part.strip()]
        if parsed.benchmark_dirs else [parsed.benchmark_dir]
    )
    return parsed


def _with_overrides(config: ExperimentConfig, args: argparse.Namespace) -> ExperimentConfig:
    """The preset, with whichever benchmark axes the CLI overrode.

    Applied to the REFERENCE config, so every agent inherits the same override - an override that
    reached only some of them would be the exact incomparability this script exists to refuse.
    """
    benchmark = config.environment.benchmark
    overrides = {name: getattr(args, name) for name in ("grid", "macro_budget", "canvas")
                 if getattr(args, name) is not None}
    # The two environment axes an agent can REQUIRE. They sit beside grid and canvas rather than
    # beside the agent for the reason the whole config is split in two: what an action is, and
    # what the run starts from, are the task's to fix, not one competitor's.
    environment = {name: Spec.parse(getattr(args, name))
                   for name in ("action_space", "initial_placement")
                   if getattr(args, name) is not None}
    if not overrides and not environment:
        return config
    return dataclasses.replace(config, environment=dataclasses.replace(
        config.environment, benchmark=dataclasses.replace(benchmark, **overrides), **environment
    ))


def _budget(args: argparse.Namespace) -> Budget:
    if args.env_steps is None and args.n_iterations is None:
        raise SystemExit("pass --env_steps (preferred) or --n_iterations to set the compute budget")
    return Budget(env_steps=args.env_steps, iterations=args.n_iterations)


def _agent_spec(name: str, reference: ExperimentConfig, population: int,
                agent_kwargs: dict | None = None) -> AgentSpec:
    """The agent half of a config; everything else is inherited from the reference unchanged."""
    kwargs = dict((agent_kwargs or {}).get(name, {}))
    if name == "ppo":
        # PPO's whole agent half - policy, optimizer, loop - comes from the preset; --agent_kwargs
        # tunes its algorithm hyperparameters only.
        if not kwargs:
            return reference.agent
        algorithm = reference.agent.algorithm
        return dataclasses.replace(reference.agent, algorithm=dataclasses.replace(
            algorithm, kwargs={**algorithm.kwargs, **kwargs}
        ))
    if name == "random_search":
        kwargs = {"population": population, **kwargs}
    return AgentSpec(algorithm=Spec(name, kwargs))


def _agent_environment(reference: ExperimentConfig, name: str, overrides: dict | None):
    """The reference environment, with the axes THIS agent's paradigm forces.

    Only two axes may differ, and only at `--level=paradigm`: how a macro moves, and therefore
    where an episode has to start. Everything else is the reference's, which is what keeps the
    rows a comparison rather than a collection.
    """
    axes = (overrides or {}).get(name, {})
    if not axes:
        return reference.environment
    return dataclasses.replace(reference.environment, **{
        axis: Spec.parse(value) for axis, value in axes.items()
    })


def build_comparison(
    reference: ExperimentConfig, agents: list[str], seeds: int, population: int,
    level: str = "environment", agent_kwargs: dict | None = None,
    agent_environment: dict | None = None,
) -> list[ExperimentConfig]:
    """One config per (agent, seed), all sharing the reference's environment exactly."""
    configs = []
    for name in agents:
        for seed in range(seeds):
            configs.append(ExperimentConfig(
                name=f"{name}-seed{seed}",
                seed=seed,
                environment=_agent_environment(reference, name, agent_environment),
                agent=_agent_spec(name, reference, population, agent_kwargs),
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
    lines.extend(_paradigm_note(results))
    return "\n".join(lines)


def _paradigm_note(results: dict[str, list[dict]]) -> list[str]:
    """What a cross-paradigm table does NOT claim, printed under the rows that make it one.

    `--level=paradigm` drops two axes, and neither loss is cosmetic:

      * **env_steps stop being one unit.** A constructive env step PLACES a macro; a perturbation
        env step MOVES one. Both are one `step()` call and one reward evaluation, so the budget
        matches INTERACTION - which is real, and is the only thing it matches. It does not match
        work, and the README's "sample-matched, not compute-matched" caveat gets a second half
        here rather than being quietly stretched to cover this too.
      * **the warm start differs.** A perturbation agent has to start from a complete placement,
        so it is typically handed one a constructive agent was never given. If that placement
        came from `greedy_wiremask`, the row is reporting what local search added to a strong
        heuristic, not what it achieves alone - and `greedy_wiremask`'s own row is the number to
        read it against.

    Printed only when the rows actually differ, so an ordinary comparison is not lectured.
    """
    spaces = {run.get("action_space") for runs in results.values() for run in runs}
    starts = {run.get("initial_placement") for runs in results.values() for run in runs}
    if len(spaces - {None}) <= 1 and len(starts - {None}) <= 1:
        return []

    lines = ["", "THESE ROWS MOVE MACROS DIFFERENTLY (--level=paradigm)", ""]
    header = f"{'agent':<22s}{'action space':>16s}{'warm start':>28s}"
    lines.extend([header, "-" * len(header)])
    for name, runs in sorted(results.items()):
        row = runs[0]
        lines.append(f"{name:<22s}{str(row.get('action_space')):>16s}"
                     f"{str(row.get('initial_placement')):>28s}")
    lines.extend([
        "",
        "One env step is one macro PLACED under a constructive space and one macro MOVED under a "
        "perturbation one.",
        "The budget matches environment INTERACTION across these rows - one step() call, one "
        "reward evaluation each - and nothing else:",
        "it is not work-matched, and 'grad steps' is not the only axis it fails to price.",
        "A perturbation agent also starts from a complete placement a constructive agent was "
        "never given; where that placement",
        "came from a heuristic, read its row against that heuristic's own.",
    ])
    return lines


def _rank_summary(by_design: dict[str, dict[str, list[dict]]]) -> str:
    """Agents ranked per design, then averaged - the only defensible way to aggregate designs.

    Averaging HPWL across benchmarks is meaningless: adaptec1 and bigblue1 differ by orders of
    magnitude, so a mean is dominated by whichever design happens to be largest and says nothing
    about which method is better. Ranking within each design and averaging the ranks is what the
    comparison actually supports, and it is the aggregation BBOPlace-Bench-style claims need.
    """
    agents = sorted({name for designs in by_design.values() for name in designs})
    ranks: dict[str, list[int]] = {name: [] for name in agents}
    for design, results in by_design.items():
        ordered = sorted(results, key=lambda name: statistics.fmean(
            run["real_hpwl"] for run in results[name]))
        for position, name in enumerate(ordered, start=1):
            ranks[name].append(position)

    header = (f"{'agent':<22s}{'mean rank':>11s}{'wins':>7s}{'designs':>9s}"
              f"{'all legal':>11s}")
    lines = ["", f"ACROSS {len(by_design)} DESIGNS", "=" * len(header), header, "-" * len(header)]
    for name in sorted(agents, key=lambda n: statistics.fmean(ranks[n]) if ranks[n] else 99):
        placings = ranks[name]
        wins = sum(1 for position in placings if position == 1)
        legal = all(
            run["is_legal"]
            for results in by_design.values() for run in results.get(name, [])
        )
        lines.append(f"{name:<22s}{statistics.fmean(placings):>11.2f}{wins:>7d}"
                     f"{len(placings):>9d}{('yes' if legal else 'NO'):>11s}")
    lines.append("")
    lines.append("Ranked within each design, then averaged - HPWL is not comparable ACROSS "
                 "designs, so a mean of it would be dominated by the largest one.")
    lines.append("'all legal' is yes only when every design and every seed was legal; a row "
                 "marked NO failed to produce a result somewhere, whatever its rank says.")
    return "\n".join(lines)


def _write_results(path: pathlib.Path, by_design: dict[str, dict[str, list[dict]]],
                   references: dict[str, ExperimentConfig], budget: Budget,
                   level: str, protocol_hash: str) -> pathlib.Path:
    """The tables as data, so they can be regenerated and checked without re-running anything.

    The script's own docstring has always promised this ("so the table can be regenerated"), and
    a printed table plus a directory of manifests did not deliver it: reconstructing a row meant
    re-running the agent. Every number shown is written here beside the hash of the environment it
    was produced under, the protocol shared across designs, and each run's own full hash.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({
        "level": level,
        "protocol_hash": protocol_hash,
        "budget": budget.to_dict(),
        "metrics": METRICS,
        "fingerprint": fingerprint(),
        "designs": {
            design: {
                "shared_hash": references[design].hash_at(level),
                "runs": {name: sorted(runs, key=lambda run: run["seed"])
                         for name, runs in results.items()},
            }
            for design, results in by_design.items()
        },
    }, indent=2, sort_keys=True) + "\n")
    return path


def _run_design(benchmark_dir, args, budget, agents, on_result=None
                ) -> tuple[ExperimentConfig, dict, pathlib.Path]:
    """Every agent x seed on ONE design, sharing one loaded netlist and one asserted environment.

    `on_result` is called after each run finishes, so the results file is written incrementally.
    Learned the hard way: a comparison that only writes at the end throws away every completed
    agent when a later one dies - eight finished runs lost to the ninth exhausting GPU memory.
    A run that cost real compute should survive whatever happens to the next one.
    """
    reference = _with_overrides(build_preset(args.preset, benchmark_dir, budget=budget), args)
    configs = build_comparison(reference, agents, args.seeds, args.population, args.level,
                               args.agent_kwargs, args.agent_environment)
    output_root = (args.output_dir or (benchmark_dir / "comparison"))
    if len(args.benchmark_dirs) > 1 and args.output_dir is not None:
        output_root = output_root / benchmark_dir.name

    Log.info(f"{benchmark_dir.name}: {len(agents)} agents x {args.seeds} seed(s)")
    Log.info(f"  shared {args.level}: {reference.hash_at(args.level)}")

    # Loaded once and shared: two agents parsing the same netlist twice would still be comparable,
    # but sharing it removes the possibility entirely and saves the parse.
    benchmark = build_benchmark(reference)
    results: dict[str, list[dict]] = {}
    built = None
    for config in configs:
        agent_name = config.name.rsplit("-seed", 1)[0]
        built = build(config, benchmark=benchmark)
        state, log = run_experiment(
            config, output_root / config.name, built=built,
            eval_every=args.eval_every, log_every=max(1, args.eval_every or 1),
        )
        # Scored exactly as run_experiment scores it: the agent's orientations, its action space
        # and its warm start all change what a placement IS worth, and a table that dropped them
        # would disagree with the per-run logs beside it.
        measured = score(
            benchmark, built.agent.best_positions(state), built.n_placed,
            best_orientations(built.agent, state), built.action_space, built.initial_positions,
        )
        # What this run actually spent, from its own last log line - not what the budget offered.
        # An agent that reports convergence stops early, and a table that prints the budget over
        # such a row claims a compute match the run itself declined.
        final = log[-1] if log else {}
        results.setdefault(agent_name, []).append({
            **measured,
            # The two axes a paradigm comparison lets vary, recorded per row: a reader of
            # results.json has to be able to see which rows moved macros which way.
            "action_space": built.config.environment.action_space.name,
            "initial_placement": built.config.environment.initial_placement.name,
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
        if on_result is not None:
            on_result(benchmark_dir.name, built.config, results)
    return built.config, results, output_root


def main() -> None:
    Log.configure()
    args = _parse_args(sys.argv)
    missing = [str(d) for d in args.benchmark_dirs if not d.exists()]
    if missing:
        Log.error(f"not found: {', '.join(missing)} - run scripts/download_benchmarks.py first.")
        sys.exit(1)

    budget = _budget(args)
    agents = [name.strip() for name in args.agents.split(",") if name.strip()]
    if not agents:
        raise SystemExit("--agents is empty; name at least one agent to compare")

    # Across designs the environments CANNOT match - they are different netlists - so the suite
    # asserts the protocol instead: same grid, order, budget, reward, observation, mask, warm
    # start, legalization and physical stack, on every design. Without it "we ran the same
    # experiment on five benchmarks" is a claim rather than a check.
    references = {d.name: _with_overrides(build_preset(args.preset, d, budget=budget), args)
                  for d in args.benchmark_dirs}
    if len(references) > 1:
        assert_comparable(*references.values(), level="protocol")
        Log.info(f"suite of {len(references)} designs, shared protocol: "
                 f"{next(iter(references.values())).protocol_hash()}")
    Log.info(f"  {describe_determinism()}")

    by_design: dict[str, dict[str, list[dict]]] = {}
    resolved: dict[str, ExperimentConfig] = {}
    roots = []
    results_root = args.output_dir or (args.benchmark_dirs[0] / "comparison")

    def checkpoint_results(design: str, config: ExperimentConfig, results: dict) -> None:
        """Write the results file after every finished run, not only after the last one."""
        by_design[design] = results
        resolved[design] = config
        _write_results(results_root / RESULTS_NAME, by_design, resolved, budget, args.level,
                       config.protocol_hash())

    for benchmark_dir in args.benchmark_dirs:
        config, results, root = _run_design(benchmark_dir, args, budget, agents, checkpoint_results)
        by_design[benchmark_dir.name] = results
        resolved[benchmark_dir.name] = config
        roots.append(root)

    print()
    for design, results in by_design.items():
        if len(by_design) > 1:
            print(f"--- {design} ---")
        print(_format_table(results, budget, args.level))
        print()
    if len(by_design) > 1:
        print(_rank_summary(by_design))
        print()

    results_path = _write_results(
        results_root / RESULTS_NAME, by_design, resolved, budget, args.level,
        next(iter(resolved.values())).protocol_hash(),
    )
    print(f"per-run manifests and logs: {', '.join(str(root) for root in dict.fromkeys(roots))}")
    print(f"the tables as data:         {results_path}")


if __name__ == "__main__":
    main()
