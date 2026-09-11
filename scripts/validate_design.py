"""Runs the full physical flow on a macro-placed DEF: standard-cell placement, then real PPA.

This is the entry point for the last box in the architecture, the one that had no code behind it:
every number this project reported before was a geometric half-perimeter proxy, never a measured
physical metric. `place_and_validate` composes a CellPlacer and a Validator without naming either,
so swapping DREAMPlace for AutoDMP or OpenROAD for another signoff tool is a change here and
nowhere else.

    python -m scripts.validate_design --def_path=placed.def --lef=tech.lef --lef=cells.lef \\
        --use_docker --liberty=cells.lib --clock_period_ns=2.0

Timing is only reported when BOTH --liberty and --clock_period_ns are given. Without them area and
utilization still come back and `timing_slack` is None - not computed, not guessed.

The Bookshelf benchmarks in benchmarks/ cannot reach this: they carry no LEF/DEF, which is
exactly why validation was never wired up. Use `scripts/run_pipeline.py` for those, which stops
at DREAMPlace and reports HPWL, and bring a DEF/LEF design here.

**Prefer --config.** Passing a run's own `manifest.json` takes the cell placer and validator from
`EnvironmentSpec.physical` and writes `ppa.json` next to that run's manifest, carrying its
`full_hash`. Without it the tools come from the flags below and the resulting number is
attributable to nothing - which is how a PPA measurement ends up sitting outside the
reproducibility envelope everything else in this project lives in.
"""
import argparse
import pathlib
import sys

from placax import _device  # noqa: F401  must precede jax imports
from placax.log import Log
from placax_tools.dreamplace.cell_placer import DREAMPlaceCellPlacer
from placax_tools.openroad.validator import OpenROADValidator
from placax_tools.pipeline import place_and_validate, validate_only


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--def_path", type=pathlib.Path, required=True,
                        help="A DEF with every macro already placed - the agent's output.")
    parser.add_argument("--lef", type=pathlib.Path, action="append", default=[], dest="lef_paths",
                        help="A tech or cell LEF. Repeat for several; at least one is required.")
    parser.add_argument("--output_dir", type=pathlib.Path, default=None,
                        help="Where the placed DEF, TCL script and reports go "
                             "(default: <def_path's directory>/validate).")
    parser.add_argument("--skip_cell_placement", action="store_true",
                        help="Validate the DEF as given, without running a cell placer first - for "
                             "a design whose standard cells are already placed.")
    parser.add_argument("--dreamplace_root", type=pathlib.Path, default=None,
                        help="Path to a DREAMPlace checkout. Required unless --use_docker or "
                             "--skip_cell_placement.")
    parser.add_argument("--use_docker", action="store_true",
                        help="Run DREAMPlace via its official Docker image instead of a local "
                             "checkout.")
    parser.add_argument("--gpu", action="store_true", help="Run DREAMPlace on GPU.")
    parser.add_argument("--target_density", type=float, default=1.0)
    parser.add_argument("--liberty", type=pathlib.Path, default=None,
                        help="Liberty file. Required together with --clock_period_ns for timing; "
                             "without both, timing is not attempted and reports as None.")
    parser.add_argument("--clock_period_ns", type=float, default=None,
                        help="Clock period in ns, for timing analysis. See --liberty.")
    parser.add_argument("--openroad_binary", default="openroad",
                        help="OpenROAD executable (default: %(default)s).")
    parser.add_argument("--config", type=pathlib.Path, default=None,
                        help="A run's manifest.json (or a bare ExperimentConfig JSON). Takes the cell "
                             "placer and validator from its EnvironmentSpec.physical instead of "
                             "the flags above, and writes ppa.json carrying the run's full_hash "
                             "- so the PPA number is attributable to the run whose placement it "
                             "measured. Strongly preferred over the bare flags.")
    parser.add_argument("--output_run_dir", type=pathlib.Path, default=None,
                        help="Where to write ppa.json with --config (default: --output_dir).")
    parsed = parser.parse_args(argv[1:])
    # Kept so the --config path can tell "left at its default" from "asked for", which is the
    # difference between a flag that is irrelevant there and one that is quietly ignored.
    parsed.parser_defaults = {
        action.dest: action.default for action in parser._actions  # noqa: SLF001
    }
    return parsed


CONFIG_OWNED_FLAGS = ("target_density", "liberty", "clock_period_ns")
"""Flags that describe the EXPERIMENT, not this machine, and so belong in the config.

With --config these used to be read and then ignored, silently: someone following this script's
own documented timing recipe (`--config=... --liberty=... --clock_period_ns=...`) got area and
utilization back with timing reported as a dash, and nothing said why. They are refused now, with
the Spec to write instead - the alternative, letting a flag override the config, would put a
number in ppa.json under a full_hash whose config says something else."""


def _reject_config_owned_flags(args, parser_defaults: dict) -> None:
    """Refuses flags that would change the result while the config claims otherwise."""
    given = [name for name in CONFIG_OWNED_FLAGS
             if getattr(args, name) != parser_defaults.get(name)]
    if not given:
        return
    settings = ", ".join(f"{name}={getattr(args, name)!r}" for name in given)
    raise SystemExit(
        f"--{' and --'.join(given)} describe the EXPERIMENT, not this machine, so with --config "
        f"they belong in the config rather than on the command line - otherwise ppa.json would "
        f"carry a run's full_hash beside a number its config cannot account for. Put them in "
        f"EnvironmentSpec.physical, e.g. Spec('openroad', {{'liberty': ..., "
        f"'clock_period_ns': ...}}) and Spec('dreamplace', {{'target_density': ...}}), and re-run "
        f"without {settings}. --dreamplace_root, --use_docker, --gpu and --openroad_binary stay "
        f"here: where a tool is installed is this machine's business and is deliberately not "
        f"hashed."
    )


def _run_from_config(args, output_dir: pathlib.Path, parser_defaults: dict) -> None:
    """The attributable path: tools named by the config, result written back beside its manifest."""
    from placax_agents.experiment.build import build
    from placax_agents.experiment.config import ExperimentConfig
    from placax_agents.experiment.physical import evaluate_physical, write_ppa

    _reject_config_owned_flags(args, parser_defaults)
    config = ExperimentConfig.read(args.config)
    built = build(config)
    # Where the binaries live on this host - deliberately not part of the config or its hash.
    machine = {
        "dreamplace_root": args.dreamplace_root, "use_docker": args.use_docker, "gpu": args.gpu,
        "openroad_binary": args.openroad_binary,
    }
    result = evaluate_physical(
        built, args.def_path, args.lef_paths, output_dir,
        skip_cell_placement=args.skip_cell_placement, machine=machine,
    )
    ppa_path = write_ppa(args.output_run_dir or output_dir, result)

    print()
    print(f"placed design: {result.def_path}")
    print(f"  design area:  {_or_dash(result.design_area, 'u^2')}")
    print(f"  utilization:  {_or_dash(result.utilization_pct, '%')}")
    print(f"  timing slack: {_or_dash(result.timing_slack, 'ns')}")
    print(f"  cell placer:  {result.cell_placer or '- (skipped)'}")
    print(f"  validator:    {result.validator}")
    print(f"  run:          {result.full_hash}")
    print(f"reports: {output_dir}")
    print(f"ppa: {ppa_path}")


def main() -> None:
    Log.configure()
    args = _parse_args(sys.argv)

    if not args.def_path.exists():
        Log.error(f"'{args.def_path}' not found.")
        sys.exit(1)
    if not args.lef_paths:
        Log.error("at least one --lef is required: OpenROAD cannot read a DEF without its LEFs.")
        sys.exit(1)
    missing = [p for p in args.lef_paths if not p.exists()]
    if missing:
        Log.error(f"LEF file(s) not found: {', '.join(str(p) for p in missing)}")
        sys.exit(1)

    output_dir = args.output_dir or (args.def_path.parent / "validate")
    if args.config is not None:
        _run_from_config(args, output_dir, args.parser_defaults)
        return

    Log.warning(
        "no --config: the cell placer and validator are coming from CLI flags, so the numbers "
        "below will not be attributable to any recorded run. Pass --config=<a manifest's config> "
        "to name the tools in the experiment's own configuration instead."
    )
    validator = OpenROADValidator(
        liberty_path=args.liberty, clock_period_ns=args.clock_period_ns,
        openroad_binary=args.openroad_binary,
    )
    if args.liberty is None or args.clock_period_ns is None:
        Log.info("no --liberty/--clock_period_ns: reporting area and utilization only, no timing")

    if args.skip_cell_placement:
        Log.info(f"validating {args.def_path} as given ...")
        ppa = validate_only(args.def_path, args.lef_paths, output_dir, validator)
        placed_def = args.def_path
    else:
        if args.dreamplace_root is None and not args.use_docker:
            Log.error("pass --dreamplace_root or --use_docker to place standard cells, or "
                      "--skip_cell_placement to validate the DEF as it stands.")
            sys.exit(1)
        cell_placer = DREAMPlaceCellPlacer(
            dreamplace_root=args.dreamplace_root, gpu=args.gpu,
            target_density=args.target_density, use_docker=args.use_docker,
        )
        Log.info(f"placing standard cells around the macros in {args.def_path} ...")
        result = place_and_validate(
            args.def_path, args.lef_paths, output_dir, cell_placer, validator
        )
        placed_def, ppa = result.def_path, result.ppa

    print()
    print(f"placed design: {placed_def}")
    print(f"  design area:  {_or_dash(ppa.design_area, 'u^2')}")
    print(f"  utilization:  {_or_dash(ppa.utilization_pct, '%')}")
    print(f"  timing slack: {_or_dash(ppa.timing_slack, 'ns')}")
    print(f"reports: {output_dir}")


def _or_dash(value: float | None, unit: str) -> str:
    """A metric that wasn't computed prints as a dash, never as a plausible-looking zero."""
    return f"{value:,.3f} {unit}" if value is not None else "- (not computed)"


if __name__ == "__main__":
    main()
