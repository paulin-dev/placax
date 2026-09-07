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
    return parser.parse_args(argv[1:])


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
