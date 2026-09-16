"""Turns an OpenROAD-flow-scripts design into a macro-placement benchmark with a real PDK behind it.

    python -m scripts.make_orfs_benchmark --design=nangate45/ariane133

The benchmarks that ship here carry no technology. adaptec1 and bigblue1 are ISPD 2005 Bookshelf:
geometry and connectivity only, so a converted design can be placed and measured for HPWL,
legality and area, but not routed or timed. ariane133's protobuf is Circuit Training's CLUSTERED
netlist - its standard cells are grouped into ~800 soft blocks with no cell types - so no real
cell can be recovered from it at all.

This builds the other kind. ORFS synthesizes the design's RTL with Yosys and floorplans it in its
own PDK (stage `2_1_floorplan`: die, rows and routing tracks, every instance still UNPLACED), and
that floorplan becomes the benchmark. An agent places its `CLASS BLOCK` macros; OpenROAD places
the standard cells around them and measures routed wirelength, vias and timing on real layers
with real liberty data.

Written to `benchmarks/<design>-orfs/`:

  * `<design>.def` - the floorplan,
  * its LEFs (technology, standard cells, each macro), which `load_def` reads for sizes and pins,
  * `physical.json` - the `EnvironmentSpec.physical` that measures it the way ORFS would: the
    OpenROAD cell placer with the platform's pin layers, density and IO constraints, and the
    OpenROAD validator with its liberty files, clock and routing layers,
  * `provenance.json` - which image, which design config, which variables.

Liberty files and IO constraints are referenced at their paths INSIDE the pinned image, which are
the same on every machine running it - a host path in `physical.json` would put this machine's
directory layout into every experiment hash. Run the physical flow with `--use_docker`.

Synthesis is the slow part: tens of minutes and several GB of memory for ariane133.
"""
import argparse
import datetime
import json
import os
import pathlib
import re
import shutil
import subprocess
import sys

from placax_tools.openroad.docker import OPENROAD_BINARY_IN_IMAGE, OPENROAD_IMAGE, image_available

FLOW_IN_IMAGE = "/OpenROAD-flow-scripts/flow"
VARIABLES = (
    "DESIGN_NAME", "DESIGN_NICKNAME", "PLATFORM", "TECH_LEF", "SC_LEF", "ADDITIONAL_LEFS",
    "LIB_FILES", "SDC_FILE", "IO_CONSTRAINTS", "PLACE_DENSITY", "PLACE_DENSITY_LB_ADDON",
    "IO_PLACER_H", "IO_PLACER_V", "MIN_ROUTING_LAYER", "MAX_ROUTING_LAYER", "RESULTS_DIR",
    "CORE_UTILIZATION",
)
"""The ORFS variables the benchmark is built from, read with its own `make print-<VAR>`."""

WIRE_RC_LAYERS = {"nangate45": "metal3", "sky130hd": "met2", "asap7": "M3"}
"""A mid-stack routing layer per platform, for the validator's placement-parasitics estimate."""


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--design", default="nangate45/ariane133",
                        help="<platform>/<design> under the image's flow/designs (default: "
                             "%(default)s).")
    parser.add_argument("--output_dir", type=pathlib.Path, default=None,
                        help="Default: benchmarks/<design>-orfs.")
    parser.add_argument("--work_dir", type=pathlib.Path, default=None,
                        help="Where ORFS writes its logs, reports and results (default: "
                             "runs/orfs-<design>). Kept: a rerun reuses a finished synthesis.")
    parser.add_argument("--route", default="global", choices=("none", "global", "detailed"),
                        help="How far physical.json's validator routes (default: %(default)s).")
    parser.add_argument("--image", default=OPENROAD_IMAGE)
    return parser.parse_args(argv[1:])


def _docker(image: str, work_dir: pathlib.Path, script: str) -> subprocess.CompletedProcess:
    """Runs `script` in bash inside the image, with ORFS's environment and `work_dir` mounted."""
    argv = ["docker", "run", "--rm"]
    if hasattr(os, "getuid"):
        argv += ["--user", f"{os.getuid()}:{os.getgid()}"]
    argv += ["-e", "HOME=/tmp", "-v", f"{work_dir}:{work_dir}", image, "bash", "-c",
             "source /OpenROAD-flow-scripts/env.sh > /dev/null 2>&1; " + script]
    return subprocess.run(argv, capture_output=True, text=True)


def _make(design_config: str, work_dir: pathlib.Path, targets: str) -> str:
    return (f"make --no-print-directory -C {FLOW_IN_IMAGE} DESIGN_CONFIG={design_config} "
            f"WORK_HOME={work_dir} {targets}")


def read_variables(output: str) -> dict[str, str]:
    """`make print-VAR` lines (`VAR: value`) as a dict, values whitespace-normalized."""
    variables = {}
    for line in output.splitlines():
        match = re.match(r"^([A-Z_][A-Z0-9_]*): ?(.*)$", line)
        if match and match.group(1) in VARIABLES:
            variables[match.group(1)] = " ".join(match.group(2).split())
    return variables


def physical_spec(variables: dict[str, str], route: str | None) -> dict:
    """The physical stack that measures this design the way its ORFS flow would."""
    routing_layers = f"{variables['MIN_ROUTING_LAYER']}-{variables['MAX_ROUTING_LAYER']}"
    placer = {
        "density": float(variables.get("PLACE_DENSITY") or 0.7),
        "pin_hor_layers": variables["IO_PLACER_H"],
        "pin_ver_layers": variables["IO_PLACER_V"],
        "routing_layers": routing_layers,
    }
    if variables.get("PLACE_DENSITY_LB_ADDON"):
        placer["density_lb_addon"] = float(variables["PLACE_DENSITY_LB_ADDON"])
    if variables.get("IO_CONSTRAINTS"):
        placer["io_constraints"] = variables["IO_CONSTRAINTS"]
    clock_port, clock_period = read_clock(variables.get("SDC_FILE_TEXT", ""))
    validator = {
        "liberty_path": variables["LIB_FILES"].split(),
        "wire_rc_layer": WIRE_RC_LAYERS.get(variables["PLATFORM"], "metal3"),
        "routing_layers": routing_layers,
        "route": route,
    }
    if clock_port is not None:
        validator.update(clock_port=clock_port, clock_period_ns=clock_period)
    return {
        "cell_placer": {"name": "openroad", "kwargs": placer},
        "validator": {"name": "openroad", "kwargs": validator},
    }


def read_clock(sdc_text: str) -> tuple[str | None, float | None]:
    """The clock port and period an ORFS SDC declares, following its `set` variables."""
    values = dict(re.findall(r"^\s*set\s+(\w+)\s+(\S+)", sdc_text, re.MULTILINE))
    port, period = values.get("clk_port_name"), values.get("clk_period")
    try:
        return port, (float(period) if period is not None else None)
    except ValueError:
        return port, None


def main() -> None:
    args = _parse_args(sys.argv)
    platform, design = args.design.split("/", 1)
    output_dir = (args.output_dir or pathlib.Path("benchmarks") / f"{design}-orfs").resolve()
    work_dir = (args.work_dir or pathlib.Path("runs") / f"orfs-{design}").resolve()
    work_dir.mkdir(parents=True, exist_ok=True)
    if not image_available(args.image):
        sys.exit(f"Docker image {args.image} is not available: docker pull {args.image}")
    design_config = f"./designs/{platform}/{design}/config.mk"

    printed = _docker(args.image, work_dir, _make(
        design_config, work_dir, " ".join(f"print-{name}" for name in VARIABLES)))
    variables = read_variables(printed.stdout)
    if "RESULTS_DIR" not in variables:
        sys.exit(f"could not read the design's variables:\n{printed.stdout}\n{printed.stderr}")
    floorplan = pathlib.Path(variables["RESULTS_DIR"]) / "2_1_floorplan.odb"

    if not floorplan.exists():
        print(f"synthesizing and floorplanning {args.design} in {work_dir} "
              f"(the slow part - Yosys on the whole design) ...", flush=True)
        built = _docker(args.image, work_dir, _make(design_config, work_dir, str(floorplan)))
        (work_dir / "make.log").write_text(built.stdout + built.stderr)
        if built.returncode != 0 or not floorplan.exists():
            tail = "\n".join((built.stdout + built.stderr).strip().splitlines()[-25:])
            sys.exit(f"ORFS failed; full log in {work_dir / 'make.log'}:\n{tail}")

    output_dir.mkdir(parents=True, exist_ok=True)
    def_name = f"{variables['DESIGN_NAME']}.def"
    export = work_dir / "export.tcl"
    export.write_text(
        f"read_db {floorplan}\nwrite_def {work_dir / def_name}\n"
    )
    exported = _docker(args.image, work_dir,
                       f"{OPENROAD_BINARY_IN_IMAGE} -no_splash -exit {export}")
    if exported.returncode != 0:
        sys.exit(f"exporting the floorplan failed:\n{exported.stdout}\n{exported.stderr}")
    shutil.copy(work_dir / def_name, output_dir / def_name)

    # The LEFs live in the benchmark directory - load_def reads them for every macro's size and
    # pins - and are copied out of the image through the mounted work directory.
    lefs = [variables["TECH_LEF"], variables["SC_LEF"], *variables["ADDITIONAL_LEFS"].split()]
    copied = _docker(args.image, work_dir,
                     f"mkdir -p {work_dir}/lef && cp {' '.join(lefs)} "
                     f"{variables.get('SDC_FILE', '')} {work_dir}/lef/")
    if copied.returncode != 0:
        sys.exit(f"copying the LEFs failed:\n{copied.stderr}")
    for lef in lefs:
        shutil.copy(work_dir / "lef" / pathlib.Path(lef).name, output_dir)
    sdc_path = work_dir / "lef" / pathlib.Path(variables.get("SDC_FILE", "none.sdc")).name
    variables["SDC_FILE_TEXT"] = sdc_path.read_text() if sdc_path.exists() else ""

    route = None if args.route == "none" else args.route
    physical = physical_spec(variables, route)
    (output_dir / "physical.json").write_text(json.dumps(physical, indent=2) + "\n")
    del variables["SDC_FILE_TEXT"]
    (output_dir / "provenance.json").write_text(json.dumps({
        "image": args.image,
        "design_config": f"{FLOW_IN_IMAGE}/designs/{platform}/{design}/config.mk",
        "stage": "2_1_floorplan",
        "created": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
        "variables": variables,
    }, indent=2) + "\n")

    from placax.netlist.def_reader import load_def
    from placax_agents.experiment.physical import design_lefs

    macros, nets = load_def(output_dir / def_name, design_lefs(output_dir))
    print(f"{output_dir}: {len(macros)} macros, {len(nets)} macro-to-macro nets")
    print(f"physical stack: {output_dir / 'physical.json'}")


if __name__ == "__main__":
    main()
