"""OpenROAD-specific Validator: area/utilization always; timing and routing only when asked for.

**Routing is why this matters.** Every number this project reports is a half-perimeter proxy, and
ChiPBench's finding is that proxy rankings do not survive a real flow - two placements can swap
order once they are actually routed. `route="global"` reports the routed wirelength HPWL is a
proxy for; `route="detailed"` additionally reports DRC violations, which is where a placement with
excellent wirelength turns out to be unroutable.

**The output regexes below are unverified against a real OpenROAD run**, for the same reason the
rest of the physical path is: no DEF/LEF design ships here and OpenROAD is not a dependency. They
degrade the way this module already degrades everywhere else - a pattern that does not match
yields None, never a plausible-looking substitute for a number nobody measured. Someone with a
real design should check them against their tool's actual report text before trusting a value.
"""
import pathlib
import re
import subprocess

from placax_tools.validator import PPAResult, Validator


ROUTE_MODES = (None, "global", "detailed")
"""How far to route before measuring. `global` gives routed wirelength; `detailed` additionally
gives DRC violations, at substantially more runtime."""


def build_openroad_script(
    def_path: pathlib.Path,
    lef_paths: list[pathlib.Path],
    liberty_path: pathlib.Path | None = None,
    clock_period_ns: float | None = None,
    wire_rc_layer: str = "metal3",
    clock_name: str = "core_clock",
    route: str | None = None,
) -> str:
    """Builds OpenROAD TCL: area always, timing if liberty+clock are given, routing if asked."""
    if route not in ROUTE_MODES:
        raise ValueError(
            f"unknown route mode {route!r}; choose one of "
            f"{', '.join(repr(mode) for mode in ROUTE_MODES)}"
        )
    # 1. Load the physical design: tech/cell LEFs, then the placed DEF.
    lines = [f"read_lef {p}" for p in lef_paths]
    lines.append(f"read_def {def_path}")
    lines.append("report_design_area")

    # 2. Only attempt timing analysis if we have both a liberty file and a clock period.
    if liberty_path is not None and clock_period_ns is not None:
        lines.append(f"read_liberty {liberty_path}")
        lines.append(f"create_clock -period {clock_period_ns} [get_ports *] -name {clock_name}")
        lines.append(f"set_wire_rc -layer {wire_rc_layer}")
        lines.append("estimate_parasitics -placement")
        lines.append("report_checks -path_delay max")

    # 3. Route, if asked. Global route is enough for a routed-wirelength number; detailed route is
    #    what produces real DRC violations, and costs far more time.
    if route is not None:
        lines.append("global_route")
        if route == "detailed":
            lines.append("detailed_route")
            lines.append("report_drc")

    return "\n".join(lines) + "\n"


_AREA_RE = re.compile(r"Design area\s+([\d.]+)\s+u\^2\s+([\d.]+)%\s+utilization")
_SLACK_RE = re.compile(r"slack\s+\(?(?:MET|VIOLATED)?\)?\s*(-?[\d.]+)", re.IGNORECASE)
_WIRELENGTH_RE = re.compile(r"Total wire ?length:?\s*([\d.]+)", re.IGNORECASE)
_VIAS_RE = re.compile(r"Total number of vias:?\s*(\d+)", re.IGNORECASE)
_DRC_RE = re.compile(r"(?:total\s+)?violations?(?:\s+found)?:?\s*(\d+)", re.IGNORECASE)


def parse_openroad_output(raw_output: str) -> PPAResult:
    """Extracts what is actually present: area always, timing and routing only if they ran.

    Every field is a `search`, not a `match`, and every miss becomes None - a validator that
    reports a number nobody computed is worse than one that reports nothing.
    """
    area_match = _AREA_RE.search(raw_output)
    slack_match = _SLACK_RE.search(raw_output)
    wirelength_match = _WIRELENGTH_RE.search(raw_output)
    vias_match = _VIAS_RE.search(raw_output)
    drc_match = _DRC_RE.search(raw_output)
    return PPAResult(
        design_area=float(area_match.group(1)) if area_match else None,
        utilization_pct=float(area_match.group(2)) if area_match else None,
        timing_slack=float(slack_match.group(1)) if slack_match else None,
        raw_output=raw_output,
        routed_wirelength=float(wirelength_match.group(1)) if wirelength_match else None,
        via_count=int(vias_match.group(1)) if vias_match else None,
        drc_violations=int(drc_match.group(1)) if drc_match else None,
    )


class OpenROADValidator(Validator):
    """Default Validator, requiring a real OpenROAD install."""

    def __init__(
        self,
        liberty_path: pathlib.Path | None = None,
        clock_period_ns: float | None = None,
        wire_rc_layer: str = "metal3",
        clock_name: str = "core_clock",
        openroad_binary: str = "openroad",
        route: str | None = None,
    ):
        self.liberty_path = liberty_path
        self.clock_period_ns = clock_period_ns
        self.wire_rc_layer = wire_rc_layer
        self.clock_name = clock_name
        self.openroad_binary = openroad_binary
        # Part of the experiment, not of this machine: routing changes the result, so it belongs
        # in the validator Spec's kwargs and therefore in the environment hash.
        self.route = route

    def _write_script(
        self, def_path: pathlib.Path, lef_paths: list[pathlib.Path], output_dir: pathlib.Path
    ) -> pathlib.Path:
        """Writes the TCL script OpenROAD will execute."""
        script_path = output_dir / "validate.tcl"
        script_path.write_text(
            build_openroad_script(
                def_path, lef_paths, self.liberty_path, self.clock_period_ns,
                self.wire_rc_layer, self.clock_name, self.route,
            )
        )
        return script_path

    def _run_openroad(self, script_path: pathlib.Path) -> str:
        """Runs OpenROAD on the script, returns its stdout report."""
        result = subprocess.run(
            [self.openroad_binary, "-exit", str(script_path)],
            capture_output=True, text=True, check=True,
        )
        return result.stdout

    def validate(
        self, def_path: pathlib.Path, lef_paths: list[pathlib.Path], output_dir: pathlib.Path
    ) -> PPAResult:
        """Writes the TCL script, runs OpenROAD, parses its output."""
        # 1. Make sure the output directory exists before anything writes to it.
        output_dir.mkdir(parents=True, exist_ok=True)
        # 2. Generate the TCL script driving this specific validation run.
        script_path = self._write_script(def_path, lef_paths, output_dir)
        # 3. Run OpenROAD and capture its textual report.
        raw_output = self._run_openroad(script_path)
        # 4. Turn that free-form text into structured numbers.
        return parse_openroad_output(raw_output)
