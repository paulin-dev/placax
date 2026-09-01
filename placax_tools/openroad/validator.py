"""OpenROAD-specific Validator: area/utilization always computed, timing only if liberty + clock are given."""
import pathlib
import re
import subprocess

from placax_tools.validator import PPAResult, Validator


def build_openroad_script(
    def_path: pathlib.Path,
    lef_paths: list[pathlib.Path],
    liberty_path: pathlib.Path | None = None,
    clock_period_ns: float | None = None,
    wire_rc_layer: str = "metal3",
    clock_name: str = "core_clock",
) -> str:
    """Builds OpenROAD TCL for area reports, plus timing if both liberty_path and clock_period_ns are given."""
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

    return "\n".join(lines) + "\n"


_AREA_RE = re.compile(r"Design area\s+([\d.]+)\s+u\^2\s+([\d.]+)%\s+utilization")
_SLACK_RE = re.compile(r"slack\s+\(?(?:MET|VIOLATED)?\)?\s*(-?[\d.]+)", re.IGNORECASE)


def parse_openroad_output(raw_output: str) -> PPAResult:
    """Extracts what's actually there in raw_output: area/utilization always, timing slack only if it ran."""
    # Search rather than require a match, since timing lines may simply be absent.
    area_match = _AREA_RE.search(raw_output)
    slack_match = _SLACK_RE.search(raw_output)
    return PPAResult(
        design_area=float(area_match.group(1)) if area_match else None,
        utilization_pct=float(area_match.group(2)) if area_match else None,
        timing_slack=float(slack_match.group(1)) if slack_match else None,
        raw_output=raw_output,
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
    ):
        self.liberty_path = liberty_path
        self.clock_period_ns = clock_period_ns
        self.wire_rc_layer = wire_rc_layer
        self.clock_name = clock_name
        self.openroad_binary = openroad_binary

    def _write_script(
        self, def_path: pathlib.Path, lef_paths: list[pathlib.Path], output_dir: pathlib.Path
    ) -> pathlib.Path:
        """Writes the TCL script OpenROAD will execute."""
        script_path = output_dir / "validate.tcl"
        script_path.write_text(
            build_openroad_script(
                def_path, lef_paths, self.liberty_path, self.clock_period_ns,
                self.wire_rc_layer, self.clock_name,
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
