"""OpenROAD-specific Validator implementation.

Honestly scoped, not a full PPA report: area/utilization are always
computed (report_design_area, confirmed real - verified output format
"Design area N u^2 M% utilization." across multiple real OpenROAD log
excerpts). Timing is computed only if a liberty file and clock period
are given - real timing needs a real technology library, not something
placax has by default. Power is deliberately NOT included: a
meaningful power number needs real switching activity data (SAIF/VCD),
which nothing in this pipeline produces - faking a number would be
worse than omitting it.

Not verified end-to-end in this environment - OpenROAD isn't installed
here. The TCL script generation follows OpenROAD's own documented
command set exactly (read_lef, read_def, report_design_area,
report_checks - all confirmed directly, not guessed), but only running
OpenROAD confirms the full flow works."""
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
    """Real OpenROAD TCL commands, confirmed against OpenROAD's own
    documentation - not guessed. Timing lines only included if both
    liberty_path and clock_period_ns are given. Kept as a standalone,
    independently testable function - OpenROADValidator just calls it.

    wire_rc_layer defaults to "metal3", a common but not universal
    layer name - real PDKs vary (M3, metal2, etc.), so this must be
    overridable, not hardcoded to one technology's convention."""
    lines = [f"read_lef {p}" for p in lef_paths]
    lines.append(f"read_def {def_path}")
    lines.append("report_design_area")

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
    """Extracts what's actually there - area/utilization always,
    timing slack only if a timing report was actually run."""
    area_match = _AREA_RE.search(raw_output)
    slack_match = _SLACK_RE.search(raw_output)
    return PPAResult(
        design_area=float(area_match.group(1)) if area_match else None,
        utilization_pct=float(area_match.group(2)) if area_match else None,
        timing_slack=float(slack_match.group(1)) if slack_match else None,
        raw_output=raw_output,
    )


class OpenROADValidator(Validator):
    """Default Validator implementation. Requires a real OpenROAD
    install - liberty_path/clock_period_ns/wire_rc_layer/clock_name are
    specific to how this particular validator does timing analysis and
    live here in __init__, not in validate()'s signature, so any other
    Validator subclass can have completely different construction
    needs while still being called identically via validate()."""

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

    def validate(
        self, def_path: pathlib.Path, lef_paths: list[pathlib.Path], output_dir: pathlib.Path
    ) -> PPAResult:
        output_dir.mkdir(parents=True, exist_ok=True)
        script = build_openroad_script(
            def_path, lef_paths, self.liberty_path, self.clock_period_ns,
            self.wire_rc_layer, self.clock_name,
        )
        script_path = output_dir / "validate.tcl"
        script_path.write_text(script)

        result = subprocess.run(
            [self.openroad_binary, "-exit", str(script_path)],
            capture_output=True, text=True, check=True,
        )
        return parse_openroad_output(result.stdout)
