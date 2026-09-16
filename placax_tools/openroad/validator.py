"""OpenROAD-specific Validator: area, utilization, legality and wirelength always; timing and routing
when asked for.

**Routing is why this matters.** Every number this project reports is a half-perimeter proxy, and
ChiPBench's finding is that proxy rankings do not survive a real flow - two placements can swap
order once they are actually routed. `route="global"` reports the routed wirelength HPWL is a
proxy for; `route="detailed"` additionally reports DRC violations, which is where a placement with
excellent wirelength turns out to be unroutable.

**Verified against a real OpenROAD, and the first run found every parser here wrong.** The
previous version scraped OpenROAD's human-readable reports with patterns nobody had checked
against the tool. Driven through the pinned ORFS image (`openroad/docker.py`) on a real sky130
design, it extracted *nothing*: area printed `um^2` where the pattern wanted `u^2`, slack printed
its number BEFORE the word "slack", and - the one that would have mattered most - `detailed_route`
prints "Number of violations" once per optimization iteration (615, 297, 235, 116, 7, 0 on that
run), so a first-match parser would have reported a clean route as 615 DRC violations.

So the design is now:

  * **What the Tcl API can answer, the script prints itself** as `PLACAX_METRIC <key> <value>`
    lines - area and utilization from `rsz`, slack and TNS from `sta`, the placement checker's
    verdict, and a full-design HPWL summed from the database. Nothing about those depends on how
    a report happens to be laid out this release.
  * **Router results come from the router's own summaries, and always the LAST occurrence**, since
    both routers print progress before they print a result. DRC is counted from the `-output_drc`
    report, which reflects the final state and is empty when the route is clean.
  * **A failed placement check says which rules failed**, from the checker's own per-rule
    summary, because the verdict alone cannot be read: on adaptec1 it is False for a rule
    (one-site gaps) that the ISPD 2005 benchmark never had.
  * **Timing and routing run inside `catch`.** A design whose technology cannot route - a Bookshelf
    benchmark's derived LEF has one layer and no vias - still returns its area, HPWL and legality,
    with the router's reason recorded in `notes` instead of the whole measurement failing.

Anything that did not run is None, never a plausible-looking stand-in for a number nobody measured.
"""
import pathlib
import re
import subprocess

from placax_tools.validator import PPAResult, Validator

ROUTE_MODES = (None, "global", "detailed")
"""How far to route before measuring. `global` gives routed wirelength and via count; `detailed`
additionally gives DRC violations, at far more runtime - 83s for a 9k-instance sky130 design here."""

METRIC_PREFIX = "PLACAX_METRIC"
NOTE_PREFIX = "PLACAX_NOTE"
INFINITE_SLACK = 1e30
"""`sta::worst_slack` returns ~1e39 when there is no constrained path at all - no clock, or a clock
on a port that drives nothing. That is "not measured", and must not reach a results file as a
number."""


def _tcl_quote(path) -> str:
    """A path as a Tcl word: braces keep spaces and brackets literal."""
    return "{" + str(path) + "}"


def build_openroad_script(
    def_path: pathlib.Path,
    lef_paths: list[pathlib.Path],
    liberty_path: pathlib.Path | None = None,
    clock_period_ns: float | None = None,
    wire_rc_layer: str = "metal3",
    clock_name: str = "core_clock",
    route: str | None = None,
    clock_port: str = "clk",
    drc_report: pathlib.Path | None = None,
) -> str:
    """The Tcl that loads a placed design and prints what it measured, one metric per line."""
    if route not in ROUTE_MODES:
        raise ValueError(
            f"unknown route mode {route!r}; choose one of "
            f"{', '.join(repr(mode) for mode in ROUTE_MODES)}"
        )
    metric = f'puts "{METRIC_PREFIX}'
    note = f'puts "{NOTE_PREFIX}'

    lines = [f"read_lef {_tcl_quote(path)}" for path in lef_paths]
    lines.append(f"read_def {_tcl_quote(def_path)}")
    lines += [
        f'{metric} tool_version [ord::openroad_version]"',
        # rsz reports area in square metres; everything here is in microns.
        f'{metric} design_area_um2 [expr {{[rsz::design_area] * 1e12}}]"',
        f'{metric} utilization [rsz::utilization]"',
        # The checker raises on the first class of problem it finds, so its verdict IS the catch.
        "if {[catch {check_placement} placax_err]} {",
        f'  {metric} placement_legal 0"',
        f'  {note} check_placement [string map {{"\\n" " "}} $placax_err]"',
        "} else {",
        f'  {metric} placement_legal 1"',
        "}",
        # Full-design HPWL from the database: every signal net with at least two terminals, from
        # the pins' real positions. Power and ground are skipped - a supply net's bounding box is
        # the die, and it is not wirelength anyone placed.
        "set placax_block [ord::get_db_block]",
        "set placax_hpwl 0",
        "foreach placax_net [$placax_block getNets] {",
        "  if {[$placax_net getSigType] in {POWER GROUND}} { continue }",
        "  if {[llength [$placax_net getITerms]] + [llength [$placax_net getBTerms]] < 2} "
        "{ continue }",
        "  set placax_bb [$placax_net getTermBBox]",
        "  set placax_hpwl [expr {$placax_hpwl + [$placax_bb dx] + [$placax_bb dy]}]",
        "}",
        f'{metric} hpwl_um [expr {{double($placax_hpwl) / [$placax_block getDbUnitsPerMicron]}}]"',
    ]

    # Timing only with both a library and a clock period - never a guessed one. The clock goes on
    # the design's clock PORT: the previous script put it on `[get_ports *]`, which declares every
    # input a clock and times a circuit that does not exist.
    if liberty_path is not None and clock_period_ns is not None:
        lines += [
            "if {[catch {",
            f"  read_liberty {_tcl_quote(liberty_path)}",
            f"  set placax_clock [get_ports -quiet {clock_port}]",
            "  if {[llength $placax_clock] == 0} {",
            f'    error "no port named {clock_port} to put the clock on"',
            "  }",
            f"  create_clock -period {clock_period_ns} -name {clock_name} $placax_clock",
            f"  set_wire_rc -layer {wire_rc_layer}",
            "  estimate_parasitics -placement",
            f'  {metric} worst_slack [sta::worst_slack -max]"',
            f'  {metric} total_negative_slack [sta::total_negative_slack -max]"',
            "} placax_err]} {",
            f'  {note} timing [string map {{"\\n" " "}} $placax_err]"',
            "}",
        ]

    if route is not None:
        detailed = ""
        if route == "detailed":
            report = drc_report if drc_report is not None else "drc.rpt"
            detailed = f"\n  detailed_route -output_drc {_tcl_quote(report)}"
        lines += [
            "if {[catch {",
            "  global_route -verbose" + detailed,
            "} placax_err]} {",
            f'  {note} route [string map {{"\\n" " "}} $placax_err]"',
            "}",
        ]

    lines.append(f'puts "{METRIC_PREFIX}_DONE"')
    return "\n".join(lines) + "\n"


# The router's own summary lines. Anchored on message IDs where OpenROAD has them - those are
# stable across releases in a way free text is not - and always read LAST-first, because both
# routers report progress before they report a result.
_GLOBAL_WIRELENGTH_RE = re.compile(r"GRT-0018\]\s+Total wirelength:\s*([\d.]+)\s*um")
_GLOBAL_VIAS_RE = re.compile(r"GRT-0111\]\s+Final number of vias:\s*(\d+)")
_DETAILED_WIRELENGTH_RE = re.compile(r"^Total wire length\s*=\s*([\d.]+)\s*um", re.MULTILINE)
_DETAILED_VIAS_RE = re.compile(r"^Total number of vias\s*=\s*(\d+)", re.MULTILINE)
_DRC_ITERATION_RE = re.compile(r"DRT-0199\]\s+Number of violations\s*=\s*(\d+)")
_DRC_MARKER_RE = re.compile(r"violation type", re.IGNORECASE)


def _last(pattern: re.Pattern, text: str):
    matches = pattern.findall(text)
    return matches[-1] if matches else None


def _metrics(raw_output: str) -> dict[str, str]:
    metrics = {}
    for line in raw_output.splitlines():
        if line.startswith(METRIC_PREFIX + " "):
            _prefix, key, *value = line.split(" ", 2)
            metrics[key] = value[0] if value else ""
    return metrics


def _float(value: str | None) -> float | None:
    try:
        return float(value) if value is not None else None
    except ValueError:
        return None


def count_drc_violations(report_text: str) -> int:
    """Violations in a `detailed_route -output_drc` report. An empty report is a clean route."""
    return len(_DRC_MARKER_RE.findall(report_text))


_PLACEMENT_CHECK_RE = re.compile(r"\[WARNING DPL-\d+\]\s+(.+?) check failed \((\d+)\)")
"""`check_placement`'s per-rule summary, e.g. `[WARNING DPL-0007] One site gap check failed (5728).`"""


def placement_violations(raw_output: str) -> tuple[tuple[str, int], ...]:
    """Each failed placement rule with its count, in the order the checker reported them."""
    return tuple((name, int(count)) for name, count in _PLACEMENT_CHECK_RE.findall(raw_output))


_MESSAGE_ID_RE = re.compile(r"\b([A-Z]{2,4}-\d{4})\b")


def _explain(note: str, raw_output: str) -> str:
    """A note carrying only a message ID, expanded with the tool's own sentence for it.

    A Tcl `catch` hands back just the ID ("GRT-0701"); the log has "[ERROR GRT-0701] Missing track
    structure for routing layers." - which is the part a reader of ppa.json actually needs.
    """
    match = _MESSAGE_ID_RE.search(note)
    if match is None:
        return note
    for line in raw_output.splitlines():
        if f"{match.group(1)}]" in line and ("ERROR" in line or "WARNING" in line):
            return f"{note}: {line.split(']', 1)[1].strip()}"
    return note


def parse_openroad_output(raw_output: str, drc_report_text: str | None = None) -> PPAResult:
    """What the script measured, and nothing it did not.

    `drc_report_text` is the `-output_drc` file when detailed routing ran. It is the authority on
    the final violation count; the router's per-iteration log is only a fallback, and then only
    its last line.
    """
    metrics = _metrics(raw_output)
    notes = tuple(
        _explain(line[len(NOTE_PREFIX) + 1:], raw_output) for line in raw_output.splitlines()
        if line.startswith(NOTE_PREFIX + " ")
    )

    slack = _float(metrics.get("worst_slack"))
    if slack is not None and abs(slack) >= INFINITE_SLACK:
        slack = None
        notes += ("timing no constrained path: worst slack was reported as unbounded",)
    tns = _float(metrics.get("total_negative_slack")) if slack is not None else None

    utilization = _float(metrics.get("utilization"))
    legal = metrics.get("placement_legal")

    detailed_length = _last(_DETAILED_WIRELENGTH_RE, raw_output)
    global_length = _last(_GLOBAL_WIRELENGTH_RE, raw_output)
    detailed_vias = _last(_DETAILED_VIAS_RE, raw_output)
    global_vias = _last(_GLOBAL_VIAS_RE, raw_output)
    if drc_report_text is not None:
        drc = count_drc_violations(drc_report_text)
    else:
        last_iteration = _last(_DRC_ITERATION_RE, raw_output)
        drc = int(last_iteration) if last_iteration is not None else None

    # A detailed route's numbers supersede the global route's: they describe real wires.
    wirelength = detailed_length if detailed_length is not None else global_length
    vias = detailed_vias if detailed_vias is not None else global_vias

    return PPAResult(
        design_area=_float(metrics.get("design_area_um2")),
        utilization_pct=utilization * 100.0 if utilization is not None else None,
        timing_slack=slack,
        raw_output=raw_output,
        routed_wirelength=float(wirelength) if wirelength is not None else None,
        via_count=int(vias) if vias is not None else None,
        drc_violations=drc,
        hpwl=_float(metrics.get("hpwl_um")),
        placement_legal=None if legal is None else legal == "1",
        placement_violations=placement_violations(raw_output),
        total_negative_slack=tns,
        tool_version=metrics.get("tool_version") or None,
        notes=notes,
    )


class OpenROADValidator(Validator):
    """Default Validator: OpenROAD on this host, or the pinned ORFS image through Docker."""

    def __init__(
        self,
        liberty_path: pathlib.Path | None = None,
        clock_period_ns: float | None = None,
        wire_rc_layer: str = "metal3",
        clock_name: str = "core_clock",
        openroad_binary: str = "openroad",
        route: str | None = None,
        clock_port: str = "clk",
        use_docker: bool = False,
        docker_image: str | None = None,
    ):
        self.liberty_path = liberty_path
        self.clock_period_ns = clock_period_ns
        self.wire_rc_layer = wire_rc_layer
        self.clock_name = clock_name
        self.clock_port = clock_port
        self.openroad_binary = openroad_binary
        # Part of the experiment, not of this machine: routing changes the result, so it belongs
        # in the validator Spec's kwargs and therefore in the environment hash.
        self.route = route
        # Where OpenROAD comes from is this machine's business - but WHICH OpenROAD it is still
        # changes the numbers, which is why the reported tool version travels with every result.
        self.use_docker = use_docker
        self.docker_image = docker_image

    def _write_script(
        self, def_path: pathlib.Path, lef_paths: list[pathlib.Path], output_dir: pathlib.Path
    ) -> pathlib.Path:
        """Writes the Tcl script OpenROAD will execute."""
        script_path = output_dir / "validate.tcl"
        script_path.write_text(
            build_openroad_script(
                def_path, lef_paths, self.liberty_path, self.clock_period_ns,
                self.wire_rc_layer, self.clock_name, self.route, self.clock_port,
                drc_report=self._drc_report(output_dir),
            )
        )
        return script_path

    @staticmethod
    def _drc_report(output_dir: pathlib.Path) -> pathlib.Path:
        return output_dir / "drc.rpt"

    def _command(self, script_path: pathlib.Path, def_path, lef_paths) -> list[str]:
        if not self.use_docker:
            return [self.openroad_binary, "-no_splash", "-exit", str(script_path)]
        from placax_tools.openroad.docker import OPENROAD_IMAGE, host_mounts, run_openroad_command

        mounts = host_mounts([script_path, def_path, self.liberty_path, *lef_paths])
        return run_openroad_command(script_path, mounts, self.docker_image or OPENROAD_IMAGE)

    def _run_openroad(self, command: list[str], output_dir: pathlib.Path) -> str:
        """Runs OpenROAD, keeps its full log beside the results, returns it."""
        result = subprocess.run(command, capture_output=True, text=True)
        log = result.stdout + (("\n" + result.stderr) if result.stderr else "")
        (output_dir / "openroad.log").write_text(log)
        if result.returncode != 0:
            # An uncaught Tcl error exits 1 - reading the design failed, which nothing downstream
            # can recover from. Surfaced with the tool's own last lines rather than a bare code.
            tail = "\n".join(log.strip().splitlines()[-15:])
            raise subprocess.CalledProcessError(
                result.returncode, command, output=log,
                stderr=f"OpenROAD failed; last lines of {output_dir / 'openroad.log'}:\n{tail}",
            )
        return log

    def validate(
        self, def_path: pathlib.Path, lef_paths: list[pathlib.Path], output_dir: pathlib.Path
    ) -> PPAResult:
        """Writes the Tcl script, runs OpenROAD, parses what it printed."""
        output_dir = pathlib.Path(output_dir).resolve()
        output_dir.mkdir(parents=True, exist_ok=True)
        def_path = pathlib.Path(def_path).resolve()
        lef_paths = [self._resolve(path) for path in lef_paths]
        drc_report = self._drc_report(output_dir)
        drc_report.unlink(missing_ok=True)   # a stale report would count as this run's

        script_path = self._write_script(def_path, lef_paths, output_dir)
        raw_output = self._run_openroad(self._command(script_path, def_path, lef_paths),
                                        output_dir)
        drc_text = drc_report.read_text() if drc_report.exists() else None
        if self.route == "detailed" and drc_text is None and "Complete detail routing" in raw_output:
            drc_text = ""   # the router finished and wrote nothing: that is a clean route
        return parse_openroad_output(raw_output, drc_text)

    @staticmethod
    def _resolve(path) -> pathlib.Path:
        """A host path made absolute; an in-image path (one that does not exist here) left alone."""
        path = pathlib.Path(path)
        return path.resolve() if path.exists() else path
