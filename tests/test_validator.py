import pathlib

import pytest

from placax_tools.validator import PPAResult, Validator
from placax_tools.openroad import docker
from placax_tools.openroad.validator import (
    OpenROADValidator, build_openroad_script, count_drc_violations, parse_openroad_output,
)

FIXTURES = pathlib.Path(__file__).parent / "fixtures" / "openroad"
"""Logs captured from the pinned ORFS image's real OpenROAD (26Q3-2130-g90e29809c3), not written
by hand - the previous parser tests were, and matched a format the tool does not print."""


def test_validator_cannot_be_instantiated_directly() -> None:
    with pytest.raises(TypeError):
        Validator()


def test_any_concrete_validator_is_callable_through_the_generic_interface() -> None:
    class MockValidator(Validator):
        def validate(self, def_path, lef_paths, output_dir):
            return PPAResult(design_area=1.0, utilization_pct=50.0, timing_slack=None, raw_output="")

    def run_check(validator: Validator, def_path, lef_paths, output_dir):
        return validator.validate(def_path, lef_paths, output_dir)

    mock = MockValidator()
    result = run_check(mock, pathlib.Path("/tmp/d.def"), [pathlib.Path("/tmp/t.lef")], pathlib.Path("/tmp/o"))
    assert result.design_area == 1.0
    assert result.notes == () and result.hpwl is None


def test_openroad_validator_config_lives_in_init_not_validate() -> None:
    validator = OpenROADValidator(
        liberty_path=pathlib.Path("/tmp/lib.lib"), clock_period_ns=2.0
    )
    assert validator.liberty_path == pathlib.Path("/tmp/lib.lib")
    assert validator.clock_period_ns == 2.0
    import inspect

    sig = inspect.signature(validator.validate)
    assert list(sig.parameters) == ["def_path", "lef_paths", "output_dir"]


# ------------------------------------------------------------------ script

def test_script_reads_the_design_and_prints_its_metrics() -> None:
    script = build_openroad_script(pathlib.Path("/tmp/d.def"), [pathlib.Path("/tmp/t.lef")])
    assert "read_lef {/tmp/t.lef}" in script
    assert "read_def {/tmp/d.def}" in script
    for key in ("tool_version", "design_area_um2", "utilization", "placement_legal", "hpwl_um"):
        assert f"PLACAX_METRIC {key} " in script
    assert "check_placement" in script
    assert "worst_slack" not in script and "read_liberty" not in script  # no timing requested
    assert script.rstrip().endswith('puts "PLACAX_METRIC_DONE"')


def test_paths_are_brace_quoted_so_spaces_survive() -> None:
    script = build_openroad_script(pathlib.Path("/tmp/my run/d.def"), [pathlib.Path("/tmp/t.lef")])
    assert "read_def {/tmp/my run/d.def}" in script


def test_multiple_lef_files_are_read_in_order() -> None:
    script = build_openroad_script(
        pathlib.Path("/tmp/d.def"), [pathlib.Path("/tmp/a.lef"), pathlib.Path("/tmp/b.lef")]
    )
    assert script.index("read_lef {/tmp/a.lef}") < script.index("read_lef {/tmp/b.lef}")
    assert script.index("read_lef {/tmp/b.lef}") < script.index("read_def")


def test_timing_puts_the_clock_on_the_clock_port_only() -> None:
    # Regression: the old script ran `create_clock [get_ports *]`, declaring every input a clock.
    script = build_openroad_script(
        pathlib.Path("/tmp/d.def"), [pathlib.Path("/tmp/t.lef")],
        liberty_path=pathlib.Path("/tmp/lib.lib"), clock_period_ns=2.0, clock_port="clk_i",
    )
    assert "read_liberty {/tmp/lib.lib}" in script
    assert "get_ports -quiet clk_i" in script
    assert "create_clock -period 2.0" in script
    assert "get_ports *" not in script
    assert "PLACAX_METRIC worst_slack" in script and "PLACAX_METRIC total_negative_slack" in script


def test_timing_needs_both_a_library_and_a_period() -> None:
    only_lib = build_openroad_script(
        pathlib.Path("d.def"), [pathlib.Path("t.lef")], liberty_path=pathlib.Path("l.lib"),
    )
    only_period = build_openroad_script(
        pathlib.Path("d.def"), [pathlib.Path("t.lef")], clock_period_ns=2.0,
    )
    assert "create_clock" not in only_lib and "create_clock" not in only_period


def test_timing_and_routing_failures_are_caught_not_fatal() -> None:
    # A converted Bookshelf design cannot route; its area, HPWL and legality must still come back.
    script = build_openroad_script(
        pathlib.Path("d.def"), [pathlib.Path("t.lef")],
        liberty_path=pathlib.Path("l.lib"), clock_period_ns=2.0, route="global",
    )
    assert script.count("catch {") == 3   # placement check, timing, routing
    assert 'PLACAX_NOTE timing' in script and 'PLACAX_NOTE route' in script


def test_wire_rc_layer_and_clock_name_are_configurable() -> None:
    script = build_openroad_script(
        pathlib.Path("/tmp/d.def"), [pathlib.Path("/tmp/t.lef")],
        liberty_path=pathlib.Path("/tmp/lib.lib"), clock_period_ns=2.0,
        wire_rc_layer="M4", clock_name="my_clk",
    )
    assert "set_wire_rc -layer M4" in script
    assert "-name my_clk" in script
    assert "metal3" not in script
    assert "core_clock" not in script


def test_openroad_validator_binary_is_configurable() -> None:
    validator = OpenROADValidator(openroad_binary="/opt/openroad/bin/openroad")
    assert validator.openroad_binary == "/opt/openroad/bin/openroad"
    assert validator._command(pathlib.Path("/x/v.tcl"), pathlib.Path("/x/d.def"), [])[0] == \
        "/opt/openroad/bin/openroad"


# ------------------------------------------------------------------ routing script

def test_routing_is_off_unless_asked_for() -> None:
    script = build_openroad_script(pathlib.Path("a.def"), [pathlib.Path("t.lef")])
    assert "global_route" not in script and "detailed_route" not in script


def test_global_routing_stops_before_detailed() -> None:
    script = build_openroad_script(pathlib.Path("a.def"), [pathlib.Path("t.lef")], route="global")
    assert "global_route" in script
    assert "detailed_route" not in script


def test_detailed_routing_writes_its_violations_to_a_report() -> None:
    script = build_openroad_script(
        pathlib.Path("a.def"), [pathlib.Path("t.lef")], route="detailed",
        drc_report=pathlib.Path("/out/drc.rpt"),
    )
    assert "global_route" in script
    assert "detailed_route -output_drc {/out/drc.rpt}" in script


def test_an_unknown_route_mode_is_refused() -> None:
    with pytest.raises(ValueError, match="unknown route mode"):
        build_openroad_script(pathlib.Path("a.def"), [pathlib.Path("t.lef")], route="sideways")


# ------------------------------------------------------------------ parsing real output

def test_a_real_sky130_global_route_log_parses_completely() -> None:
    parsed = parse_openroad_output((FIXTURES / "gcd_sky130_global_route.log").read_text())
    assert parsed.tool_version == "26Q3-2130-g90e29809c3"
    assert parsed.design_area == pytest.approx(4987.283, abs=1e-3)
    assert parsed.utilization_pct == pytest.approx(7.4393, abs=1e-4)
    assert parsed.placement_legal is True
    # OpenROAD's own detailed placer reports 27629.4 um for this design.
    assert parsed.hpwl == pytest.approx(27629.352)
    assert parsed.timing_slack == pytest.approx(1.49723, abs=1e-5)
    assert parsed.total_negative_slack == 0.0
    assert parsed.routed_wirelength == 40206.0
    assert parsed.via_count == 3370
    assert parsed.drc_violations is None   # global route only: nobody checked
    assert parsed.notes == ()


def test_a_converted_bookshelf_design_reports_what_it_can_and_why_not_the_rest() -> None:
    parsed = parse_openroad_output((FIXTURES / "converted_bookshelf.log").read_text())
    assert parsed.design_area == pytest.approx(2096.0)
    assert parsed.utilization_pct == pytest.approx(21.8333, abs=1e-4)
    assert parsed.placement_legal is True
    assert parsed.hpwl == pytest.approx(184.5)   # checked by hand against the .nets offsets
    assert parsed.routed_wirelength is None and parsed.via_count is None
    assert parsed.timing_slack is None
    # The catch hands back only the message ID; the note carries the tool's sentence for it.
    assert parsed.notes == ("route GRT-0701: Missing track structure for routing layers.",)


def test_unrelated_output_yields_nothing_rather_than_an_error() -> None:
    result = parse_openroad_output("some unrelated output")
    assert result.design_area is None
    assert result.utilization_pct is None
    assert result.timing_slack is None
    assert result.placement_legal is None and result.hpwl is None and result.tool_version is None


def test_human_readable_reports_are_not_mistaken_for_metrics() -> None:
    # What the previous parser scraped. It is not a measurement this script asked for.
    result = parse_openroad_output("Design area 899 um^2 83% utilization.\n")
    assert result.design_area is None and result.utilization_pct is None


def test_an_illegal_placement_is_reported_with_the_checkers_reason() -> None:
    result = parse_openroad_output(
        "PLACAX_METRIC placement_legal 0\nPLACAX_NOTE check_placement DPL-0033 detailed placement checks failed.\n"
    )
    assert result.placement_legal is False
    assert result.notes == ("check_placement DPL-0033 detailed placement checks failed.",)


def test_unbounded_slack_is_not_a_number() -> None:
    # sta reports ~1e39 when no path is constrained. That must not reach ppa.json as a slack.
    result = parse_openroad_output(
        "PLACAX_METRIC worst_slack 1.0000000331813535e+39\nPLACAX_METRIC total_negative_slack 0.0\n"
    )
    assert result.timing_slack is None and result.total_negative_slack is None
    assert any("no constrained path" in note for note in result.notes)


def test_router_summaries_are_read_last_first() -> None:
    # Both routers print progress before a result; the final lines are the answer.
    raw = (
        "[INFO GRT-0018] Total wirelength: 111 um\n"
        "[INFO GRT-0111] Final number of vias: 11\n"
        "[INFO GRT-0018] Total wirelength: 222 um\n"
        "[INFO GRT-0111] Final number of vias: 22\n"
    )
    result = parse_openroad_output(raw)
    assert result.routed_wirelength == 222.0 and result.via_count == 22


def test_detailed_route_numbers_supersede_global_ones() -> None:
    raw = (
        "[INFO GRT-0018] Total wirelength: 40206 um\n"
        "[INFO GRT-0111] Final number of vias: 3370\n"
        "Total wire length = 35000 um.\n"
        "Total number of vias = 3100.\n"
    )
    result = parse_openroad_output(raw, drc_report_text="")
    assert result.routed_wirelength == 35000.0 and result.via_count == 3100
    assert result.drc_violations == 0


def test_drc_is_the_final_count_not_the_first_iteration() -> None:
    # The real run this was found on printed 615, 297, 235, 116, 7, 0: a first-match parser
    # reported a clean route as 615 violations.
    iterations = "".join(
        f"[INFO DRT-0199]   Number of violations = {n}.\n" for n in (615, 297, 235, 116, 7, 0)
    )
    assert parse_openroad_output(iterations).drc_violations == 0


def test_the_drc_report_is_the_authority_when_present() -> None:
    report = (
        "violation type: Short\n  srcs: net1 net2\n  bbox = ( 1, 2 ) - ( 3, 4 ) on Layer met1\n"
        "violation type: EOL\n  srcs: net3\n  bbox = ( 5, 6 ) - ( 7, 8 ) on Layer met2\n"
    )
    assert count_drc_violations(report) == 2
    assert count_drc_violations("") == 0
    raw = "[INFO DRT-0199]   Number of violations = 0.\n"
    assert parse_openroad_output(raw, drc_report_text=report).drc_violations == 2


def test_routing_metrics_are_none_when_routing_did_not_run() -> None:
    parsed = parse_openroad_output("PLACAX_METRIC design_area_um2 1234.5\n")
    assert parsed.design_area == 1234.5
    assert parsed.routed_wirelength is None
    assert parsed.via_count is None
    assert parsed.drc_violations is None


# ------------------------------------------------------------------ running it

def test_a_failed_run_raises_with_the_tools_last_lines(tmp_path, monkeypatch) -> None:
    import subprocess

    def fake_run(command, capture_output, text):
        return subprocess.CompletedProcess(command, 1, stdout="[ERROR ODB-0001] bad def\n", stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)
    validator = OpenROADValidator()
    with pytest.raises(subprocess.CalledProcessError) as caught:
        validator.validate(tmp_path / "d.def", [], tmp_path / "out")
    assert "ODB-0001" in caught.value.stderr
    assert (tmp_path / "out" / "openroad.log").read_text().startswith("[ERROR ODB-0001]")


def test_a_stale_drc_report_is_not_counted_as_this_runs(tmp_path, monkeypatch) -> None:
    import subprocess

    out = tmp_path / "out"
    out.mkdir()
    (out / "drc.rpt").write_text("violation type: Short\n")

    def fake_run(command, capture_output, text):
        return subprocess.CompletedProcess(command, 0, stdout="Complete detail routing.\n", stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)
    result = OpenROADValidator(route="detailed").validate(tmp_path / "d.def", [], out)
    assert result.drc_violations == 0


# ------------------------------------------------------------------ docker

def test_docker_command_mounts_host_files_at_the_same_path(tmp_path) -> None:
    (tmp_path / "design").mkdir()
    def_path = tmp_path / "design" / "d.def"
    def_path.write_text("")
    lef = tmp_path / "design" / "t.lef"
    lef.write_text("")
    script = tmp_path / "out" / "validate.tcl"
    script.parent.mkdir()
    script.write_text("")
    in_image = pathlib.Path(docker.platform_files("sky130hd")["tech_lef"])

    validator = OpenROADValidator(use_docker=True)
    argv = validator._command(script, def_path, [lef, in_image])
    assert argv[:3] == ["docker", "run", "--rm"]
    mounts = [argv[i + 1] for i, arg in enumerate(argv) if arg == "-v"]
    assert sorted(mounts) == sorted(
        [f"{tmp_path / 'design'}:{tmp_path / 'design'}", f"{tmp_path / 'out'}:{tmp_path / 'out'}"]
    )
    assert docker.OPENROAD_IMAGE in argv
    assert argv[-4:] == [docker.OPENROAD_BINARY_IN_IMAGE, "-no_splash", "-exit", str(script)]
    assert "HOME=/tmp" in argv


def test_host_mounts_are_deduplicated_to_the_outermost_directory(tmp_path) -> None:
    (tmp_path / "a" / "b").mkdir(parents=True)
    mounts = docker.host_mounts([tmp_path / "a" / "b", tmp_path / "a", None, "/not/on/this/host"])
    assert mounts == [(tmp_path / "a").resolve()]


def test_the_image_is_pinned_and_overridable() -> None:
    assert ":" in docker.OPENROAD_IMAGE and not docker.OPENROAD_IMAGE.endswith(":latest")
    validator = OpenROADValidator(use_docker=True, docker_image="my/orfs:1")
    argv = validator._command(pathlib.Path("/x/v.tcl"), pathlib.Path("/x/d.def"), [])
    assert "my/orfs:1" in argv


def test_unknown_platform_names_the_known_ones() -> None:
    with pytest.raises(KeyError, match="nangate45"):
        docker.platform_files("gf180")


def test_a_real_sky130_detailed_route_log_parses_completely() -> None:
    # 10 ns clock on `clk`, routed to completion; the router wrote an empty DRC report.
    raw = (FIXTURES / "gcd_sky130_detailed_route.log").read_text()
    parsed = parse_openroad_output(raw, drc_report_text="")
    assert parsed.timing_slack == pytest.approx(6.49723, abs=1e-5)
    # The detailed router's final summary, not the global router's estimate earlier in the log -
    # and not the per-layer "Total wire length on LAYER ..." lines under it.
    assert parsed.routed_wirelength == 31290.0
    assert parsed.via_count == 3218
    assert parsed.drc_violations == 0
    assert parse_openroad_output(raw).drc_violations == 0   # the log's last iteration agrees
    assert parsed.notes == ()


def test_a_failed_placement_check_says_which_rules_failed() -> None:
    # Real output from adaptec1 after DREAMPlace: legal by ISPD 2005's rules, not by OpenROAD's.
    raw = (
        "PLACAX_METRIC placement_legal 0\n"
        "[WARNING DPL-0011] Padding check failed (29).\n"
        "[WARNING DPL-0007] One site gap check failed (5728).\n"
        "[ERROR DPL-0033] detailed placement checks failed during check placement.\n"
        "PLACAX_NOTE check_placement DPL-0033\n"
    )
    result = parse_openroad_output(raw)
    assert result.placement_legal is False
    assert result.placement_violations == (("Padding", 29), ("One site gap", 5728))
    assert result.notes == (
        "check_placement DPL-0033: detailed placement checks failed during check placement.",
    )
    assert parse_openroad_output("PLACAX_METRIC placement_legal 1\n").placement_violations == ()
