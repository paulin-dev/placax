import pathlib

import pytest

from placax_tools.validator import PPAResult, Validator
from placax_tools.openroad.validator import (
    OpenROADValidator, build_openroad_script, parse_openroad_output,
)


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


def test_openroad_validator_config_lives_in_init_not_validate() -> None:
    validator = OpenROADValidator(
        liberty_path=pathlib.Path("/tmp/lib.lib"), clock_period_ns=2.0
    )
    assert validator.liberty_path == pathlib.Path("/tmp/lib.lib")
    assert validator.clock_period_ns == 2.0
    import inspect

    sig = inspect.signature(validator.validate)
    assert list(sig.parameters) == ["def_path", "lef_paths", "output_dir"]


def test_build_openroad_script_basic() -> None:
    script = build_openroad_script(pathlib.Path("/tmp/d.def"), [pathlib.Path("/tmp/t.lef")])
    assert "read_lef /tmp/t.lef" in script
    assert "read_def /tmp/d.def" in script
    assert "report_design_area" in script
    assert "report_checks" not in script  # no timing requested


def test_build_openroad_script_with_timing() -> None:
    script = build_openroad_script(
        pathlib.Path("/tmp/d.def"), [pathlib.Path("/tmp/t.lef")],
        liberty_path=pathlib.Path("/tmp/lib.lib"), clock_period_ns=2.0,
    )
    assert "read_liberty /tmp/lib.lib" in script
    assert "create_clock -period 2.0" in script
    assert "report_checks" in script


def test_build_openroad_script_multiple_lef_files() -> None:
    script = build_openroad_script(
        pathlib.Path("/tmp/d.def"), [pathlib.Path("/tmp/a.lef"), pathlib.Path("/tmp/b.lef")]
    )
    assert "read_lef /tmp/a.lef" in script
    assert "read_lef /tmp/b.lef" in script


def test_build_openroad_script_wire_rc_layer_and_clock_name_are_configurable() -> None:
    # Regression test: "metal3" (a technology-specific layer name that
    # won't work on a different PDK) and "core_clock" were hardcoded.
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


def test_parse_openroad_output_matches_real_format() -> None:
    # Real, documented OpenROAD output format, confirmed against actual
    # log excerpts (multiple independent real bug reports/tutorials).
    raw = """
==========================================================================
global route report_design_area
--------------------------------------------------------------------------
Design area 899 u^2 83% utilization.
"""
    result = parse_openroad_output(raw)
    assert result.design_area == 899.0
    assert result.utilization_pct == 83.0
    assert result.timing_slack is None  # no timing report in this output


def test_parse_openroad_output_no_match_returns_none_not_error() -> None:
    result = parse_openroad_output("some unrelated output")
    assert result.design_area is None
    assert result.utilization_pct is None
    assert result.timing_slack is None
