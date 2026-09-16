"""The validator against the REAL OpenROAD in the pinned ORFS image.

Skipped when Docker or the image is missing - the image is 1.6 GB and is never pulled implicitly.
Pull it once with `docker pull openroad/orfs:<tag in placax_tools/openroad/docker.py>`.
"""
import pathlib

import pytest

from placax.netlist.def_export import export_bookshelf_as_def
from placax_tools.openroad import docker
from placax_tools.openroad.validator import OpenROADValidator

pytestmark = pytest.mark.skipif(
    not docker.image_available(), reason=f"Docker image {docker.OPENROAD_IMAGE} not available"
)

TINY = pathlib.Path(__file__).parent / "fixtures" / "openroad" / "tiny_bookshelf"
MACROS = {"macro_a", "macro_b"}
LEGAL = {
    "macro_a": (10.0, 24.0), "macro_b": (60.0, 12.0),
    "cell_x": (0.0, 0.0), "cell_y": (2.0, 0.0), "cell_z": (30.0, 0.0),
}


def _validate(tmp_path, placement):
    def_path, lef_path = export_bookshelf_as_def(TINY, tmp_path / "design", placement, fixed=MACROS)
    return OpenROADValidator(route="global", use_docker=True).validate(
        def_path, [lef_path], tmp_path / "out"
    )


def test_a_converted_bookshelf_design_is_measured_by_real_openroad(tmp_path) -> None:
    result = _validate(tmp_path, LEGAL)
    assert result.tool_version and result.tool_version.startswith("26Q3")
    assert result.design_area == pytest.approx(2096.0, abs=1e-3)   # 40*20 + 20*60 + 2*12 + 2*12 + 4*12
    assert result.utilization_pct == pytest.approx(2096.0 / (100 * 96) * 100, abs=1e-3)
    assert result.placement_legal is True
    assert result.hpwl == pytest.approx(184.5)
    # The derived technology has one layer and no tracks: routing is reported as not measured.
    assert result.routed_wirelength is None
    assert any("GRT-0701" in note for note in result.notes)
    assert (tmp_path / "out" / "openroad.log").exists()


def test_real_openroad_catches_an_overlap_the_export_let_through(tmp_path) -> None:
    result = _validate(tmp_path, {**LEGAL, "cell_y": (1.0, 0.0)})   # on top of cell_x
    assert result.placement_legal is False
    assert any(note.startswith("check_placement") for note in result.notes)
    assert result.hpwl is not None   # the rest of the measurement still came back


def test_validate_design_measures_a_real_sky130_design_inside_the_image(tmp_path) -> None:
    # Every input lives in the image - its PDK and its tutorial design - so nothing about this
    # host's files is involved: the CLI has to accept paths that do not exist here.
    import subprocess
    import sys

    sky130 = docker.platform_files("sky130hd")
    completed = subprocess.run(
        [sys.executable, "-m", "scripts.validate_design",
         "--def_path=/OpenROAD-flow-scripts/flow/tutorials/scripts/drt/gcd/4_cts.def",
         f"--lef={sky130['tech_lef']}", f"--lef={sky130['cell_lef']}",
         f"--liberty={sky130['liberty']}", "--clock_period_ns=10",
         f"--wire_rc_layer={sky130['wire_rc_layer']}",
         "--skip_cell_placement", "--use_docker", f"--output_dir={tmp_path}"],
        capture_output=True, text=True, cwd=pathlib.Path(__file__).parents[1], timeout=300,
    )
    assert completed.returncode == 0, completed.stderr[-2000:]
    out = completed.stdout
    assert "4,987.283 um^2" in out
    assert "27,629.352 um" in out          # OpenROAD's detailed placer reports 27629.4
    assert "6.497 ns" in out               # the same slack the detailed-route run measured
    assert "legal:        True" in out
