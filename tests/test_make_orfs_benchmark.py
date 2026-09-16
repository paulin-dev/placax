"""An ORFS design as a benchmark: what is read from ORFS, and the physical stack derived from it."""
from scripts.make_orfs_benchmark import physical_spec, read_clock, read_variables

PRINTED = """TECH_LEF: /p/lef/NangateOpenCellLibrary.tech.lef
SC_LEF: /p/lef/NangateOpenCellLibrary.macro.mod.lef
ADDITIONAL_LEFS: /p/lef/fakeram45_256x16.lef  
LIB_FILES: /p/lib/NangateOpenCellLibrary_typical.lib /p/lib/fakeram45_256x16.lib  
PLACE_DENSITY: 0.30
PLACE_DENSITY_LB_ADDON: 
IO_PLACER_H: metal5
IO_PLACER_V: metal6
MIN_ROUTING_LAYER: metal2
MAX_ROUTING_LAYER: metal10
IO_CONSTRAINTS: /d/io.tcl
SDC_FILE: /d/ariane.sdc
DESIGN_NAME: ariane
PLATFORM: nangate45
RESULTS_DIR: /w/results/nangate45/ariane133/base
make: some unrelated line
"""

SDC = """current_design ariane

set clk_name core_clock
set clk_port_name clk_i
set clk_period 3.0
"""


def test_orfs_variables_are_read_from_its_own_print_targets() -> None:
    variables = read_variables(PRINTED)
    assert variables["LIB_FILES"] == "/p/lib/NangateOpenCellLibrary_typical.lib /p/lib/fakeram45_256x16.lib"
    assert variables["PLACE_DENSITY_LB_ADDON"] == ""
    assert variables["DESIGN_NAME"] == "ariane"
    assert "make" not in variables


def test_the_clock_is_read_through_the_sdcs_variables() -> None:
    assert read_clock(SDC) == ("clk_i", 3.0)
    assert read_clock("") == (None, None)


def test_the_physical_stack_measures_the_design_the_way_orfs_would() -> None:
    variables = {**read_variables(PRINTED), "SDC_FILE_TEXT": SDC}
    spec = physical_spec(variables, route="global")
    placer, validator = spec["cell_placer"], spec["validator"]
    assert placer["name"] == "openroad" and validator["name"] == "openroad"
    assert placer["kwargs"] == {
        "density": 0.30, "pin_hor_layers": "metal5", "pin_ver_layers": "metal6",
        "routing_layers": "metal2-metal10", "io_constraints": "/d/io.tcl",
    }
    assert validator["kwargs"] == {
        "liberty_path": ["/p/lib/NangateOpenCellLibrary_typical.lib", "/p/lib/fakeram45_256x16.lib"],
        "wire_rc_layer": "metal3", "routing_layers": "metal2-metal10", "route": "global",
        "clock_port": "clk_i", "clock_period_ns": 3.0,
    }


def test_the_stack_loads_as_a_physical_spec() -> None:
    from placax_agents.experiment.config import PhysicalSpec

    variables = {**read_variables(PRINTED), "SDC_FILE_TEXT": SDC}
    physical = PhysicalSpec.from_dict(physical_spec(variables, route=None))
    assert physical.cell_placer.kwargs["density"] == 0.30
    assert physical.validator.kwargs["clock_port"] == "clk_i"
