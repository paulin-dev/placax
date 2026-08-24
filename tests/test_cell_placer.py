import pathlib

import pytest

from placax_tools.cell_placer import CellPlacer
from placax_tools.dreamplace.cell_placer import DREAMPlaceCellPlacer, build_dreamplace_config


def test_cell_placer_cannot_be_instantiated_directly() -> None:
    with pytest.raises(TypeError):
        CellPlacer()


def test_any_concrete_cell_placer_is_callable_through_the_generic_interface() -> None:
    # The actual point: a function that only knows about CellPlacer's
    # generic contract should work identically with any implementation.
    class MockCellPlacer(CellPlacer):
        def place(self, def_path, lef_paths, output_dir):
            return def_path  # trivial mock, pretends input is already placed

    def run_pipeline(placer: CellPlacer, def_path, lef_paths, output_dir):
        return placer.place(def_path, lef_paths, output_dir)

    mock = MockCellPlacer()
    result = run_pipeline(mock, pathlib.Path("/tmp/d.def"), [pathlib.Path("/tmp/t.lef")], pathlib.Path("/tmp/o"))
    assert result == pathlib.Path("/tmp/d.def")


def test_dreamplace_cell_placer_config_lives_in_init_not_place() -> None:
    placer = DREAMPlaceCellPlacer(
        dreamplace_root=pathlib.Path("/opt/dreamplace"), gpu=True, target_density=0.8
    )
    assert placer.gpu is True
    assert placer.target_density == 0.8
    # place() itself only takes the generic arguments, matching CellPlacer's contract
    import inspect

    sig = inspect.signature(placer.place)
    assert list(sig.parameters) == ["def_path", "lef_paths", "output_dir"]


def test_build_dreamplace_config_real_fields() -> None:
    config = build_dreamplace_config(
        pathlib.Path("/tmp/design.def"), [pathlib.Path("/tmp/tech.lef")], pathlib.Path("/tmp/out")
    )
    assert config["def_input"] == "/tmp/design.def"
    assert config["lef_input"] == "/tmp/tech.lef"
    assert config["result_dir"] == "/tmp/out"
    assert config["gpu"] == 0


def test_build_dreamplace_config_gpu_flag() -> None:
    config = build_dreamplace_config(
        pathlib.Path("/tmp/d.def"), [pathlib.Path("/tmp/t.lef")], pathlib.Path("/tmp/o"), gpu=True
    )
    assert config["gpu"] == 1


def test_build_dreamplace_config_multiple_lef_files() -> None:
    config = build_dreamplace_config(
        pathlib.Path("/tmp/d.def"),
        [pathlib.Path("/tmp/a.lef"), pathlib.Path("/tmp/b.lef")],
        pathlib.Path("/tmp/o"),
    )
    assert config["lef_input"] == "/tmp/a.lef;/tmp/b.lef"


def test_dreamplace_cell_placer_extra_config_and_python_executable() -> None:
    # Regression test: many real DREAMPlace knobs (num_bins_x,
    # random_seed, density_weight, etc.) were hardcoded with no
    # override, and the python binary was hardcoded to "python".
    placer = DREAMPlaceCellPlacer(
        dreamplace_root=pathlib.Path("/opt/dreamplace"),
        extra_config={"num_bins_x": 1024, "random_seed": 42},
        python_executable="python3",
    )
    assert placer.extra_config == {"num_bins_x": 1024, "random_seed": 42}
    assert placer.python_executable == "python3"
