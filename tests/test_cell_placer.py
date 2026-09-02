import pathlib

import pytest

from placax_tools.cell_placer import CellPlacer
from placax_tools.dreamplace.cell_placer import (
    DREAMPlaceCellPlacer, build_dreamplace_config, build_dreamplace_config_bookshelf,
)


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


def test_expect_result_pl_finds_dreamplaces_nested_output(tmp_path) -> None:
    # DREAMPlace writes into a <result_dir>/<design_name>/ subdirectory, not directly into result_dir -
    # confirmed against a real end-to-end run, not just its README/docs.
    placer = DREAMPlaceCellPlacer(dreamplace_root=pathlib.Path("/opt/dreamplace"))
    (tmp_path / "adaptec1").mkdir()
    (tmp_path / "adaptec1" / "adaptec1.gp.pl").write_text("")
    result = placer._expect_result_pl(pathlib.Path("/tmp/adaptec1.aux"), tmp_path)
    assert result == tmp_path / "adaptec1" / "adaptec1.gp.pl"


def test_expect_result_pl_raises_if_not_nested(tmp_path) -> None:
    placer = DREAMPlaceCellPlacer(dreamplace_root=pathlib.Path("/opt/dreamplace"))
    (tmp_path / "adaptec1.gp.pl").write_text("")  # the old, wrong (unnested) assumption
    with pytest.raises(FileNotFoundError):
        placer._expect_result_pl(pathlib.Path("/tmp/adaptec1.aux"), tmp_path)


def test_expect_result_def_finds_dreamplaces_nested_output(tmp_path) -> None:
    placer = DREAMPlaceCellPlacer(dreamplace_root=pathlib.Path("/opt/dreamplace"))
    (tmp_path / "design").mkdir()
    (tmp_path / "design" / "design.gp.def").write_text("")
    result = placer._expect_result_def(pathlib.Path("/tmp/design.def"), tmp_path)
    assert result == tmp_path / "design" / "design.gp.def"


def test_dreamplace_cell_placer_place_bookshelf_signature() -> None:
    # place_bookshelf is a DREAMPlace-specific addition, not part of the generic CellPlacer ABC
    # (which stays DEF/LEF-only, the only format OpenROAD reads) - its own contract, checked separately.
    placer = DREAMPlaceCellPlacer(dreamplace_root=pathlib.Path("/opt/dreamplace"))
    import inspect

    sig = inspect.signature(placer.place_bookshelf)
    assert list(sig.parameters) == ["aux_path", "output_dir"]


def test_build_dreamplace_config_bookshelf_real_fields() -> None:
    config = build_dreamplace_config_bookshelf(
        pathlib.Path("/tmp/adaptec1.aux"), pathlib.Path("/tmp/out")
    )
    assert config["aux_input"] == "/tmp/adaptec1.aux"
    assert config["result_dir"] == "/tmp/out"
    assert config["gpu"] == 0
    assert "def_input" not in config
    assert "lef_input" not in config


def test_build_dreamplace_config_bookshelf_gpu_and_density() -> None:
    config = build_dreamplace_config_bookshelf(
        pathlib.Path("/tmp/a.aux"), pathlib.Path("/tmp/o"), gpu=True, target_density=0.8
    )
    assert config["gpu"] == 1
    assert config["target_density"] == 0.8


def test_dreamplace_cell_placer_docker_mode_stores_flag_and_mounts() -> None:
    placer = DREAMPlaceCellPlacer(
        dreamplace_root=pathlib.Path("/tmp/DREAMPlace"), use_docker=True,
        extra_mounts=(pathlib.Path("/tmp/bench"), pathlib.Path("/tmp/out")),
    )
    assert placer.use_docker is True
    assert placer.extra_mounts == (pathlib.Path("/tmp/bench"), pathlib.Path("/tmp/out"))


def test_dreamplace_cell_placer_defaults_to_local_checkout_not_docker() -> None:
    placer = DREAMPlaceCellPlacer(dreamplace_root=pathlib.Path("/opt/dreamplace"))
    assert placer.use_docker is False
    assert placer.extra_mounts == ()


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
