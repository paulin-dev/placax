"""OpenROAD as the cell placer: the script it runs, and how it is chosen from a config."""
import pathlib

import pytest

from placax_tools.cell_placer import CellPlacer
from placax_tools.openroad.cell_placer import OpenROADCellPlacer, build_placement_script


def test_the_script_follows_orfs_order_pins_between_two_global_placements() -> None:
    script = build_placement_script(
        pathlib.Path("/d/in.def"), [pathlib.Path("/d/tech.lef"), pathlib.Path("/d/cells.lef")],
        pathlib.Path("/o/placed.def"), density=0.6, pin_hor_layers="metal5",
        pin_ver_layers="metal6", routing_layers="metal2-metal10", seed=3,
        io_constraints="/d/io.tcl",
    )
    order = [script.index(marker) for marker in (
        "read_lef {/d/tech.lef}", "read_lef {/d/cells.lef}", "read_def {/d/in.def}",
        "set_routing_layers -signal metal2-metal10", "set placax_density 0.6",
        "global_placement -density $placax_density -random_seed 3 -skip_io",
        "source {/d/io.tcl}",
        "place_pins -hor_layers metal5 -ver_layers metal6",
        "global_placement -density $placax_density -random_seed 3\n",
        "detailed_placement", "write_def {/o/placed.def}",
    )]
    assert order == sorted(order)


def test_without_pin_layers_the_pins_are_left_where_the_design_has_them() -> None:
    script = build_placement_script(pathlib.Path("a.def"), [pathlib.Path("t.lef")],
                                    pathlib.Path("p.def"))
    assert "place_pins" not in script and "-skip_io" not in script
    assert "set_routing_layers" not in script and "-random_seed" not in script
    assert script.count("global_placement ") == 1


def test_orfs_density_lower_bound_addon() -> None:
    script = build_placement_script(pathlib.Path("a.def"), [pathlib.Path("t.lef")],
                                    pathlib.Path("p.def"), density_lb_addon=0.2)
    assert "gpl::get_global_placement_uniform_density" in script
    assert "(1.0 - $placax_lb) * 0.2 + 0.01" in script
    assert "set placax_density 0.7" not in script


def test_it_is_a_cell_placer_and_runs_through_docker_when_asked(tmp_path, monkeypatch) -> None:
    import subprocess

    calls = []

    def fake_run(command, capture_output, text):
        calls.append(command)
        (tmp_path / "out" / "placed.def").write_text("DESIGN x ;\n")
        return subprocess.CompletedProcess(command, 0, stdout="PLACAX_PLACEMENT_DONE\n", stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)
    (tmp_path / "in.def").write_text("")
    placer = OpenROADCellPlacer(use_docker=True)
    assert isinstance(placer, CellPlacer)
    placed = placer.place(tmp_path / "in.def", [], tmp_path / "out")
    assert placed == (tmp_path / "out" / "placed.def").resolve()
    assert calls[0][:3] == ["docker", "run", "--rm"]
    assert (tmp_path / "out" / "place.log").exists()


def test_a_run_that_writes_no_design_is_an_error(tmp_path, monkeypatch) -> None:
    import subprocess

    monkeypatch.setattr(subprocess, "run", lambda command, capture_output, text:
                        subprocess.CompletedProcess(command, 0, stdout="", stderr=""))
    (tmp_path / "in.def").write_text("")
    with pytest.raises(FileNotFoundError, match="without writing"):
        OpenROADCellPlacer().place(tmp_path / "in.def", [], tmp_path / "out")


def test_the_config_can_name_it_and_its_settings_are_hashed() -> None:
    from placax_agents.experiment.build import build_physical
    from placax_agents.experiment.config import PhysicalSpec, Spec
    from placax_agents.experiment.registry import CELL_PLACERS
    import dataclasses

    from placax_agents.experiment.presets import maskplace

    assert "openroad" in CELL_PLACERS
    identity = Spec("openroad").identity("cell_placer")["kwargs"]
    assert {"density", "pin_hor_layers", "pin_ver_layers", "routing_layers", "seed",
            "io_constraints", "density_lb_addon"} <= set(identity)
    assert "openroad_binary" not in identity and "use_docker" not in identity

    config = maskplace("benchmarks/adaptec1")
    config = dataclasses.replace(config, environment=dataclasses.replace(
        config.environment, physical=PhysicalSpec(Spec("openroad", {"density": 0.5}),
                                                  Spec("openroad")),
    ))
    placer, validator = build_physical(config, {"use_docker": True, "openroad_binary": "/x",
                                                "dreamplace_root": "/ignored"})
    assert isinstance(placer, OpenROADCellPlacer)
    assert placer.density == 0.5 and placer.use_docker is True and validator.use_docker is True
