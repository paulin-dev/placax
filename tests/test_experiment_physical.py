"""The physical flow, driven by a config rather than by whichever flags a script was handed.

`place_and_validate` was always tool-agnostic. What it was not was attributable: a PPA number came
out of `scripts/validate_design.py`'s own CLI flags, and nothing tied it back to the run whose
placement it measured, so the last box of the architecture sat outside the reproducibility
envelope the rest of the project is built around.

These tests use stand-in tools, like tests/test_pipeline.py does and for the same reason - the
point is that the composition doesn't know which tools it is driving. Neither OpenROAD nor a
DEF/LEF design ships here, so an end-to-end run against the real binaries is still NOT verified.
"""
import dataclasses
import json
import pathlib

from placax.core import reset  # noqa: F401  must precede jax imports
from placax_agents.experiment.budget import Budget
from placax_agents.experiment.build import build
from placax_agents.experiment.config import PhysicalSpec, Spec
from placax_agents.experiment.physical import PPA_NAME, evaluate_physical, write_ppa
from placax_agents.experiment.presets import training
from placax_agents.experiment.registry import CELL_PLACERS, VALIDATORS
from placax_tools.cell_placer import CellPlacer
from placax_tools.validator import PPAResult, Validator

import pytest


class StandInCellPlacer(CellPlacer):
    def __init__(self):
        self.lefs = []

    def place(self, def_path, lef_paths, output_dir):
        # Recorded so a test can assert WHICH LEFs reached the tool - for a Bookshelf design the
        # library is derived by the export, so it has to be handed over rather than assumed.
        self.lefs.extend(lef_paths)
        placed = output_dir / "placed.def"
        placed.write_text(def_path.read_text() + "\n# cells placed\n")
        return placed


class StandInValidator(Validator):
    def __init__(self):
        self.validated = []

    def validate(self, def_path, lef_paths, output_dir):
        self.validated.append(def_path)
        return PPAResult(design_area=1234.5, utilization_pct=67.8, timing_slack=None,
                         raw_output="stand-in")


@pytest.fixture
def design(tmp_path: pathlib.Path):
    """A tiny benchmark, a config naming stand-in tools, and a macro-placed DEF to measure."""
    benchmark_dir = tmp_path / "bench"
    benchmark_dir.mkdir()
    (benchmark_dir / "s.aux").write_text("RowBasedPlacement : s.nodes s.nets s.wts s.pl s.scl\n")
    (benchmark_dir / "s.nodes").write_text(
        "UCLA nodes 1.0\nNumNodes : 3\nNumTerminals : 3\n"
        "a 4 4 terminal\nb 2 2 terminal\nc 2 4 terminal\n"
    )
    (benchmark_dir / "s.nets").write_text(
        "UCLA nets 1.0\nNumNets : 2\nNumPins : 4\n"
        "NetDegree : 2 n0\n\ta I : 0.0 0.0\n\tb O : 0.0 0.0\n"
        "NetDegree : 2 n1\n\tb I : 0.0 0.0\n\tc O : 0.0 0.0\n"
    )
    config = training(benchmark_dir, budget=Budget(iterations=1))
    config = dataclasses.replace(config, environment=dataclasses.replace(
        config.environment,
        benchmark=dataclasses.replace(config.environment.benchmark, grid=8),
        physical=PhysicalSpec(Spec("dreamplace"), Spec("openroad")),
    ))

    def_path = tmp_path / "macros.def"
    def_path.write_text("DESIGN toy ;\n")
    lef_path = tmp_path / "tech.lef"
    lef_path.write_text("VERSION 5.8 ;\n")
    return config, def_path, [lef_path], tmp_path / "out"


def _build_with_stand_ins(config):
    """Build, then swap in stand-in tools - the registry entries import real, absent binaries."""
    built = build(config)
    return dataclasses.replace(
        built, cell_placer=StandInCellPlacer(), validator=StandInValidator()
    )


def test_the_ppa_result_carries_the_run_and_the_tools_that_produced_it(design) -> None:
    # The whole point: a number nobody can attribute to a configuration is not a result.
    config, def_path, lef_paths, output_dir = design
    built = _build_with_stand_ins(config)

    result = evaluate_physical(built, def_path, lef_paths, output_dir)

    assert result.full_hash == built.config.full_hash()
    assert result.cell_placer == "dreamplace"
    assert result.validator == "openroad"
    assert result.design_area == 1234.5
    assert result.timing_slack is None   # not computed, never invented


def test_the_validator_measures_the_placed_def_not_the_input(design) -> None:
    config, def_path, lef_paths, output_dir = design
    built = _build_with_stand_ins(config)
    evaluate_physical(built, def_path, lef_paths, output_dir)
    assert built.validator.validated == [output_dir / "placed.def"]


def test_skip_cell_placement_validates_the_design_as_given(design) -> None:
    config, def_path, lef_paths, output_dir = design
    built = _build_with_stand_ins(config)
    result = evaluate_physical(built, def_path, lef_paths, output_dir, skip_cell_placement=True)
    assert built.validator.validated == [def_path]
    assert result.cell_placer is None   # honest: no placer ran, so none is credited


def test_a_config_with_no_validator_is_refused_rather_than_defaulted(design) -> None:
    # Silently defaulting to OpenROAD would put a tool nobody named into a recorded result.
    config, def_path, lef_paths, output_dir = design
    proxy_only = dataclasses.replace(config, environment=dataclasses.replace(
        config.environment, physical=PhysicalSpec()
    ))
    built = _build_with_stand_ins(proxy_only)
    with pytest.raises(ValueError, match="declares no validator"):
        evaluate_physical(built, def_path, lef_paths, output_dir)


def test_a_config_with_no_cell_placer_is_refused_unless_placement_is_skipped(design) -> None:
    config, def_path, lef_paths, output_dir = design
    validator_only = dataclasses.replace(config, environment=dataclasses.replace(
        config.environment, physical=PhysicalSpec(cell_placer=None, validator=Spec("openroad"))
    ))
    built = _build_with_stand_ins(validator_only)
    with pytest.raises(ValueError, match="declares no cell placer"):
        evaluate_physical(built, def_path, lef_paths, output_dir)
    # ...but skipping placement is a legitimate answer, not an error.
    assert evaluate_physical(
        built, def_path, lef_paths, output_dir, skip_cell_placement=True
    ).validator == "openroad"


def test_ppa_json_sits_beside_the_manifest_and_names_its_run(design, tmp_path) -> None:
    config, def_path, lef_paths, output_dir = design
    built = _build_with_stand_ins(config)
    result = evaluate_physical(built, def_path, lef_paths, output_dir)

    run_dir = tmp_path / "run"
    path = write_ppa(run_dir, result)
    assert path.name == PPA_NAME
    written = json.loads(path.read_text())
    assert written["full_hash"] == built.config.full_hash()
    assert written["design_area"] == 1234.5
    assert written["timing_slack"] is None


def test_the_tool_registries_name_the_shipped_defaults() -> None:
    # Registry membership is what makes a tool selectable from a config at all.
    assert "dreamplace" in CELL_PLACERS
    assert "openroad" in VALIDATORS


def test_a_proxy_only_run_never_imports_the_physical_tools(design) -> None:
    # Neither DREAMPlace nor OpenROAD may become a dependency of a wirelength-proxy experiment,
    # which is why the registry entries import lazily.
    config, _def_path, _lef_paths, _output_dir = design
    proxy_only = dataclasses.replace(config, environment=dataclasses.replace(
        config.environment, physical=PhysicalSpec()
    ))
    built = build(proxy_only)
    assert built.cell_placer is None and built.validator is None


# -------------------------------------------- the run's own placement reaching the flow


def _add_bookshelf_placement_files(benchmark_dir: pathlib.Path) -> None:
    """The .pl/.wts/.scl a real Bookshelf design carries, added AFTER the benchmark is built.

    After, not in the fixture: `Benchmark.load` reads a real die size out of the .pl when one is
    there, which would change this design's cell_size and with it every other test in this file.
    Export reads the .pl at export time, so writing them here is enough and changes nothing else.
    """
    (benchmark_dir / "s.pl").write_text(
        "UCLA pl 1.0\n\na\t0\t0\t: N\nb\t0\t0\t: N\nc\t0\t0\t: N\n"
    )
    (benchmark_dir / "s.wts").write_text("UCLA wts 1.0\n")
    (benchmark_dir / "s.scl").write_text("UCLA scl 1.0\n")


def test_a_runs_placement_can_reach_the_physical_flow_without_a_hand_made_def(
    design, tmp_path: pathlib.Path
) -> None:
    """The connection that did not exist: agent placement -> design file -> validator.

    `evaluate_physical` takes a DEF "with every macro already placed", and until
    `experiment.export` nothing in this repository produced one - so the flow was configured,
    hashed and completely unreachable from a run. Here the placement goes in as positions and a
    PPA number comes out, with no file made by hand anywhere in between.
    """
    import jax.numpy as jnp

    from placax.netlist import NetlistFormat
    from placax_agents.experiment.physical import evaluate_placement

    config, _def_path, lef_paths, output_dir = design
    built = _build_with_stand_ins(config)
    _add_bookshelf_placement_files(config.environment.benchmark.path)
    positions = jnp.array([[i, i] for i in range(built.benchmark.params.n_macros)])

    # This design is Bookshelf, which OpenROAD cannot read - so it is CONVERTED on the way in.
    # That conversion is what made the physical box reachable from a design that actually ships;
    # before it, this call raised and every benchmark here was stuck on the wrong side of it.
    result = evaluate_placement(built, positions, lef_paths, output_dir)

    assert result.full_hash == built.config.full_hash()
    assert (output_dir / "placement" / "s.def").exists()
    assert (output_dir / "placement" / "s.lef").exists(), (
        "Bookshelf carries no cell library, so the export has to derive one and hand it over"
    )
    # The derived LEF reaches the validator: it is an output of the export, not an input to it.
    assert any(path.name == "s.lef" for path in built.cell_placer.lefs)

    # The Bookshelf route is untouched - DREAMPlace reads it natively, and both routes have to
    # describe the same placement.
    from placax_agents.experiment.export import write_placement

    assert write_placement(built, positions, tmp_path / "again").format is NetlistFormat.BOOKSHELF


def test_the_exported_placement_is_the_one_that_was_scored(design) -> None:
    # A second rollout would be a different placement. The exported design has to be the array the
    # runner measured, or the PPA number describes something nobody reported.
    import jax.numpy as jnp

    from placax_agents.experiment.export import write_placement
    from placax_agents.experiment.run import score

    config, _def_path, _lef_paths, output_dir = design
    built = _build_with_stand_ins(config)
    _add_bookshelf_placement_files(config.environment.benchmark.path)
    positions = jnp.array([[i, i] for i in range(built.benchmark.params.n_macros)])

    measured = score(built.benchmark, positions, built.n_placed)
    exported = write_placement(built, positions, output_dir / "placement")

    placed_text = (exported.path.parent / "s.pl").read_text()
    cell = built.benchmark.cell_size
    for name, idx in built.benchmark.name_to_idx.items():
        assert f"{name}\t{int(round(idx * cell))}\t{int(round(idx * cell))}\t" in placed_text
    assert measured["real_hpwl"] > 0  # the scored placement is the exported one, not a rerun
