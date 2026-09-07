"""place_and_validate: the composition the design document described for a year without it existing.

Every test here uses stand-in tools rather than DREAMPlace and OpenROAD, which is the point being
tested - the composition must not know or care which tools it is driving. The real OpenROAD
invocation is covered separately by tests/test_validator.py's script-building and output-parsing
tests; neither OpenROAD nor a DEF/LEF design is available in this repo, so an end-to-end run
against the real binary is NOT verified here.
"""
import pathlib

import pytest

from placax_tools.cell_placer import CellPlacer
from placax_tools.pipeline import PlacedDesign, place_and_validate, validate_only
from placax_tools.validator import PPAResult, Validator


class RecordingCellPlacer(CellPlacer):
    """Writes a 'placed' DEF and remembers what it was asked to do."""

    def __init__(self):
        self.calls = []

    def place(self, def_path, lef_paths, output_dir):
        self.calls.append((def_path, list(lef_paths), output_dir))
        placed = output_dir / "placed.def"
        placed.write_text(def_path.read_text() + "\n# cells placed\n")
        return placed


class RecordingValidator(Validator):
    def __init__(self, ppa: PPAResult | None = None):
        self.calls = []
        self.ppa = ppa or PPAResult(design_area=1234.5, utilization_pct=71.2,
                                    timing_slack=None, raw_output="ok")

    def validate(self, def_path, lef_paths, output_dir):
        self.calls.append((def_path, list(lef_paths), output_dir))
        return self.ppa


@pytest.fixture
def design(tmp_path: pathlib.Path):
    def_path = tmp_path / "macros.def"
    def_path.write_text("DESIGN top ;\nEND DESIGN\n")
    lefs = [tmp_path / "tech.lef", tmp_path / "cells.lef"]
    for lef in lefs:
        lef.write_text("VERSION 5.8 ;\n")
    return def_path, lefs, tmp_path / "out"


def test_places_cells_then_validates_the_placed_output_not_the_input(design) -> None:
    # The ordering bug this rules out: validating the ORIGINAL def would report the macro-only
    # design's PPA while looking exactly like a full-flow result.
    def_path, lefs, output_dir = design
    placer, validator = RecordingCellPlacer(), RecordingValidator()

    result = place_and_validate(def_path, lefs, output_dir, placer, validator)

    assert isinstance(result, PlacedDesign)
    assert placer.calls[0][0] == def_path
    assert validator.calls[0][0] == output_dir / "placed.def"
    assert validator.calls[0][0] != def_path
    assert result.def_path.read_text().endswith("# cells placed\n")


def test_both_tools_receive_the_same_lefs_and_output_dir(design) -> None:
    def_path, lefs, output_dir = design
    placer, validator = RecordingCellPlacer(), RecordingValidator()
    place_and_validate(def_path, lefs, output_dir, placer, validator)
    assert placer.calls[0][1] == validator.calls[0][1] == lefs
    assert placer.calls[0][2] == validator.calls[0][2] == output_dir


def test_creates_the_output_directory_so_tools_can_write_into_it(design) -> None:
    def_path, lefs, output_dir = design
    assert not output_dir.exists()
    place_and_validate(def_path, lefs, output_dir, RecordingCellPlacer(), RecordingValidator())
    assert output_dir.is_dir()


def test_ppa_is_returned_verbatim_including_uncomputed_metrics(design) -> None:
    # A validator that didn't run timing reports None, and that None must survive to the caller
    # rather than being smoothed into a number someone could mistake for a measurement.
    def_path, lefs, output_dir = design
    ppa = PPAResult(design_area=10.0, utilization_pct=None, timing_slack=None, raw_output="raw")
    result = place_and_validate(def_path, lefs, output_dir, RecordingCellPlacer(),
                                RecordingValidator(ppa))
    assert result.ppa is ppa
    assert result.ppa.utilization_pct is None
    assert result.ppa.timing_slack is None


def test_works_with_any_pair_of_tools_without_the_composition_changing(design) -> None:
    # The design rule this exists to satisfy: substituting a different placer or validator is a
    # change at the call site and nowhere else.
    def_path, lefs, output_dir = design

    class OtherPlacer(CellPlacer):
        def place(self, def_path, lef_paths, output_dir):
            out = output_dir / "other.def"
            out.write_text("placed by someone else\n")
            return out

    class OtherValidator(Validator):
        def validate(self, def_path, lef_paths, output_dir):
            return PPAResult(design_area=99.0, utilization_pct=1.0, timing_slack=-0.5,
                             raw_output="")

    result = place_and_validate(def_path, lefs, output_dir, OtherPlacer(), OtherValidator())
    assert result.def_path.name == "other.def"
    assert result.ppa.timing_slack == -0.5


def test_validate_only_skips_cell_placement_entirely(design) -> None:
    # The Bookshelf flow's case: DREAMPlace already placed the cells natively, so running a
    # placer again would redo the expensive half of the flow for nothing.
    def_path, lefs, output_dir = design
    validator = RecordingValidator()
    ppa = validate_only(def_path, lefs, output_dir, validator)
    assert ppa is validator.ppa
    assert validator.calls[0][0] == def_path
    assert output_dir.is_dir()
