"""The bridge from a run's placement to a file an external tool can open.

`write_placed_def` shipped for months with exactly one caller in the tree: its own unit test.
There was no way to get from an agent's placement to a design file, so `scripts/validate_design.py`
asked for a macro-placed DEF that nothing here produced, and the PPA box at the bottom of the
architecture was not merely unverified - it was unreachable. These tests are what say it is
connected now, in both formats the downstream tools actually read.
"""
import pathlib

from placax.core import reset  # noqa: F401  must precede jax imports
from placax.netlist import NetlistFormat
from placax_agents.experiment.budget import Budget
from placax_agents.experiment.build import build
from placax_agents.experiment.export import BOOKSHELF_SIBLINGS, write_placement
from placax_agents.experiment.presets import training

import dataclasses
import jax.numpy as jnp
import pytest

NODES = ("UCLA nodes 1.0\nNumNodes : 3\nNumTerminals : 3\n"
         "a 4 4 terminal\nb 2 2 terminal\nc 2 4 terminal\n")
NETS = ("UCLA nets 1.0\nNumNets : 2\nNumPins : 4\n"
        "NetDegree : 2 n0\n\ta I : 0.0 0.0\n\tb O : 0.0 0.0\n"
        "NetDegree : 2 n1\n\tb I : 0.0 0.0\n\tc O : 0.0 0.0\n")
PL = ("UCLA pl 1.0\n\n"
      "a\t0\t0\t: N\n"
      "b\t0\t0\t: N\n"
      "c\t0\t0\t: N\n"
      "filler\t0\t0\t: N\n")


def _bookshelf(directory: pathlib.Path) -> pathlib.Path:
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "s.aux").write_text("RowBasedPlacement : s.nodes s.nets s.wts s.pl s.scl\n")
    (directory / "s.nodes").write_text(NODES)
    (directory / "s.nets").write_text(NETS)
    (directory / "s.pl").write_text(PL)
    (directory / "s.wts").write_text("UCLA wts 1.0\n")
    (directory / "s.scl").write_text("UCLA scl 1.0\n")
    return directory


def _built(directory: pathlib.Path):
    config = training(directory, budget=Budget(iterations=1))
    config = dataclasses.replace(config, environment=dataclasses.replace(
        config.environment,
        benchmark=dataclasses.replace(config.environment.benchmark, grid=8),
    ))
    return build(config)


def _positions(built):
    """A placement in grid cells, one macro per row, in the environment's own placement order."""
    return jnp.array([[i, i] for i in range(built.benchmark.params.n_macros)])


def test_a_placement_reaches_a_file_a_cell_placer_can_open(tmp_path: pathlib.Path) -> None:
    built = _built(_bookshelf(tmp_path / "bench"))
    exported = write_placement(built, _positions(built), tmp_path / "out")

    assert exported.format is NetlistFormat.BOOKSHELF
    assert exported.path.name == "s.aux"
    assert exported.path.exists()
    # The .aux must reference bare filenames: Limbo's Bookshelf grammar rejects a leading '/'.
    assert exported.path.read_text().strip().endswith("s.nodes s.nets s.wts s.pl s.scl")


def test_every_placed_macro_is_written_fixed_at_its_own_position(tmp_path: pathlib.Path) -> None:
    built = _built(_bookshelf(tmp_path / "bench"))
    exported = write_placement(built, _positions(built), tmp_path / "out")

    placed = (exported.path.parent / "s.pl").read_text()
    cell = built.benchmark.cell_size
    for name, idx in built.benchmark.name_to_idx.items():
        expected = int(round(idx * cell))
        assert f"{name}\t{expected}\t{expected}\t: N /FIXED" in placed, name
    assert exported.n_macros == built.benchmark.params.n_macros


def test_everything_the_run_did_not_place_passes_through_untouched(tmp_path: pathlib.Path) -> None:
    # A macro budget and every standard cell are exactly this case: the writer rewrites only the
    # instances it is handed, so the rest of the design has to survive verbatim.
    built = _built(_bookshelf(tmp_path / "bench"))
    exported = write_placement(built, _positions(built), tmp_path / "out")

    placed = (exported.path.parent / "s.pl").read_text()
    assert "filler\t0\t0\t: N" in placed
    assert "/FIXED" not in placed.split("filler")[1]


def test_the_unchanged_files_are_linked_rather_than_copied(tmp_path: pathlib.Path) -> None:
    # .nets alone runs to hundreds of megabytes on the larger ISPD designs; copying it per
    # placement would make exporting a run cost more than producing it.
    built = _built(_bookshelf(tmp_path / "bench"))
    exported = write_placement(built, _positions(built), tmp_path / "out")

    for suffix in BOOKSHELF_SIBLINGS:
        assert (exported.path.parent / f"s.{suffix}").is_symlink()


def test_the_export_names_the_run_it_came_from(tmp_path: pathlib.Path) -> None:
    # An exported design that cannot be traced back to a run is the same unattributable artifact
    # the whole experiment layer exists to stop producing.
    built = _built(_bookshelf(tmp_path / "bench"))
    exported = write_placement(built, _positions(built), tmp_path / "out")
    assert exported.full_hash == built.config.full_hash()


def test_exporting_twice_is_stable(tmp_path: pathlib.Path) -> None:
    # The symlinks already exist on the second pass; that must not raise.
    built = _built(_bookshelf(tmp_path / "bench"))
    first = write_placement(built, _positions(built), tmp_path / "out")
    second = write_placement(built, _positions(built), tmp_path / "out")
    assert first.path == second.path
    assert (first.path.parent / "s.pl").read_text() == (second.path.parent / "s.pl").read_text()


def test_a_format_with_no_placement_file_says_so_rather_than_writing_nothing(
    tmp_path: pathlib.Path, monkeypatch
) -> None:
    built = _built(_bookshelf(tmp_path / "bench"))
    monkeypatch.setattr(
        "placax_agents.experiment.export.detect_format", lambda _dir: NetlistFormat.PROTOBUF
    )
    with pytest.raises(NotImplementedError, match="protobuf"):
        write_placement(built, _positions(built), tmp_path / "out")


def test_a_def_placement_is_written_in_the_designs_database_units(tmp_path: pathlib.Path) -> None:
    """DEF coordinates are database units; macro geometry from LEF is microns.

    Getting this wrong is silent - the file parses, every macro is simply in the wrong place by a
    factor of the design's UNITS - so the conversion is pinned here rather than trusted.
    """
    from placax_agents.experiment.export import _export_def

    directory = _bookshelf(tmp_path / "bench")
    built = _built(directory)
    (directory / "s.def").write_text(
        "VERSION 5.8 ;\nUNITS DISTANCE MICRONS 1000 ;\n"
        "COMPONENTS 1 ;\n- a AND2 + PLACED ( 0 0 ) ;\nEND COMPONENTS\nEND DESIGN\n"
    )
    # _export_def is exercised directly: this is about the unit conversion, and reaching it
    # through write_placement would need detect_format to prefer DEF over the .aux beside it.
    out = tmp_path / "def_out"
    out.mkdir()
    exported = _export_def(built, {"a": (3, 5)}, out)

    assert exported.format is NetlistFormat.DEF
    assert "- a AND2 + PLACED ( 3000 5000 )" in exported.path.read_text()
