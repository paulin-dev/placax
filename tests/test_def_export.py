"""Bookshelf to DEF/LEF, checked by reading it back with this project's own parsers.

The physical box was configured, hashed, documented and tested, and no design that ships could
reach it: OpenROAD reads DEF/LEF, every benchmark here is Bookshelf or protobuf, and nothing
converted between them. `evaluate_placement` raised.

There is no OpenROAD on this machine, so what can be verified here is self-consistency rather than
tool acceptance - and that distinction is the point of the round trip. The exporter writes the
design; `netlist/lef.py` and `netlist/def_reader.py` read it back; the geometry, the placement and
the connectivity that come back have to be the ones that went in. That catches the errors that
actually happen in a converter - a pin offset measured from the wrong corner, a unit scale applied
once too often, a macro silently dropped - without claiming anything about whether a real tool
likes the file.

What it cannot check is stated in `netlist/def_export.py`: the generated LEF is not a technology.
"""
import pathlib
import re

import pytest

from placax.netlist.bookshelf import parse_all_node_sizes, parse_nets
from placax.netlist.def_export import (
    CellLibrary, DB_UNITS_PER_MICRON, collect_pin_offsets, export_bookshelf_as_def, write_def,
    write_lef,
)
from placax.netlist.def_reader import parse_components, parse_nets as parse_def_nets
from placax.netlist.lef import parse_lef_pin_offsets, parse_lef_sizes
from placax.netlist.rows import load_placement_rows

NODES = """UCLA nodes 1.0
NumNodes : 5
NumTerminals : 2

  macro_a 40 20 terminal
  macro_b 20 60 terminal
  cell_x 2 12
  cell_y 2 12
  cell_z 4 12
"""

NETS = """UCLA nets 1.0
NumNets : 3
NumPins : 7

NetDegree : 3 n0
\tmacro_a I : 5.0 -2.0
\tcell_x O : 0.5 1.0
\tcell_y I : -0.5 0.0
NetDegree : 2 n1
\tmacro_b I : -3.0 4.0
\tcell_z O : 1.0 -1.0
NetDegree : 2 n2
\tmacro_a O : -5.0 2.0
\tmacro_b O : 3.0 -4.0
"""

SCL = """UCLA scl 1.0
NumRows : 2

CoreRow Horizontal
  Coordinate :   0
  Height :   12
  Sitewidth :  1
  Sitespacing :  1
  Siteorient :  N
  Sitesymmetry :  Y
  SubrowOrigin :  0  NumSites :  100
End
CoreRow Horizontal
  Coordinate :   12
  Height :   12
  Sitewidth :  1
  Sitespacing :  1
  Siteorient :  N
  Sitesymmetry :  Y
  SubrowOrigin :  0  NumSites :  100
End
"""


@pytest.fixture
def design(tmp_path: pathlib.Path) -> pathlib.Path:
    directory = tmp_path / "bench"
    directory.mkdir()
    (directory / "s.aux").write_text("RowBasedPlacement : s.nodes s.nets s.wts s.pl s.scl\n")
    (directory / "s.nodes").write_text(NODES)
    (directory / "s.nets").write_text(NETS)
    (directory / "s.scl").write_text(SCL)
    return directory


PLACEMENT = {"macro_a": (10.0, 24.0), "macro_b": (60.0, 12.0)}


# --------------------------------------------------------------- the cell library


def test_instances_with_identical_geometry_share_a_cell_type(design) -> None:
    """Bookshelf has no cell types; LEF needs them. This is the grouping that invents them.

    `cell_x` and `cell_y` are both 2x12 but carry different pin offsets, so they are NOT the same
    cell - geometry means footprint AND pins. Measured on adaptec1, this collapses 211,447
    instances to 520 types, which is what keeps the generated library small enough to be a library.
    """
    sizes = parse_all_node_sizes(design / "s.nodes")
    nets = parse_nets(design / "s.nets", set(sizes))
    library = CellLibrary(sizes, collect_pin_offsets(nets), {"macro_a", "macro_b"})

    assert library.cell_of["cell_x"] != library.cell_of["cell_y"], "different pins, different cell"
    assert library.is_block[library.cell_of["macro_a"]] is True
    assert library.is_block[library.cell_of["cell_x"]] is False


def test_a_macro_and_a_cell_of_one_size_are_still_different_cells() -> None:
    # CLASS BLOCK vs CLASS CORE is the difference between something a placer routes around and
    # something it places, so geometry alone must not merge them.
    sizes = {"m": (4.0, 12.0), "c": (4.0, 12.0)}
    library = CellLibrary(sizes, {}, macro_names={"m"})
    assert library.cell_of["m"] != library.cell_of["c"]


# --------------------------------------------------------------- the round trip


def test_every_nodes_size_survives_the_round_trip(design, tmp_path) -> None:
    """Macro or standard cell, the geometry that comes back is the geometry that went in.

    Read two different ways on purpose: `parse_components` returns only nodes with a POSITION, so
    the unplaced standard cells are looked up through the component lines directly. That asymmetry
    is the export's own design - cells are UNPLACED because placing them is the cell placer's job.
    """
    def_path, lef_path = export_bookshelf_as_def(design, tmp_path / "out", PLACEMENT)
    sizes = parse_all_node_sizes(design / "s.nodes")
    cell_sizes = parse_lef_sizes(lef_path)
    text = def_path.read_text()

    placed = parse_components(text)
    assert set(placed) == set(PLACEMENT), "only the macros carry a position"

    cell_of = dict(re.findall(r"-\s+(\S+)\s+(PLACAX_CELL_\d+)\s+\+", text))
    assert set(cell_of) == set(sizes), "every node reaches the DEF"
    for name, (width, height) in sizes.items():
        assert cell_sizes[cell_of[name]] == pytest.approx((width, height))


def test_pin_offsets_survive_the_round_trip(design, tmp_path) -> None:
    """The conversion most likely to be wrong, and the one a tool would not complain about.

    Bookshelf measures a pin from its macro's CENTRE; LEF geometry is measured from the macro's
    lower-left CORNER. Getting that backwards shifts every pin by half a macro and produces a
    design that reads perfectly and has the wrong wirelength.
    """
    _def_path, lef_path = export_bookshelf_as_def(design, tmp_path / "out", PLACEMENT)
    sizes = parse_all_node_sizes(design / "s.nodes")
    nets = parse_nets(design / "s.nets", set(sizes))
    expected = collect_pin_offsets(nets)

    library = CellLibrary(sizes, expected, {"macro_a", "macro_b"})
    read_back = parse_lef_pin_offsets(lef_path)
    for instance, offsets in expected.items():
        cell = library.cell_of[instance]
        recovered = sorted(read_back[cell].values())
        assert recovered == pytest.approx(sorted(offsets), abs=1e-3)


def test_the_agents_placement_survives_the_round_trip(design, tmp_path) -> None:
    # Positions go out in microns and come back in database units; applying the scale twice, or
    # not at all, is the other classic converter bug.
    def_path, _lef_path = export_bookshelf_as_def(design, tmp_path / "out", PLACEMENT)
    components = parse_components(def_path.read_text())
    for name, (x, y) in PLACEMENT.items():
        _cell, read_x, read_y = components[name]
        assert read_x == pytest.approx(x * DB_UNITS_PER_MICRON)
        assert read_y == pytest.approx(y * DB_UNITS_PER_MICRON)


def test_every_node_is_exported_not_only_the_macros(design, tmp_path) -> None:
    """A DEF of macros and no logic measures a die that does not exist.

    The standard cells have to be there for utilization or timing to mean anything - UNPLACED,
    because placing them is the cell placer's job, which is exactly the input
    `place_and_validate` expects.
    """
    def_path, _lef_path = export_bookshelf_as_def(design, tmp_path / "out", PLACEMENT)
    text = def_path.read_text()
    sizes = parse_all_node_sizes(design / "s.nodes")

    assert f"COMPONENTS {len(sizes)} ;" in text
    for cell in ("cell_x", "cell_y", "cell_z"):
        assert f"- {cell} " in text and "UNPLACED" in text
    # The placed macros are FIXED, which is what tells a cell placer not to move them.
    assert "+ FIXED" in text
    # ...and `parse_components` sees them, which it did not before FIXED was recognised.
    components = parse_components(text)
    assert set(PLACEMENT) <= set(components)


def test_connectivity_survives_the_round_trip(design, tmp_path) -> None:
    def_path, _lef_path = export_bookshelf_as_def(design, tmp_path / "out", PLACEMENT)
    sizes = parse_all_node_sizes(design / "s.nodes")
    source = parse_nets(design / "s.nets", set(sizes))

    read_back = parse_def_nets(def_path.read_text())
    assert len(read_back) == len(source)
    for original, recovered in zip(source, read_back):
        assert {name for name, _x, _y in original} == {name for name, _port in recovered}


def test_the_die_and_rows_come_from_the_designs_own_scl(design, tmp_path) -> None:
    # The row geometry is what tells a placer where a cell may legally sit; inventing it would
    # make the utilization number meaningless.
    def_path, _lef_path = export_bookshelf_as_def(design, tmp_path / "out", PLACEMENT)
    rows = load_placement_rows(design)
    text = def_path.read_text()

    assert rows is not None
    assert f"DIEAREA ( {round(rows.x0 * DB_UNITS_PER_MICRON)}" in text
    assert text.count("ROW ROW_") == rows.n_rows


def test_a_design_without_rows_still_exports(tmp_path) -> None:
    # No .scl is a real case (it is why `canvas="core"` refuses some designs), and it must not
    # take the physical flow down with it.
    directory = tmp_path / "bench"
    directory.mkdir()
    (directory / "s.aux").write_text("RowBasedPlacement : s.nodes s.nets s.wts s.pl s.scl\n")
    (directory / "s.nodes").write_text(NODES)
    (directory / "s.nets").write_text(NETS)

    def_path, lef_path = export_bookshelf_as_def(directory, tmp_path / "out", PLACEMENT)
    assert "DIEAREA" in def_path.read_text()
    assert "ROW ROW_" not in def_path.read_text()
    assert parse_lef_sizes(lef_path)


def test_orientation_reaches_the_def(design, tmp_path) -> None:
    def_path, _lef_path = export_bookshelf_as_def(
        design, tmp_path / "out", PLACEMENT, orientations={"macro_a": "W"}
    )
    text = def_path.read_text()
    assert "- macro_a " in text and " W ;" in text
    assert " N ;" in text, "an unturned macro is still north"


# --------------------------------------------------------------- the generated LEF's shape


def test_the_lef_carries_what_a_pin_needs_to_sit_on(design, tmp_path) -> None:
    # Not a technology - see netlist/def_export.py - but a PIN needs a LAYER and a CORE cell needs
    # a SITE, or the file is not readable at all.
    _def_path, lef_path = export_bookshelf_as_def(design, tmp_path / "out", PLACEMENT)
    text = lef_path.read_text()
    for required in ("VERSION", "UNITS", "DATABASE MICRONS", "LAYER metal1", "SITE core",
                     "END LIBRARY"):
        assert required in text, required
    assert "CLASS BLOCK" in text, "macros are blockages"
    assert "CLASS CORE" in text, "standard cells are placeable"


# --------------------------------------------------------------- at real scale


def test_a_real_benchmark_converts_and_reads_back(tmp_path) -> None:
    """adaptec1 end to end, because the tiny fixture cannot show the two things that matter.

    Scale: 211,447 instances and 216,932 nets is where a per-instance cell library would have been
    the wrong choice, and where a quadratic mistake would show. And completeness: this is a design
    nobody wrote by hand, so it exercises the cases a fixture is too tidy to contain.

    Takes a few seconds; it is the only test that touches the real netlist, and the conversion it
    covers is the one standing between a run and a PPA number.
    """
    from tests.real_benchmarks import ADAPTEC1

    if not ADAPTEC1.exists():
        pytest.skip("adaptec1 is not checked out")

    from placax.netlist.bookshelf import parse_nodes, parse_pl_positions

    macros = parse_nodes(ADAPTEC1 / "adaptec1.nodes")
    pl = parse_pl_positions(ADAPTEC1 / "adaptec1.pl")
    placement = {name: (pl[name][0], pl[name][1]) for name in macros}

    def_path, lef_path = export_bookshelf_as_def(ADAPTEC1, tmp_path / "out", placement)
    text = def_path.read_text()
    components = parse_components(text)
    cells = parse_lef_sizes(lef_path)

    # Every macro placed, and placed where it was put.
    assert set(components) == set(macros)
    for name, (x, y) in placement.items():
        _cell, read_x, read_y = components[name]
        assert read_x == pytest.approx(x * DB_UNITS_PER_MICRON, abs=1.0)
        assert read_y == pytest.approx(y * DB_UNITS_PER_MICRON, abs=1.0)

    # The library really does collapse - a per-instance one would have 211,447 entries here.
    assert len(cells) < 1000, f"{len(cells)} cell types is not a library"
    assert len(parse_def_nets(text)) > 200_000
