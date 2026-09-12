"""Bookshelf to DEF/LEF: the bridge that lets a shipped design reach the validator at all.

The physical box at the bottom of the architecture has been configured, hashed, documented and
tested for a while, and **structurally unreachable from any run**. `evaluate_placement` reads
DEF/LEF because that is what OpenROAD reads; every benchmark in this repository is Bookshelf or
protobuf; so "every experiment uses the same PPA evaluation" was a claim no shipped design could
test. This module is the missing conversion.

**It exports the WHOLE design, not just the macros, and that is the point.** A DEF holding
adaptec1's 543 macros and nothing else would read fine and measure nothing: utilization of a die
containing no logic, timing of a netlist with no cells. So every node is exported -

  * **macros** at the agent's placement, marked `FIXED`, because that is the result being measured;
  * **standard cells** marked `UNPLACED`, because placing them is the cell placer's job.

which is exactly the shape `placax_tools.pipeline.place_and_validate` expects: macros already
placed, cells for DREAMPlace (or whatever `CELL_PLACERS` names), then the validator on the result.

**A cell library falls out of the netlist, and it is small.** Bookshelf has no notion of cell
types - every node carries its own width, height and pin offsets - so the naive conversion emits
one LEF `MACRO` per instance. Measured on adaptec1 before choosing: its 211,447 instances use only
**55 distinct sizes and 520 distinct (size + pin-offset) signatures**, a 406x collapse. Grouping by
geometry therefore produces a real, small cell library rather than a 211k-entry one, and instances
that genuinely differ still get their own type.

**What the generated LEF is, stated plainly, because it is not a technology.** It carries the one
routing layer pins need, a site matching the design's own rows, and the cell geometry. It has no
via rules, no routing tracks, no antenna rules and no timing library. That is enough for a tool to
read the design, place the cells and report area and utilization; it is NOT enough to route, and a
routed wirelength or a DRC count taken against it would be a number about this file rather than
about a chip. Timing needs a real `.lib` and is already reported as None without one.

**Units are a convention, not a measurement.** Bookshelf coordinates are dimensionless integers.
They are written here as microns with a 1000x database factor, which keeps LEF (microns) and DEF
(database units) consistent with each other and with this project's own readers. Nothing in the
Bookshelf files says a unit is a micron; the exported design is self-consistent and its absolute
scale is arbitrary.
"""
import collections
import pathlib

from placax.netlist.bookshelf import parse_all_node_sizes, parse_nets
from placax.netlist.rows import PlacementRows
from placax.types import Nets, SizeMap

DB_UNITS_PER_MICRON = 1000
"""DEF database units per LEF micron. One Bookshelf unit is written as one micron."""

ROUTING_LAYER = "metal1"
SITE_NAME = "core"
PIN_SIZE = 0.1
"""Side of the square each pin is drawn as, in microns.

A Bookshelf pin is a point offset, and LEF needs a rectangle. Small enough not to overlap its
neighbours on the tightest cell here, large enough to be on the manufacturing grid."""


class CellLibrary:
    """Distinct cell geometries in a design, and which one each instance uses.

    Bookshelf stores geometry per instance; LEF stores it per cell type. This is the grouping that
    turns one into the other - instances whose footprint AND pin offsets agree share a type.
    """

    def __init__(self, sizes: SizeMap, pin_offsets: dict[str, list[tuple[float, float]]],
                 macro_names: set[str]):
        self.cell_of: dict[str, str] = {}
        self.geometry: dict[str, tuple[float, float]] = {}
        self.pins: dict[str, list[tuple[float, float]]] = {}
        self.is_block: dict[str, bool] = {}

        by_signature: dict[tuple, str] = {}
        for name, (width, height) in sizes.items():
            pins = tuple(sorted(pin_offsets.get(name, ())))
            # A macro and a standard cell of identical geometry are still different CLASSes to a
            # placer - one is a blockage it must route around, the other is something it places.
            signature = (width, height, pins, name in macro_names)
            cell = by_signature.get(signature)
            if cell is None:
                cell = f"PLACAX_CELL_{len(by_signature)}"
                by_signature[signature] = cell
                self.geometry[cell] = (width, height)
                self.pins[cell] = list(pins)
                self.is_block[cell] = name in macro_names
            self.cell_of[name] = cell

    def pin_name(self, cell: str, offset: tuple[float, float]) -> str:
        """The LEF pin name for one offset on one cell - positional, since Bookshelf has none."""
        return f"P{self.pins[cell].index(offset)}"

    def __len__(self) -> int:
        return len(self.geometry)


def collect_pin_offsets(nets: Nets) -> dict[str, list[tuple[float, float]]]:
    """{instance: [distinct pin offsets]} - every place a net attaches to that instance.

    One instance can carry many offsets (217 at most on adaptec1), and two instances of the same
    size can carry different ones, which is why the cell library groups on geometry rather than on
    size alone.
    """
    offsets: dict[str, set[tuple[float, float]]] = collections.defaultdict(set)
    for net in nets:
        for name, x_offset, y_offset in net:
            offsets[name].add((round(x_offset, 6), round(y_offset, 6)))
    return {name: sorted(values) for name, values in offsets.items()}


def write_lef(library: CellLibrary, rows: PlacementRows | None) -> str:
    """The technology-and-cells LEF for a Bookshelf design. See this module's docstring for limits."""
    site_width = rows.site_width if rows is not None else 1.0
    site_height = rows.row_pitch if rows is not None else 1.0

    out = [
        "VERSION 5.8 ;",
        'BUSBITCHARS "[]" ;',
        'DIVIDERCHAR "/" ;',
        "UNITS",
        f"  DATABASE MICRONS {DB_UNITS_PER_MICRON} ;",
        "END UNITS",
        f"MANUFACTUREGRID {1.0 / DB_UNITS_PER_MICRON} ;",
        "",
        # One routing layer, because a PIN needs a LAYER to sit on. Enough to read and place;
        # not a technology - see the module docstring.
        f"LAYER {ROUTING_LAYER}",
        "  TYPE ROUTING ;",
        "  DIRECTION HORIZONTAL ;",
        f"  PITCH {site_width} ;",
        f"  WIDTH {PIN_SIZE} ;",
        f"END {ROUTING_LAYER}",
        "",
        f"SITE {SITE_NAME}",
        "  CLASS CORE ;",
        "  SYMMETRY Y ;",
        f"  SIZE {site_width} BY {site_height} ;",
        f"END {SITE_NAME}",
        "",
    ]

    for cell, (width, height) in library.geometry.items():
        block = library.is_block[cell]
        out.append(f"MACRO {cell}")
        # BLOCK is a hard macro the placer must work around; CORE is a cell it places in a row.
        out.append(f"  CLASS {'BLOCK' if block else 'CORE'} ;")
        out.append("  ORIGIN 0 0 ;")
        out.append(f"  SIZE {width} BY {height} ;")
        out.append("  SYMMETRY X Y ;")
        if not block:
            out.append(f"  SITE {SITE_NAME} ;")
        for index, (x_offset, y_offset) in enumerate(library.pins[cell]):
            # Bookshelf offsets are from the macro's CENTER; LEF geometry is from its lower-left
            # corner. `lef.parse_lef_pin_offsets` inverts exactly this, which is what makes the
            # round trip a real check rather than a formality.
            centre_x = x_offset + width / 2.0
            centre_y = y_offset + height / 2.0
            half = PIN_SIZE / 2.0
            out.extend([
                f"  PIN P{index}",
                "    DIRECTION INOUT ;",
                "    USE SIGNAL ;",
                "    PORT",
                f"      LAYER {ROUTING_LAYER} ;",
                f"        RECT {centre_x - half:.4f} {centre_y - half:.4f} "
                f"{centre_x + half:.4f} {centre_y + half:.4f} ;",
                "    END",
                f"  END P{index}",
            ])
        out.append(f"END {cell}")
        out.append("")

    out.append("END LIBRARY")
    return "\n".join(out) + "\n"


def _die_area(sizes: SizeMap, rows: PlacementRows | None) -> tuple[float, float, float, float]:
    """The die rectangle in microns - the design's own rows where it has them."""
    if rows is not None:
        return rows.x0, rows.y0, rows.x1, rows.y1
    # No .scl: fall back to the extent the macros themselves imply, which is what the `die` canvas
    # has always used.
    span = max(max(w, h) for w, h in sizes.values()) if sizes else 1.0
    return 0.0, 0.0, span, span


def write_def(
    design: str,
    sizes: SizeMap,
    library: CellLibrary,
    nets: Nets,
    placement: dict[str, tuple[float, float]],
    rows: PlacementRows | None = None,
    orientations: dict[str, str] | None = None,
    fixed: set[str] | None = None,
) -> str:
    """The design as DEF: what is placed, how firmly, and what is still waiting to be.

    `placement` is {node: (x, y)} lower-left corners in the same units as `sizes` - what
    `experiment.export` already computes. Anything absent from it is written UNPLACED, which is
    the cell placer's cue to place it.

    `fixed` says which of the placed nodes a downstream tool may NOT move; the rest are written
    `PLACED`. Defaults to all of them, which is the macro-placement case. The distinction is not
    cosmetic - it is how a cell placer is told "these macros are the result, work around them",
    and how a validator is told "these cells may still be legalized".
    """
    fixed = set(placement) if fixed is None else fixed
    orientations = orientations or {}
    x0, y0, x1, y1 = _die_area(sizes, rows)
    scale = DB_UNITS_PER_MICRON

    out = [
        "VERSION 5.8 ;",
        'DIVIDERCHAR "/" ;',
        'BUSBITCHARS "[]" ;',
        f"DESIGN {design} ;",
        f"UNITS DISTANCE MICRONS {scale} ;",
        f"DIEAREA ( {round(x0 * scale)} {round(y0 * scale)} ) "
        f"( {round(x1 * scale)} {round(y1 * scale)} ) ;",
        "",
    ]

    if rows is not None and rows.site_width > 0:
        # One ROW per row of sites, which is what tells a placer where cells may legally go. The
        # same geometry `netlist/rows.py` reads back out of a DEF.
        n_sites = max(int(rows.width / rows.site_width), 1)
        for index in range(rows.n_rows):
            y = rows.y0 + index * rows.row_pitch
            out.append(
                f"ROW ROW_{index} {SITE_NAME} {round(rows.x0 * scale)} {round(y * scale)} N "
                f"DO {n_sites} BY 1 STEP {round(rows.site_width * scale)} 0 ;"
            )
        out.append("")

    out.append(f"COMPONENTS {len(sizes)} ;")
    for name in sizes:
        cell = library.cell_of[name]
        if name in placement:
            x, y = placement[name]
            orient = orientations.get(name, "N")
            status = "FIXED" if name in fixed else "PLACED"
            out.append(
                f"    - {name} {cell} + {status} ( {round(x * scale)} {round(y * scale)} ) "
                f"{orient} ;"
            )
        else:
            # The cell placer's input: a component with a type and no position yet.
            out.append(f"    - {name} {cell} + UNPLACED ;")
    out.append("END COMPONENTS")
    out.append("")

    out.append(f"NETS {len(nets)} ;")
    for index, net in enumerate(nets):
        pins = " ".join(
            f"( {name} {library.pin_name(library.cell_of[name], (round(x, 6), round(y, 6)))} )"
            for name, x, y in net
        )
        out.append(f"    - N{index} {pins} + USE SIGNAL ;")
    out.append("END NETS")
    out.append("")
    out.append("END DESIGN")
    return "\n".join(out) + "\n"


def export_bookshelf_as_def(
    benchmark_dir: pathlib.Path,
    output_dir: pathlib.Path,
    placement: dict[str, tuple[float, float]],
    orientations: dict[str, str] | None = None,
    fixed: set[str] | None = None,
) -> tuple[pathlib.Path, pathlib.Path]:
    """Writes `(def_path, lef_path)` for a Bookshelf design carrying `placement` for its macros.

    Reads the design's own files rather than a loaded `Benchmark`, deliberately: a Benchmark is
    macro-only (and possibly truncated by a macro budget), and a DEF that dropped the standard
    cells would describe a die containing no logic. What the agent placed comes in through
    `placement`; everything else comes from the netlist on disk.
    """
    from placax.netlist.rows import load_placement_rows

    benchmark_dir = pathlib.Path(benchmark_dir)
    design = next(benchmark_dir.glob("*.aux")).stem
    sizes = parse_all_node_sizes(benchmark_dir / f"{design}.nodes")
    nets = parse_nets(benchmark_dir / f"{design}.nets", set(sizes))
    # What counts as a MACRO for the cell library is the design's own terminals, not whatever
    # happens to be placed - after a cell placer runs, everything is placed, and CLASS BLOCK would
    # otherwise swallow the whole netlist.
    from placax.netlist.bookshelf import parse_nodes

    macro_names = set(parse_nodes(benchmark_dir / f"{design}.nodes"))

    library = CellLibrary(sizes, collect_pin_offsets(nets), macro_names)
    rows = load_placement_rows(benchmark_dir)

    output_dir.mkdir(parents=True, exist_ok=True)
    lef_path = output_dir / f"{design}.lef"
    def_path = output_dir / f"{design}.def"
    lef_path.write_text(write_lef(library, rows))
    def_path.write_text(
        write_def(design, sizes, library, nets, placement, rows, orientations, fixed)
    )
    return def_path, lef_path


__all__ = [
    "CellLibrary", "DB_UNITS_PER_MICRON", "collect_pin_offsets", "export_bookshelf_as_def",
    "write_def", "write_lef",
]
