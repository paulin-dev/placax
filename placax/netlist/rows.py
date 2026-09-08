"""The design's placement rows: where a macro may legally sit, in real units.

Nothing in this project read a row until now. `.scl` was carried around as a filename and
symlinked next to an exported placement; DEF's `ROW` statements were never parsed. The canvas was
sized from the DIE extent (max over macros of x+w, y+h - MaskPlace's own derivation) and anchored
at the origin, which produces two measurable errors on a real design:

  adaptec1: core area x 459..11151, y 459..11139;  canvas 0..11589 at either shipped grid
    * **15% of the canvas is off-core margin.** 17 of 224 grid columns and 17 of 224 rows fall
      outside the row region entirely, and the action space can choose them freely. A macro
      placed there is not "slightly misaligned", it is outside the placeable area.
    * **0 of 224 grid rows land on a real row.** Row pitch is 12; cell_size is 51.737, so no grid
      row is a multiple of the pitch from y=459. Worst-case snap is pitch/2 = 6 units, which is
      0.116 of a grid cell - so this half is a rounding footnote, not a wirelength effect. Both
      numbers are worth stating precisely, because they point at different fixes: the first is a
      canvas definition (see `BenchmarkSpec.canvas`), the second is a legalizer.

Row geometry is uniform in every benchmark this project handles - one pitch, one site width, one
core rectangle - so that is what this models. A design with mixed row heights or fragmented
subrows would need more, and `from_bookshelf` says so rather than quietly averaging.
"""
import pathlib
import re
from dataclasses import dataclass

_COORDINATE_RE = re.compile(r"Coordinate\s*:\s*(-?[\d.]+)")
_HEIGHT_RE = re.compile(r"Height\s*:\s*([\d.]+)")
_SITE_WIDTH_RE = re.compile(r"Sitewidth\s*:\s*([\d.]+)")
_SUBROW_RE = re.compile(r"SubrowOrigin\s*:\s*(-?[\d.]+)\s+NumSites\s*:\s*(\d+)")


@dataclass(frozen=True)
class PlacementRows:
    """The core area and its row/site grid, in the same real units as macro sizes."""

    x0: float
    y0: float
    x1: float
    y1: float
    row_pitch: float
    """Row-to-row spacing in y. A macro's y is legal only at y0 + k * row_pitch."""

    site_width: float
    """Site-to-site spacing in x. A macro's x is legal only at x0 + k * site_width."""

    n_rows: int

    @property
    def width(self) -> float:
        return self.x1 - self.x0

    @property
    def height(self) -> float:
        return self.y1 - self.y0

    @property
    def span(self) -> float:
        """The square canvas side that covers the core - what a grid should be scaled against."""
        return max(self.width, self.height)

    def snap(self, x: float, y: float, macro_width: float = 0.0,
             macro_height: float = 0.0) -> tuple[float, float]:
        """The nearest legal (site, row) origin for a macro, clamped to keep it inside the core.

        Clamping happens after snapping and is itself re-snapped, so a macro pushed in from the
        edge still lands on a row rather than flush against the boundary between two.
        """
        def snap_axis(value, origin, pitch, limit, extent):
            highest = origin + max(0.0, limit - extent - origin)
            snapped = origin + round((value - origin) / pitch) * pitch
            clamped = min(max(snapped, origin), highest)
            # Re-snap after clamping, then step back one pitch if that pushed past the limit.
            resnapped = origin + round((clamped - origin) / pitch) * pitch
            if resnapped + extent > limit and resnapped - pitch >= origin:
                resnapped -= pitch
            return resnapped

        return (
            snap_axis(x, self.x0, self.site_width, self.x1, macro_width),
            snap_axis(y, self.y0, self.row_pitch, self.y1, macro_height),
        )

    def is_legal(self, x: float, y: float, macro_width: float = 0.0,
                 macro_height: float = 0.0, tolerance: float = 1e-6) -> bool:
        """Whether a macro at (x, y) sits on a row, on a site, and wholly inside the core."""
        if x < self.x0 - tolerance or y < self.y0 - tolerance:
            return False
        if x + macro_width > self.x1 + tolerance or y + macro_height > self.y1 + tolerance:
            return False
        return (
            _on_grid(x - self.x0, self.site_width, tolerance)
            and _on_grid(y - self.y0, self.row_pitch, tolerance)
        )

    def to_dict(self) -> dict:
        return {"x0": self.x0, "y0": self.y0, "x1": self.x1, "y1": self.y1,
                "row_pitch": self.row_pitch, "site_width": self.site_width,
                "n_rows": self.n_rows}


def _on_grid(offset: float, pitch: float, tolerance: float) -> bool:
    if pitch <= 0:
        return True
    remainder = abs(offset) % pitch
    return remainder <= tolerance or abs(remainder - pitch) <= tolerance


def from_bookshelf(scl_path: pathlib.Path) -> PlacementRows | None:
    """Parses a Bookshelf `.scl`. None if the file is absent or carries no CoreRow."""
    if not scl_path.exists():
        return None
    text = scl_path.read_text()
    coordinates = [float(value) for value in _COORDINATE_RE.findall(text)]
    heights = [float(value) for value in _HEIGHT_RE.findall(text)]
    site_widths = [float(value) for value in _SITE_WIDTH_RE.findall(text)]
    subrows = [(float(origin), int(sites)) for origin, sites in _SUBROW_RE.findall(text)]
    if not coordinates or not heights or not subrows:
        return None

    # One pitch and one site width, per this module's docstring. Taking the first and checking the
    # rest keeps a mixed-height design from being silently averaged into a wrong answer.
    pitch, site_width = heights[0], (site_widths[0] if site_widths else 1.0)
    if any(abs(height - pitch) > 1e-6 for height in heights):
        raise NotImplementedError(
            f"{scl_path} mixes row heights {sorted(set(heights))}; PlacementRows models one "
            f"uniform row pitch. Model the rows individually before using this."
        )

    x0 = min(origin for origin, _sites in subrows)
    x1 = max(origin + sites * site_width for origin, sites in subrows)
    return PlacementRows(
        x0=x0, y0=min(coordinates), x1=x1, y1=max(coordinates) + pitch,
        row_pitch=pitch, site_width=site_width, n_rows=len(coordinates),
    )


_DEF_ROW_RE = re.compile(
    r"ROW\s+\S+\s+\S+\s+(-?\d+)\s+(-?\d+)\s+\w+\s+DO\s+(\d+)\s+BY\s+(\d+)\s+STEP\s+(\d+)\s+(\d+)"
)
_DEF_UNITS_RE = re.compile(r"UNITS\s+DISTANCE\s+MICRONS\s+(\d+)")


def from_def(def_path: pathlib.Path) -> PlacementRows | None:
    """Parses DEF `ROW` statements into the same shape, converted to microns.

    Macro geometry reaches this project from LEF, which is in microns, while DEF coordinates are
    in database units - so rows are converted here for the same reason `experiment.export`
    converts the other way when writing one back out.
    """
    if not def_path.exists():
        return None
    text = def_path.read_text()
    rows = _DEF_ROW_RE.findall(text)
    if not rows:
        return None
    units_match = _DEF_UNITS_RE.search(text)
    units = float(units_match.group(1)) if units_match else 1.0

    x0 = y0 = float("inf")
    x1 = y1 = float("-inf")
    pitches, site_widths, n_rows = set(), set(), 0
    for x, y, num_x, num_y, step_x, step_y in rows:
        x, y = float(x) / units, float(y) / units
        num_x, num_y = int(num_x), int(num_y)
        step_x, step_y = float(step_x) / units, float(step_y) / units
        x0, y0 = min(x0, x), min(y0, y)
        x1 = max(x1, x + max(num_x - 1, 0) * step_x)
        y1 = max(y1, y + max(num_y - 1, 0) * step_y)
        if step_x:
            site_widths.add(round(step_x, 9))
        if step_y:
            pitches.add(round(step_y, 9))
        n_rows += max(num_y, 1)

    # A single-row-per-statement DEF (the common shape) carries its pitch between statements
    # rather than inside one, so recover it from the distinct row y coordinates.
    if not pitches:
        ys = sorted({float(y) / units for _x, y, *_rest in rows})
        pitches = {round(ys[1] - ys[0], 9)} if len(ys) > 1 else {1.0}
    pitch = min(pitches)
    site_width = min(site_widths) if site_widths else 1.0
    return PlacementRows(x0=x0, y0=y0, x1=x1, y1=y1 + pitch, row_pitch=pitch,
                         site_width=site_width, n_rows=n_rows)


def load_placement_rows(benchmark_dir: pathlib.Path) -> PlacementRows | None:
    """The placement rows for a benchmark directory, or None if its format carries none.

    None is a real answer, not a failure: a protobuf netlist has no row geometry, and a caller is
    expected to fall back to the die extent rather than invent a core area.
    """
    from placax.netlist import NetlistFormat, detect_format

    match detect_format(benchmark_dir):
        case NetlistFormat.BOOKSHELF:
            aux_path = next(benchmark_dir.glob("*.aux"))
            return from_bookshelf(benchmark_dir / f"{aux_path.stem}.scl")
        case NetlistFormat.DEF:
            return from_def(next(benchmark_dir.glob("*.def")))
        case _:
            return None


__all__ = ["PlacementRows", "from_bookshelf", "from_def", "load_placement_rows"]
