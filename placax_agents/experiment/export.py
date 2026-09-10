"""Writing a run's macro placement back out into the design's own format.

This is the step the architecture always assumed and never had. `placax/netlist/def_writer.py`
has shipped `write_placed_def` since before the experiment layer existed, and its only caller in
the entire tree was its own unit test - so there was no way to get from an agent's placement to a
file any external tool could read. `scripts/validate_design.py` therefore asked for a
`--def_path` "with every macro already placed", which nothing in this repository produced: the
PPA box at the bottom of the architecture was not merely unverified, it was **disconnected**.

That is what this closes. One function turns `(built experiment, positions)` into the file a cell
placer or a validator actually opens, dispatching on the design's own format:

  Bookshelf  a new `.pl` with every placed macro at its RL position and marked `/FIXED`, plus a
             `.aux` pointing at it and symlinks to the untouched `.nodes`/`.nets`/`.wts`/`.scl`.
             This is what DREAMPlace reads natively, and what `scripts/run_pipeline.py` used to
             assemble inline - it lives here now so the pipeline and the physical evaluation
             cannot drift into two different notions of "the placement we measured".
  DEF        the original DEF with every placed macro's `PLACED` coordinate rewritten, in the
             design's own database units. This is what OpenROAD reads.

Macros the run did not place - anything outside a macro budget, and every standard cell - pass
through untouched, because both writers rewrite only the instances they are handed.
"""
import pathlib
import re
from dataclasses import dataclass, replace

from placax.log import Log  # must precede jax imports
from placax.netlist import NetlistFormat, detect_format
from placax.netlist.bookshelf import write_aux, write_pl
from placax.netlist.def_writer import write_placed_def
from placax.extras import orientation as orientation_module
from placax_agents.policy.scale import to_real_lower_left

import numpy as np

_DEF_UNITS_RE = re.compile(r"UNITS\s+DISTANCE\s+MICRONS\s+(\d+)")

BOOKSHELF_SIBLINGS = ("nodes", "nets", "wts", "scl")
"""The Bookshelf files a placement does not change, symlinked beside the new .pl rather than
copied - .nets alone runs to hundreds of megabytes on the larger ISPD designs."""


@dataclass(frozen=True)
class ExportedPlacement:
    """Where a run's placement was written, and what produced it."""

    format: NetlistFormat
    path: pathlib.Path
    """The file a downstream tool opens: the .aux for Bookshelf, the .def for DEF."""

    full_hash: str
    """The run this placement came from, so the exported design is attributable on its own."""

    n_macros: int
    """How many macro positions were written - fewer than the design's total under a budget."""

    legalizer: str | None = None
    """Which legalizer ran, or None if the config named none."""

    max_displacement: float = 0.0
    mean_displacement: float = 0.0
    """How far legalization moved macros, in real units. Reported rather than assumed: this is
    the cost of making a placement realizable, and a run should be able to say whether that cost
    was a rounding error or a redesign. On adaptec1 a row snap moves a macro at most 6 units -
    0.116 of a grid cell - which is worth having as a number instead of a belief."""

    off_rows_before: int = 0
    off_rows_after: int = 0
    """Macros not sitting on a legal site/row, before and after legalization. `after` should be
    zero; anything else means the legalizer could not place them and the design is not realizable.
    Both are 0 when the design carries no rows to check against."""


def _export_bookshelf(built, named, output_dir: pathlib.Path) -> ExportedPlacement:
    """A new .pl/.aux pair with the run's macros FIXED, beside symlinks to the untouched rest."""
    benchmark_dir = built.config.environment.benchmark.path.resolve()
    design = next(benchmark_dir.glob("*.aux")).stem

    pl_path = output_dir / f"{design}.pl"
    pl_path.write_text(write_pl((benchmark_dir / f"{design}.pl").read_text(), named))

    # Limbo's Bookshelf .aux grammar requires filenames to start with a letter, so an absolute
    # path fails to parse. Symlinking the unchanged files next to the new .pl under their bare
    # names - every real Bookshelf benchmark's own convention - sidesteps that rather than
    # relying on a lexer quirk, and avoids copying the multi-hundred-megabyte .nets.
    for suffix in BOOKSHELF_SIBLINGS:
        link = output_dir / f"{design}.{suffix}"
        if not link.exists():
            link.symlink_to((benchmark_dir / f"{design}.{suffix}").resolve())

    aux_path = output_dir / f"{design}.aux"
    aux_path.write_text(write_aux(
        f"{design}.nodes", f"{design}.nets", f"{design}.wts", pl_path.name, f"{design}.scl",
    ))
    return ExportedPlacement(NetlistFormat.BOOKSHELF, aux_path, built.config.full_hash(), len(named))


def _export_def(built, named, output_dir: pathlib.Path) -> ExportedPlacement:
    """The original DEF with every placed macro's PLACED coordinate rewritten."""
    benchmark_dir = built.config.environment.benchmark.path.resolve()
    source = next(benchmark_dir.glob("*.def"))
    def_text = source.read_text()

    # Macro sizes come from LEF in microns, so a placement is in microns too - but DEF coordinates
    # are in database units. Converting here rather than at the writer keeps `write_placed_def`
    # what it is: a text rewriter that takes the numbers it is given.
    units_match = _DEF_UNITS_RE.search(def_text)
    db_units = int(units_match.group(1)) if units_match else 1
    scaled = {name: (int(round(placement[0] * db_units)), int(round(placement[1] * db_units)),
                     *placement[2:])
              for name, placement in named.items()}

    def_path = output_dir / source.name
    def_path.write_text(write_placed_def(def_text, scaled))
    return ExportedPlacement(NetlistFormat.DEF, def_path, built.config.full_hash(), len(named))


def _real_placement(built, positions) -> dict[str, tuple[float, float]]:
    """{macro: (x, y)} real-unit lower-left corners, with the canvas origin applied.

    The origin is added here and nowhere else. HPWL is translation-invariant, so shifting every
    macro by a constant cannot change a reward, a wiremask or a grid legality check - which is why
    the environment never carries it. It matters only once a coordinate has to mean something to a
    tool other than this one, and this is that boundary.
    """
    benchmark = built.benchmark
    lower_left = np.asarray(to_real_lower_left(positions, benchmark.cell_size), dtype=float)
    origin_x, origin_y = benchmark.origin
    return {
        name: (float(lower_left[idx, 0]) + origin_x, float(lower_left[idx, 1]) + origin_y)
        for name, idx in benchmark.name_to_idx.items()
    }


def _off_rows(placement: dict, macro_sizes: dict, rows) -> int:
    """How many macros are not on a legal site and row, wholly inside the core."""
    if rows is None:
        return 0
    return sum(
        not rows.is_legal(x, y, *macro_sizes.get(name, (0.0, 0.0)))
        for name, (x, y) in placement.items()
    )


def _oriented_macro_sizes(built, orientations) -> dict:
    """{name: (w, h)} AS PLACED, so the legalizer and the row check see the real footprint.

    A macro on its side is its height by its width; snapping it against the core with its
    unrotated size would push it to the wrong place and call an illegal placement legal.
    """
    import numpy as np

    sizes = np.asarray(
        orientation_module.effective_sizes(built.benchmark.sizes_array, orientations)
    )
    return {
        name: (float(sizes[idx, 0]), float(sizes[idx, 1]))
        for name, idx in built.benchmark.name_to_idx.items()
    }


def write_placement(built, positions, output_dir: pathlib.Path,
                    orientations=None) -> ExportedPlacement:
    """Writes `positions` back into the design's own format, and says where it landed.

    `positions` are the grid cells an agent handed the runner - the same array `score()` measures,
    so the exported design is the placement that was reported, not a second rollout of it.

    The run's configured legalizer, if it has one, is applied here: this is the boundary between a
    placement that is legal on the GRID (guaranteed by masking) and one that is legal on the DIE
    (rows, sites, core area), and how far the two differ is reported rather than assumed.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    benchmark = built.benchmark
    placement = _real_placement(built, positions)
    # Footprints as placed: with no orientation this is `benchmark.macro_sizes` unchanged.
    macro_sizes = (
        benchmark.macro_sizes if orientations is None
        else _oriented_macro_sizes(built, orientations)
    )
    rows = benchmark.rows

    off_before = _off_rows(placement, macro_sizes, rows)
    legalizer = built.config.environment.legalization
    max_move = mean_move = 0.0
    if built.legalize_fn is not None:
        legalized = built.legalize_fn(placement, macro_sizes)
        moves = [
            float(np.hypot(legalized[name][0] - x, legalized[name][1] - y))
            for name, (x, y) in placement.items()
        ]
        max_move = max(moves, default=0.0)
        mean_move = float(np.mean(moves)) if moves else 0.0
        placement = legalized
    off_after = _off_rows(placement, macro_sizes, rows)

    # Rounded last: both Bookshelf .pl and DEF PLACED take integers, and rounding before
    # legalizing would snap to a row and then step off it again. The orientation letter rides
    # along so a turned macro reaches the file turned, rather than losing the axis on the way out.
    letters = (
        orientation_module.names(orientations, benchmark.params.n_macros)
        if orientations is not None else None
    )
    named = {
        name: ((int(round(x)), int(round(y))) if letters is None
               else (int(round(x)), int(round(y)), letters[benchmark.name_to_idx[name]]))
        for name, (x, y) in placement.items()
    }

    benchmark_dir = built.config.environment.benchmark.path
    design_format = detect_format(benchmark_dir)
    if design_format is NetlistFormat.BOOKSHELF:
        exported = _export_bookshelf(built, named, output_dir)
    elif design_format is NetlistFormat.DEF:
        exported = _export_def(built, named, output_dir)
    else:
        raise NotImplementedError(
            f"{benchmark_dir} is a {design_format.value} design, and only Bookshelf and DEF can "
            f"be written back out - those are the two formats the cell placers and validators in "
            f"placax_tools actually read. A protobuf netlist carries no placement file to rewrite."
        )
    exported = replace(
        exported, legalizer=legalizer.name if legalizer else None,
        max_displacement=max_move, mean_displacement=mean_move,
        off_rows_before=off_before, off_rows_after=off_after,
    )
    Log.info(f"  wrote {exported.n_macros} macro positions to {exported.path} "
             f"(run {exported.full_hash})")
    if rows is not None:
        Log.info(f"  off-row macros: {off_before} before legalization, {off_after} after"
                 + (f"; moved at most {max_move:.1f} units" if built.legalize_fn else
                    " (no legalizer configured)"))
    return exported


__all__ = ["ExportedPlacement", "write_placement"]
