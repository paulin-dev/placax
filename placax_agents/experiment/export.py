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
from dataclasses import dataclass

from placax.log import Log  # must precede jax imports
from placax.netlist import NetlistFormat, detect_format
from placax.netlist.bookshelf import write_aux, write_pl
from placax.netlist.def_writer import write_placed_def
from placax_agents.ops.inference import positions_to_named_lower_left

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
    scaled = {name: (int(round(x * db_units)), int(round(y * db_units)))
              for name, (x, y) in named.items()}

    def_path = output_dir / source.name
    def_path.write_text(write_placed_def(def_text, scaled))
    return ExportedPlacement(NetlistFormat.DEF, def_path, built.config.full_hash(), len(named))


def write_placement(built, positions, output_dir: pathlib.Path) -> ExportedPlacement:
    """Writes `positions` back into the design's own format, and says where it landed.

    `positions` are the grid cells an agent handed the runner - the same array `score()` measures,
    so the exported design is the placement that was reported, not a second rollout of it.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    benchmark = built.benchmark
    named = positions_to_named_lower_left(
        positions, benchmark.sizes_array, benchmark.cell_size, benchmark.name_to_idx
    )

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
    Log.info(f"  wrote {exported.n_macros} macro positions to {exported.path} "
             f"(run {exported.full_hash})")
    return exported


__all__ = ["ExportedPlacement", "write_placement"]
