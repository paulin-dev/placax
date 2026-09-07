"""place_and_validate: macros in, standard cells placed, real PPA out - with neither tool named.

The design document has presented this composition as working, tested code since v1, and it did
not exist: the `CellPlacer` and `Validator` interfaces were both defined and one implementation of
each was written, but nothing ever called them together, and no pipeline in the repository ever
called the validator at all. So the "true PPA" box in the architecture had no code behind it, and
every number this project reported was a geometric HPWL proxy.

This is that composition, and it is deliberately three lines of real work: the value is entirely
in what it does NOT know. It never mentions DREAMPlace or OpenROAD, so substituting RePlAce,
AutoDMP, Innovus or a different signoff tool is a change at the call site and nowhere else.
"""
import pathlib
from dataclasses import dataclass

from placax_tools.cell_placer import CellPlacer
from placax_tools.validator import PPAResult, Validator


@dataclass(frozen=True)
class PlacedDesign:
    """The output of a full flow: where the placed design landed, and what it measures."""

    def_path: pathlib.Path
    """The DEF with macros where the agent put them and standard cells placed around them."""

    ppa: PPAResult
    """Real physical metrics from the validator - area and utilization always, timing only if
    the validator was given a liberty file and a clock period. Never fabricated: a metric that
    wasn't computed comes back as None rather than a plausible-looking number."""


def place_and_validate(
    def_path: pathlib.Path,
    lef_paths: list[pathlib.Path],
    output_dir: pathlib.Path,
    cell_placer: CellPlacer,
    validator: Validator,
) -> PlacedDesign:
    """Places standard cells around already-placed macros, then measures the result.

    `def_path` must already have every macro placed - that is the RL agent's output. Both tools
    are arguments rather than imports, per the project's one rule: if a piece of logic could
    plausibly be done differently by a different team, it is a parameter, not a hard-coded call.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    placed_def = cell_placer.place(def_path, lef_paths, output_dir)
    ppa = validator.validate(placed_def, lef_paths, output_dir)
    return PlacedDesign(def_path=placed_def, ppa=ppa)


def validate_only(
    def_path: pathlib.Path,
    lef_paths: list[pathlib.Path],
    output_dir: pathlib.Path,
    validator: Validator,
) -> PPAResult:
    """Measures an already-complete placement, for when cell placement happened elsewhere.

    The Bookshelf flow is the reason this exists separately: DREAMPlace reads and writes Bookshelf
    natively there, so cell placement has already happened by the time there is anything to
    validate, and forcing it back through place() would run the placer twice.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    return validator.validate(def_path, lef_paths, output_dir)
