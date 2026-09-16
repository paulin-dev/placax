"""The generic validator contract, independent of any specific tool."""
import pathlib
from abc import ABC, abstractmethod

from flax import struct


@struct.dataclass
class PPAResult:
    """What a validator measured. Every field is None unless it was actually computed.

    The list is deliberately longer than area/utilization/slack. A macro placement is judged in
    this literature by half-perimeter wirelength, and ChiPBench's result is that proxy rankings do
    not survive a real flow - routed wirelength and DRC are where a good-HPWL placement is found
    to be unroutable. A validator that cannot report them leaves this project open to the exact
    criticism it exists to make.
    """

    design_area: float | None
    utilization_pct: float | None
    timing_slack: float | None  # None if no liberty/clock given - not computed, not faked
    raw_output: str
    routed_wirelength: float | None = None
    """Total wirelength after routing, in the tool's own units. The number HPWL is a proxy FOR,
    and the one that decides whether the proxy ranked two placements correctly."""

    via_count: int | None = None
    drc_violations: int | None = None
    """Design-rule violations after detailed routing. Zero is the only acceptable value for a
    real result; a placement that routes with violations has not been shown to work."""

    hpwl: float | None = None
    """Half-perimeter wirelength of the WHOLE design, measured by the tool, in microns.

    The number every `real_hpwl` in this project is a proxy for, taken by something other than
    this project: every signal net, standard cells included, from the pins' real positions. It
    needs no technology and no routing, so it is the one wirelength a converted Bookshelf design
    can report meaningfully."""

    placement_legal: bool | None = None
    """The tool's own verdict on the placement - overlap, off-row, off-site, out of the core.

    This project measures legality itself on a grid; this is the same question asked of the real
    design by a real placer's checker, which is the one that decides whether it can be built."""

    placement_violations: tuple[tuple[str, int], ...] = ()
    """Which of the checker's rules failed, and how often - `(("One site gap", 5728), ...)`.

    Needed to read `placement_legal=False` at all. The checker applies the tool's full rule set,
    including rules a benchmark may never have had: adaptec1 (ISPD 2005) knows nothing of
    OpenROAD's one-site-gap rule, so a DREAMPlace result that is legal by the benchmark's own
    rules still fails it. The verdict stays the tool's; this says what it was about."""

    total_negative_slack: float | None = None
    """Sum of negative slack over all endpoints, in the liberty's time unit. None when timing did
    not run; 0.0 when it ran and nothing failed."""

    tool_version: str | None = None
    """What reported itself as having produced these numbers. A validator's results change between
    releases, so a number without this is not reproducible."""

    notes: tuple[str, ...] = ()
    """Steps that were asked for and could not run, with the tool's own reason - a route that
    failed on a technology with no vias, a clock port that does not exist. Recorded rather than
    swallowed: a None beside its reason is a result, a None on its own is a mystery."""


class Validator(ABC):
    """Validates a fully-placed design and reports real physical metrics."""

    @abstractmethod
    def validate(
        self, def_path: pathlib.Path, lef_paths: list[pathlib.Path], output_dir: pathlib.Path
    ) -> PPAResult:
        raise NotImplementedError
