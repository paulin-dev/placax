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


class Validator(ABC):
    """Validates a fully-placed design and reports real physical metrics."""

    @abstractmethod
    def validate(
        self, def_path: pathlib.Path, lef_paths: list[pathlib.Path], output_dir: pathlib.Path
    ) -> PPAResult:
        raise NotImplementedError
