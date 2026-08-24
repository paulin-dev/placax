"""The generic validator contract - lives here, stable, independent of
any specific tool. See openroad/validator.py for the default, concrete
implementation; ready for a second implementation to sit alongside it
without touching this file at all.

Validator is an ABC, not a bare Callable type alias: an earlier version
used Callable[..., PPAResult] - the "..." means "any arguments at
all", which isn't a real contract, just giving up on having one. Tool-
specific config (liberty_path, clock_period_ns) belongs in each
subclass's __init__, so validate() itself stays uniform across any
validator - genuinely swappable at the call site.

PPAResult stays here too, not in openroad/: it's the shared return
type of the generic contract, not OpenROAD-specific."""
import pathlib
from abc import ABC, abstractmethod

from flax import struct


@struct.dataclass
class PPAResult:
    design_area: float | None
    utilization_pct: float | None
    timing_slack: float | None  # None if no liberty/clock given - not computed, not faked
    raw_output: str


class Validator(ABC):
    """The generic contract: validate a fully-placed design, report real
    physical metrics. Any tool-specific setup (liberty files, clock
    constraints, install location) belongs in a concrete subclass's
    __init__, not in validate() itself."""

    @abstractmethod
    def validate(
        self, def_path: pathlib.Path, lef_paths: list[pathlib.Path], output_dir: pathlib.Path
    ) -> PPAResult:
        raise NotImplementedError
