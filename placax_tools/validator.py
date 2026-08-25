"""The generic validator contract, independent of any specific tool."""
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
    """Validates a fully-placed design and reports real physical metrics."""

    @abstractmethod
    def validate(
        self, def_path: pathlib.Path, lef_paths: list[pathlib.Path], output_dir: pathlib.Path
    ) -> PPAResult:
        raise NotImplementedError
