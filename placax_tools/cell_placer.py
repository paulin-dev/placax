"""The generic cell-placer contract, independent of any specific tool.
An ABC rather than a Callable alias: tool-specific config (install root,
GPU) lives in each concrete subclass's __init__, so place() stays
uniform across any placer."""
import pathlib
from abc import ABC, abstractmethod


class CellPlacer(ABC):
    """Places standard cells given a macro-placed DEF and tech LEF files."""

    @abstractmethod
    def place(
        self, def_path: pathlib.Path, lef_paths: list[pathlib.Path], output_dir: pathlib.Path
    ) -> pathlib.Path:
        """Returns the path to a new DEF with standard cells placed."""
        raise NotImplementedError
