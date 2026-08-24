"""The generic cell placer contract - lives here, stable, independent of
any specific tool. See dreamplace/cell_placer.py for the default,
concrete implementation; ready for a second implementation to sit
alongside it without touching this file at all.

CellPlacer is an ABC, not a bare Callable type alias: an earlier
version used Callable[[def_path, lef_paths, output_dir], ...], but the
one real implementation (DREAMPlace) needed extra tool-specific
arguments (dreamplace_root, gpu) that didn't fit that signature at
all - the "generic interface" didn't actually describe its own
implementation. Tool-specific config now lives in each subclass's
__init__, so place() itself stays uniform across any concrete placer -
genuinely swappable at the call site, not just similarly-named
functions."""
import pathlib
from abc import ABC, abstractmethod


class CellPlacer(ABC):
    """The generic contract: place standard cells given a macro-placed
    DEF and tech LEF files, return the path to a fully-placed DEF.
    Any tool-specific setup (install location, GPU/algorithm settings)
    belongs in a concrete subclass's __init__, not in place() itself."""

    @abstractmethod
    def place(
        self, def_path: pathlib.Path, lef_paths: list[pathlib.Path], output_dir: pathlib.Path
    ) -> pathlib.Path:
        """Returns the path to a new DEF with standard cells placed."""
        raise NotImplementedError
