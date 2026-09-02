"""The generic cell-placer contract, independent of any specific tool."""
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

    def place_bookshelf(self, aux_path: pathlib.Path, output_dir: pathlib.Path) -> pathlib.Path:
        """Returns the path to a new Bookshelf .pl with standard cells placed, given a macro-placed
        .aux (macros marked /FIXED in its .pl). Not every tool reads Bookshelf natively (unlike
        place()'s DEF/LEF, which every placer here is required to support since that's the only
        format OpenROAD reads too) - default raises, concrete placers override this only if they can.
        Kept optional rather than @abstractmethod so implementing place() alone is always enough to
        satisfy this ABC."""
        raise NotImplementedError(f"{type(self).__name__} doesn't support Bookshelf-native placement")
