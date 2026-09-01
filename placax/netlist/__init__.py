"""Detects a benchmark's format, then routes to the right loader - all resolve to the same shape."""
import enum
import pathlib

from placax.netlist.bookshelf import load_bookshelf, parse_pl_die_size
from placax.netlist.def_reader import load_def, parse_die_size as parse_def_die_size
from placax.netlist.protobuf_reader import load_protobuf
from placax.types import Nets, SizeMap


class NetlistFormat(enum.Enum):
    BOOKSHELF = "bookshelf"
    DEF = "def"
    PROTOBUF = "protobuf"
    VERILOG_RTL = "verilog_rtl"
    UNKNOWN = "unknown"


def detect_format(benchmark_dir: pathlib.Path) -> NetlistFormat:
    """Identifies the format by extension (checked in priority order)."""
    if any(benchmark_dir.glob("*.aux")):
        return NetlistFormat.BOOKSHELF
    if any(benchmark_dir.glob("*.def")):
        return NetlistFormat.DEF
    if any(benchmark_dir.glob("*.pb.txt")):
        return NetlistFormat.PROTOBUF
    if any(benchmark_dir.glob("*.v")) or any(benchmark_dir.glob("*.sv")):
        return NetlistFormat.VERILOG_RTL
    return NetlistFormat.UNKNOWN


def load_netlist(
    benchmark_dir: pathlib.Path, lef_paths: list[pathlib.Path] | None = None
) -> tuple[SizeMap, Nets]:
    """Returns (macro_sizes, nets), the identical shape regardless of the detected source format."""
    # Detect the format first, then dispatch to the loader that understands it;
    # DEF additionally needs its companion .lef files for macro geometry.
    match detect_format(benchmark_dir):
        case NetlistFormat.DEF:
            def_path = next(benchmark_dir.glob("*.def"))
            lef_paths = lef_paths or list(benchmark_dir.glob("*.lef"))
            if not lef_paths:
                raise ValueError(f"no .lef files found or provided for {def_path}")
            return load_def(def_path, lef_paths)
        case NetlistFormat.BOOKSHELF:
            aux_path = next(benchmark_dir.glob("*.aux"))
            return load_bookshelf(benchmark_dir, aux_path.stem)
        case NetlistFormat.PROTOBUF:
            return load_protobuf(next(benchmark_dir.glob("*.pb.txt")))
        case NetlistFormat.VERILOG_RTL:
            # Unsynthesized RTL has no macro geometry yet, so we can't load it directly.
            raise NotImplementedError(
                f"{benchmark_dir} contains unconverted Verilog RTL. Synthesize it first "
                "(yosys synth against a target cell library, e.g. NanGate45) to produce "
                "a DEF file, which loads natively."
            )
        case NetlistFormat.UNKNOWN:
            raise ValueError(
                f"no recognizable netlist format (.aux, .def, .pb.txt, .v, .sv) "
                f"found in {benchmark_dir}"
            )


def load_die_size(benchmark_dir: pathlib.Path, macro_sizes: SizeMap) -> float | None:
    """The real (square) die side length for benchmark_dir, from whichever source format carries physical
    dimensions: Bookshelf's .pl file (MaskPlace's own die-sizing source) or DEF's own DIEAREA statement.
    None for protobuf (upstream itself has no general derivation there - see parse_macros's docstring
    discussion) or a Bookshelf dir missing its .pl file, so callers can fall back to an area-based
    heuristic instead of inventing a physically-meaningless canvas size."""
    match detect_format(benchmark_dir):
        case NetlistFormat.BOOKSHELF:
            aux_path = next(benchmark_dir.glob("*.aux"))
            return parse_pl_die_size(benchmark_dir / f"{aux_path.stem}.pl", macro_sizes)
        case NetlistFormat.DEF:
            def_path = next(benchmark_dir.glob("*.def"))
            return parse_def_die_size(def_path.read_text())
        case _:
            return None
