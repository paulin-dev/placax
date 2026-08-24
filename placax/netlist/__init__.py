"""Detects a benchmark's format, then routes to the right loader. Bookshelf,
DEF, and protobuf all resolve to the identical (macro_sizes, nets) shape -
real research repos bridge multiple formats routinely, this isn't unusual
overhead - so nothing downstream needs to know which format a benchmark
came from."""
import enum
import pathlib

from placax.netlist.bookshelf import load_bookshelf
from placax.netlist.def_reader import load_def
from placax.netlist.protobuf_reader import load_protobuf
from placax.types import Nets, SizeMap


class NetlistFormat(enum.Enum):
    BOOKSHELF = "bookshelf"
    DEF = "def"
    PROTOBUF = "protobuf"
    VERILOG_RTL = "verilog_rtl"
    UNKNOWN = "unknown"


def detect_format(benchmark_dir: pathlib.Path) -> NetlistFormat:
    """Bookshelf has an .aux manifest; DEF, protobuf, and unconverted RTL
    are identified by extension. Order matters: check .aux before .def/
    .pb.txt/.v, since a real Bookshelf dir could plausibly have unrelated
    stray files sitting near it."""
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
    """Returns (macro_sizes, nets), identical shape regardless of format."""
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
