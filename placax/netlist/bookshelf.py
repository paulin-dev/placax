"""Reads Bookshelf .nodes/.nets into the same shape the DEF loader produces, keeping only macros."""
import pathlib
import re

from placax.types import Nets, SizeMap

_NODE_RE = re.compile(r"^\s*(\S+)\s+(\d+)\s+(\d+)\s*(terminal)?\s*$")
_NET_HEADER_RE = re.compile(r"NetDegree\s*:\s*(\d+)")
_NET_PIN_RE = re.compile(r"^\s*(\S+)\s+[IO]\s*:\s*(-?[\d.]+)\s+(-?[\d.]+)")
_PL_RE = re.compile(r"^\s*(\S+)\s+(-?\d+)\s+(-?\d+)")
_PL_LINE_RE = re.compile(r"^(\S+)\s+(-?\d+)\s+(-?\d+)\s*:\s*(\S+)(?:\s*/FIXED)?\s*$", re.MULTILINE)


def parse_nodes(nodes_path: pathlib.Path) -> SizeMap:
    """Returns {name: (width, height)} for terminal (macro) nodes only."""
    macros = {}
    for line in nodes_path.read_text().splitlines():
        match = _NODE_RE.match(line)
        if match and match.group(4) == "terminal":
            name, w, h, _ = match.groups()
            macros[name] = (float(w), float(h))
    return macros


def parse_all_node_sizes(nodes_path: pathlib.Path) -> SizeMap:
    """Returns {name: (width, height)} for every node - terminals (macros) and standard cells alike;
    for full-placement rendering only, never for training (which stays macro-only via parse_nodes)."""
    sizes = {}
    for line in nodes_path.read_text().splitlines():
        match = _NODE_RE.match(line)
        if match:
            name, w, h, _ = match.groups()
            sizes[name] = (float(w), float(h))
    return sizes


def parse_nets(nets_path: pathlib.Path, macro_names: set[str]) -> Nets:
    """Returns a list of (macro_name, x_offset, y_offset) pins per net, dropping nets with < 2 macros; keeps only the first offset per (net, macro)."""
    nets: Nets = []
    # macro name -> first-seen (x_offset, y_offset) for the net being read right now; dedupes and stores the offset in one structure.
    current: dict[str, tuple[float, float]] = {}

    def _flush() -> None:
        # Only keep a net if it actually connects two or more distinct macros.
        if len(current) >= 2:
            nets.append([(name, x_off, y_off) for name, (x_off, y_off) in current.items()])

    # Walk the file line by line: a "NetDegree :" line starts a new net, flushing the previous one's pins.
    for line in nets_path.read_text().splitlines():
        if _NET_HEADER_RE.search(line):
            _flush()
            current = {}
            continue
        pin_match = _NET_PIN_RE.match(line)
        if pin_match and pin_match.group(1) in macro_names:  # skip non-macro (standard cell) pins
            name, x_off, y_off = pin_match.groups()
            if name not in current:
                current[name] = (float(x_off), float(y_off))

    _flush()  # the last net in the file has no following header to trigger its flush
    return nets


def parse_pl_positions(pl_path: pathlib.Path) -> dict[str, tuple[float, float, bool]]:
    """Returns {name: (x, y, is_fixed)} for every node in a Bookshelf .pl file (lower-left corner in
    real/database units); is_fixed reflects a trailing '/FIXED' marker on that node's line."""
    positions = {}
    for line in pl_path.read_text().splitlines():
        match = _PL_RE.match(line)
        if not match:
            continue
        name, x_str, y_str = match.groups()
        positions[name] = (float(x_str), float(y_str), "/FIXED" in line)
    return positions


def parse_pl_die_size(pl_path: pathlib.Path, macro_sizes: SizeMap) -> float | None:
    """The real (square) die side length implied by every macro's ORIGINAL position in the .pl file, matching MaskPlace's own read_pl_file: max over all macros of (x + width) and (y + height). None if the file is missing or none of macro_sizes appears in it."""
    if not pl_path.exists():
        return None
    max_extent = 0.0
    found = False
    for name, (x, y, _fixed) in parse_pl_positions(pl_path).items():
        if name not in macro_sizes:
            continue
        found = True
        width, height = macro_sizes[name]
        max_extent = max(max_extent, x + width, y + height)
    return max_extent if found else None


def write_pl(original_pl_text: str, positions: dict[str, tuple[int, int]]) -> str:
    """Rewrites x/y and marks '/FIXED' for every name in positions; every other line (standard cells,
    comments, header) passes through unchanged. positions: {name: (x, y)} lower-left corner, integer
    real/database units - the mechanism DREAMPlace uses to keep RL-placed macros put while it places cells."""
    def replace_line(match: re.Match) -> str:
        name, _x, _y, orient = match.groups()
        if name not in positions:
            return match.group(0)
        new_x, new_y = positions[name]
        return f"{name}\t{int(new_x)}\t{int(new_y)}\t: {orient} /FIXED"

    return _PL_LINE_RE.sub(replace_line, original_pl_text)


def write_aux(nodes: str, nets: str, wts: str, pl: str, scl: str) -> str:
    """Builds a Bookshelf .aux file's RowBasedPlacement line, referencing the given component file names/paths."""
    return f"RowBasedPlacement : {nodes} {nets} {wts} {pl} {scl}\n"


def load_bookshelf(benchmark_dir: pathlib.Path, name: str) -> tuple[SizeMap, Nets]:
    """Returns (node_sizes, nets) for one Bookshelf benchmark's macros."""
    macros = parse_nodes(benchmark_dir / f"{name}.nodes")
    nets = parse_nets(benchmark_dir / f"{name}.nets", set(macros))
    return macros, nets
