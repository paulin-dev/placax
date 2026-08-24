"""Reads Bookshelf .nodes/.nets, converging on the same shape def_reader
produces. Only terminal nodes (macros) are kept - the thousands of
standard cells are out of scope, same filter MaskPlace itself uses.

Pin offsets are kept, not discarded: two placements that differ only in
where a pin actually sits within its macro can have genuinely different
real wirelength but identical HPWL if every pin is snapped to its macro's
center - a real blind spot MaskPlace's own paper documents directly."""
import pathlib
import re

from placax.types import NetPin, Nets, SizeMap

_NODE_RE = re.compile(r"^\s*(\S+)\s+(\d+)\s+(\d+)\s*(terminal)?\s*$")
_NET_HEADER_RE = re.compile(r"NetDegree\s*:\s*(\d+)")
_NET_PIN_RE = re.compile(r"^\s*(\S+)\s+[IO]\s*:\s*(-?[\d.]+)\s+(-?[\d.]+)")


def parse_nodes(nodes_path: pathlib.Path) -> SizeMap:
    """Returns {name: (width, height)} for terminal (macro) nodes only."""
    macros = {}
    for line in nodes_path.read_text().splitlines():
        match = _NODE_RE.match(line)
        if match and match.group(4) == "terminal":
            name, w, h, _ = match.groups()
            macros[name] = (float(w), float(h))
    return macros


def parse_nets(nets_path: pathlib.Path, macro_names: set[str]) -> Nets:
    """Returns a list of pins per net, each pin (macro_name, x_offset,
    y_offset). Every pin is kept, even multiple pins on the same macro -
    each is a real, separate connection point. A net is dropped only if
    fewer than 2 *distinct* macros appear in it (an intra-macro-only net
    has no wirelength that any placement decision can change)."""
    nets = []
    current: list[NetPin] = []

    def _flush() -> None:
        if len({name for name, _x, _y in current}) >= 2:
            nets.append(current)

    for line in nets_path.read_text().splitlines():
        if _NET_HEADER_RE.search(line):
            _flush()
            current = []
            continue
        pin_match = _NET_PIN_RE.match(line)
        if pin_match and pin_match.group(1) in macro_names:
            name, x_off, y_off = pin_match.groups()
            current.append((name, float(x_off), float(y_off)))

    _flush()
    return nets


def load_bookshelf(benchmark_dir: pathlib.Path, name: str) -> tuple[SizeMap, Nets]:
    """Returns (node_sizes, nets) for one Bookshelf benchmark's macros."""
    macros = parse_nodes(benchmark_dir / f"{name}.nodes")
    nets = parse_nets(benchmark_dir / f"{name}.nets", set(macros))
    return macros, nets
