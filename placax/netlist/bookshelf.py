"""Reads Bookshelf .nodes/.nets into the same shape the DEF loader produces, keeping only macros."""
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
    """Returns a list of (macro_name, x_offset, y_offset) pins per net, dropping nets with < 2 macros."""
    nets: Nets = []
    current: list[NetPin] = []  # pins collected for the net being read right now

    def _flush() -> None:
        # Only keep a net if it actually connects two or more distinct macros.
        if len({name for name, _x, _y in current}) >= 2:
            nets.append(current)

    # Walk the file line by line: a "NetDegree :" line starts a new net, so we
    # flush the previous one's pins and start collecting again.
    for line in nets_path.read_text().splitlines():
        if _NET_HEADER_RE.search(line):
            _flush()
            current = []
            continue
        pin_match = _NET_PIN_RE.match(line)
        if pin_match and pin_match.group(1) in macro_names:  # skip non-macro (standard cell) pins
            name, x_off, y_off = pin_match.groups()
            current.append((name, float(x_off), float(y_off)))

    _flush()  # the last net in the file has no following header to trigger its flush
    return nets


def load_bookshelf(benchmark_dir: pathlib.Path, name: str) -> tuple[SizeMap, Nets]:
    """Returns (node_sizes, nets) for one Bookshelf benchmark's macros."""
    macros = parse_nodes(benchmark_dir / f"{name}.nodes")
    nets = parse_nets(benchmark_dir / f"{name}.nets", set(macros))
    return macros, nets
