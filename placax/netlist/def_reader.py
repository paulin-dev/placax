"""Reads DEF COMPONENTS/NETS + LEF geometry directly into the same shape
the Bookshelf loader produces. Pin offsets come from a second LEF lookup
(cell_type, port_name) -> offset, since DEF's NETS section only gives pin
names, not geometry - that lives in LEF's per-cell PIN blocks."""
import pathlib
import re

from placax.netlist.lef import parse_lef_pin_offsets, parse_lef_sizes
from placax.types import NetPin, Nets, PinOffsets, SizeMap

_COMPONENT_RE = re.compile(r"-\s+(\S+)\s+(\S+)\s+\+\s+PLACED\s+\(\s*(-?\d+)\s+(-?\d+)\s*\)")
_NET_LINE_RE = re.compile(r"-\s+(\S+)\s+(.*)\+\s+USE\s+SIGNAL\s*;")
_NET_PIN_RE = re.compile(r"\(\s*(\S+)\s+(\S+)\s*\)")


def parse_components(def_text: str) -> dict[str, tuple[str, float, float]]:
    """Returns {instance_name: (cell_type, x, y)} from the COMPONENTS section."""
    start = def_text.index("\nCOMPONENTS")
    end = def_text.index("\nEND COMPONENTS")
    section = def_text[start:end]

    components = {}
    for name, cell_type, x, y in _COMPONENT_RE.findall(section):
        components[name] = (cell_type, float(x), float(y))
    return components


def parse_nets(def_text: str) -> list[list[tuple[str, str]]]:
    """Returns a list of [(instance_name, port_name)] per net, skipping
    top-level PIN references. A list, not a dict keyed by name: net names
    legitimately repeat for buffered segments of the same logical net
    (e.g. a high-fanout net repowered by synthesis) - keying by name
    would silently drop duplicates."""
    start = def_text.index("\nNETS")
    end = def_text.index("\nEND NETS")
    section = def_text[start:end]

    nets = []
    for line in section.split("\n"):
        match = _NET_LINE_RE.search(line)
        if not match:
            continue
        _net_name, body = match.groups()
        pins = [(inst, port) for inst, port in _NET_PIN_RE.findall(body) if inst != "PIN"]
        if len({inst for inst, _port in pins}) >= 2:
            nets.append(pins)
    return nets


def resolve_macro_sizes(
    components: dict[str, tuple[str, float, float]], cell_sizes: SizeMap
) -> SizeMap:
    """Looks up each instance's cell_type in cell_sizes, converting a
    type-keyed LEF lookup table into an instance-keyed one - the same
    shape Bookshelf's .nodes file already gives directly."""
    macro_sizes = {}
    for name, (cell_type, _x, _y) in components.items():
        if cell_type in cell_sizes:
            macro_sizes[name] = cell_sizes[cell_type]
    return macro_sizes


def resolve_net_pin_offsets(
    nets: list[list[tuple[str, str]]],
    components: dict[str, tuple[str, float, float]],
    pin_offsets: PinOffsets,
) -> Nets:
    """Converts (instance, port) pairs into (instance, x_offset, y_offset),
    looking up each instance's cell_type then that type's port offset.
    Pins with no matching LEF pin geometry fall back to (0, 0) - center,
    the same simplification as never having offsets at all, rather than
    silently dropping the pin (which would change net degree)."""
    resolved: Nets = []
    for net in nets:
        pins: list[NetPin] = []
        for inst, port in net:
            cell_type = components[inst][0]
            x_off, y_off = pin_offsets.get(cell_type, {}).get(port, (0.0, 0.0))
            pins.append((inst, x_off, y_off))
        resolved.append(pins)
    return resolved


def load_def(def_path: pathlib.Path, lef_paths: list[pathlib.Path]) -> tuple[SizeMap, Nets]:
    """Returns (macro_sizes, nets) - same shape as load_bookshelf."""
    def_text = def_path.read_text()
    components = parse_components(def_text)
    raw_nets = parse_nets(def_text)

    cell_sizes: SizeMap = {}
    pin_offsets: PinOffsets = {}
    for lef in lef_paths:
        cell_sizes.update(parse_lef_sizes(lef))
        pin_offsets.update(parse_lef_pin_offsets(lef))

    macro_sizes = resolve_macro_sizes(components, cell_sizes)
    nets = resolve_net_pin_offsets(raw_nets, components, pin_offsets)
    return macro_sizes, nets
