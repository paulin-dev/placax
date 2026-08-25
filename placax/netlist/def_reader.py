"""Reads DEF COMPONENTS/NETS + LEF geometry into the same shape the Bookshelf loader produces."""
import pathlib
import re

from placax.netlist.lef import parse_lef_pin_offsets, parse_lef_sizes
from placax.types import NetPin, Nets, PinOffsets, SizeMap

_COMPONENT_RE = re.compile(r"-\s+(\S+)\s+(\S+)\s+\+\s+PLACED\s+\(\s*(-?\d+)\s+(-?\d+)\s*\)")
_NET_LINE_RE = re.compile(r"-\s+(\S+)\s+(.*)\+\s+USE\s+SIGNAL\s*;")
_NET_PIN_RE = re.compile(r"\(\s*(\S+)\s+(\S+)\s*\)")


def parse_components(def_text: str) -> dict[str, tuple[str, float, float]]:
    """Returns {instance_name: (cell_type, x, y)} from the COMPONENTS section."""
    section = def_text[def_text.index("\nCOMPONENTS") : def_text.index("\nEND COMPONENTS")]
    return {
        name: (cell_type, float(x), float(y))
        for name, cell_type, x, y in _COMPONENT_RE.findall(section)
    }


def parse_nets(def_text: str) -> list[list[tuple[str, str]]]:
    """Returns [(instance_name, port_name)] per net, as a list since net names can legitimately repeat."""
    section = def_text[def_text.index("\nNETS") : def_text.index("\nEND NETS")]

    nets = []
    for line in section.split("\n"):
        match = _NET_LINE_RE.search(line)
        if not match:
            continue
        # Drop the special top-level "PIN" pseudo-instance and keep only nets
        # that actually connect two or more distinct instances.
        pins = [(inst, port) for inst, port in _NET_PIN_RE.findall(match.group(2)) if inst != "PIN"]
        if len({inst for inst, _port in pins}) >= 2:
            nets.append(pins)
    return nets


def resolve_macro_sizes(
    components: dict[str, tuple[str, float, float]], cell_sizes: SizeMap
) -> SizeMap:
    """Looks up each instance's cell_type in cell_sizes: type-keyed -> instance-keyed."""
    return {
        name: cell_sizes[cell_type]
        for name, (cell_type, _x, _y) in components.items()
        if cell_type in cell_sizes
    }


def resolve_net_pin_offsets(
    nets: list[list[tuple[str, str]]],
    components: dict[str, tuple[str, float, float]],
    pin_offsets: PinOffsets,
) -> Nets:
    """Resolves (instance, port) pairs to (instance, x_offset, y_offset), defaulting to (0, 0) if unknown."""
    resolved: Nets = []
    for net in nets:
        pins: list[NetPin] = []
        for inst, port in net:
            # Look up this instance's cell type, then that type's LEF-declared pin
            # offset; pins with no matching geometry fall back to (0, 0) rather than
            # being dropped, since the net topology should still be preserved.
            cell_type = components[inst][0]
            x_off, y_off = pin_offsets.get(cell_type, {}).get(port, (0.0, 0.0))
            pins.append((inst, x_off, y_off))
        resolved.append(pins)
    return resolved


def load_def(def_path: pathlib.Path, lef_paths: list[pathlib.Path]) -> tuple[SizeMap, Nets]:
    """Returns (macro_sizes, nets) - the same shape as load_bookshelf."""
    # 1. Parse instance placements/types and pin connectivity out of the DEF text.
    def_text = def_path.read_text()
    components = parse_components(def_text)

    # 2. Merge every LEF's cell geometry - a design can split cells across multiple LEFs.
    cell_sizes: SizeMap = {}
    pin_offsets: PinOffsets = {}
    for lef in lef_paths:
        cell_sizes.update(parse_lef_sizes(lef))
        pin_offsets.update(parse_lef_pin_offsets(lef))

    # 3. Translate cell-type-keyed geometry into instance-keyed sizes/offsets.
    macro_sizes = resolve_macro_sizes(components, cell_sizes)
    nets = resolve_net_pin_offsets(parse_nets(def_text), components, pin_offsets)
    return macro_sizes, nets
