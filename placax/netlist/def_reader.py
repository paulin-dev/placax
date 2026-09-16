"""Reads DEF COMPONENTS/NETS + LEF geometry into the same shape the Bookshelf loader produces."""
import pathlib
import re

from placax.netlist.lef import parse_lef_classes, parse_lef_pin_offsets, parse_lef_sizes
from placax.types import NetPin, Nets, PinOffsets, SizeMap

_COMPONENT_RE = re.compile(
    r"-\s+(\S+)\s+(\S+)(?:\s+\+\s+(?:SOURCE|WEIGHT|REGION)\s+\S+)*"
    r"\s+\+\s+(?:PLACED|FIXED)\s+\(\s*(-?\d+)\s+(-?\d+)\s*\)"
)
"""A component with a position, however it was fixed there.

`FIXED` used to be invisible here, which is a real omission rather than a stylistic one: every
tool in this flow marks a placed MACRO `FIXED` - that is what tells the cell placer not to move
it - so a DEF produced by OpenROAD, by DREAMPlace, or by this project's own
`netlist/def_export.py` had its macros silently skipped on the way back in. `UNPLACED` components
deliberately still do not match: they carry no coordinates to read."""
_UNPLACED_RE = re.compile(
    r"^(\s*)-\s+(\S+)\s+(\S+)((?:\s+\+\s+(?:SOURCE|WEIGHT|REGION)\s+\S+)*)"
    r"(?:\s+\+\s+UNPLACED)?\s*;",
    re.MULTILINE,
)
"""A component that has no position yet - every instance of a fresh floorplan.

OpenROAD writes one as `- name master ;` - no `UNPLACED` keyword at all - or, for an instance the
synthesis flow inserted, `- name master + SOURCE TIMING + UNPLACED ;`. Both forms."""
_NET_PIN_RE = re.compile(r"\(\s*(\S+)\s+(\S+)\s*\)")
_UNITS_RE = re.compile(r"UNITS\s+DISTANCE\s+MICRONS\s+(\d+)")
_DIEAREA_RE = re.compile(
    r"DIEAREA\s*\(\s*(-?\d+)\s+(-?\d+)\s*\)\s*\(\s*(-?\d+)\s+(-?\d+)\s*\)"
)


def parse_die_size(def_text: str) -> float | None:
    """The real (square) die side length from DEF's own DIEAREA statement, converted from database units to microns (matching LEF's macro sizes) via UNITS DISTANCE MICRONS. None if DIEAREA is absent."""
    diearea_match = _DIEAREA_RE.search(def_text)
    if not diearea_match:
        return None
    units_match = _UNITS_RE.search(def_text)
    db_units_per_micron = float(units_match.group(1)) if units_match else 1.0
    x1, y1, x2, y2 = (float(v) for v in diearea_match.groups())
    return max(x2 - x1, y2 - y1) / db_units_per_micron


def parse_components(def_text: str) -> dict[str, tuple[str, float, float]]:
    """Returns {instance_name: (cell_type, x, y)} from the COMPONENTS section."""
    section = def_text[def_text.index("\nCOMPONENTS") : def_text.index("\nEND COMPONENTS")]
    return {
        name: (cell_type, float(x), float(y))
        for name, cell_type, x, y in _COMPONENT_RE.findall(section)
    }


def parse_unplaced_components(def_text: str) -> dict[str, str]:
    """Returns {instance_name: cell_type} for the COMPONENTS that carry no position."""
    section = def_text[def_text.index("\nCOMPONENTS") : def_text.index("\nEND COMPONENTS")]
    return {match.group(2): match.group(3) for match in _UNPLACED_RE.finditer(section)}


PIN_PREFIX = "PIN:"
"""How an IO pin appears among instances in a full-design view."""

_PIN_STATEMENT_RE = re.compile(r"^\s*-\s+(\S+)(.*?);", re.MULTILINE | re.DOTALL)
_PIN_PLACED_RE = re.compile(r"\+\s*(?:PLACED|FIXED|COVER)\s+\(\s*(-?\d+)\s+(-?\d+)\s*\)")
_PIN_RECT_RE = re.compile(
    r"\+\s*LAYER\s+\S+\s+\(\s*(-?\d+)\s+(-?\d+)\s*\)\s*\(\s*(-?\d+)\s+(-?\d+)\s*\)"
)


def parse_pins(def_text: str) -> dict[str, tuple[float, float]]:
    """{pin: (x, y)} for every placed IO pin, in database units - its shape's center."""
    if "\nPINS" not in def_text:
        return {}
    section = def_text[def_text.index("\nPINS") : def_text.index("\nEND PINS")]
    section = section[section.index("\n", 1):]
    pins = {}
    for match in _PIN_STATEMENT_RE.finditer(section):
        placed = _PIN_PLACED_RE.search(match.group(2))
        if placed is None:
            continue
        x, y = float(placed.group(1)), float(placed.group(2))
        rect = _PIN_RECT_RE.search(match.group(2))
        if rect is not None:
            x1, y1, x2, y2 = (float(v) for v in rect.groups())
            x, y = x + (x1 + x2) / 2.0, y + (y1 + y2) / 2.0
        pins[match.group(1)] = (x, y)
    return pins


_NET_STATEMENT_RE = re.compile(r"^\s*-\s+(\S+)(.*?);", re.MULTILINE | re.DOTALL)
_NET_USE_RE = re.compile(r"\+\s*USE\s+(\S+)")


def parse_nets(def_text: str, keep_pins: bool = False) -> list[list[tuple[str, str]]]:
    """Returns [(instance_name, port_name)] per signal net, as a list since net names can repeat;
    keeps only the first port per (net, instance).

    Read statement by statement, not line by line: OpenROAD wraps a high-fanout net over as many
    lines as it needs, and a line-based reader dropped every such net without a word - on the
    ORFS ariane133 floorplan that was 21,503 continuation lines, and the full-design HPWL came out
    at 57% of the tool's. A net with no USE is a signal net, as the DEF standard says.

    `keep_pins` keeps the design's IO pins as pseudo-instances named `PIN:<pin>`, for a
    full-design wirelength; the macro view drops them.
    """
    section = def_text[def_text.index("\nNETS") : def_text.index("\nEND NETS")]
    section = section[section.index("\n", 1):]   # past the "NETS <count> ;" header

    nets = []
    for match in _NET_STATEMENT_RE.finditer(section):
        body = match.group(2)
        use = _NET_USE_RE.search(body)
        if use is not None and use.group(1) != "SIGNAL":
            continue
        # Drop the special top-level "PIN" pseudo-instance and dedupe by instance, keeping the first port seen.
        seen: dict[str, str] = {}
        for inst, port in _NET_PIN_RE.findall(body.split("+", 1)[0]):
            if inst == "PIN":
                if not keep_pins:
                    continue
                inst, port = f"{PIN_PREFIX}{port}", ""
            if inst not in seen:
                seen[inst] = port
        # Keep only nets that actually connect two or more distinct instances.
        if len(seen) >= 2:
            nets.append(list(seen.items()))
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
            # Look up this instance's cell type and its LEF-declared pin offset; unmatched pins fall back to (0, 0) to preserve topology.
            cell_type = components[inst][0]
            x_off, y_off = pin_offsets.get(cell_type, {}).get(port, (0.0, 0.0))
            pins.append((inst, x_off, y_off))
        resolved.append(pins)
    return resolved


def load_def(def_path: pathlib.Path, lef_paths: list[pathlib.Path]) -> tuple[SizeMap, Nets]:
    """Returns (macro_sizes, nets) - the same shape as load_bookshelf.

    Which instances are MACROS depends on what the LEFs say. When they declare classes - every
    real PDK does - a macro is an instance of a `CLASS BLOCK` master, placed or not: a design
    straight out of a floorplanner has every instance UNPLACED, and reading only positioned
    components found nothing in it. The nets are then the macro-to-macro connections, as the
    Bookshelf loader's are. A LEF set with no BLOCK at all describes a design that carries only
    its macros, and every positioned component is one - the original behaviour.
    """
    # 1. Parse instance placements/types and pin connectivity out of the DEF text.
    def_text = def_path.read_text()
    components = parse_components(def_text)

    # 2. Merge every LEF's cell geometry - a design can split cells across multiple LEFs.
    cell_sizes: SizeMap = {}
    pin_offsets: PinOffsets = {}
    classes: dict[str, str] = {}
    for lef in lef_paths:
        cell_sizes.update(parse_lef_sizes(lef))
        pin_offsets.update(parse_lef_pin_offsets(lef))
        classes.update(parse_lef_classes(lef))

    blocks = {name for name, kind in classes.items() if kind == "BLOCK"}
    if blocks:
        unplaced = parse_unplaced_components(def_text)
        every = {**{name: (kind, 0.0, 0.0) for name, kind in unplaced.items()}, **components}
        components = {name: value for name, value in every.items() if value[0] in blocks}
        nets = [
            [(inst, port) for inst, port in net if inst in components]
            for net in parse_nets(def_text)
        ]
        nets = [net for net in nets if len(net) >= 2]
    else:
        nets = parse_nets(def_text)

    # 3. Translate cell-type-keyed geometry into instance-keyed sizes/offsets.
    macro_sizes = resolve_macro_sizes(components, cell_sizes)
    nets = resolve_net_pin_offsets(nets, components, pin_offsets)
    return macro_sizes, nets


def load_placed_design(def_path: pathlib.Path, lef_paths: list[pathlib.Path]):
    """Every positioned instance of a placed DEF, in microns: (positions, sizes, nets).

    `positions` are lower-left corners, `sizes` come from the LEFs, and `nets` are ALL the signal
    nets between instances with their pin offsets - the whole design, for a full-design HPWL, not
    the macro-only view `load_def` gives an agent.
    """
    def_text = def_path.read_text()
    units = _UNITS_RE.search(def_text)
    scale = float(units.group(1)) if units else 1.0
    components = parse_components(def_text)
    cell_sizes: SizeMap = {}
    pin_offsets: PinOffsets = {}
    for lef in lef_paths:
        cell_sizes.update(parse_lef_sizes(lef))
        pin_offsets.update(parse_lef_pin_offsets(lef))
    positions = {name: (x / scale, y / scale) for name, (_kind, x, y) in components.items()}
    sizes = resolve_macro_sizes(components, cell_sizes)
    # IO pins are points: a zero-size pseudo-instance each, which is how the tool counts them too.
    for pin, (x, y) in parse_pins(def_text).items():
        name = f"{PIN_PREFIX}{pin}"
        positions[name], sizes[name] = (x / scale, y / scale), (0.0, 0.0)
        components[name] = (name, x, y)
    nets = [
        [(inst, port) for inst, port in net if inst in components]
        for net in parse_nets(def_text, keep_pins=True)
    ]
    nets = resolve_net_pin_offsets([net for net in nets if len(net) >= 2], components, pin_offsets)
    return positions, sizes, nets
