"""Reads Circuit Training's .pb.txt netlist format (a TensorFlow GraphDef
text representation) into the same shape as the other loaders."""
import pathlib
import re

from placax.types import NetPin, Nets, SizeMap

_NAME_RE = re.compile(r'name:\s*"([^"]+)"')
_TYPE_RE = re.compile(r'key:\s*"type"\s*\n\s*value\s*\{\s*\n\s*placeholder:\s*"([^"]+)"')
_FLOAT_ATTR_RE = re.compile(r'key:\s*"(\w+)"\s*\n\s*value\s*\{\s*\n\s*f:\s*(-?[\d.]+)')
_STR_ATTR_RE = re.compile(r'key:\s*"(\w+)"\s*\n\s*value\s*\{\s*\n\s*placeholder:\s*"([^"]+)"')


def _split_nodes(pb_text: str) -> list[str]:
    """Splits the file into one text block per top-level `node { ... }`."""
    starts = [m.start() for m in re.finditer(r"^node \{", pb_text, re.MULTILINE)]
    ends = starts[1:] + [len(pb_text)]
    return [pb_text[s:e] for s, e in zip(starts, ends)]


def parse_macros(pb_text: str) -> SizeMap:
    """Returns {macro_name: (width, height)} for MACRO-type nodes only."""
    macros = {}
    for block in _split_nodes(pb_text):
        type_match = _TYPE_RE.search(block)
        if not type_match or type_match.group(1) != "MACRO":
            continue
        name = _NAME_RE.search(block).group(1)
        attrs = dict(_FLOAT_ATTR_RE.findall(block))
        if "width" in attrs and "height" in attrs:
            macros[name] = (float(attrs["width"]), float(attrs["height"]))
    return macros


def _macro_pin_entries(pb_text: str) -> list[tuple[str, float, float, str]]:
    """(macro_name, x_offset, y_offset, pin_node_name) per MACRO_PIN node
    - the node name is what other nodes' input: lists reference."""
    entries = []
    for block in _split_nodes(pb_text):
        type_match = _TYPE_RE.search(block)
        if not type_match or type_match.group(1) != "MACRO_PIN":
            continue
        pin_name = _NAME_RE.search(block).group(1)
        str_attrs = dict(_STR_ATTR_RE.findall(block))
        float_attrs = dict(_FLOAT_ATTR_RE.findall(block))
        macro_name = str_attrs.get("macro_name")
        if macro_name and "x_offset" in float_attrs and "y_offset" in float_attrs:
            entries.append(
                (macro_name, float(float_attrs["x_offset"]), float(float_attrs["y_offset"]), pin_name)
            )
    return entries


def parse_macro_pins(pb_text: str) -> list[NetPin]:
    """Returns [(macro_name, x_offset, y_offset)] for every MACRO_PIN node."""
    return [(macro, x, y) for macro, x, y, _pin_name in _macro_pin_entries(pb_text)]


def parse_nets(pb_text: str) -> Nets:
    """Returns a list of pins per net, from `input:` references on hub
    nodes (any node - only which macro pins it references matters).
    Kept only if >= 2 distinct macros."""
    pin_lookup = {name: (macro, x, y) for macro, x, y, name in _macro_pin_entries(pb_text)}

    nets = []
    for block in _split_nodes(pb_text):
        # A net is: this node's own name, plus everything it lists as input: -
        # whichever of those names happen to be MACRO_PIN nodes are this net's pins.
        hub_name_match = _NAME_RE.search(block)
        referenced = re.findall(r'input:\s*"([^"]+)"', block)
        if hub_name_match:
            referenced = referenced + [hub_name_match.group(1)]

        pins = [pin_lookup[n] for n in referenced if n in pin_lookup]
        if len({macro for macro, _x, _y in pins}) >= 2:
            nets.append(pins)
    return nets


def load_protobuf(pb_path: pathlib.Path) -> tuple[SizeMap, Nets]:
    """Returns (macro_sizes, nets) - same shape as the other loaders."""
    pb_text = pb_path.read_text()
    return parse_macros(pb_text), parse_nets(pb_text)
