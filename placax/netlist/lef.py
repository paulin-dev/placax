"""Parses cell/macro physical dimensions and pin geometry out of a LEF file."""
import pathlib
import re

from placax.types import PinOffsets, SizeMap

_SIZE_RE = re.compile(r"MACRO\s+(\S+).*?SIZE\s+([\d.]+)\s+BY\s+([\d.]+)", re.DOTALL)
_MACRO_BLOCK_RE = re.compile(r"MACRO\s+(\S+)(.*?)\n\s*END\s+\1\s*\n", re.DOTALL)
_PIN_BLOCK_RE = re.compile(r"PIN\s+(\S+)(.*?)\n\s*END\s+\1\s*\n", re.DOTALL)
_RECT_RE = re.compile(r"RECT\s+(-?[\d.]+)\s+(-?[\d.]+)\s+(-?[\d.]+)\s+(-?[\d.]+)")


def parse_lef_sizes(lef_path: pathlib.Path) -> SizeMap:
    """Returns {macro_name: (width, height)} from a LEF file's SIZE lines."""
    content = lef_path.read_text()
    return {name: (float(w), float(h)) for name, w, h in _SIZE_RE.findall(content)}


def _parse_macro_pins(macro_block: str, width: float, height: float) -> dict[str, tuple[float, float]]:
    """Returns {pin_name: (x_offset, y_offset)} for one MACRO block's pins."""
    pins = {}
    for pin_name, pin_block in _PIN_BLOCK_RE.findall(macro_block):
        rect_match = _RECT_RE.search(pin_block)
        if not rect_match:
            continue
        x1, y1, x2, y2 = (float(v) for v in rect_match.groups())
        pin_center_x, pin_center_y = (x1 + x2) / 2, (y1 + y2) / 2
        pins[pin_name] = (pin_center_x - width / 2, pin_center_y - height / 2)
    return pins


def parse_lef_pin_offsets(lef_path: pathlib.Path) -> PinOffsets:
    """Returns {macro_name: {pin_name: (x_offset, y_offset)}}, offset from
    the macro's center. LEF's RECT geometry is relative to the macro's
    lower-left origin, not center: offset = rect_center - size/2. Only
    the first RECT per pin is used (a pin can have several, for multi-
    layer geometry - the first is a fair approximation of its location)."""
    content = lef_path.read_text()
    result: PinOffsets = {}

    for macro_name, block in _MACRO_BLOCK_RE.findall(content):
        size_match = re.search(r"SIZE\s+([\d.]+)\s+BY\s+([\d.]+)", block)
        if not size_match:
            continue
        width, height = float(size_match.group(1)), float(size_match.group(2))
        pins = _parse_macro_pins(block, width, height)
        if pins:
            result[macro_name] = pins

    return result
