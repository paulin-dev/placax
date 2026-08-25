"""Parses cell/macro physical dimensions and pin geometry from LEF files."""
import pathlib
import re

from placax.types import PinOffsets, SizeMap

_SIZE_RE = re.compile(r"MACRO\s+(\S+).*?SIZE\s+([\d.]+)\s+BY\s+([\d.]+)", re.DOTALL)
_MACRO_BLOCK_RE = re.compile(r"MACRO\s+(\S+)(.*?)\n\s*END\s+\1\s*\n", re.DOTALL)
_PIN_BLOCK_RE = re.compile(r"PIN\s+(\S+)(.*?)\n\s*END\s+\1\s*\n", re.DOTALL)
_RECT_RE = re.compile(r"RECT\s+(-?[\d.]+)\s+(-?[\d.]+)\s+(-?[\d.]+)\s+(-?[\d.]+)")
_SIZE_IN_BLOCK_RE = re.compile(r"SIZE\s+([\d.]+)\s+BY\s+([\d.]+)")


def parse_lef_sizes(lef_path: pathlib.Path) -> SizeMap:
    """Returns {macro_name: (width, height)} from a LEF file's SIZE lines."""
    return {
        name: (float(w), float(h))
        for name, w, h in _SIZE_RE.findall(lef_path.read_text())
    }


def _parse_macro_pins(macro_block: str, width: float, height: float) -> dict[str, tuple[float, float]]:
    """Returns {pin_name: (x_offset, y_offset)} for one MACRO block."""
    pins = {}
    for pin_name, pin_block in _PIN_BLOCK_RE.findall(macro_block):
        rect_match = _RECT_RE.search(pin_block)
        if not rect_match:
            continue
        x1, y1, x2, y2 = (float(v) for v in rect_match.groups())
        pins[pin_name] = (
            (x1 + x2) / 2 - width / 2,
            (y1 + y2) / 2 - height / 2,
        )
    return pins


def parse_lef_pin_offsets(lef_path: pathlib.Path) -> PinOffsets:
    """Returns {macro_name: {pin_name: (x_offset, y_offset)}} relative to
    the macro center. Only the first RECT per pin is used."""
    result: PinOffsets = {}
    for macro_name, block in _MACRO_BLOCK_RE.findall(lef_path.read_text()):
        size_match = _SIZE_IN_BLOCK_RE.search(block)
        if not size_match:
            continue
        width, height = float(size_match.group(1)), float(size_match.group(2))
        pins = _parse_macro_pins(block, width, height)
        if pins:
            result[macro_name] = pins
    return result
