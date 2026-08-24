"""Writes placed positions back into a DEF file - the reverse of
def_reader.py, and the actual handoff artifact for a downstream cell
placer (DREAMPlace) or validator (OpenROAD), matching the architecture:
placax's kernel outputs placed macro positions, which get handed to
external, swappable tools that consume standard DEF.

Rewrites only the COMPONENTS section's PLACED coordinates in an
existing DEF's text - everything else (header, rows, tracks, nets)
passes through unchanged, since a downstream cell placer needs that
real floorplan structure too, not just positions. This is deliberately
NOT synthesizing a DEF from scratch: a from-scratch file would be
missing rows/tracks a real tool needs (confirmed by inspecting a real
DEF - ariane.def carries hundreds of ROW statements placax never reads
or needs itself, but a cell placer does)."""
import re

_COMPONENT_RE = re.compile(r"-\s+(\S+)\s+(\S+)\s+\+\s+PLACED\s+\(\s*(-?\d+)\s+(-?\d+)\s*\)")


def write_placed_def(original_def_text: str, positions: dict[str, tuple[int, int]]) -> str:
    """positions: {instance_name: (x, y)} - lower-left corner, in the
    same integer DEF database units the original file already used
    (matching what PLACED itself represents - see def_reader.py).
    Instances not in positions are left with their original coordinates
    unchanged, not dropped."""

    def replace_component(match: re.Match) -> str:
        name, cell_type = match.group(1), match.group(2)
        if name not in positions:
            return match.group(0)
        new_x, new_y = positions[name]
        return f"- {name} {cell_type} + PLACED ( {int(new_x)} {int(new_y)} )"

    return _COMPONENT_RE.sub(replace_component, original_def_text)
