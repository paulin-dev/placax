"""Writes placed positions back into a DEF file - the reverse of
def_reader.py, and the handoff artifact for a downstream cell placer
(DREAMPlace) or validator (OpenROAD). Rewrites only the COMPONENTS
section's PLACED coordinates; everything else passes through unchanged,
since downstream tools need the real floorplan structure (rows/tracks)
too - synthesizing a DEF from scratch would lose that."""
import re

_COMPONENT_RE = re.compile(r"-\s+(\S+)\s+(\S+)\s+\+\s+PLACED\s+\(\s*(-?\d+)\s+(-?\d+)\s*\)")


def write_placed_def(original_def_text: str, positions: dict[str, tuple[int, int]]) -> str:
    """positions: {instance_name: (x, y)} lower-left corner in integer DEF
    database units. Instances not in positions keep their original
    coordinates."""
    def replace_component(match: re.Match) -> str:
        name, cell_type = match.group(1), match.group(2)
        if name not in positions:
            return match.group(0)
        new_x, new_y = positions[name]
        return f"- {name} {cell_type} + PLACED ( {int(new_x)} {int(new_y)} )"

    return _COMPONENT_RE.sub(replace_component, original_def_text)
