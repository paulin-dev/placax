"""Writes placed positions back into a DEF file, rewriting only PLACED coordinates and passing the rest through."""
import re

_COMPONENT_RE = re.compile(r"-\s+(\S+)\s+(\S+)\s+\+\s+PLACED\s+\(\s*(-?\d+)\s+(-?\d+)\s*\)")


def write_placed_def(original_def_text: str, positions: dict[str, tuple[int, int]]) -> str:
    """positions: {instance_name: (x, y)} lower-left corner in integer DEF database units."""
    def replace_component(match: re.Match) -> str:
        # Leave any instance we weren't given a new position for exactly as-is.
        name, cell_type = match.group(1), match.group(2)
        if name not in positions:
            return match.group(0)
        new_x, new_y = positions[name]
        return f"- {name} {cell_type} + PLACED ( {int(new_x)} {int(new_y)} )"

    return _COMPONENT_RE.sub(replace_component, original_def_text)
