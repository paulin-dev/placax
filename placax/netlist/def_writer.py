"""Writes placed positions back into a DEF file, rewriting only PLACED coordinates and passing the rest through."""
import re

_COMPONENT_RE = re.compile(
    r"-\s+(\S+)\s+(\S+)\s+\+\s+PLACED\s+\(\s*(-?\d+)\s+(-?\d+)\s*\)(\s+(?:F?[NSEW]))?"
)


def write_placed_def(original_def_text: str, positions: dict[str, tuple]) -> str:
    """Rewrites PLACED coordinates, and the orientation when the caller chose one.

    `positions` maps an instance to `(x, y)` or `(x, y, orientation)`, lower-left corner in
    integer DEF database units. The orientation is captured by the pattern so that it can be
    replaced; with the two-element form it is written back exactly as it was, which is what every
    run did before macros could be turned.
    """
    def replace_component(match: re.Match) -> str:
        # Leave any instance we weren't given a new position for exactly as-is.
        name, cell_type = match.group(1), match.group(2)
        if name not in positions:
            return match.group(0)
        placement = positions[name]
        new_x, new_y = placement[0], placement[1]
        source_orient = match.group(5)
        orient = f" {placement[2]}" if len(placement) > 2 else (source_orient or "")
        return f"- {name} {cell_type} + PLACED ( {int(new_x)} {int(new_y)} ){orient}"

    return _COMPONENT_RE.sub(replace_component, original_def_text)
