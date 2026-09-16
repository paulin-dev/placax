"""Writes placed positions back into a DEF file, rewriting only PLACED coordinates and passing the rest through."""
import re

_COMPONENT_RE = re.compile(
    r"-\s+(\S+)\s+(\S+)((?:\s+\+\s+(?:SOURCE|WEIGHT|REGION)\s+\S+)*)"
    r"\s+\+\s+(PLACED|FIXED)\s+\(\s*(-?\d+)\s+(-?\d+)\s*\)"
    r"(\s+(?:F?[NSEW]))?"
)
"""Same widening as the reader's, and for the same reason: a placed macro is written `FIXED`.

The status is captured rather than hard-coded into the replacement so that rewriting a position
does not quietly promote a FIXED macro to PLACED, or demote a placed cell - a cell placer reads
that difference as "may I move this"."""

_UNPLACED_RE = re.compile(
    r"^(\s*)-\s+(\S+)\s+(\S+)((?:\s+\+\s+(?:SOURCE|WEIGHT|REGION)\s+\S+)*)"
    r"(?:\s+\+\s+UNPLACED)?\s*;",
    re.MULTILINE,
)
"""A component with no position yet. Written FIXED once the agent gives it one."""


def write_placed_def(original_def_text: str, positions: dict[str, tuple]) -> str:
    """Rewrites PLACED coordinates, and the orientation when the caller chose one.

    `positions` maps an instance to `(x, y)` or `(x, y, orientation)`, lower-left corner in
    integer DEF database units. The orientation is captured by the pattern so that it can be
    replaced; with the two-element form it is written back exactly as it was, which is what every
    run did before macros could be turned.
    """
    def replace_component(match: re.Match) -> str:
        # Leave any instance we weren't given a new position for exactly as-is.
        name, cell_type, attributes, status = match.group(1, 2, 3, 4)
        if name not in positions:
            return match.group(0)
        placement = positions[name]
        new_x, new_y = placement[0], placement[1]
        source_orient = match.group(7)
        orient = f" {placement[2]}" if len(placement) > 2 else (source_orient or "")
        return (f"- {name} {cell_type}{attributes} + {status} "
                f"( {int(new_x)} {int(new_y)} ){orient}")

    def place_unplaced(match: re.Match) -> str:
        # A macro the floorplan left UNPLACED has a position now, and it is the agent's decision:
        # FIXED, so the cell placer works around it rather than moving it.
        indent, name, cell_type, attributes = match.group(1, 2, 3, 4)
        if name not in positions:
            return match.group(0)
        placement = positions[name]
        orient = placement[2] if len(placement) > 2 else "N"
        return (f"{indent}- {name} {cell_type}{attributes} + FIXED "
                f"( {int(placement[0])} {int(placement[1])} ) {orient} ;")

    placed = _COMPONENT_RE.sub(replace_component, original_def_text)
    return _UNPLACED_RE.sub(place_unplaced, placed)
