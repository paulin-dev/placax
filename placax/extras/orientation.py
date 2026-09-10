"""Macro orientation - the half of "placement representation" that was never represented.

A placement was `(n_macros, 2)` integers: where each macro's lower-left corner sits, and nothing
else. Real macro placement chooses an orientation too - a tall SRAM rotated 90 degrees is a
different shape with its pins somewhere else, and every flow in this field emits one per instance
(`N`, `W`, `S`, `E` and their mirrors in both Bookshelf `.pl` and DEF). This project's writers
preserved whatever the source file happened to say and the agent never had an opinion, so the axis
did not exist rather than being fixed at north.

**The design that keeps this from touching everything.** Orientation is not a new parameter on
`hpwl`, `render` and `legality`. It is a transform on their INPUTS:

    effective_sizes(sizes, orientations)      w and h swap under a quarter turn
    rotate_offsets(offsets, orientations)     a pin rotates about its macro's center

so a caller that knows the orientations rewrites the geometry once and hands the existing
functions exactly what they already take. Nothing downstream needs a new argument, and code that
does not use orientation cannot be broken by it - `NORTH` everywhere is the identity on both.

Only the four rotations are modelled, not the four mirrors. A mirror leaves the footprint alone
and reflects the pins, so it changes wirelength but not legality; adding it is another entry in
the tables below rather than a different shape of code. Stated because "we support orientation"
would otherwise imply all eight.
"""
from placax import _device  # noqa: F401  must run before any `import jax` below

import jax
import jax.numpy as jnp

NORTH, WEST, SOUTH, EAST = 0, 1, 2, 3
"""Quarter turns counter-clockwise. NORTH is the identity and the default everywhere, so a
placement that never mentions orientation behaves exactly as it did before this existed."""

N_ORIENTATIONS = 4

ORIENTATION_NAMES = ("N", "W", "S", "E")
"""Bookshelf `.pl` and DEF spell them the same way, so one table serves both writers."""


def default_orientations(n_macros: int) -> jax.Array:
    """Every macro north - what a placement means when nobody chose."""
    return jnp.zeros((n_macros,), dtype=jnp.int32)


def resolve(orientations: jax.Array | None, n_macros: int) -> jax.Array:
    """`orientations`, or all-north when a state carries none.

    States store `None` rather than a zeros array so that the un-oriented path allocates nothing
    and stays bit-identical; this is the one place that difference is turned back into an array.
    """
    return default_orientations(n_macros) if orientations is None else orientations


def is_quarter_turned(orientations: jax.Array) -> jax.Array:
    """Whether each macro's width and height are swapped: true for WEST and EAST."""
    return (orientations % 2) == 1


def effective_sizes(sizes: jax.Array, orientations: jax.Array | None) -> jax.Array:
    """`(n_macros, 2)` footprints as they actually sit on the canvas, with rotations applied.

    Feed this wherever an unrotated `sizes_array` used to go - `to_grid_units`, `render`,
    `legality`, `to_real_centers` - and every one of them becomes orientation-aware without
    knowing that orientation exists.
    """
    if orientations is None:
        return sizes
    swapped = jnp.stack([sizes[:, 1], sizes[:, 0]], axis=-1)
    return jnp.where(is_quarter_turned(orientations)[:, None], swapped, sizes)


def rotate_offsets(offsets: jax.Array, orientations: jax.Array | None) -> jax.Array:
    """Pin offsets rotated about their macro's center. Shapes broadcast, so padded arrays work.

    `offsets` is `(..., 2)` and `orientations` is `(...)` - for the padded net arrays that means
    `padded_pin_offset` with `orientations[padded_pin_idx]`, so each pin turns with the macro it
    belongs to rather than with the net it is on.

    Counter-clockwise, matching NORTH/WEST/SOUTH/EAST above:
        N  ( dx,  dy)      W  (-dy,  dx)      S  (-dx, -dy)      E  ( dy, -dx)
    """
    if orientations is None:
        return offsets
    dx, dy = offsets[..., 0], offsets[..., 1]
    quarter = orientations % N_ORIENTATIONS
    # Selected rather than branched so this stays one jittable expression over the whole array.
    rotated_x = jnp.select(
        [quarter == NORTH, quarter == WEST, quarter == SOUTH, quarter == EAST],
        [dx, -dy, -dx, dy],
    )
    rotated_y = jnp.select(
        [quarter == NORTH, quarter == WEST, quarter == SOUTH, quarter == EAST],
        [dy, dx, -dy, -dx],
    )
    return jnp.stack([rotated_x, rotated_y], axis=-1)


def oriented_pin_offsets(
    padded_pin_offset: jax.Array, padded_pin_idx: jax.Array, orientations: jax.Array | None
) -> jax.Array:
    """`padded_pin_offset` with every pin turned by ITS OWN macro's orientation.

    The one call an HPWL site needs: `hpwl(centers, idx, oriented_pin_offsets(offsets, idx, o),
    mask)` is the whole of making wirelength orientation-aware.
    """
    if orientations is None:
        return padded_pin_offset
    return rotate_offsets(padded_pin_offset, orientations[padded_pin_idx])


def names(orientations: jax.Array | None, n_macros: int) -> list[str]:
    """`["N", "W", ...]` for a finished placement, for the `.pl` and DEF writers."""
    resolved = resolve(orientations, n_macros)
    return [ORIENTATION_NAMES[int(value) % N_ORIENTATIONS] for value in resolved]


__all__ = [
    "NORTH", "WEST", "SOUTH", "EAST", "N_ORIENTATIONS", "ORIENTATION_NAMES",
    "default_orientations", "effective_sizes", "is_quarter_turned", "names",
    "oriented_pin_offsets", "resolve", "rotate_offsets",
]
