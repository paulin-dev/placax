"""Measuring how legal a finished placement actually is.

Legality in this environment is enforced by MASKING the action distribution, so placements are
legal by construction - which is a genuinely good property, and the reason nothing here measured
it for a long time. But `legal_action_logits` carries a two-stage relaxation valve: if an extra
quality rule leaves no legal cell it drops that rule, and if bare legality leaves no legal cell it
drops legality too and lets the macro overlap. That second stage is silent. A run that hit it
would report a better HPWL - overlapping macros have shorter wires - with nothing in the log to
say the placement was not physically realizable.

So legality is measured, every time a placement is scored, rather than assumed. The other reason
is comparability: MaskPlace's own results table reports overlap beside HPWL (adaptec1 0%,
ariane 1.94%), and a wirelength number without one is not comparable to it.

Everything here is pure JAX and jittable, so it costs a rounding error next to the rollout that
produced the placement.
"""
from placax import _device  # noqa: F401  must run before any `import jax` below
from placax.extras.render import render
from placax.types import EnvParams

import jax
import jax.numpy as jnp
from flax import struct


@struct.dataclass
class Legality:
    """How much of a placement is physically impossible, in grid cells.

    Areas are counted with multiplicity: a cell covered by three macros contributes 2 to
    `overlap_area`, because two of those three macros have nowhere real to be.
    """

    overlap_area: jax.Array
    """Grid cells of macro footprint that land on top of another macro."""

    out_of_bounds_area: jax.Array
    """Grid cells of macro footprint that fall outside the canvas."""

    placed_area: jax.Array
    """Total footprint of every placed macro, the denominator for the ratios below."""

    n_unplaced: jax.Array
    """Macros still at the (-1, -1) sentinel - a placement that never finished."""

    @property
    def overlap_ratio(self) -> jax.Array:
        """Overlapping fraction of total macro area - the number MaskPlace reports as "overlap"."""
        return jnp.where(self.placed_area > 0, self.overlap_area / self.placed_area, 0.0)

    @property
    def out_of_bounds_ratio(self) -> jax.Array:
        return jnp.where(self.placed_area > 0, self.out_of_bounds_area / self.placed_area, 0.0)

    @property
    def is_legal(self) -> jax.Array:
        """True only for a placement that is complete, in bounds, and free of overlap."""
        return (
            (self.overlap_area == 0) & (self.out_of_bounds_area == 0) & (self.n_unplaced == 0)
        )

    def to_dict(self) -> dict:
        """Plain Python floats for a log line. Not jittable - call it outside the traced region."""
        return {
            "overlap_area": float(self.overlap_area),
            "overlap_ratio": float(self.overlap_ratio),
            "out_of_bounds_area": float(self.out_of_bounds_area),
            "out_of_bounds_ratio": float(self.out_of_bounds_ratio),
            "n_unplaced": int(self.n_unplaced),
            "is_legal": bool(self.is_legal),
        }


def _clipped_area(positions: jax.Array, grid_sizes: jax.Array, grid_x: int, grid_y: int) -> jax.Array:
    """Per-macro footprint area that actually lands on the canvas (0 for unplaced macros)."""
    x, y = positions[:, 0], positions[:, 1]
    w, h = grid_sizes[:, 0], grid_sizes[:, 1]
    # Overlap of [x, x+w) with [0, grid_x), and the same in y - a negative width means the macro
    # missed the canvas entirely, so clamp at zero rather than letting it subtract.
    visible_w = jnp.clip(jnp.minimum(x + w, grid_x) - jnp.maximum(x, 0), 0, None)
    visible_h = jnp.clip(jnp.minimum(y + h, grid_y) - jnp.maximum(y, 0), 0, None)
    return jnp.where(x >= 0, visible_w * visible_h, 0)


def legality(positions: jax.Array, grid_sizes: jax.Array, params: EnvParams) -> Legality:
    """Measures overlap, out-of-bounds area and completeness for one finished placement.

    `positions` are lower-left grid cells with -1 for unplaced; `grid_sizes` the macro footprints
    in the same grid units (see policy.scale.to_grid_units).
    """
    grid_x, grid_y = params.grid_x, params.effective_grid_y
    placed = positions[:, 0] >= 0

    # Total footprint the macros claim, versus how much of it is on the canvas at all.
    nominal_area = jnp.where(placed, grid_sizes[:, 0] * grid_sizes[:, 1], 0).sum()
    on_canvas_area = _clipped_area(positions, grid_sizes, grid_x, grid_y).sum()

    # render() unions the clipped footprints, so anything claimed on-canvas beyond what the union
    # covers is a cell two or more macros both wanted.
    covered_area = render(positions, grid_sizes, grid_x, grid_y).sum()

    return Legality(
        overlap_area=on_canvas_area - covered_area,
        out_of_bounds_area=nominal_area - on_canvas_area,
        placed_area=nominal_area,
        n_unplaced=(~placed).sum(),
    )


jitted_legality = jax.jit(legality, static_argnames=("params",))
"""legality() precompiled, since it is called once per evaluation on the same shapes."""
