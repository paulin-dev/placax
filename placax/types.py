"""Shared type contracts every part of placax is built around."""
from typing import Callable

from placax import _device  # noqa: F401  must run before any `import jax` below

import jax
from flax import struct


@struct.dataclass
class EnvState:
    """Dynamic per-episode state: changes every step() call."""

    positions: jax.Array
    step: int
    orientations: jax.Array | None = None
    """Per-macro quarter turns, or None for "every macro north" - see extras/orientation.py.

    None rather than a zeros array so the un-oriented path allocates nothing and stays exactly
    what it was: a placement that never mentions orientation is bit-identical to one from before
    the axis existed. `orientation.resolve()` turns the absence back into an array where one is
    needed."""


@struct.dataclass
class EnvParams:
    """Static per-run config; fields are pytree_node=False (static shapes, not traced values)."""

    grid: int = struct.field(pytree_node=False, default=4)
    grid_y: int | None = struct.field(pytree_node=False, default=None)
    n_macros: int = struct.field(pytree_node=False, default=4)

    @property
    def grid_x(self) -> int:
        return self.grid

    @property
    def effective_grid_y(self) -> int:
        return self.grid_y if self.grid_y is not None else self.grid


RewardFn = Callable[..., jax.Array]
"""reward_fn(old_positions, new_positions, old_placed, new_placed[, orientations]) -> scalar.

Called by `step()` every step. The fifth argument is the placement's per-macro quarter turns, and
it is passed ONLY when the run's action space produces them (`oriented_grid` today). So:

  * a reward that never mentions orientation keeps the four-parameter signature it always had,
    and every un-oriented run - which is every run this project has produced - calls it exactly
    as before;
  * a reward used under an orientation-bearing space must accept `orientations=None` as a fifth
    parameter, because a turned macro's center and pins move and a reward blind to that scores a
    placement nobody made. `build()` checks this up front rather than letting it fail mid-episode.

`extras/orientation.py` has the two transforms that make a reward orientation-aware without any
new geometry code: `effective_sizes` for the footprint, `oriented_pin_offsets` for the pins."""

SizeMap = dict[str, tuple[float, float]]  # instance or cell-type name -> (width, height)
NetPin = tuple[str, float, float]  # (macro_name, x_offset, y_offset) - offset from macro center
Nets = list[list[NetPin]]  # one list of pins per net
PinOffsets = dict[str, dict[str, tuple[float, float]]]  # cell_type -> port_name -> (x, y) offset

OrderFn = Callable[[SizeMap, Nets], list[str]]
"""order_fn(macro_sizes, nets) -> macro names in placement order (must return every key exactly once)."""
