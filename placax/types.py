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


@struct.dataclass
class EnvParams:
    """Static per-run config. grid/grid_y/n_macros are pytree_node=False
    (static shapes, not traced values). grid_y=None means square canvas."""

    grid: int = struct.field(pytree_node=False, default=4)
    grid_y: int | None = struct.field(pytree_node=False, default=None)
    n_macros: int = struct.field(pytree_node=False, default=4)

    @property
    def grid_x(self) -> int:
        return self.grid

    @property
    def effective_grid_y(self) -> int:
        return self.grid_y if self.grid_y is not None else self.grid


RewardFn = Callable[[jax.Array, jax.Array, jax.Array, jax.Array], jax.Array]
"""reward_fn(old_positions, new_positions, old_placed, new_placed) ->
scalar array. old/new_positions: (n_macros, 2) int, grid-cell coordinates
(lower-left corner) before/after this step's action - the raw values
EnvState tracks, NOT real units. old/new_placed: (n_macros,) bool,
positions[:, 0] >= 0 before/after - which macros already have a real
position (pass this in rather than re-deriving it downstream, since a
real-unit conversion loses the -1 sentinel). core.step() calls this every
step, unconditionally; whether/how much reward is granted on any given
step is entirely reward_fn's choice - a sparse reward that only fires
once new_placed.all() and a dense reward that fires every step are both
just reward_fn implementations, not different code paths in step().
To compute in real units (e.g. matching a netlist's pin offsets), convert
positions with placax_agents.policy.scale.to_real_centers - or start from
placax_agents.training.reward.make_scaled_hpwl_reward, which already does."""

SizeMap = dict[str, tuple[float, float]]  # instance or cell-type name -> (width, height)
NetPin = tuple[str, float, float]  # (macro_name, x_offset, y_offset) - offset from macro center
Nets = list[list[NetPin]]  # one list of pins per net
PinOffsets = dict[str, dict[str, tuple[float, float]]]  # cell_type -> port_name -> (x, y) offset

OrderFn = Callable[[SizeMap, Nets], list[str]]
"""order_fn(macro_sizes, nets) -> macro names in placement order. Macro
i's name becomes row i everywhere downstream (positions, sizes_array,
padded_pin_idx) - see placax.netlist.order for built-in choices. Must
return every key of macro_sizes exactly once."""
