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
scalar. positions are grid-cell coords (lower-left corner), NOT real
units. placed = positions[:, 0] >= 0, passed in rather than re-derived
(a real-unit conversion would lose the -1 sentinel). Called every step by
core.step(); sparse vs. dense reward is entirely reward_fn's choice.
For real units, convert with placax_agents.policy.scale.to_real_centers,
or use placax_agents.training.reward.make_scaled_hpwl_reward directly."""

SizeMap = dict[str, tuple[float, float]]  # instance or cell-type name -> (width, height)
NetPin = tuple[str, float, float]  # (macro_name, x_offset, y_offset) - offset from macro center
Nets = list[list[NetPin]]  # one list of pins per net
PinOffsets = dict[str, dict[str, tuple[float, float]]]  # cell_type -> port_name -> (x, y) offset

OrderFn = Callable[[SizeMap, Nets], list[str]]
"""order_fn(macro_sizes, nets) -> macro names in placement order. Macro
i's name becomes row i everywhere downstream (positions, sizes_array,
padded_pin_idx); see placax.netlist.order for built-ins. Must return
every key of macro_sizes exactly once."""
