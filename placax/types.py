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
    """Static config: fixed for a run, can be swept independently of state.

    grid/n_macros determine array shapes, so they're marked pytree_node=False -
    static metadata JAX treats as a compile-time constant, not a traced value.
    A field like a reward weight, which doesn't affect any shape, would stay
    a normal pytree_node=True field and could be vmapped over freely.

    grid_y defaults to None, meaning "same as grid" (a square canvas) -
    every existing EnvParams(grid=64) call keeps working identically.
    Real chip die areas aren't always square, so a genuinely generic
    environment needs to support grid_x != grid_y (EnvParams(grid=128,
    grid_y=64)); grid itself is the x dimension in that case."""

    grid: int = struct.field(pytree_node=False, default=4)
    grid_y: int | None = struct.field(pytree_node=False, default=None)
    n_macros: int = struct.field(pytree_node=False, default=4)

    @property
    def grid_x(self) -> int:
        return self.grid

    @property
    def effective_grid_y(self) -> int:
        return self.grid_y if self.grid_y is not None else self.grid


RewardFn = Callable[[jax.Array], jax.Array]

SizeMap = dict[str, tuple[float, float]]  # instance or cell-type name -> (width, height)
NetPin = tuple[str, float, float]  # (macro_name, x_offset, y_offset) - offset from macro center
Nets = list[list[NetPin]]  # one list of pins per net
PinOffsets = dict[str, dict[str, tuple[float, float]]]  # cell_type -> port_name -> (x, y) offset
