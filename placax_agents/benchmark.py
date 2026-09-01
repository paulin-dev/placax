"""Bundles a loaded netlist into everything training needs."""
import pathlib
from collections.abc import Callable
from dataclasses import dataclass

from placax.core import reset  # must precede jax imports
from placax.netlist import load_netlist
from placax.netlist.budget import freeze_order, truncate_to_budget
from placax.netlist.order import alphabetical_order
from placax.netlist.padding import build_padded_arrays
from placax.types import EnvParams, Nets, OrderFn, RewardFn, SizeMap
from placax_agents.policy.observation import observation
from placax_agents.policy.scale import compute_grid_scale
from placax_agents.training.reward import make_scaled_hpwl_reward

import jax

RewardFnFactory = Callable[[jax.Array, jax.Array, jax.Array, jax.Array, float], RewardFn]
"""Builds a RewardFn from padded pin/size arrays and cell size; result expects grid-unit centers."""


@dataclass(frozen=True)
class Benchmark:
    """A netlist loaded and ready for training/eval."""

    macro_sizes: SizeMap
    nets: Nets
    params: EnvParams
    sizes_array: jax.Array
    cell_size: float
    reward_fn: RewardFn
    padded_pin_idx: jax.Array
    padded_pin_offset: jax.Array
    valid_mask: jax.Array

    @staticmethod
    def _load_and_truncate(
        benchmark_dir: pathlib.Path, order_fn: OrderFn, macro_budget: int | None
    ) -> tuple[SizeMap, Nets, OrderFn]:
        """Loads the raw netlist and, if macro_budget is set, truncates and freezes the ordering used."""
        # Load everything first...
        macro_sizes, nets = load_netlist(benchmark_dir)
        # ...then optionally drop all but the first macro_budget macros (by order_fn's ordering).
        if macro_budget is None:
            return macro_sizes, nets, order_fn
        macro_sizes, nets, order = truncate_to_budget(macro_sizes, nets, macro_budget, order_fn=order_fn)
        return macro_sizes, nets, freeze_order(order)

    @classmethod
    def load(
        cls,
        benchmark_dir: pathlib.Path,
        grid: int = 64,
        make_reward_fn: RewardFnFactory = make_scaled_hpwl_reward,
        order_fn: OrderFn = alphabetical_order,
        macro_budget: int | None = None,
    ) -> "Benchmark":
        """Loads a netlist directory into a fully-built, ready-to-train Benchmark."""
        # 1. Load the netlist, optionally truncated to a macro budget, with its frozen ordering.
        macro_sizes, nets, frozen_order_fn = cls._load_and_truncate(benchmark_dir, order_fn, macro_budget)
        # 2. Pad/index everything into the fixed-shape arrays the JAX code operates on.
        _, sizes_array, padded_pin_idx, padded_pin_offset, valid_mask = build_padded_arrays(
            macro_sizes, nets, order_fn=frozen_order_fn
        )
        # 3. Set up env params and pick a grid cell size that fits all macros at the target utilization.
        params = EnvParams(grid=grid, n_macros=len(macro_sizes))
        cell_size = compute_grid_scale(sizes_array, params.grid_x, params.effective_grid_y)
        # 4. Build the reward function for this specific netlist's wiring.
        reward_fn = make_reward_fn(padded_pin_idx, padded_pin_offset, valid_mask, sizes_array, cell_size)
        return cls(
            macro_sizes, nets, params, sizes_array, cell_size, reward_fn,
            padded_pin_idx, padded_pin_offset, valid_mask,
        )

    def init_policy(self, policy, key: jax.Array):
        """Initializes policy's variables from one fresh observation of this benchmark."""
        # Build a "step 0" observation just to get the right shapes for Flax's init.
        obs0 = observation(reset(self.params), self.params, self.sizes_array, cell_size=self.cell_size)
        return policy.init(key, obs0)
