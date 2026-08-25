"""Bundles a loaded netlist into everything training needs."""
import pathlib
from collections.abc import Callable
from dataclasses import dataclass

from placax.core import reset  # must precede jax imports
from placax.netlist import load_netlist
from placax.netlist.padding import build_padded_arrays
from placax.types import EnvParams, Nets, RewardFn, SizeMap
from placax_agents.policy.observation import observation
from placax_agents.policy.scale import compute_grid_scale
from placax_agents.training.reward import make_scaled_hpwl_reward

import jax

RewardFnFactory = Callable[[jax.Array, jax.Array, jax.Array, jax.Array, float], RewardFn]


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

    @classmethod
    def load(
        cls,
        benchmark_dir: pathlib.Path,
        grid: int = 64,
        make_reward_fn: RewardFnFactory = make_scaled_hpwl_reward,
    ) -> "Benchmark":
        """Loads benchmark_dir (any supported netlist format). Pass
        make_reward_fn for a different reward - same signature as
        make_scaled_hpwl_reward."""
        macro_sizes, nets = load_netlist(benchmark_dir)  # raw, name-keyed
        # Pad/index into the fixed-shape arrays JAX code operates on.
        _, sizes_array, padded_pin_idx, padded_pin_offset, valid_mask = build_padded_arrays(
            macro_sizes, nets
        )
        params = EnvParams(grid=grid, n_macros=len(macro_sizes))
        cell_size = compute_grid_scale(sizes_array, params.grid_x, params.effective_grid_y)
        reward_fn = make_reward_fn(padded_pin_idx, padded_pin_offset, valid_mask, sizes_array, cell_size)
        return cls(
            macro_sizes, nets, params, sizes_array, cell_size, reward_fn,
            padded_pin_idx, padded_pin_offset, valid_mask,
        )

    def init_policy(self, policy, key: jax.Array):
        """variables for policy (any nn.Module taking an obs dict), from
        one fresh observation of this benchmark."""
        obs0 = observation(reset(self.params), self.params, self.sizes_array)
        return policy.init(key, obs0)
