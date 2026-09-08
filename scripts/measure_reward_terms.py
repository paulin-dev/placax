"""Reports the per-step magnitude of each reward term on a benchmark, to size --regularity_weight.

make_expert_reward() deliberately does NOT auto-balance its two terms: the HPWL term keeps whatever
units reward_scale leaves it in, while the regularity term is always [0, 1] * regularity_weight. So
"what weight is EXPlace's 0.45?" has no benchmark-independent answer - it depends on the HPWL term's
actual scale here. This script measures that, using a wiremask-greedy rollout (place each macro at
its lowest-HPWL-increase legal cell) as a stand-in for a trained policy's behaviour.

    python -m scripts.measure_reward_terms --benchmark_dir=benchmarks/adaptec1 --macro_budget=128
"""
import argparse
import pathlib
import sys

from placax import _device  # noqa: F401  must precede jax imports
from placax.core import reset
from placax.extras.masks import boundary_mask, occupancy_mask, regularity_cost, regularity_max
from placax.extras.render import render
from placax.extras.rewards import wiremask
from placax.netlist.padding import build_macro_net_index
from placax.types import EnvState
from placax_agents.policy.scale import to_grid_units, to_real_centers
from placax_agents.experiment.build import build_benchmark
from placax_agents.experiment.presets import maskplace
from placax_agents.experiment.registry import MASKPLACE_REWARD_DIVISOR

import jax
import jax.numpy as jnp
import numpy as np


def _greedy_wiremask_rollout(benchmark):
    """Places every macro at its lowest-HPWL-increase legal cell, returning (positions, per-step HPWL delta)."""
    params, sizes_array, cell_size = benchmark.params, benchmark.sizes_array, benchmark.cell_size
    macro_net_idx, macro_net_offset, macro_net_valid = build_macro_net_index(
        benchmark.padded_pin_idx, benchmark.padded_pin_offset, benchmark.valid_mask, n_macros=params.n_macros
    )
    grid_sizes = to_grid_units(sizes_array, cell_size)

    def scan_step(state, _):
        # The wiremask is exactly the per-cell HPWL increase, so its value at the chosen cell IS
        # the dense reward's raw delta - no separate reward evaluation needed.
        real_state = EnvState(positions=to_real_centers(state.positions, sizes_array, cell_size), step=state.step)
        wm = wiremask(
            real_state, params, benchmark.padded_pin_idx, benchmark.padded_pin_offset, benchmark.valid_mask,
            macro_net_idx, macro_net_offset, macro_net_valid, sizes_array, cell_size=cell_size,
        )
        canvas = render(state.positions, grid_sizes, params.grid_x, params.effective_grid_y)
        size = grid_sizes[state.step]
        illegal = occupancy_mask(canvas, (size[0], size[1])) | boundary_mask(params, (size[0], size[1]))
        flat = jnp.argmin(jnp.where(illegal, jnp.inf, wm).ravel())
        action = jnp.array([flat // params.effective_grid_y, flat % params.effective_grid_y])
        positions = state.positions.at[state.step].set(action)
        return EnvState(positions=positions, step=state.step + 1), wm[action[0], action[1]]

    final_state, deltas = jax.lax.scan(scan_step, reset(params), jnp.arange(params.n_macros))
    return final_state.positions, deltas


def _regularity_costs(benchmark, positions, mode):
    """Normalized [0, 1] regularity cost at each macro's chosen cell."""
    grid_sizes = to_grid_units(benchmark.sizes_array, benchmark.cell_size)

    def one(size, pos):
        cost = regularity_cost(benchmark.params, size, pos, benchmark.cell_size, mode)
        scale = regularity_max(benchmark.params, size, benchmark.cell_size, mode)
        return jnp.where(scale > 0, cost / jnp.where(scale > 0, scale, 1.0), 0.0)

    return jax.vmap(one)(grid_sizes, positions)


def _describe(name, values):
    v = np.asarray(values, dtype=np.float64)
    print(f"  {name:<28} mean {np.abs(v).mean():>12.4f}   median {np.median(np.abs(v)):>12.4f}   max {np.abs(v).max():>12.4f}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--benchmark_dir", type=pathlib.Path, default=pathlib.Path("benchmarks/adaptec1"))
    parser.add_argument("--macro_budget", default="128")
    parser.add_argument("--regularity_mode", choices=("corner", "edge"), default="corner")
    args = parser.parse_args()
    if not args.benchmark_dir.exists():
        print(f"'{args.benchmark_dir}' not found - run scripts/download_benchmarks.py first.", file=sys.stderr)
        sys.exit(1)

    macro_budget = None if args.macro_budget.lower() == "all" else int(args.macro_budget)
    # Through the config, like every other consumer - not a training script's private helper.
    benchmark = build_benchmark(maskplace(args.benchmark_dir, macro_budget=macro_budget))
    reward_scale = 1.0 / (benchmark.cell_size * MASKPLACE_REWARD_DIVISOR)
    print(
        f"{args.benchmark_dir.name}: {len(benchmark.macro_sizes)} macros, grid {benchmark.params.grid_x}"
        f"x{benchmark.params.effective_grid_y}, cell_size {benchmark.cell_size:.2f}, "
        f"reward_scale {reward_scale:.3e}"
    )

    positions, hpwl_deltas = _greedy_wiremask_rollout(benchmark)
    reg = _regularity_costs(benchmark, positions, args.regularity_mode)

    print("\nper-step magnitudes under a wiremask-greedy rollout:")
    _describe("HPWL reward term", np.asarray(hpwl_deltas) * reward_scale)
    _describe(f"regularity cost ({args.regularity_mode}, [0,1])", reg)

    # EXPlace weights regularity at 0.45 and wire at 0.15, i.e. 3x - but only after normalizing
    # both to a common scale, which is what this ratio reconstructs for placax's units.
    hpwl_mean = float(np.abs(np.asarray(hpwl_deltas) * reward_scale).mean())
    reg_mean = float(np.abs(np.asarray(reg)).mean())
    parity = hpwl_mean / reg_mean if reg_mean > 0 else float("nan")
    print(
        f"\n  --regularity_weight={parity:.3f}  makes the two terms equal on average"
        f"\n  --regularity_weight={parity * 3.0:.3f}  matches EXPlace's own 3:1 regularity:wire ratio"
    )
    # Under a greedy rollout most macros land at near-zero HPWL increase and a handful cost a lot,
    # so the mean sits well above the median. The mean is still the right statistic for balancing
    # episode *returns* (a sum over steps), but the gap is worth seeing before trusting one number.
    hpwl_abs = np.abs(np.asarray(hpwl_deltas) * reward_scale)
    skew = hpwl_abs.mean() / max(np.median(hpwl_abs), 1e-12)
    if skew > 2.0:
        print(
            f"\n  note: the HPWL term is heavily skewed (mean/median = {skew:.1f}x) - most macros cost"
            f"\n  almost nothing and a few dominate. Sweep around the suggestion rather than trusting it."
        )


if __name__ == "__main__":
    main()
