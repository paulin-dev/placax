"""Swap same-footprint macros while it shortens the wires. The move the agents cannot make.

    python -m multiagent.swap --benchmark_dir=benchmarks/ariane133 --canvas=die
    python -m multiagent.swap --benchmark_dir=benchmarks/adaptec1 --positions=<run>/best_positions.npy

**Why this baseline exists.** Every method in this directory moves macros continuously, by small
steps. Two macros can then only trade places by passing THROUGH each other, which the overlap
terms forbid - so a placement whose best improvement is a reordering is out of reach for all of
them, however well trained. ariane133 is exactly that: 128 identical 17 x 11-cell SRAMs covering
half the canvas. This does the reordering directly: every step, try every pair of macros with the
same integer footprint, apply the single swap that lowers real HPWL most, and stop when none does.
Swapping two equal footprints leaves the occupied cells unchanged, so a legal placement stays
legal by construction and no repair is involved.

It is deliberately dumb - best-improvement descent, no annealing, no learning - so that anything it
reaches is a floor, not a competitor's best effort.

`--positions` starts from another method's LEGAL output instead of the greedy placement, which asks
whether swaps and continuous nudges are complementary.
"""
import argparse
import json
import pathlib
import time

from placax import _device  # noqa: F401  must precede jax imports
from placax.extras.rewards import hpwl
from placax_agents.policy.scale import to_grid_units, to_real_centers

import jax
import jax.numpy as jnp
import numpy as np

from multiagent import context, objective as objective_mod


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--benchmark_dir", type=pathlib.Path, default=pathlib.Path("benchmarks/adaptec1"))
    parser.add_argument("--grid", type=int, default=context.DEFAULT_GRID)
    parser.add_argument("--macro_budget", type=int, default=context.DEFAULT_MACRO_BUDGET)
    parser.add_argument("--canvas", default="core", choices=("die", "core"))
    parser.add_argument("--positions", type=pathlib.Path, default=None,
                        help="A legal (n_macros, 2) .npy placement to start from. Default: greedy.")
    parser.add_argument("--max_swaps", type=int, default=1000)
    parser.add_argument("--out", type=pathlib.Path, default=None)
    return parser.parse_args(argv)


def swap_descent(ctx, positions: np.ndarray, max_swaps: int = 1000) -> tuple[np.ndarray, int]:
    """Best-improvement pairwise swaps among equal footprints. Returns `(positions, n_swaps)`."""
    benchmark = ctx.benchmark
    footprints = np.asarray(to_grid_units(benchmark.sizes_array, benchmark.cell_size))
    n = ctx.n_macros
    i, j = np.triu_indices(n, k=1)
    same = np.all(footprints[i] == footprints[j], axis=1)
    i, j = i[same], j[same]
    if len(i) == 0:
        return positions, 0

    @jax.jit
    def score_all(pos):
        """HPWL of every one-swap neighbour of `pos`, in one batched call."""
        batch = jnp.repeat(pos[None], len(i), axis=0)
        rows = jnp.arange(len(i))
        batch = batch.at[rows, i].set(pos[j]).at[rows, j].set(pos[i])
        return jax.vmap(lambda p: hpwl(
            to_real_centers(p, benchmark.sizes_array, benchmark.cell_size),
            benchmark.padded_pin_idx, benchmark.padded_pin_offset, benchmark.valid_mask,
        ))(batch)

    current = jnp.asarray(positions, dtype=jnp.float32)
    base = float(hpwl(to_real_centers(current, benchmark.sizes_array, benchmark.cell_size),
                      benchmark.padded_pin_idx, benchmark.padded_pin_offset, benchmark.valid_mask))
    swaps = 0
    while swaps < max_swaps:
        values = np.asarray(score_all(current))
        best = int(values.argmin())
        if values[best] >= base * (1 - 1e-9):
            break
        a, b = int(i[best]), int(j[best])
        current = current.at[a].set(current[b]).at[b].set(current[a])
        base = float(values[best])
        swaps += 1
    return np.asarray(current), swaps


def main(argv=None) -> None:
    args = parse_args(argv)
    ctx = context.build(args.benchmark_dir, grid=args.grid,
                        macro_budget=args.macro_budget if args.macro_budget > 0 else None,
                        canvas=args.canvas)
    objective = objective_mod.make(ctx)
    warm = objective_mod.report(ctx, objective, ctx.warm_start)
    start = (np.load(args.positions).astype(np.float32) if args.positions is not None
             else np.asarray(jnp.round(ctx.warm_start)))
    before = objective_mod.report(ctx, objective, jnp.asarray(start))
    if not before["is_legal"]:
        raise SystemExit(f"{args.positions} is not legal; swap descent keeps legality, it does not "
                         f"create it. Pass a repaired placement (best_positions.npy).")
    started = time.perf_counter()
    final, swaps = swap_descent(ctx, start, args.max_swaps)
    after = objective_mod.report(ctx, objective, jnp.asarray(final))
    elapsed = time.perf_counter() - started
    improvement = 1.0 - after["real_hpwl_snapped"] / warm["real_hpwl_snapped"]
    print(f"{args.benchmark_dir.name}: start {before['real_hpwl_snapped']:,.0f} -> {swaps} swaps -> "
          f"{after['real_hpwl_snapped']:,.0f}  ({improvement:+.2%} vs greedy)  "
          f"legal={after['is_legal']}  in {elapsed:.1f}s")
    if args.out is not None:
        args.out.mkdir(parents=True, exist_ok=True)
        np.save(args.out / "positions.npy", final)
        (args.out / "summary.json").write_text(json.dumps({
            "kind": "multiagent.swap", "benchmark_dir": str(args.benchmark_dir),
            "started_from": str(args.positions) if args.positions else "greedy",
            "swaps": swaps, "warm_start": warm, "before": before, "after": after,
            "hpwl_improvement": improvement, "wall_clock_s": round(elapsed, 2),
        }, indent=2))


if __name__ == "__main__":
    main()
