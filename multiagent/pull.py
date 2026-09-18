"""The untrained control: every macro steps toward the weighted centre of its wired partners.

    python -m multiagent.pull --benchmark_dir=benchmarks/adaptec1 --steps=32

The question it answers is whether the trained policy learned anything beyond the obvious rule.
Wirelength is a sum over pairs, so "move toward whoever you are wired to" is the first thing any
per-macro rule would discover - and the training curves show the policy contracting the placement
steadily. If this hand-written rule, through the SAME rollout, the SAME step bound and the SAME
repair, matches the policy, the learning added nothing; if it falls well short, it added something.

`--k` sets which partners count: `4` is exactly what an `m1` macro sees (its top-4 by clique
weight), `0` is every macro it shares a net with - more than any trained view is given.
"""
import argparse
import json
import pathlib

from placax import _device  # noqa: F401  must precede jax imports

import jax
import jax.numpy as jnp
import numpy as np

from multiagent import context, legalize, moves, objective as objective_mod


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--benchmark_dir", type=pathlib.Path, default=pathlib.Path("benchmarks/adaptec1"))
    parser.add_argument("--grid", type=int, default=context.DEFAULT_GRID)
    parser.add_argument("--macro_budget", type=int, default=context.DEFAULT_MACRO_BUDGET)
    parser.add_argument("--canvas", default="core", choices=("die", "core"))
    parser.add_argument("--steps", type=int, nargs="+", default=[32],
                        help="Episode lengths to evaluate; 32 is the policy's.")
    parser.add_argument("--max_step", type=float, default=1.0)
    parser.add_argument("--k", type=int, default=4,
                        help="Partners that count: 4 = what m1 sees, 0 = every wired macro.")
    parser.add_argument("--density_weight", type=float, default=1.0)
    parser.add_argument("--out", type=pathlib.Path, default=None)
    return parser.parse_args(argv)


def partner_weights(ctx, k: int) -> jnp.ndarray:
    """`(n, n)` pull weights: the full clique matrix, or only each macro's top-k row entries."""
    weights = np.asarray(ctx.connection_weights, dtype=np.float32)
    if k > 0:
        keep = np.zeros_like(weights, dtype=bool)
        idx = np.asarray(ctx.neighbor_idx)[:, :k]
        valid = np.asarray(ctx.neighbor_valid)[:, :k]
        rows = np.repeat(np.arange(len(weights))[:, None], idx.shape[1], axis=1)
        keep[rows[valid], idx[valid]] = True
        weights = np.where(keep, weights, 0.0)
    return jnp.asarray(weights)


def main(argv=None) -> None:
    args = parse_args(argv)
    ctx = context.build(args.benchmark_dir, grid=args.grid,
                        macro_budget=args.macro_budget if args.macro_budget > 0 else None,
                        canvas=args.canvas, k_neighbors=max(args.k, 4))
    objective = objective_mod.make(ctx, density_weight=args.density_weight)
    warm = objective_mod.report(ctx, objective, ctx.warm_start)
    weights = partner_weights(ctx, args.k)
    total = weights.sum(axis=1, keepdims=True)
    half = ctx.sizes_grid / 2

    def act(positions, _parts, _progress, _key):
        centers = positions + half
        target = (weights @ centers) / jnp.maximum(total, 1e-9)
        pull = target - centers
        distance = jnp.linalg.norm(pull, axis=1, keepdims=True)
        # A full step toward the target, or exactly onto it if it is closer than one step; a macro
        # with no partners stays put.
        step = pull / jnp.maximum(distance, 1e-9) * jnp.minimum(distance, args.max_step)
        return jnp.where(total > 0, step, 0.0).astype(positions.dtype)

    results = []
    print(f"warm start: {warm['real_hpwl_snapped']:,.0f}   pulling toward "
          f"{'every wired partner' if args.k <= 0 else f'the top-{args.k} partners (= m1)'}")
    for steps in args.steps:
        final, _costs = jax.jit(lambda: moves.rollout(
            ctx.warm_start, jax.random.PRNGKey(0), act, objective, ctx.lo, ctx.hi, steps, steps,
        ))()
        raw = objective_mod.report(ctx, objective, final)
        _repaired_positions, repaired = legalize.repair_and_report(ctx, objective, final)
        hpwl = repaired["repaired_real_hpwl_snapped"]
        improvement = 1.0 - hpwl / warm["real_hpwl_snapped"]
        print(f"steps {steps:4d}  raw={raw['real_hpwl_snapped']:,.0f} "
              f"(overlap {raw['overlap_ratio']:.1%})  ->  repaired={hpwl:,.0f} ({improvement:+.2%})"
              f"  legal={repaired['repaired_is_legal']}  "
              f"repair move {repaired['repair_mean_displacement_cells']:.1f} cells")
        results.append({"steps": steps, "raw": raw, "repaired": repaired,
                        "hpwl_improvement": improvement})

    if args.out is not None:
        args.out.mkdir(parents=True, exist_ok=True)
        (args.out / "summary.json").write_text(json.dumps({
            "kind": "multiagent.pull", "k": args.k, "benchmark_dir": str(args.benchmark_dir),
            "warm_start": warm, "results": results,
        }, indent=2))


if __name__ == "__main__":
    main()
