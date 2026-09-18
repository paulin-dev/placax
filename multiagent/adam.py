"""The baseline that decides whether any of this was worth it: optimize the positions directly.

    python -m multiagent.adam --benchmark_dir=benchmarks/adaptec1 --steps=2000

No policy, no observation, no learning. The macro positions ARE the parameters, and Adam walks them
downhill on the same objective the agents are trained on, from the same warm start, under the same
bounds. This is what DREAMPlace's mixed-size placement does in spirit, and it is the honest
competitor for a method whose gradient comes from the same place.

**Why it has to be in every table.** A shared per-macro policy that maps a local view to a
displacement, trained by differentiating the objective, is close enough to gradient descent on the
positions that "we beat PPO" would prove nothing. If Adam reaches the same wirelength, the policy's
only remaining claim is the one Adam cannot make - being reusable on a design it was not optimized
on (`transfer.py`). If Adam beats it outright, that is the result, and it is worth reporting.

**Projected gradient, not a penalty, for the canvas** - `clip` onto the same `[lo, hi]` bounds
`moves.apply_deltas` uses, so both methods are confined to exactly the same feasible set and the
comparison is about the search, not about who was allowed to leave.

**One Adam step is one all-macro move**, which is the unit `train.py` logs as `moves`. So a policy
trained for 200 iterations of 32 steps has driven 6,400 moves, and `--steps=6400` is the
compute-matched row; `--steps=32` is the single-episode row, which is what the policy actually gets
at evaluation time. Both are worth having, and they answer different questions.
"""
import argparse
import json
import pathlib
import time

from placax import _device  # noqa: F401  must precede jax imports

import jax
import jax.numpy as jnp
import numpy as np
import optax

from multiagent import context, legalize, objective as objective_mod
from multiagent.train import git_sha, prepare_output


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    env = parser.add_argument_group("environment (must match the policy runs it is compared to)")
    env.add_argument("--benchmark_dir", type=pathlib.Path, default=pathlib.Path("benchmarks/adaptec1"))
    env.add_argument("--grid", type=int, default=context.DEFAULT_GRID)
    env.add_argument("--macro_budget", type=int, default=context.DEFAULT_MACRO_BUDGET,
                     help="0 or negative means every macro in the design.")
    env.add_argument("--canvas", default="core", choices=("die", "core"))
    env.add_argument("--density_weight", type=float, default=1.0,
                     help="Legality weight at the START of the run.")
    env.add_argument("--density_weight_end", type=float, default=None,
                     help="Legality weight at the END, ramped geometrically. Unset = constant.")
    env.add_argument("--target_density", type=float, default=objective_mod.DEFAULT_TARGET_DENSITY)
    env.add_argument("--gamma_cells", type=float, default=objective_mod.DEFAULT_GAMMA_CELLS)

    search = parser.add_argument_group("search")
    search.add_argument("--steps", type=int, default=2000, help="Adam steps = all-macro moves.")
    search.add_argument("--lr", type=float, default=0.05, help="Step size in grid cells.")
    search.add_argument("--eval_every", type=int, default=50)
    search.add_argument("--out", type=pathlib.Path, default=None)
    # Accepted so a sweep script can pass the same flags to both entry points; unused here, and
    # recorded in the manifest so the run says plainly that they did nothing.
    search.add_argument("--seed", type=int, default=0,
                        help="Recorded only: Adam from a fixed warm start is deterministic.")
    return parser.parse_args(argv)


def main(argv=None) -> None:
    args = parse_args(argv)
    args.view = "adam"  # so the default run directory names itself like the policy runs
    out = prepare_output(args)

    ctx = context.build(
        args.benchmark_dir, grid=args.grid,
        macro_budget=args.macro_budget if args.macro_budget > 0 else None,
        canvas=args.canvas,
    )
    objective = objective_mod.make(
        ctx, density_weight=args.density_weight, target_density=args.target_density,
        gamma_cells=args.gamma_cells,
    )
    warm = objective_mod.report(ctx, objective, ctx.warm_start)
    print(f"warm start (greedy wiremask): real_hpwl={warm['real_hpwl']:,.0f}  "
          f"legal={warm['is_legal']}")
    print(f"THE NUMBER TO BEAT: {warm['real_hpwl']:,.0f}\n")

    optimizer = optax.adam(args.lr)
    positions = ctx.warm_start
    opt_state = optimizer.init(positions)

    @jax.jit
    def step(positions, opt_state, density_weight):
        cost, grads = jax.value_and_grad(objective.total)(positions, density_weight)
        updates, opt_state = optimizer.update(grads, opt_state, positions)
        # Projection onto the same feasible set the policy's moves are clipped to.
        moved = jnp.clip(optax.apply_updates(positions, updates), ctx.lo, ctx.hi)
        return moved, opt_state, cost

    manifest = {
        "kind": "multiagent.adam",
        "method": "adam_on_positions",
        "args": {name: str(value) if isinstance(value, pathlib.Path) else value
                 for name, value in vars(args).items()},
        "git_sha": git_sha(),
        "n_macros": ctx.n_macros,
        "cell_size": ctx.benchmark.cell_size,
        "netlist_digest": ctx.benchmark.netlist_digest,
        "objective": {
            "wl0": objective.wl0, "area_bins": objective.area_bins,
            "density_weight": objective.density_weight,
            "target_density": objective.target_density,
            "gamma_cells": objective.gamma_cells,
        },
        "warm_start": warm,
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))

    best = {"real_hpwl_snapped": float("inf"), "iteration": -1}
    started = time.perf_counter()
    with (out / "log.jsonl").open("w") as log:
        for iteration in range(1, args.steps + 1):
            weight = objective_mod.ramp(
                args.density_weight, args.density_weight_end or args.density_weight,
                (iteration - 1) / max(args.steps - 1, 1),
            )
            positions, opt_state, cost = step(positions, opt_state, jnp.float32(weight))
            line = {
                "iteration": iteration,
                "density_weight": weight,
                "moves": iteration,
                "macro_moves": iteration * ctx.n_macros,
                "cost": float(cost),
                "wall_clock_s": round(time.perf_counter() - started, 2),
            }
            if iteration % args.eval_every == 0 or iteration == args.steps:
                metrics = objective_mod.report(ctx, objective, positions)
                repaired_positions, repaired = legalize.repair_and_report(
                    ctx, objective, positions
                )
                line.update({f"eval_{name}": value for name, value in metrics.items()})
                line.update({f"eval_{name}": value for name, value in repaired.items()})
                hpwl_after = repaired["repaired_real_hpwl_snapped"]
                improvement = 1.0 - hpwl_after / warm["real_hpwl_snapped"]
                line["eval_hpwl_improvement"] = improvement
                snapshots = out / "snapshots"
                snapshots.mkdir(exist_ok=True)
                np.save(snapshots / f"iter_{iteration:05d}.npy", np.asarray(positions))
                print(f"step {iteration:5d}  cost={float(cost):.4f}  "
                      f"wl_norm={metrics['wl_norm']:.4f}  "
                      f"raw={metrics['real_hpwl_snapped']:,.0f} (overlap {metrics['overlap_ratio']:.2%})"
                      f"  ->  repaired={hpwl_after:,.0f} ({improvement:+.2%})  "
                      f"legal={repaired['repaired_is_legal']}")
                if repaired["repaired_is_legal"] and hpwl_after < best["real_hpwl_snapped"]:
                    best = {**{name.removeprefix("repaired_"): value
                               for name, value in repaired.items()}, "iteration": iteration}
                    np.save(out / "best_positions.npy", repaired_positions)
                    np.save(out / "best_positions_raw.npy", np.asarray(positions))
            log.write(json.dumps(line) + "\n")
            log.flush()

    summary = {
        "warm_start": warm,
        "best_legal": best if best["iteration"] >= 0 else None,
        "steps": args.steps,
        "wall_clock_s": round(time.perf_counter() - started, 2),
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    if best["iteration"] < 0:
        print("\nno legal placement survived REPAIR at any checkpoint - check "
              "repair_unplaceable_macros in the log. This is worth chasing BEFORE reading any "
              "policy run, since both methods are scored through the same repair.")
    else:
        print(f"\nbest legal placement: step {best['iteration']}  "
              f"real_hpwl={best['real_hpwl_snapped']:,.0f}  "
              f"({1.0 - best['real_hpwl_snapped'] / warm['real_hpwl_snapped']:+.2%})")
    print(f"run written to {out}")


if __name__ == "__main__":
    main()
