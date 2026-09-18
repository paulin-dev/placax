"""Trains the shared macro policy by differentiating the placement objective through the episode.

    python -m multiagent.train --benchmark_dir=benchmarks/adaptec1 --view=m1 --iterations=200

**What the update is.** One episode per iteration: every macro nudges itself, `--steps` times, and
the loss is the mean objective over the placements the episode passed through. The gradient of that
loss with respect to the policy's weights is taken straight through the simulator - through
`clip`, through the smoothed wirelength, through the density cost - and truncated every
`--horizon` steps. This is SHAC's short window, and it is what makes 128 cooperating agents
trainable at all: each macro's displacement receives its OWN derivative of the shared objective,
so there is no credit-assignment problem to solve with sampled returns.

**What it is not, stated plainly.** Xu et al.'s SHAC bootstraps each window with a learned value
function, and this has no critic - the window's own costs are the objective. That is a deliberate
simplification for a one-week experiment (a critic is the first thing to add if the horizon turns
out to be the binding constraint), and it is why the log calls the method `short_horizon` rather
than claiming to be SHAC. The comparison against the shipped `shac` agent is still apples to
apples on the axis that matters here - centralized, one macro per step, against decentralized, all
macros per step - because both differentiate the same objective through the same simulator.

**The number to beat is printed before training starts**: the warm start's own real HPWL. An
iteration that does not get below it has not done anything, and `wl_norm` in every log line says
so directly (1.0 is the warm start, 0.98 is 2% better).
"""
import argparse
import json
import pathlib
import pickle
import subprocess
import time

from placax import _device  # noqa: F401  must precede jax imports

import jax
import jax.numpy as jnp
import numpy as np
import optax

from multiagent import context, legalize, moves, objective as objective_mod, view as view_mod
from multiagent.policy import SharedMacroPolicy, displacements


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    env = parser.add_argument_group("environment (must match between compared runs)")
    env.add_argument("--benchmark_dir", type=pathlib.Path, default=pathlib.Path("benchmarks/adaptec1"))
    env.add_argument("--grid", type=int, default=context.DEFAULT_GRID)
    env.add_argument("--macro_budget", type=int, default=context.DEFAULT_MACRO_BUDGET,
                     help="0 or negative means every macro in the design.")
    env.add_argument("--canvas", default="core", choices=("die", "core"))
    env.add_argument("--k_neighbors", type=int, default=4,
                     help="Wired partners each macro sees under --view=m1/m2.")
    env.add_argument("--steps", type=int, default=32, help="Simultaneous moves per episode.")
    env.add_argument("--max_step", type=float, default=1.0,
                     help="Largest displacement per macro per step, in grid cells.")
    env.add_argument("--density_weight", type=float, default=1.0,
                     help="Legality weight at the START of the run.")
    env.add_argument("--density_weight_end", type=float, default=None,
                     help="Legality weight at the END, ramped geometrically from --density_weight. "
                          "Leave unset for a constant weight. A low weight finds a better "
                          "arrangement and a high one makes it legal; see objective.ramp.")
    env.add_argument("--target_density", type=float, default=objective_mod.DEFAULT_TARGET_DENSITY)
    env.add_argument("--gamma_cells", type=float, default=objective_mod.DEFAULT_GAMMA_CELLS)

    agent = parser.add_argument_group("agent (the thing under test)")
    agent.add_argument("--view", default="m1", choices=view_mod.LEVELS,
                       help="What one macro sees. This is the ablation: m0 local only, m1 + wired "
                            "neighbours, m2 + global summary.")
    agent.add_argument("--hidden", default="64,64", help="Shared policy's hidden widths.")
    agent.add_argument("--horizon", type=int, default=8,
                       help="Steps differentiated before the gradient path is cut.")
    agent.add_argument("--iterations", type=int, default=200)
    agent.add_argument("--lr", type=float, default=3e-4)
    agent.add_argument("--max_grad_norm", type=float, default=1.0)
    agent.add_argument("--seed", type=int, default=0)

    out = parser.add_argument_group("output")
    out.add_argument("--eval_every", type=int, default=5,
                     help="Iterations between deterministic (mean-action) evaluations.")
    out.add_argument("--out", type=pathlib.Path, default=None,
                     help="Run directory (default: multiagent/runs/<design>-<view>-s<seed>).")
    return parser.parse_args(argv)


def git_sha() -> str:
    """Best-effort commit id, so a result can be traced back to the code that made it."""
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True,
                              check=True).stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def prepare_output(args: argparse.Namespace) -> pathlib.Path:
    """One directory per run, and it refuses to become a second one - the core's rule, kept here."""
    out = args.out or pathlib.Path("multiagent/runs") / (
        f"{args.benchmark_dir.name}-{args.view}-s{args.seed}"
    )
    if (out / "log.jsonl").exists():
        raise SystemExit(
            f"{out} already holds a run ({out / 'log.jsonl'}). One directory holds one run: pass "
            f"--out with a new path, or delete that one if it was a mistake."
        )
    out.mkdir(parents=True, exist_ok=True)
    return out


def main(argv=None) -> None:
    args = parse_args(argv)
    out = prepare_output(args)

    # 1. The design, the warm start, and the score everything is judged by.
    ctx = context.build(
        args.benchmark_dir, grid=args.grid,
        macro_budget=args.macro_budget if args.macro_budget > 0 else None,
        canvas=args.canvas, k_neighbors=args.k_neighbors,
    )
    objective = objective_mod.make(
        ctx, density_weight=args.density_weight, target_density=args.target_density,
        gamma_cells=args.gamma_cells,
    )
    view_fn, n_features = view_mod.make(ctx, args.view)

    warm = objective_mod.report(ctx, objective, ctx.warm_start)
    print(f"design {args.benchmark_dir.name}: {ctx.n_macros} macros on a {args.grid} grid "
          f"({args.canvas} canvas), {n_features} features per macro under {args.view}")
    print(f"warm start (greedy wiremask): real_hpwl={warm['real_hpwl']:,.0f}  "
          f"legal={warm['is_legal']}  overlap={warm['overlap_ratio']:.2%}  "
          f"legal_norm={warm['legal_norm']:.4f}")
    print(f"THE NUMBER TO BEAT: {warm['real_hpwl']:,.0f}\n")

    # 2. The shared policy: one set of weights, applied to every macro's own view.
    hidden = tuple(int(width) for width in args.hidden.split(",") if width)
    policy = SharedMacroPolicy(features=hidden)
    key = jax.random.PRNGKey(args.seed)
    key, init_key = jax.random.split(key)
    obs0 = view_fn(ctx.warm_start, objective.parts(ctx.warm_start), jnp.float32(0.0))
    variables = policy.init(init_key, obs0)

    optimizer = optax.chain(
        optax.clip_by_global_norm(args.max_grad_norm), optax.adam(args.lr)
    )
    opt_state = optimizer.init(variables)

    def episode(variables, key, density_weight, stochastic: bool):
        """One episode driven by the policy, returning (final positions, per-step costs)."""
        def act(positions, parts, progress, step_key):
            mean_raw, log_std = policy.apply(variables, view_fn(positions, parts, progress))
            return displacements(mean_raw, log_std, step_key, args.max_step, stochastic)

        return moves.rollout(
            ctx.warm_start, key, act, objective, ctx.lo, ctx.hi, args.steps, args.horizon,
            density_weight,
        )

    @jax.jit
    def train_step(variables, opt_state, key, density_weight):
        def loss_fn(variables):
            _final, costs = episode(variables, key, density_weight, stochastic=True)
            # The mean over the episode's placements, not just the last one: a policy that dives
            # and then wanders back up has not learned to hold a good placement.
            return costs.mean(), costs

        (loss, costs), grads = jax.value_and_grad(loss_fn, has_aux=True)(variables)
        updates, opt_state = optimizer.update(grads, opt_state, variables)
        return optax.apply_updates(variables, updates), opt_state, loss, costs

    @jax.jit
    def evaluate(variables, density_weight):
        """The policy's actual recommendation: mean actions, no noise."""
        final, costs = episode(variables, jax.random.PRNGKey(0), density_weight, stochastic=False)
        return final, costs

    # 3. The manifest, written BEFORE training, so a crashed run is still attributable.
    manifest = {
        "kind": "multiagent.train",
        "method": "short_horizon",
        "args": {name: str(value) if isinstance(value, pathlib.Path) else value
                 for name, value in vars(args).items()},
        "git_sha": git_sha(),
        "n_macros": ctx.n_macros,
        "n_features": n_features,
        "cell_size": ctx.benchmark.cell_size,
        "netlist_digest": ctx.benchmark.netlist_digest,
        "objective": {
            "wl0": objective.wl0, "area_bins": objective.area_bins,
            "density_weight": objective.density_weight,
            "target_density": objective.target_density,
            "gamma_cells": objective.gamma_cells,
        },
        "warm_start": warm,
        "metric_meanings": {
            "wl_norm": "smoothed wirelength / the warm start's own; 1.0 = no better than greedy",
            "legal_norm": "overlap + off-canvas area, as a fraction of total macro footprint",
            "real_hpwl": "unsmoothed HPWL of the float placement (experiment.run.score_placement)",
            "real_hpwl_snapped": "the same, after rounding to the grid - what a tool would receive",
            "moves": "simultaneous all-macro moves so far",
            "macro_moves": "moves * n_macros - individual macro displacements, for cost accounting",
        },
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))

    # 4. Train.
    log_path = out / "log.jsonl"
    best = {"real_hpwl_snapped": float("inf"), "iteration": -1}
    started = time.perf_counter()
    with log_path.open("w") as log:
        for iteration in range(1, args.iterations + 1):
            key, step_key = jax.random.split(key)
            # The legality weight for THIS iteration. Traced, not static, so ramping it does not
            # retrigger compilation every iteration.
            weight = objective_mod.ramp(
                args.density_weight, args.density_weight_end or args.density_weight,
                (iteration - 1) / max(args.iterations - 1, 1),
            )
            variables, opt_state, loss, costs = train_step(
                variables, opt_state, step_key, jnp.float32(weight)
            )
            line = {
                "iteration": iteration,
                "density_weight": weight,
                "moves": iteration * args.steps,
                "macro_moves": iteration * args.steps * ctx.n_macros,
                "loss": float(loss),
                "cost_first": float(costs[0]),
                "cost_last": float(costs[-1]),
                "wall_clock_s": round(time.perf_counter() - started, 2),
            }
            if iteration % args.eval_every == 0 or iteration == args.iterations:
                positions, eval_costs = evaluate(variables, jnp.float32(weight))
                metrics = objective_mod.report(ctx, objective, positions)
                repaired_positions, repaired = legalize.repair_and_report(
                    ctx, objective, positions
                )
                line.update({f"eval_{name}": value for name, value in metrics.items()})
                line.update({f"eval_{name}": value for name, value in repaired.items()})
                line["eval_cost_last"] = float(eval_costs[-1])
                hpwl_after = repaired["repaired_real_hpwl_snapped"]
                improvement = 1.0 - hpwl_after / warm["real_hpwl_snapped"]
                line["eval_hpwl_improvement"] = improvement
                print(f"iter {iteration:4d}  loss={float(loss):.4f}  "
                      f"wl_norm={metrics['wl_norm']:.4f}  "
                      f"raw={metrics['real_hpwl_snapped']:,.0f} (overlap {metrics['overlap_ratio']:.2%})"
                      f"  ->  repaired={hpwl_after:,.0f} ({improvement:+.2%})  "
                      f"legal={repaired['repaired_is_legal']}")
                # Best by the metric the paper reports: the REPAIRED placement's HPWL, and only
                # when that placement is actually legal.
                if repaired["repaired_is_legal"] and hpwl_after < best["real_hpwl_snapped"]:
                    best = {**{name.removeprefix("repaired_"): value
                               for name, value in repaired.items()}, "iteration": iteration}
                    np.save(out / "best_positions.npy", repaired_positions)
                    np.save(out / "best_positions_raw.npy", np.asarray(positions))
                    with (out / "best_params.pkl").open("wb") as handle:
                        pickle.dump(jax.tree_util.tree_map(np.asarray, variables), handle)
            log.write(json.dumps(line) + "\n")
            log.flush()

    # 5. What happened, in one place, whatever happened.
    summary = {
        "warm_start": warm,
        "best_legal": best if best["iteration"] >= 0 else None,
        "iterations": args.iterations,
        "wall_clock_s": round(time.perf_counter() - started, 2),
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    if best["iteration"] < 0:
        print("\nno legal placement survived REPAIR, which is a stronger failure than overlap in "
              "the raw output: the greedy repositioning could not find legal spots at all. Check "
              "repair_unplaceable_macros in the log before reading anything into the wirelength.")
    else:
        print(f"\nbest legal placement: iter {best['iteration']}  "
              f"real_hpwl={best['real_hpwl_snapped']:,.0f}  "
              f"vs warm start {warm['real_hpwl_snapped']:,.0f}  "
              f"({1.0 - best['real_hpwl_snapped'] / warm['real_hpwl_snapped']:+.2%})")
    print(f"run written to {out}")


if __name__ == "__main__":
    main()
