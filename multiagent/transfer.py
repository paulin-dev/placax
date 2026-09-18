"""Runs a trained policy on a design it never saw. The one claim Adam cannot match.

    python -m multiagent.transfer --run=multiagent/runs/adaptec1-m1-s0 \\
        --benchmark_dir=benchmarks/bigblue1

Adam optimizes one placement and knows nothing afterwards: a new design means starting over.
A shared per-macro policy is a rule - "given what I can see around me, move this way" - so it can
be applied to another netlist with no training at all, in a single rollout. If it improves
bigblue1's greedy placement after being trained only on adaptec1, that is a result direct
optimization structurally cannot produce, and it is the strongest argument this line of work has.

**Why the shapes line up for free.** The policy reads a fixed-size per-macro view, not the canvas,
so the network does not depend on how many macros a design has or how large its die is. That is
also why every feature in `view.py` is normalized to the canvas and every neighbour position is
relative: a policy trained on one design's absolute coordinates would transfer to nothing.

**What has to match, and is checked here:** the view level and `k_neighbors` (they set the input
width) and the hidden widths (they are the weights' shape). Everything else - design, macro count,
grid, canvas - is free, which is the point. The objective's weights default to the training run's,
so the transferred run is scored exactly the way the trained one was.
"""
import argparse
import json
import pathlib
import pickle
import time

from placax import _device  # noqa: F401  must precede jax imports

import jax
import jax.numpy as jnp
import numpy as np

from multiagent import context, legalize, moves, objective as objective_mod, view as view_mod
from multiagent.policy import SharedMacroPolicy, make_act


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--run", type=pathlib.Path, required=True,
                        help="A completed multiagent.train run directory - its manifest supplies "
                             "the architecture and its best_params.pkl the weights.")
    parser.add_argument("--benchmark_dir", type=pathlib.Path, required=True,
                        help="The design to place. The whole point is that this is NOT the design "
                             "the policy was trained on.")
    parser.add_argument("--grid", type=int, default=None, help="Default: the training run's grid.")
    parser.add_argument("--macro_budget", type=int, default=None,
                        help="Default: the training run's budget. 0 or negative means every macro.")
    parser.add_argument("--canvas", default=None, choices=("die", "core"))
    parser.add_argument("--steps", type=int, default=None,
                        help="Moves in the rollout. Default: the training run's episode length.")
    parser.add_argument("--out", type=pathlib.Path, default=None)
    return parser.parse_args(argv)


def main(argv=None) -> None:
    args = parse_args(argv)
    manifest = json.loads((args.run / "manifest.json").read_text())
    if manifest.get("method") != "short_horizon":
        raise SystemExit(f"{args.run} is a {manifest.get('method')} run; only a trained policy "
                         f"has weights to transfer.")
    trained = manifest["args"]
    params_path = args.run / "best_params.pkl"
    if not params_path.exists():
        raise SystemExit(f"{params_path} does not exist - that run never produced a legal "
                         f"placement, so it has no weights worth transferring.")

    # Everything not overridden on the command line comes from the training run, so a transfer is
    # described by two paths and nothing else.
    grid = args.grid or int(trained["grid"])
    budget = int(trained["macro_budget"] if args.macro_budget is None else args.macro_budget)
    steps = args.steps or int(trained["steps"])
    out = args.out or pathlib.Path("multiagent/runs") / (
        f"transfer-{args.run.name}-to-{args.benchmark_dir.name}"
    )
    if (out / "summary.json").exists():
        raise SystemExit(f"{out} already holds a transfer result; pass --out with a new path.")
    out.mkdir(parents=True, exist_ok=True)

    ctx = context.build(
        args.benchmark_dir, grid=grid, macro_budget=budget if budget > 0 else None,
        canvas=args.canvas or trained["canvas"], k_neighbors=int(trained["k_neighbors"]),
    )
    objective = objective_mod.make(
        ctx, density_weight=float(trained["density_weight"]),
        target_density=float(trained["target_density"]),
        gamma_cells=float(trained["gamma_cells"]),
        overlap_weight=float(trained.get("overlap_weight", 0.0)),
    )
    view_fn, n_features = view_mod.make(ctx, trained["view"])

    with params_path.open("rb") as handle:
        variables = jax.tree_util.tree_map(jnp.asarray, pickle.load(handle))
    policy = SharedMacroPolicy(
        features=tuple(int(width) for width in str(trained["hidden"]).split(",") if width)
    )
    # The first layer's kernel says what the policy was trained to read. A mismatch here is a
    # silently wrong experiment, so it is a hard error rather than a broadcast.
    trained_features = variables["params"]["Dense_0"]["kernel"].shape[0]
    if trained_features != n_features:
        raise SystemExit(
            f"the policy reads {trained_features} features per macro and this design's "
            f"{trained['view']} view produces {n_features}. Same --view and --k_neighbors are "
            f"required for a transfer; everything else may differ."
        )

    warm = objective_mod.report(ctx, objective, ctx.warm_start)
    print(f"policy trained on {pathlib.Path(trained['benchmark_dir']).name}, applied to "
          f"{args.benchmark_dir.name} ({ctx.n_macros} macros, {grid} grid) with NO retraining")
    print(f"warm start (greedy wiremask): real_hpwl={warm['real_hpwl_snapped']:,.0f}  "
          f"legal={warm['is_legal']}")

    started = time.perf_counter()

    rule = make_act(policy, view_fn, trained, ctx.connection_weights)

    def act(positions, parts, progress, key):
        return rule(variables, positions, parts, progress, key, False)

    positions, costs = jax.jit(lambda: moves.rollout(
        ctx.warm_start, jax.random.PRNGKey(0), act, objective, ctx.lo, ctx.hi, steps,
        int(trained["horizon"]),
    ))()
    metrics = objective_mod.report(ctx, objective, positions)
    repaired_positions, repaired = legalize.repair_and_report(ctx, objective, positions)
    elapsed = time.perf_counter() - started

    hpwl_after = repaired["repaired_real_hpwl_snapped"]
    improvement = 1.0 - hpwl_after / warm["real_hpwl_snapped"]
    print(f"after {steps} moves: raw={metrics['real_hpwl_snapped']:,.0f} "
          f"(overlap {metrics['overlap_ratio']:.2%})  ->  repaired={hpwl_after:,.0f} "
          f"({improvement:+.2%})  legal={repaired['repaired_is_legal']}  in {elapsed:.1f}s")
    print("\nthe comparison this run exists for: Adam on THIS design, from scratch, at a matched\n"
          "move budget - python -m multiagent.adam "
          f"--benchmark_dir={args.benchmark_dir} --steps={steps}")

    np.save(out / "positions.npy", repaired_positions)
    (out / "summary.json").write_text(json.dumps({
        "kind": "multiagent.transfer",
        "trained_on": trained["benchmark_dir"],
        "trained_run": str(args.run),
        "applied_to": str(args.benchmark_dir),
        "view": trained["view"],
        "steps": steps,
        "n_macros": ctx.n_macros,
        "warm_start": warm,
        "raw": metrics,
        "repaired": repaired,
        "hpwl_improvement": improvement,
        "final_cost": float(costs[-1]),
        "wall_clock_s": round(elapsed, 2),
    }, indent=2))
    print(f"written to {out}")


if __name__ == "__main__":
    main()
