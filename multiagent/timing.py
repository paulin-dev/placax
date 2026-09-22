"""Wall-clock cost of every method, measured the same way on every design.

    python -m multiagent.timing --out=multiagent/runs/timing

A method that needs ten minutes to match one that needs one second is not the same result, and
the swap swarm's whole argument is that its rounds are parallel. Neither claim can be checked
without times, so this module measures them in one place, from the same greedy start, on the same
machine.

**What is timed.** For each design: loading the netlist and building the greedy start (paid by
every method), then the pull rule, the central swap search, the swap swarm, one episode of a
trained policy, and Adam. Simulated annealing sets its own budget, so it is timed by construction
(`anneal.py`) and reported beside these numbers rather than measured here.

**Compiled code is timed twice.** The JAX methods pay a one-off compilation on their first call.
Reporting only the second call would flatter them, and reporting only the first would flatter the
plain-numpy competitors, so both are recorded: `first_s` includes compilation, `repeat_s` does
not. The honest single number for a one-shot use is `first_s`.
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

from multiagent import context, legalize, moves, objective as objective_mod, results, swap, swarm_swap
from multiagent.pull import partner_weights

CHIPS = {"adaptec1": "core", "bigblue1": "core", "ariane133": "die"}
POLICY = pathlib.Path("multiagent/runs/sweep2/m1-s0")


class Timer:
    """Times a call twice: with compilation (first) and without (repeat)."""

    @staticmethod
    def measure(fn, repeats: int = 1) -> dict:
        started = time.perf_counter()
        value = fn()
        first = time.perf_counter() - started
        started = time.perf_counter()
        for _ in range(repeats):
            fn()
        repeat = (time.perf_counter() - started) / repeats
        return {"first_s": round(first, 3), "repeat_s": round(repeat, 3), "value": value}


def time_design(chip: str, canvas: str, policy: pathlib.Path) -> dict:
    out = {}
    started = time.perf_counter()
    ctx = context.build(f"benchmarks/{chip}", canvas=canvas)
    objective = objective_mod.make(ctx)
    out["load_and_greedy_s"] = round(time.perf_counter() - started, 3)

    warm = objective_mod.report(ctx, objective, ctx.warm_start)["real_hpwl_snapped"]
    start = np.asarray(jnp.round(ctx.warm_start), dtype=np.float32)
    score = lambda positions: 1 - objective_mod.report(
        ctx, objective, jnp.asarray(positions, dtype=jnp.float32))["real_hpwl_snapped"] / warm

    def legalized(raw):
        placed, _ = legalize.repair_and_report(ctx, objective, jnp.asarray(raw))
        return np.asarray(placed, dtype=np.float32)

    # 1. The untrained pull rule: 32 steps, then legalization.
    weights = partner_weights(ctx, 4)
    total = weights.sum(axis=1, keepdims=True)
    half = ctx.sizes_grid / 2

    def pull_act(positions, _parts, _progress, _key):
        centers = positions + half
        pull = (weights @ centers) / jnp.maximum(total, 1e-9) - centers
        distance = jnp.linalg.norm(pull, axis=1, keepdims=True)
        return jnp.where(total > 0, pull / jnp.maximum(distance, 1e-9) * jnp.minimum(distance, 1.0), 0.0
                         ).astype(positions.dtype)

    def run_pull():
        final, _ = moves.rollout(ctx.warm_start, jax.random.PRNGKey(0), pull_act, objective,
                                 ctx.lo, ctx.hi, 32, 32)
        return score(legalized(final))

    out["pull_rule"] = Timer.measure(run_pull)

    # 2. The central swap search and 3. the swap swarm, both on the rounded greedy start.
    exact = swarm_swap.ExactGain(ctx)
    out["swap_search"] = Timer.measure(lambda: score(swap.swap_descent(ctx, start)[0]))
    out["swap_swarm"] = Timer.measure(
        lambda: score(swarm_swap.swarm(ctx, start, exact, k=0, max_rounds=200, exact=exact,
                                       resolve=True)["positions"]))

    # 4. One episode of the trained policy (transfer on designs it never saw), then legalization.
    if policy.exists():
        nudge = swarm_swap.nudger(ctx, objective, policy)
        out["policy_episode"] = Timer.measure(lambda: score(nudge(start)))

    # 5. Adam on the positions: the same loop adam.py runs, at its published setting.
    optimizer = optax.adam(0.05)

    @jax.jit
    def adam_steps(positions):
        def step(carry, _):
            positions, state = carry
            grads = jax.grad(objective.total)(positions)
            updates, state = optimizer.update(grads, state, positions)
            return (jnp.clip(optax.apply_updates(positions, updates), ctx.lo, ctx.hi), state), None
        (positions, _), _ = jax.lax.scan(step, (positions, optimizer.init(positions)), None, length=2000)
        return positions

    out["adam_2000_steps"] = Timer.measure(lambda: score(legalized(adam_steps(ctx.warm_start))))
    return out


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=pathlib.Path, default=results.RUNS / "timing")
    parser.add_argument("--policy", type=pathlib.Path, default=POLICY)
    args = parser.parse_args(argv)

    measured = {}
    for chip, canvas in CHIPS.items():
        print(f"timing {chip} ...", flush=True)
        measured[chip] = time_design(chip, canvas, args.policy)
        for name, entry in measured[chip].items():
            if isinstance(entry, dict):
                print(f"  {name:18s} {entry['first_s']:7.2f}s first, {entry['repeat_s']:7.2f}s repeat"
                      f"   -> {entry['value']:+.2%}", flush=True)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "summary.json").write_text(json.dumps({"designs": measured}, indent=2))
    print(f"written to {args.out}")


if __name__ == "__main__":
    main()
