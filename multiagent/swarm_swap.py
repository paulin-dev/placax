"""Macros that swap places themselves: a leaderless swap swarm, exact or learned.

    python -m multiagent.swarm_swap run --benchmark_dir=benchmarks/ariane133 --canvas=die --k=8
    python -m multiagent.swarm_swap train --view=m1all --seed=0 --out=multiagent/runs/swapnet-m1all-s0
    python -m multiagent.swarm_swap run --scorer=multiagent/runs/swapnet-m1all-s0 --benchmark_dir=benchmarks/bigblue1

**Why.** Every continuous method here - the nudging agents, Adam, the boids variants - is blind
to one move: two blocks trading places, which small steps can't do without passing through each
other. `swap.py` makes that move centrally (one best swap per step, chosen by looking at every
pair) and beats every continuous method. This asks whether the AGENTS can make it:

* **Local candidates.** A macro only considers its `k` nearest macros with the same integer
  footprint (so a swap keeps the placement legal by construction).
* **Mutual consent, all at once.** Each macro names the candidate it most wants to trade with; a
  swap happens when two macros name each other. Every agreed swap in a round is applied at the
  same time - no coordinator, no global ordering. Swaps that share nets can interfere, so a round
  can make the chip worse; the swarm isn't told.
* **Two ways to judge a swap.** `exact`: the macro computes the true HPWL change of the nets it
  and its candidate sit on - local information, but exact pin geometry. `learned`: one shared
  network reads only the same view the nudging policy reads (`view.py`: m0 / m1 / m1all, for both
  macros, plus their relative position) and predicts that change. Trained on adaptec1, applied to
  any design without retraining - Q2 and Q4 again, for swaps instead of nudges.
"""
import argparse
import json
import pathlib
import pickle
import time

from placax import _device  # noqa: F401  must precede jax imports
from placax.extras.rewards import hpwl
from placax_agents.policy.scale import to_grid_units, to_real_centers

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax import linen as nn

from multiagent import context, objective as objective_mod, view as view_mod


# ----------------------------------------------------------------------------------------------
# Candidates and exact gains


def candidates(ctx, positions: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    """`(n, k)` indices of each macro's k nearest same-footprint macros, and which slots are real.

    `k <= 0` means every same-footprint macro. Nearest by centre distance in the CURRENT placement,
    so a macro's candidates change as the swarm rearranges itself.
    """
    footprints = np.asarray(to_grid_units(ctx.benchmark.sizes_array, ctx.benchmark.cell_size))
    same = np.all(footprints[:, None, :] == footprints[None, :, :], axis=-1)
    np.fill_diagonal(same, False)
    centers = positions + footprints / 2.0
    distance = np.linalg.norm(centers[:, None, :] - centers[None, :, :], axis=-1)
    distance = np.where(same, distance, np.inf)
    width = int(same.sum(axis=1).max()) if k <= 0 else k
    width = max(width, 1)
    order = np.argsort(distance, axis=1)[:, :width]
    valid = np.isfinite(np.take_along_axis(distance, order, axis=1))
    return order, valid


class ExactGain:
    """HPWL reduction of every candidate swap, in one batched call per round."""

    def __init__(self, ctx):
        benchmark = ctx.benchmark

        def total(pos):
            return hpwl(to_real_centers(pos, benchmark.sizes_array, benchmark.cell_size),
                        benchmark.padded_pin_idx, benchmark.padded_pin_offset, benchmark.valid_mask)

        self.total = jax.jit(total)

        @jax.jit
        def gains(pos, first, second):
            batch = jnp.repeat(pos[None], first.shape[0], axis=0)
            rows = jnp.arange(first.shape[0])
            batch = batch.at[rows, first].set(pos[second]).at[rows, second].set(pos[first])
            return total(pos) - jax.vmap(total)(batch)

        self._gains = gains

    def __call__(self, positions: np.ndarray, idx: np.ndarray, valid: np.ndarray) -> np.ndarray:
        n, k = idx.shape
        first = np.repeat(np.arange(n), k)
        second = idx.ravel()
        out = np.asarray(self._gains(jnp.asarray(positions, dtype=jnp.float32),
                                     jnp.asarray(first), jnp.asarray(second))).reshape(n, k)
        return np.where(valid, out, -np.inf)


# ----------------------------------------------------------------------------------------------
# The learned judge


class SwapScorer(nn.Module):
    """`(pairs, features) -> (pairs,)` predicted gain, in units of the placement's mean per-macro
    HPWL - scale-free, so one scorer can be applied to designs of any size."""

    features: tuple[int, ...] = (64, 64)

    @nn.compact
    def __call__(self, x):
        for width in self.features:
            x = nn.tanh(nn.Dense(width)(x))
        return nn.Dense(1)(x)[..., 0]


def pair_features(ctx, view_fn, positions: np.ndarray, idx: np.ndarray) -> np.ndarray:
    """`(n, k, 2F + 2)`: my view, my candidate's view, and where it sits relative to me.

    Only what the nudging policy could already see. In particular NO wirelength: under m1/m1all a
    macro knows where its partners are (m1all: their weighted centre), so it can in principle work
    out whether it would be closer to them at the candidate's spot - that is the thing to learn.
    """
    pos = jnp.asarray(positions, dtype=jnp.float32)
    views = np.asarray(view_fn(pos, None, jnp.float32(0.0)))           # (n, F)
    canvas = np.asarray(ctx.canvas)
    rel = (positions[idx] - positions[:, None, :]) / canvas            # (n, k, 2)
    mine = np.repeat(views[:, None, :], idx.shape[1], axis=1)
    theirs = views[idx]
    return np.concatenate([mine, theirs, rel], axis=-1).astype(np.float32)


class LearnedGain:
    """The shared scorer applied to every candidate pair: `(positions, idx, valid) -> scores`."""

    def __init__(self, ctx, run_dir: pathlib.Path):
        meta = json.loads((run_dir / "scorer.json").read_text())
        self.view_fn, n_features = view_mod.make(ctx, meta["view"])
        if 2 * n_features + 2 != meta["n_inputs"]:
            raise SystemExit(f"scorer reads {meta['n_inputs']} inputs; {meta['view']} on this "
                             f"design gives {2 * n_features + 2}")
        with (run_dir / "scorer.pkl").open("rb") as handle:
            self.params = jax.tree_util.tree_map(jnp.asarray, pickle.load(handle))
        self.model = SwapScorer(tuple(meta["hidden"]))
        self.apply = jax.jit(self.model.apply)
        self.ctx = ctx

    def __call__(self, positions, idx, valid):
        x = pair_features(self.ctx, self.view_fn, positions, idx)
        out = np.asarray(self.apply(self.params, jnp.asarray(x)))
        return np.where(valid, out, -np.inf)


# ----------------------------------------------------------------------------------------------
# The swarm


def _independent(pairs, scores, wired):
    """Keeps an agreed swap only if it outscores every other agreed swap it is wired to.

    Measured reason: with every agreed swap applied at once, adaptec1 got 15% WORSE - its
    same-size macros share buses, and each macro scored its swap assuming the others stay put.
    This is the local fix from distributed algorithms (a Luby-style local maximum): a swap waits
    if a wired partner is part of a better one this round. Every macro consults only its own
    partners, so it stays leaderless; the kept swaps touch disjoint nets and their gains add.
    """
    if not pairs:
        return pairs
    members = [np.array(p) for p in pairs]
    kept = []
    for a, (pa, sa) in enumerate(zip(members, scores)):
        best = True
        for b, (pb, sb) in enumerate(zip(members, scores)):
            if a != b and wired[np.ix_(pa, pb)].any() and (sb > sa or (sb == sa and b < a)):
                best = False
                break
        if best:
            kept.append(pairs[a])
    return kept


def swarm(ctx, positions: np.ndarray, judge, k: int = 8, max_rounds: int = 60,
          exact: ExactGain | None = None, resolve: bool = False, threshold: float = 0.0) -> dict:
    """Rounds of mutual-consent swaps until nobody wants to trade. Returns the trajectory.

    `judge(positions, idx, valid) -> (n, k)` scores; a macro wants its highest-scoring candidate
    if that score is positive. `resolve` applies `_independent` to the agreed swaps. `exact`
    measures the true HPWL after each round, for the record - the agents never see it.
    """
    wired = np.asarray(ctx.connection_weights) > 0
    exact = exact or ExactGain(ctx)
    pos = np.asarray(positions, dtype=np.float32).copy()
    hpwls = [float(exact.total(jnp.asarray(pos)))]
    swaps_per_round = []
    path, swapped = [pos.copy()], [[]]
    seen = {pos.tobytes()}
    for _ in range(max_rounds):
        idx, valid = candidates(ctx, pos, k)
        scores = judge(pos, idx, valid)
        best = scores.argmax(axis=1)
        wants = np.where(scores[np.arange(len(pos)), best] > threshold, idx[np.arange(len(pos)), best], -1)
        pairs = [(i, int(j)) for i, j in enumerate(wants) if j > i and wants[j] == i]
        if resolve:
            chosen = scores[np.arange(len(pos)), best]
            pairs = _independent(pairs, [chosen[i] + chosen[j] for i, j in pairs], wired)
        if not pairs:
            break
        new = pos.copy()
        for i, j in pairs:
            new[i], new[j] = pos[j], pos[i]
        pos = new
        swaps_per_round.append(len(pairs))
        hpwls.append(float(exact.total(jnp.asarray(pos))))
        path.append(pos.copy()); swapped.append(pairs)
        key = pos.tobytes()
        if key in seen:        # the swarm is cycling between arrangements it has already been in
            break
        seen.add(key)
    return {"positions": pos, "hpwl": hpwls, "swaps_per_round": swaps_per_round,
            "path": path, "swapped": swapped}


# ----------------------------------------------------------------------------------------------
# Training the judge on adaptec1


def training_states(ctx, rng, exact, n_random=24, k=8):
    """Placements to learn from: greedy, greedy with random swaps, and swarm trajectories."""
    footprints = np.asarray(to_grid_units(ctx.benchmark.sizes_array, ctx.benchmark.cell_size))
    groups = [np.flatnonzero(np.all(footprints == f, axis=1)) for f in np.unique(footprints, axis=0)]
    groups = [g for g in groups if len(g) > 1]
    weights = np.array([len(g) for g in groups], dtype=float)
    greedy = np.asarray(jnp.round(ctx.warm_start), dtype=np.float32)
    states = [greedy]
    for r in range(n_random):
        pos = greedy.copy()
        for _ in range(int(rng.choice([3, 10, 30, 60]))):
            g = groups[rng.choice(len(groups), p=weights / weights.sum())]
            a, b = rng.choice(g, 2, replace=False)
            pos[[a, b]] = pos[[b, a]]
        states.append(pos)
    # Swarm trajectories visit the placements a swarm will actually be in.
    trajectories = []
    for start in states[: len(states) // 2]:
        run = _trajectory(ctx, start, exact, k)
        trajectories.extend(run)
    return states + trajectories


def _trajectory(ctx, start, exact, k, rounds=12):
    out, pos = [], start.copy()
    for _ in range(rounds):
        idx, valid = candidates(ctx, pos, k)
        scores = exact(pos, idx, valid)
        best = scores.argmax(axis=1)
        wants = np.where(scores[np.arange(len(pos)), best] > 0, idx[np.arange(len(pos)), best], -1)
        pairs = [(i, int(j)) for i, j in enumerate(wants) if j > i and wants[j] == i]
        if not pairs:
            break
        new = pos.copy()
        for i, j in pairs:
            new[i], new[j] = pos[j], pos[i]
        pos = new
        out.append(pos.copy())
    return out


def train(args) -> None:
    rng = np.random.default_rng(args.seed)
    xs, ys, n_states = [], [], 0
    for benchmark_dir in [args.benchmark_dir, *args.extra_benchmarks]:
        ctx = context.build(benchmark_dir, canvas=args.canvas)
        exact = ExactGain(ctx)
        view_fn, _ = view_mod.make(ctx, args.view)
        states = training_states(ctx, rng, exact, k=args.k)
        n_states += len(states)
        for pos in states:
            idx, valid = candidates(ctx, pos, args.k)
            gains = exact(pos, idx, valid)
            scale = float(exact.total(jnp.asarray(pos))) / ctx.n_macros
            x = pair_features(ctx, view_fn, pos, idx)
            xs.append(x[valid]); ys.append((gains[valid] / scale).astype(np.float32))
    x = np.concatenate(xs); y = np.clip(np.concatenate(ys), -5.0, 5.0)
    order = rng.permutation(len(x)); x, y = x[order], y[order]
    split = int(0.9 * len(x))
    print(f"{n_states} placements -> {len(x):,} candidate swaps; {np.mean(y > 0):.1%} would help")

    model = SwapScorer(tuple(int(w) for w in args.hidden.split(",")))
    params = model.init(jax.random.PRNGKey(args.seed), jnp.zeros((1, x.shape[1])))
    optimizer = optax.adam(args.lr)
    state = optimizer.init(params)

    @jax.jit
    def step(params, state, xb, yb):
        def loss(p):
            return optax.huber_loss(model.apply(p, xb), yb, delta=1.0).mean()
        value, grads = jax.value_and_grad(loss)(params)
        updates, state = optimizer.update(grads, state, params)
        return optax.apply_updates(params, updates), state, value

    @jax.jit
    def evaluate(params, xb, yb):
        pred = model.apply(params, xb)
        return optax.huber_loss(pred, yb, delta=1.0).mean(), jnp.mean((pred > 0) == (yb > 0))

    batch = 512
    for epoch in range(args.epochs):
        for start in range(0, split, batch):
            params, state, _ = step(params, state, x[start:start + batch], y[start:start + batch])
    val_loss, val_sign = evaluate(params, x[split:], y[split:])
    print(f"view {args.view}: held-out huber {float(val_loss):.3f}, sign accuracy {float(val_sign):.1%}")
    args.out.mkdir(parents=True, exist_ok=True)
    with (args.out / "scorer.pkl").open("wb") as handle:
        pickle.dump(jax.tree_util.tree_map(np.asarray, params), handle)
    (args.out / "scorer.json").write_text(json.dumps({
        "view": args.view, "hidden": [int(w) for w in args.hidden.split(",")], "k": args.k,
        "n_inputs": int(x.shape[1]), "seed": args.seed,
        "trained_on": [str(args.benchmark_dir), *map(str, args.extra_benchmarks)],
        "samples": int(len(x)), "val_huber": float(val_loss), "val_sign_accuracy": float(val_sign),
    }, indent=2))


# ----------------------------------------------------------------------------------------------
# Nudges and swaps together


def nudger(ctx, objective, run_dir: pathlib.Path):
    """A trained nudging policy as a function `legal positions -> legal positions`.

    One deterministic episode from the given placement (not the greedy start it was trained from),
    then the same legalizer every method goes through. Built the way transfer.py builds it.
    """
    from multiagent import legalize, moves
    from multiagent.policy import SharedMacroPolicy, make_act

    trained = json.loads((run_dir / "manifest.json").read_text())["args"]
    with (run_dir / "best_params.pkl").open("rb") as handle:
        variables = jax.tree_util.tree_map(jnp.asarray, pickle.load(handle))
    policy = SharedMacroPolicy(features=tuple(int(w) for w in str(trained["hidden"]).split(",") if w))
    view_fn, _ = view_mod.make(ctx, trained["view"])
    rule = make_act(policy, view_fn, trained, ctx.connection_weights)

    def act(positions, parts, progress, key):
        return rule(variables, positions, parts, progress, key, False)

    @jax.jit
    def episode(start):
        _final, _costs, path = moves.rollout(
            start, jax.random.PRNGKey(0), act, objective, ctx.lo, ctx.hi,
            int(trained["steps"]), int(trained["horizon"]), return_path=True)
        return path

    def nudge(positions, return_path=False):
        path = np.asarray(episode(jnp.asarray(positions, dtype=jnp.float32)))
        placed, _ = legalize.repair_and_report(ctx, objective, jnp.asarray(path[-1]))
        placed = np.asarray(placed, dtype=np.float32)
        return (placed, path) if return_path else placed

    return nudge


# ----------------------------------------------------------------------------------------------
# Entry point


def run(args) -> None:
    ctx = context.build(args.benchmark_dir, canvas=args.canvas)
    objective = objective_mod.make(ctx)
    warm = objective_mod.report(ctx, objective, ctx.warm_start)["real_hpwl_snapped"]
    exact = ExactGain(ctx)
    start = (np.load(args.positions).astype(np.float32) if args.positions is not None
             else np.asarray(jnp.round(ctx.warm_start), dtype=np.float32))
    judge = exact if args.scorer is None else LearnedGain(ctx, args.scorer)
    started = time.perf_counter()
    if args.policy is None:
        result = swarm(ctx, start, judge, k=args.k, max_rounds=args.max_rounds, exact=exact,
                       resolve=args.resolve, threshold=args.threshold)
    else:
        # Alternate the nudging policy and the swap swarm. `--nudge_first` decides the order.
        nudge = nudger(ctx, objective, args.policy)
        pos, trace, swaps = start, [float(exact.total(jnp.asarray(start)))], []
        frames = [("start", start, [])]

        def do_nudge(pos):
            placed, path = nudge(pos, return_path=True)
            frames.extend(("nudge", p, []) for p in path[1:])
            frames.append(("legalize", placed, []))
            trace.append(float(exact.total(jnp.asarray(placed))))
            return placed

        for _cycle in range(args.cycles):
            if args.nudge_first:
                pos = do_nudge(pos)
            step = swarm(ctx, pos, judge, k=args.k, max_rounds=args.max_rounds, exact=exact,
                         resolve=args.resolve, threshold=args.threshold)
            frames.extend(("swap", p, pr) for p, pr in zip(step["path"][1:], step["swapped"][1:]))
            pos = step["positions"]; trace.append(step["hpwl"][-1]); swaps += step["swaps_per_round"]
            if not args.nudge_first:
                pos = do_nudge(pos)
        result = {"positions": pos, "hpwl": trace, "swaps_per_round": swaps, "frames": frames}
    final = objective_mod.report(ctx, objective, jnp.asarray(result["positions"]))
    improvement = 1.0 - final["real_hpwl_snapped"] / warm
    best = 1.0 - min(result["hpwl"]) / warm
    print(f"{args.benchmark_dir.name}: {'exact' if args.scorer is None else 'learned'} judge, k={args.k}, "
          f"{'wired swaps wait' if args.resolve else 'all at once'}: "
          f"{len(result['swaps_per_round'])} rounds, {sum(result['swaps_per_round'])} swaps -> "
          f"{improvement:+.2%} vs greedy (best round {best:+.2%})  legal={final['is_legal']}  "
          f"[{time.perf_counter() - started:.1f}s]")
    if args.gif is not None:
        from multiagent.visualize import save_swap_gif
        frames = result.get("frames") or [
            ("start" if r == 0 else "swap", p, pr)
            for r, (p, pr) in enumerate(zip(result["path"], result["swapped"]))]
        label = ("exact" if args.scorer is None else "learned") + f" judge, k={'all' if args.k <= 0 else args.k}, " + (
            "wired swaps wait" if args.resolve else "all swaps at once")
        save_swap_gif(ctx, frames, warm, args.gif, label)
        print(f"gif written to {args.gif}")
    if args.out is not None:
        args.out.mkdir(parents=True, exist_ok=True)
        np.save(args.out / "positions.npy", result["positions"])
        (args.out / "summary.json").write_text(json.dumps({
            "kind": "multiagent.swarm_swap", "benchmark_dir": str(args.benchmark_dir),
            "judge": "exact" if args.scorer is None else str(args.scorer), "k": args.k,
            "resolve": args.resolve, "threshold": args.threshold,
            "policy": str(args.policy) if args.policy else None, "cycles": args.cycles,
            "nudge_first": args.nudge_first,
            "rounds": len(result["swaps_per_round"]), "swaps_per_round": result["swaps_per_round"],
            "hpwl_trajectory": result["hpwl"], "warm_hpwl": warm,
            "hpwl_improvement": improvement, "best_round_improvement": best,
            "is_legal": final["is_legal"],
        }, indent=2))


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("run", "train"):
        p = sub.add_parser(name)
        p.add_argument("--benchmark_dir", type=pathlib.Path, default=pathlib.Path("benchmarks/adaptec1"))
        p.add_argument("--canvas", default="core", choices=("die", "core"))
        p.add_argument("--k", type=int, default=8, help="Candidates per macro; <= 0 = all same-size.")
        p.add_argument("--out", type=pathlib.Path, default=None)
    r = sub.choices["run"]
    r.add_argument("--scorer", type=pathlib.Path, default=None,
                   help="A trained scorer directory; default is the exact local gain.")
    r.add_argument("--positions", type=pathlib.Path, default=None,
                   help="A legal .npy placement to start from; default greedy.")
    r.add_argument("--max_rounds", type=int, default=60)
    r.add_argument("--policy", type=pathlib.Path, default=None,
                   help="A trained multiagent.train run: alternate its nudges with swap rounds.")
    r.add_argument("--cycles", type=int, default=1)
    r.add_argument("--gif", type=pathlib.Path, default=None,
                   help="Write an animation: one frame per nudge step and per swap round.")
    r.add_argument("--nudge_first", action="store_true")
    r.add_argument("--threshold", type=float, default=0.0,
                   help="Swap only when the judge's score exceeds this. A learned judge needs a "
                        "margin: most candidate swaps change nothing, and a slightly positive "
                        "prediction on those is noise.")
    r.add_argument("--resolve", action="store_true",
                   help="An agreed swap waits if a wired partner is in a better one this round.")
    t = sub.choices["train"]
    t.add_argument("--view", default="m1all", choices=("m0", "m1", "m1all"))
    t.add_argument("--hidden", default="64,64")
    t.add_argument("--epochs", type=int, default=30)
    t.add_argument("--lr", type=float, default=1e-3)
    t.add_argument("--seed", type=int, default=0)
    t.add_argument("--extra_benchmarks", type=pathlib.Path, nargs="*", default=[],
                   help="More designs to learn from (same --canvas).")
    return parser.parse_args(argv)


def main(argv=None) -> None:
    args = parse_args(argv)
    if args.command == "train":
        if args.out is None:
            raise SystemExit("--out is required for train")
        train(args)
    else:
        run(args)


if __name__ == "__main__":
    main()
