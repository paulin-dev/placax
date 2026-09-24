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

        @jax.jit
        def moves(pos, macros, targets):
            batch = jnp.repeat(pos[None], macros.shape[0], axis=0)
            batch = batch.at[jnp.arange(macros.shape[0]), macros].set(targets)
            return total(pos) - jax.vmap(total)(batch)

        self._moves = moves

    def for_moves(self, positions: np.ndarray, macros: np.ndarray, targets: np.ndarray) -> np.ndarray:
        """The HPWL reduction of moving each `macros[p]` to `targets[p]`, in one call."""
        if not len(macros):
            return np.zeros(0)
        return np.asarray(self._moves(jnp.asarray(positions, dtype=jnp.float32),
                                      jnp.asarray(macros), jnp.asarray(targets, dtype=jnp.float32)))

    def for_pairs(self, positions: np.ndarray, first: np.ndarray, second: np.ndarray) -> np.ndarray:
        """The HPWL reduction of each proposed swap `(first[p], second[p])`, in one call."""
        return np.asarray(self._gains(jnp.asarray(positions, dtype=jnp.float32),
                                      jnp.asarray(first), jnp.asarray(second)))

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


def anneal_swarm(ctx, positions: np.ndarray, seconds: float, rng, k: int = 0,
                 accept0: float = 0.3, greedy_tail: float = 0.1, resolve: bool = True,
                 shift_prob: float = 0.5, proposal: str = "random",
                 exact: ExactGain | None = None) -> dict:
    """The swap swarm with a temperature: a parallel annealer instead of a parallel descent.

    `swarm` stops as soon as no macro can improve, which is a local optimum of the exchange move.
    Simulated annealing passes it given enough time precisely because it accepts a worse placement
    now and then. This keeps every decision local and adds that one ability:

    1. each macro proposes either a swap (with `proposal="random"`, a random same-size candidate;
       with `proposal="best"`, its best one, which is what the greedy swarm always does) or, with probability
       `shift_prob`, a move of itself into free space within a shrinking window - the same two
       moves `anneal.py` uses;
    2. it accepts its own proposal by the Metropolis rule - always if the move shortens wires,
       otherwise with probability `exp(gain / T)`;
    3. proposals that touch the same macro, or whose source and target areas overlap another
       accepted move, are resolved by keeping the better one, so a round stays legal;
    4. `resolve` then applies the same wired-conflict rule as `swarm`.

    The temperature follows the schedule `anneal.py` uses, so the two are comparable: steered
    toward an acceptance rate falling from `accept0` to 1%, with the last `greedy_tail` of the
    budget at zero temperature, returning the best placement seen.
    """
    from placax_agents.policy.scale import to_grid_units
    exact = exact or ExactGain(ctx)
    wired = np.asarray(ctx.connection_weights) > 0
    footprints = np.asarray(to_grid_units(ctx.benchmark.sizes_array, ctx.benchmark.cell_size))
    grid = (int(ctx.params.grid_x), int(ctx.params.effective_grid_y))
    window0 = max(grid) // 2
    pos = np.asarray(positions, dtype=np.float32).copy()
    n = len(pos)

    def overlapping(positions):
        """Any two macros sharing area, checked directly rather than trusted.

        The round applies many moves at once after local checks. Those checks are conservative, but
        a placement that is wrong is worse than one that is slow, so every round is verified and a
        round that would break legality is dropped (`reverted` counts them; it should stay 0).
        """
        lo = positions.astype(int)
        hi = lo + footprints
        w = np.minimum(hi[:, None, 0], hi[None, :, 0]) - np.maximum(lo[:, None, 0], lo[None, :, 0])
        h = np.minimum(hi[:, None, 1], hi[None, :, 1]) - np.maximum(lo[:, None, 1], lo[None, :, 1])
        hit = (w > 0) & (h > 0)
        np.fill_diagonal(hit, False)
        return bool(hit.any()) or bool((lo < 0).any()) or bool((hi > np.asarray(grid)).any())

    def occupancy(positions):
        occupied = np.zeros(grid, dtype=np.int16)
        for m, (x, y) in enumerate(positions.astype(int)):
            w, h = footprints[m]
            occupied[x:x + w, y:y + h] += 1
        return occupied

    def free_targets(positions, window):
        """A random in-window target for every macro, and whether it is free right now."""
        occupied = occupancy(positions)
        offsets = rng.integers(-window, window + 1, size=(n, 2))
        targets = positions.astype(int) + offsets
        ok = np.zeros(n, dtype=bool)
        for m, (x, y) in enumerate(targets):
            w, h = footprints[m]
            if x < 0 or y < 0 or x + w > grid[0] or y + h > grid[1]:
                continue
            px, py = positions[m].astype(int)
            patch = occupied[x:x + w, y:y + h].copy()
            occupied[px:px + w, py:py + h] -= 1          # ignore the macro's own cells
            ok[m] = not occupied[x:x + w, y:y + h].any()
            occupied[px:px + w, py:py + h] += 1
            del patch
        return targets, ok
    total = float(exact.total(jnp.asarray(pos)))
    best, best_pos = total, pos.copy()
    hpwls, swaps_per_round = [total], []
    t0 = None
    reverted = 0
    started = time.perf_counter()

    while True:
        elapsed = time.perf_counter() - started
        if elapsed >= seconds:
            break
        cooling = min((elapsed / seconds) / max(1.0 - greedy_tail, 1e-9), 1.0)

        window = max(1, int(round(window0 * (0.02 ** cooling))))
        idx, valid = candidates(ctx, pos, k)
        counts = valid.sum(axis=1)
        movable = counts > 0
        first = np.arange(n)[movable]
        if proposal == "best":
            scored = exact(pos, idx, valid)                       # every candidate, then take the best
            chosen = idx[np.arange(n), scored.argmax(axis=1)]
            second = chosen[movable]
            gains = scored.max(axis=1)[movable]
        else:
            pick = (rng.random(n) * np.maximum(counts, 1)).astype(int)
            order = np.argsort(~valid, axis=1, kind="stable")     # valid slots first
            chosen = np.take_along_axis(idx, np.take_along_axis(order, pick[:, None], axis=1), axis=1)[:, 0]
            second = chosen[movable]
            gains = exact.for_pairs(pos, first, second)

        # Shifts: some macros propose moving into free space instead of trading.
        shifting = movable & (rng.random(n) < shift_prob)
        targets, free = free_targets(pos, window)
        shifting &= free
        shift_macros = np.arange(n)[shifting]
        shift_gains = exact.for_moves(pos, shift_macros, targets[shifting])

        if t0 is None:                                            # calibrate on the first round
            uphill = np.concatenate([-gains[gains < 0], -shift_gains[shift_gains < 0]])
            t0 = float(uphill.mean() / -np.log(accept0)) if uphill.size else 1.0
        temperature = 0.0 if cooling >= 1.0 else t0 * (1e-3 ** cooling)
        if cooling >= 1.0 and best < total:                       # descend from the best seen
            pos, total = best_pos.copy(), best

        def metropolis(values):
            with np.errstate(over="ignore"):
                if temperature <= 0:
                    return values >= 0
                return (values >= 0) | (rng.random(len(values)) < np.exp(np.minimum(values / temperature, 0.0)))

        keep = metropolis(gains)
        proposals = [(int(i), int(j), float(g)) for i, j, g, take in zip(first, second, gains, keep)
                     if take and i != j]
        shifts = [(int(m), targets[m].astype(int), float(g))
                  for m, g, take in zip(shift_macros, shift_gains, metropolis(shift_gains)) if take]
        # A macro may take part in one swap per round: keep the best proposal touching it.
        best_for = {}
        for i, j, g in proposals:
            for m in (i, j):
                if m not in best_for or g > best_for[m][2]:
                    best_for[m] = (i, j, g)
        pairs = sorted({(min(i, j), max(i, j)) for i, j, g in proposals
                        if best_for[i] == (i, j, g) and best_for[j] == (i, j, g)})
        if resolve:
            scores = [next(g for a, b, g in proposals if (min(a, b), max(a, b)) == pair) for pair in pairs]
            pairs = _independent(pairs, scores, wired)
        # Shifts are applied too, unless their area clashes with an accepted swap or a better shift.
        claimed = np.zeros(grid, dtype=bool)
        for i, j in pairs:
            for m in (i, j):
                x, y = pos[m].astype(int)
                w, h = footprints[m]
                claimed[x:x + w, y:y + h] = True
        applied_shifts = []
        for m, target, gain in sorted(shifts, key=lambda item: -item[2]):
            w, h = footprints[m]
            x, y = pos[m].astype(int)
            tx, ty = target
            if claimed[x:x + w, y:y + h].any() or claimed[tx:tx + w, ty:ty + h].any():
                continue
            claimed[x:x + w, y:y + h] = True
            claimed[tx:tx + w, ty:ty + h] = True
            applied_shifts.append((m, target))

        if pairs or applied_shifts:
            new = pos.copy()
            for i, j in pairs:
                new[i], new[j] = pos[j], pos[i]
            for m, target in applied_shifts:
                new[m] = target
            if overlapping(new):
                reverted += 1
            else:
                pos = new
                total = float(exact.total(jnp.asarray(pos)))
                if total < best:
                    best, best_pos = total, pos.copy()
        swaps_per_round.append(len(pairs) + len(applied_shifts))
        hpwls.append(total)

    return {"positions": best_pos, "hpwl": best, "hpwl_trace": hpwls, "reverted_rounds": reverted,
            "swaps_per_round": swaps_per_round, "rounds": len(swaps_per_round), "seconds": seconds}


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
    if args.anneal_seconds is not None:
        descent, down = None, None
        if args.descend_first:
            # Spend the first part of the budget on the greedy swarm, then heat what it found.
            # The annealer gets what is left, so the total wall clock is still `--anneal_seconds`.
            down = swarm(ctx, start, judge, k=args.k, max_rounds=args.max_rounds, exact=exact,
                         resolve=args.resolve, threshold=args.threshold)
            start = down["positions"]
            descent = time.perf_counter() - started
        budget = max(args.anneal_seconds - (descent or 0.0), 1.0)
        result = anneal_swarm(ctx, start, budget, np.random.default_rng(args.seed),
                              k=args.k, resolve=args.resolve, proposal=args.proposal, exact=exact)
        result["hpwl"] = (down["hpwl"] + result["hpwl_trace"]) if down else result["hpwl_trace"]
        result["descent_s"] = descent
        if down:
            result["swaps_per_round"] = down["swaps_per_round"] + result["swaps_per_round"]
    elif args.policy is None:
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
    elapsed = time.perf_counter() - started
    final = objective_mod.report(ctx, objective, jnp.asarray(result["positions"]))
    improvement = 1.0 - final["real_hpwl_snapped"] / warm
    best = 1.0 - min(result["hpwl"]) / warm
    print(f"{args.benchmark_dir.name}: {'exact' if args.scorer is None else 'learned'} judge, k={args.k}, "
          f"{'wired swaps wait' if args.resolve else 'all at once'}: "
          f"{len(result['swaps_per_round'])} rounds, {sum(result['swaps_per_round'])} swaps -> "
          f"{improvement:+.2%} vs greedy (best round {best:+.2%})  legal={final['is_legal']}  "
          f"[{elapsed:.1f}s]")
    if args.gif is not None and args.anneal_seconds is not None:
        raise SystemExit("--gif animates the greedy swarm's rounds; it is not wired to --anneal_seconds")
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
            "anneal_seconds": args.anneal_seconds, "seed": args.seed,
            "proposal": args.proposal, "descend_first": args.descend_first,
            "descent_s": result.get("descent_s"), "elapsed_s": round(elapsed, 2),
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
    r.add_argument("--proposal", default="random", choices=("random", "best"),
                   help="With --anneal_seconds: propose a random same-size candidate, or the best one.")
    r.add_argument("--anneal_seconds", type=float, default=None,
                   help="Run the parallel annealer (temperature, random proposals) for this long "
                        "instead of the greedy swarm.")
    r.add_argument("--descend_first", action="store_true",
                   help="With --anneal_seconds: run the greedy swarm to its local optimum first "
                        "and anneal from there, inside the same total budget.")
    r.add_argument("--seed", type=int, default=0)
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
