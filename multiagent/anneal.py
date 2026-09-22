"""Simulated annealing on the same placements: the classic competitor to our swap swarm.

    python -m multiagent.anneal --benchmark_dir=benchmarks/adaptec1 --seconds=30
    python -m multiagent.anneal --benchmark_dir=benchmarks/ariane133 --canvas=die --seconds=30 --seeds=3

**Why this baseline decides something.** Exchanging two blocks, and moving one block to a free
spot, are the moves simulated annealing has used for placement since TimberWolf (Sechen and
Sangiovanni-Vincentelli, 1985). Our swap swarm makes the same exchange move, in parallel and
without a coordinator. A reader's first question is therefore whether annealing, given the same
wall-clock time, already does as well. This module answers it.

**It never needs the legalizer.** Both moves map a legal placement to a legal placement: a swap of
two equal footprints leaves the occupied cells unchanged, and a shift is only proposed into free
space. So annealing is scored on its own output, with no repair, which is the friendliest possible
setting for it.

**Cost is incremental.** A move changes only the nets on the one or two macros it touches, so each
step re-evaluates those nets rather than the whole design. That is what makes a Python loop fast
enough to be a fair opponent: measured at roughly 20k-40k proposed moves per second on adaptec1.

**Temperature.** Rather than trusting a fixed cooling curve, the temperature is steered by the
measured acceptance rate, which is the robust version of the same idea. The target acceptance
falls from `--accept0` to 1% over the first `1 - --greedy_tail` of the budget, and after each
batch of moves T is nudged up or down to track it. The shift window shrinks on the same schedule.
The remaining time runs at zero temperature from the best placement found so far, so the answer is
always at least a local optimum of the move set. Both parts were needed: with a fixed curve and no
tail, a 120s budget returned the starting placement, because the search stayed hot and wandered.
"""
import argparse
import json
import pathlib
import statistics
import time

from placax import _device  # noqa: F401  must precede jax imports
from placax_agents.policy.scale import to_grid_units

import jax.numpy as jnp
import numpy as np

from multiagent import context, objective as objective_mod


class Cost:
    """Real HPWL of a placement, and the change caused by touching a few macros."""

    def __init__(self, ctx):
        benchmark = ctx.benchmark
        self.pin_idx = np.asarray(benchmark.padded_pin_idx)                      # (nets, pins)
        self.pin_off = np.asarray(benchmark.padded_pin_offset, dtype=np.float64)  # (nets, pins, 2)
        self.valid = np.asarray(benchmark.valid_mask)                            # (nets, pins)
        self.cell = float(benchmark.cell_size)
        self.half = np.asarray(benchmark.sizes_array, dtype=np.float64) / 2.0
        n_macros = ctx.n_macros
        # Which nets touch each macro, so a move only rescores those.
        self.nets_of = [[] for _ in range(n_macros)]
        for net in range(self.pin_idx.shape[0]):
            for macro in np.unique(self.pin_idx[net][self.valid[net]]):
                self.nets_of[int(macro)].append(net)
        self.nets_of = [np.asarray(sorted(set(v)), dtype=np.int64) for v in self.nets_of]
        self.all_nets = np.arange(self.pin_idx.shape[0])

    def of(self, positions: np.ndarray, nets: np.ndarray) -> float:
        """HPWL of the given nets, with every macro at `positions` (grid cells, lower-left)."""
        idx = self.pin_idx[nets]
        centers = positions[idx] * self.cell + self.half[idx] + self.pin_off[nets]
        valid = self.valid[nets]
        big = np.where(valid[..., None], centers, np.inf)
        small = np.where(valid[..., None], centers, -np.inf)
        return float((big.min(axis=1) * -1 + small.max(axis=1)).sum())

    def total(self, positions: np.ndarray) -> float:
        return self.of(positions, self.all_nets)

    def touched(self, macros) -> np.ndarray:
        return np.unique(np.concatenate([self.nets_of[m] for m in macros]))


class Placement:
    """A legal placement plus the occupancy grid that keeps it legal."""

    def __init__(self, ctx, positions: np.ndarray):
        self.footprints = np.asarray(to_grid_units(ctx.benchmark.sizes_array, ctx.benchmark.cell_size))
        self.grid = (int(ctx.params.grid_x), int(ctx.params.effective_grid_y))
        self.pos = positions.astype(np.int64).copy()
        self.occupied = np.zeros(self.grid, dtype=np.int32)
        for m, (x, y) in enumerate(self.pos):
            w, h = self.footprints[m]
            self.occupied[x:x + w, y:y + h] += 1
        if self.occupied.max() > 1:
            raise SystemExit("the starting placement overlaps; annealing needs a legal start")

    def free_for(self, macro: int, x: int, y: int) -> bool:
        w, h = self.footprints[macro]
        if x < 0 or y < 0 or x + w > self.grid[0] or y + h > self.grid[1]:
            return False
        return not self.occupied[x:x + w, y:y + h].any()

    def lift(self, macro: int) -> None:
        x, y = self.pos[macro]
        w, h = self.footprints[macro]
        self.occupied[x:x + w, y:y + h] -= 1

    def drop(self, macro: int, x: int, y: int) -> None:
        w, h = self.footprints[macro]
        self.occupied[x:x + w, y:y + h] += 1
        self.pos[macro] = (x, y)


def anneal(ctx, start: np.ndarray, seconds: float, rng, swap_prob: float = 0.5,
           accept0: float = 0.2, greedy_tail: float = 0.2, report_every: float = 1.0) -> dict:
    """Anneal for `seconds` of wall clock. Returns the placement and a trace of (time, HPWL)."""
    cost = Cost(ctx)
    state = Placement(ctx, start)
    footprints = state.footprints
    n = ctx.n_macros
    # Macros grouped by footprint: only equal footprints may swap.
    groups = {}
    for m, f in enumerate(map(tuple, footprints)):
        groups.setdefault(f, []).append(m)
    same_size = [np.asarray(groups[tuple(f)]) for f in footprints]
    window0 = max(state.grid) // 2

    total = cost.total(state.pos)
    best, best_pos = total, state.pos.copy()

    def propose(window):
        """One random move: (macros touched, new positions) or None if it is not legal."""
        i = int(rng.integers(n))
        if rng.random() < swap_prob and len(same_size[i]) > 1:
            j = int(rng.choice(same_size[i]))
            if j == i:
                return None
            return (i, j), (state.pos[j].copy(), state.pos[i].copy())
        dx, dy = rng.integers(-window, window + 1, size=2)
        x, y = int(state.pos[i][0] + dx), int(state.pos[i][1] + dy)
        state.lift(i)
        ok = state.free_for(i, x, y)
        state.drop(i, *state.pos[i])
        return ((i,), (np.array([x, y]),)) if ok else None

    started = time.perf_counter()
    trace = [(0.0, total)]
    next_report = report_every
    proposed = accepted = 0
    temperature = None
    in_tail = False
    batch_taken = batch_moves = 0
    calibration = []

    def restore(positions):
        """Put the search back on a known placement, occupancy included."""
        state.pos[:] = positions
        state.occupied[:] = 0
        for m, (x, y) in enumerate(state.pos):
            w, h = state.footprints[m]
            state.occupied[x:x + w, y:y + h] += 1

    while True:
        elapsed = time.perf_counter() - started
        if elapsed >= seconds:
            break
        progress = elapsed / seconds
        cooling = min(progress / max(1.0 - greedy_tail, 1e-9), 1.0)
        if temperature is None:
            pass                           # calibration phase: accept everything, measure uphill
        elif cooling >= 1.0:
            temperature = 0.0              # greedy tail: only improving moves, from the best found
            if not in_tail:
                in_tail = True
                restore(best_pos)
                total = best
        else:
            # Steer T toward the acceptance rate this point in the schedule calls for.
            target = accept0 * (0.01 / accept0) ** cooling
            if batch_moves:
                rate = batch_taken / batch_moves
                temperature *= 0.85 if rate > target else 1.1
        batch_taken = batch_moves = 0
        window = max(1, int(round(window0 * (0.02 ** cooling))))

        for _ in range(200):               # a batch between clock reads, to keep timing cheap
            move = propose(window)
            if move is None:
                continue
            macros, targets = move
            proposed += 1
            nets = cost.touched(macros)
            before = cost.of(state.pos, nets)
            saved = [state.pos[m].copy() for m in macros]
            for m in macros:
                state.lift(m)
            for m, target in zip(macros, targets):
                state.drop(m, int(target[0]), int(target[1]))
            after = cost.of(state.pos, nets)
            delta = after - before
            if temperature is None:
                if delta > 0:
                    calibration.append(delta)
                take = True
            else:
                take = delta <= 0 or (temperature > 0 and rng.random() < np.exp(-delta / temperature))
            batch_moves += 1
            if take:
                batch_taken += 1
                total += delta
                accepted += 1
                if total < best:
                    best, best_pos = total, state.pos.copy()
            else:
                for m in macros:
                    state.lift(m)
                for m, saved_pos in zip(macros, saved):
                    state.drop(m, int(saved_pos[0]), int(saved_pos[1]))
            if temperature is None and len(calibration) >= 300:
                # A starting T at which `accept0` of the measured uphill moves would be taken.
                temperature = float(-statistics.mean(calibration) / np.log(accept0))
                restore(best_pos)          # discard the random walk used for calibration
                total = best

        if elapsed >= next_report:
            trace.append((round(elapsed, 3), total))
            next_report += report_every

    trace.append((round(time.perf_counter() - started, 3), best))
    return {"positions": best_pos, "hpwl": best, "trace": trace, "proposed": proposed,
            "accepted": accepted, "seconds": seconds}


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--benchmark_dir", type=pathlib.Path, default=pathlib.Path("benchmarks/adaptec1"))
    parser.add_argument("--canvas", default="core", choices=("die", "core"))
    parser.add_argument("--grid", type=int, default=context.DEFAULT_GRID)
    parser.add_argument("--macro_budget", type=int, default=context.DEFAULT_MACRO_BUDGET)
    parser.add_argument("--seconds", type=float, default=30.0, help="Wall-clock budget per seed.")
    parser.add_argument("--seeds", type=int, default=1)
    parser.add_argument("--swap_prob", type=float, default=0.5,
                        help="Share of proposals that are swaps; the rest are shifts.")
    parser.add_argument("--accept0", type=float, default=0.3,
                        help="Target acceptance rate at the start; it falls to 1% while cooling.")
    parser.add_argument("--greedy_tail", type=float, default=0.1,
                        help="Share of the budget spent at zero temperature, from the best found.")
    parser.add_argument("--positions", type=pathlib.Path, default=None,
                        help="A legal .npy placement to start from. Default: the greedy start.")
    parser.add_argument("--out", type=pathlib.Path, default=None)
    return parser.parse_args(argv)


def main(argv=None) -> None:
    args = parse_args(argv)
    ctx = context.build(args.benchmark_dir, grid=args.grid,
                        macro_budget=args.macro_budget if args.macro_budget > 0 else None,
                        canvas=args.canvas)
    objective = objective_mod.make(ctx)
    warm = objective_mod.report(ctx, objective, ctx.warm_start)["real_hpwl_snapped"]
    start = (np.load(args.positions) if args.positions is not None
             else np.asarray(jnp.round(ctx.warm_start))).astype(np.int64)

    results = []
    for seed in range(args.seeds):
        run = anneal(ctx, start, args.seconds, np.random.default_rng(seed), args.swap_prob,
                     args.accept0, args.greedy_tail)
        metrics = objective_mod.report(ctx, objective, jnp.asarray(run["positions"], dtype=jnp.float32))
        improvement = 1.0 - metrics["real_hpwl_snapped"] / warm
        print(f"{args.benchmark_dir.name} seed {seed}: {improvement:+.2%} in {args.seconds:g}s  "
              f"({run['proposed']:,} moves proposed, {run['accepted'] / max(run['proposed'], 1):.0%} accepted)"
              f"  legal={metrics['is_legal']}")
        results.append({"seed": seed, "hpwl_improvement": improvement, "legal": bool(metrics["is_legal"]),
                        "proposed": run["proposed"], "accepted": run["accepted"], "trace": run["trace"]})
        if args.out is not None and seed == 0:
            args.out.mkdir(parents=True, exist_ok=True)
            np.save(args.out / "positions.npy", run["positions"])

    if args.out is not None:
        args.out.mkdir(parents=True, exist_ok=True)
        (args.out / "summary.json").write_text(json.dumps({
            "kind": "multiagent.anneal", "benchmark_dir": str(args.benchmark_dir),
            "canvas": args.canvas, "seconds": args.seconds, "swap_prob": args.swap_prob,
            "accept0": args.accept0, "greedy_tail": args.greedy_tail,
            "warm_hpwl": warm, "seeds": results,
            "hpwl_improvement": statistics.mean(r["hpwl_improvement"] for r in results),
        }, indent=2))


if __name__ == "__main__":
    main()
