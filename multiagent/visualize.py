"""Pictures of a run: watch the macros move, and watch the policy learn.

    python -m multiagent.visualize --run=multiagent/runs/adaptec1-m1-s0
    python -m multiagent.visualize --run=multiagent/runs/adaptec1-m1-s0 --against=multiagent/runs/adaptec1-adam-s0

Writes into `<run>/viz/`:

    episode.gif       the trained policy's own episode, one frame per simultaneous move, then the
                      repair. Every macro moves in every frame. (policy runs only)
    training.gif      the end placement at each evaluation checkpoint - how the policy's answer
                      changed as it trained, or how Adam's positions changed as it descended.
    before_after.png  warm start | raw end placement | repaired, side by side.
    curves.png        wirelength, overlap and repaired improvement against moves spent, with
                      `--against` drawn on the same axes.

**How to read a frame.** Blue macros are clear of every other macro; red ones overlap at least one
(that is what the repair will have to fix). The dashed outlines are the warm start, and the orange
line from each macro runs from where it started to where it is now - one step is at most
`max_step` cells, far too small to see on its own, so the displacement is drawn cumulatively.
`--wires` adds each macro's strongest connection, which is what an `m1` macro can see.

It works on a run that is still training: `training.gif` uses whatever snapshots exist, and
`--params=last` animates the latest weights rather than the best legal ones.
"""
import argparse
import json
import pathlib
import pickle

from placax import _device  # noqa: F401  must precede jax imports

import jax
import jax.numpy as jnp
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.animation import FuncAnimation, PillowWriter  # noqa: E402
from matplotlib.collections import LineCollection, PatchCollection  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402

from multiagent import context, legalize, moves, objective as objective_mod, view as view_mod  # noqa: E402
from multiagent.policy import SharedMacroPolicy, make_act  # noqa: E402

CLEAR = "#4C78A8"
OVERLAPPING = "#E45756"
TRAIL = "#F58518"
GHOST = "#9A9A9A"
WIRE = "#54A24B"


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--run", type=pathlib.Path, required=True,
                        help="A multiagent.train or multiagent.adam run directory.")
    parser.add_argument("--against", type=pathlib.Path, nargs="*", default=[],
                        help="Other runs to overlay on curves.png (e.g. the Adam baseline).")
    parser.add_argument("--params", default="best", choices=("best", "last"),
                        help="Which weights drive episode.gif: the best legal checkpoint, or the "
                             "latest evaluation (useful while training is still running).")
    parser.add_argument("--benchmark_dir", type=pathlib.Path, default=None,
                        help="Animate the policy on ANOTHER design, with no retraining - "
                             "transfer.py, but as a picture.")
    parser.add_argument("--wires", action="store_true",
                        help="Draw each macro's strongest wired partner.")
    parser.add_argument("--fps", type=int, default=6)
    parser.add_argument("--only", nargs="*", default=None,
                        choices=("episode", "training", "before_after", "curves"))
    return parser.parse_args(argv)


# ----------------------------------------------------------------------------------------------
# Loading


def load_log(run: pathlib.Path) -> list[dict]:
    path = run / "log.jsonl"
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def build_context(manifest: dict, benchmark_dir: pathlib.Path | None = None):
    """The run's own environment, rebuilt from its manifest (the design may be overridden)."""
    args = manifest["args"]
    budget = int(args["macro_budget"])
    ctx = context.build(
        benchmark_dir or args["benchmark_dir"], grid=int(args["grid"]),
        macro_budget=budget if budget > 0 else None, canvas=args["canvas"],
        k_neighbors=int(args.get("k_neighbors", 4)),
    )
    objective = objective_mod.make(
        ctx, density_weight=float(args["density_weight"]),
        target_density=float(args["target_density"]), gamma_cells=float(args["gamma_cells"]),
        overlap_weight=float(args.get("overlap_weight", 0.0)),
    )
    return ctx, objective


def policy_episode(run: pathlib.Path, manifest: dict, ctx, objective, which: str):
    """The deterministic episode the policy recommends: `(steps + 1, n, 2)` placements."""
    args = manifest["args"]
    params_path = run / f"{which}_params.pkl"
    if not params_path.exists():
        raise SystemExit(f"{params_path} does not exist yet - no evaluation has written it. "
                         f"Try --params={'last' if which == 'best' else 'best'}, or wait.")
    with params_path.open("rb") as handle:
        variables = jax.tree_util.tree_map(jnp.asarray, pickle.load(handle))
    policy = SharedMacroPolicy(
        features=tuple(int(width) for width in str(args["hidden"]).split(",") if width)
    )
    view_fn, n_features = view_mod.make(ctx, args["view"])
    trained_features = variables["params"]["Dense_0"]["kernel"].shape[0]
    if trained_features != n_features:
        raise SystemExit(f"the policy reads {trained_features} features and this design's "
                         f"{args['view']} view has {n_features}.")

    rule = make_act(policy, view_fn, args, ctx.connection_weights)

    def act(positions, parts, progress, key):
        return rule(variables, positions, parts, progress, key, False)

    _final, _costs, path = jax.jit(lambda: moves.rollout(
        ctx.warm_start, jax.random.PRNGKey(0), act, objective, ctx.lo, ctx.hi,
        int(args["steps"]), int(args["horizon"]), return_path=True,
    ))()
    return np.asarray(path)


# ----------------------------------------------------------------------------------------------
# Drawing


def overlapping(positions: np.ndarray, sizes: np.ndarray, tolerance: float = 1e-3) -> np.ndarray:
    """Which macros share area with at least one other - on the float placement as drawn."""
    lo = positions
    hi = positions + sizes
    width = np.minimum(hi[:, None, 0], hi[None, :, 0]) - np.maximum(lo[:, None, 0], lo[None, :, 0])
    height = np.minimum(hi[:, None, 1], hi[None, :, 1]) - np.maximum(lo[:, None, 1], lo[None, :, 1])
    hit = (width > tolerance) & (height > tolerance)
    np.fill_diagonal(hit, False)
    return hit.any(axis=1)


def draw(ax, positions, ctx, warm, title: str, wires: bool = False) -> None:
    """One placement: macros coloured by overlap, the warm start as a ghost, the trail between."""
    positions = np.asarray(positions)
    sizes = np.asarray(ctx.sizes_grid)
    canvas = np.asarray(ctx.canvas)
    ax.clear()
    margin = 0.02 * canvas.max()
    ax.set_xlim(-margin, canvas[0] + margin)
    ax.set_ylim(-margin, canvas[1] + margin)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.add_patch(Rectangle((0, 0), canvas[0], canvas[1], facecolor="none",
                           edgecolor="black", linewidth=1.2))

    ax.add_collection(PatchCollection(
        [Rectangle(xy, w, h) for xy, (w, h) in zip(warm, sizes)],
        facecolor="none", edgecolor=GHOST, linewidth=0.4, linestyle="--",
    ))
    clash = overlapping(positions, sizes)
    ax.add_collection(PatchCollection(
        [Rectangle(xy, w, h) for xy, (w, h) in zip(positions, sizes)],
        facecolor=np.where(clash[:, None], matplotlib.colors.to_rgba_array(OVERLAPPING),
                           matplotlib.colors.to_rgba_array(CLEAR)),
        edgecolor="black", linewidth=0.3, alpha=0.8,
    ))

    centers = positions + sizes / 2
    if wires:
        idx = np.asarray(ctx.neighbor_idx)[:, 0]
        valid = np.asarray(ctx.neighbor_valid)[:, 0]
        segments = np.stack([centers[valid], centers[idx[valid]]], axis=1)
        ax.add_collection(LineCollection(segments, colors=WIRE, linewidths=0.5, alpha=0.5))
    trails = np.stack([warm + sizes / 2, centers], axis=1)
    ax.add_collection(LineCollection(trails, colors=TRAIL, linewidths=0.9))
    ax.scatter(centers[:, 0], centers[:, 1], s=2, color=TRAIL, zorder=3)
    ax.set_title(title, fontsize=9, loc="left")


def describe(ctx, objective, positions, warm_hpwl: float) -> str:
    metrics = objective_mod.report(ctx, objective, jnp.asarray(positions))
    hpwl = metrics["real_hpwl_snapped"]
    return (f"HPWL {hpwl:,.0f} ({1 - hpwl / warm_hpwl:+.1%} vs warm)   "
            f"overlap {metrics['overlap_ratio']:.1%}")


def save_gif(frames: list, ctx, warm, path: pathlib.Path, fps: int, wires: bool) -> None:
    """`frames`: (positions, title) pairs."""
    fig, ax = plt.subplots(figsize=(6, 6.3))
    fig.subplots_adjust(left=0.02, right=0.98, bottom=0.02, top=0.9)

    def frame(i):
        draw(ax, frames[i][0], ctx, warm, frames[i][1], wires)

    FuncAnimation(fig, frame, frames=len(frames)).save(str(path), writer=PillowWriter(fps=fps))
    plt.close(fig)


SWAPPED = "#B279A2"


def save_swap_gif(ctx, frames: list, warm_hpwl: float, path: pathlib.Path, label: str,
                  fps: int = 3) -> None:
    """A swap swarm (optionally interleaved with nudges) as an animation.

    `frames`: `(kind, positions, swapped_pairs)` with kind in start / nudge / legalize / swap. On a
    swap frame the macros that just traded places are filled purple and linked; everything else
    reads as in `draw` (red = overlapping, orange trail = distance from the greedy start).
    """
    from multiagent import objective as objective_mod

    objective = objective_mod.make(ctx)
    warm = np.asarray(jnp.round(ctx.warm_start))
    sizes = np.asarray(ctx.sizes_grid)
    fig, ax = plt.subplots(figsize=(6, 6.4))
    fig.subplots_adjust(left=0.02, right=0.98, bottom=0.02, top=0.88)
    rounds = nudges = 0
    titled = []
    for kind, positions, pairs in frames:
        if kind == "swap":
            rounds += 1
        if kind == "nudge":
            nudges += 1
        metrics = objective_mod.report(ctx, objective, jnp.asarray(positions))
        hpwl = metrics["real_hpwl_snapped"]
        what = {"start": "greedy start", "nudge": f"policy nudge, step {nudges}",
                "legalize": "legalized", "swap": f"swap round {rounds}: {len(pairs)} swaps"}[kind]
        titled.append((positions, pairs, f"{label}\n{what}   ·   HPWL {hpwl:,.0f} "
                                          f"({1 - hpwl / warm_hpwl:+.1%} vs greedy)"))
    titled += [titled[-1]] * (2 * fps)

    def frame(i):
        positions, pairs, title = titled[i]
        draw(ax, positions, ctx, warm, title)
        if pairs:
            moved = sorted({m for pair in pairs for m in pair})
            ax.add_collection(PatchCollection(
                [Rectangle(positions[m], *sizes[m]) for m in moved],
                facecolor=SWAPPED, edgecolor="black", linewidth=0.4, alpha=0.95, zorder=4))
            centers = positions + sizes / 2
            links = [[centers[i], centers[j]] for i, j in pairs]
            ax.add_collection(LineCollection(links, colors="white", linewidths=3.2, zorder=5))
            ax.add_collection(LineCollection(links, colors="#3B1F4A", linewidths=1.6, zorder=6))
            ax.scatter(centers[moved, 0], centers[moved, 1], s=9, color="#3B1F4A", zorder=7)

    FuncAnimation(fig, frame, frames=len(titled)).save(str(path), writer=PillowWriter(fps=fps))
    plt.close(fig)


# ----------------------------------------------------------------------------------------------
# The four outputs


def episode_gif(run, manifest, ctx, objective, warm, out, args) -> np.ndarray:
    path = policy_episode(run, manifest, ctx, objective, args.params)
    warm_hpwl = float(manifest["warm_start"]["real_hpwl_snapped"]) if args.benchmark_dir is None \
        else objective_mod.report(ctx, objective, ctx.warm_start)["real_hpwl_snapped"]
    view = manifest["args"]["view"]
    frames = [(p, f"{view} policy  ·  move {t}/{len(path) - 1}  ·  all {ctx.n_macros} macros\n"
                  f"{describe(ctx, objective, p, warm_hpwl)}")
              for t, p in enumerate(path)]
    frames += [frames[-1]] * args.fps  # hold the end placement for a second
    repaired, report = legalize.repair_and_report(ctx, objective, jnp.asarray(path[-1]))
    hpwl = report["repaired_real_hpwl_snapped"]
    frames += [(np.asarray(repaired, dtype=np.float32),
                f"after repair (legalize.py)  ·  legal={report['repaired_is_legal']}\n"
                f"HPWL {hpwl:,.0f} ({1 - hpwl / warm_hpwl:+.1%} vs warm)   mean repair move "
                f"{report['repair_mean_displacement_cells']:.1f} cells")] * (2 * args.fps)
    save_gif(frames, ctx, warm, out / "episode.gif", args.fps, args.wires)
    return path


def training_gif(run, ctx, warm, out, args, label: str) -> None:
    snapshots = sorted((run / "snapshots").glob("iter_*.npy"))
    if not snapshots:
        print(f"  no snapshots in {run} (runs before this change did not save them) - skipped")
        return
    by_iteration = {line["iteration"]: line for line in load_log(run)}
    frames = []
    for snapshot in snapshots:
        iteration = int(snapshot.stem.split("_")[1])
        line = by_iteration.get(iteration, {})
        raw = line.get("eval_real_hpwl_snapped", float("nan"))
        frames.append((np.load(snapshot), (
            f"{label}  ·  iteration {iteration}  ·  {line.get('moves', '?')} moves\n"
            f"raw HPWL {raw:,.0f} (overlap {line.get('eval_overlap_ratio', float('nan')):.1%})"
            f"  ·  repaired {line.get('eval_hpwl_improvement', float('nan')):+.1%}"
        )))
    frames += [frames[-1]] * args.fps
    save_gif(frames, ctx, warm, out / "training.gif", max(args.fps // 2, 2), args.wires)


def before_after(run, ctx, objective, warm, out, args, end: np.ndarray | None) -> None:
    if end is None:
        if args.benchmark_dir is not None:
            print("  before_after on another design needs the episode - skipped")
            return
        raw_path = run / "best_positions_raw.npy"
        if not raw_path.exists():
            print("  no end placement to show yet - skipped")
            return
        end = np.load(raw_path)
    repaired, _ = legalize.repair_and_report(ctx, objective, jnp.asarray(end))
    warm_hpwl = objective_mod.report(ctx, objective, jnp.asarray(warm))["real_hpwl_snapped"]
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.8))
    fig.subplots_adjust(left=0.01, right=0.99, bottom=0.02, top=0.88, wspace=0.03)
    for ax, positions, name in zip(
        axes, (warm, end, np.asarray(repaired, dtype=np.float32)),
        ("warm start (greedy)", "end of episode (raw)", "after repair"),
    ):
        draw(ax, positions, ctx, warm, f"{name}\n{describe(ctx, objective, positions, warm_hpwl)}",
             args.wires)
    fig.savefig(out / "before_after.png", dpi=130)
    plt.close(fig)


def curves(runs: list[pathlib.Path], out: pathlib.Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    panels = (("eval_wl_norm", "smoothed wirelength / warm start (raw)"),
              ("eval_overlap_ratio", "overlap before repair"),
              ("eval_hpwl_improvement", "HPWL improvement after repair"))
    for run in runs:
        evals = [line for line in load_log(run) if "eval_wl_norm" in line]
        if not evals:
            continue
        x = [line["moves"] for line in evals]
        for ax, (key, _) in zip(axes, panels):
            ax.plot(x, [line[key] for line in evals], marker="o", markersize=2.5, linewidth=1.2,
                    label=run.name)
    for ax, (key, title) in zip(axes, panels):
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("simultaneous moves spent")
        ax.grid(alpha=0.3)
        if key != "eval_wl_norm":
            ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    axes[2].axhline(0, color="black", linewidth=0.8)
    axes[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "curves.png", dpi=130)
    plt.close(fig)


def main(argv=None) -> None:
    args = parse_args(argv)
    manifest = json.loads((args.run / "manifest.json").read_text())
    is_policy = manifest.get("method") == "short_horizon"
    wanted = set(args.only or ("episode", "training", "before_after", "curves"))
    out = args.run / "viz"
    if args.benchmark_dir is not None:
        out = out / f"on-{args.benchmark_dir.name}"
        wanted &= {"episode", "before_after"}
    out.mkdir(parents=True, exist_ok=True)

    print(f"rebuilding {manifest['args']['benchmark_dir'] if args.benchmark_dir is None else args.benchmark_dir}"
          f" (greedy warm start included)...")
    ctx, objective = build_context(manifest, args.benchmark_dir)
    warm = np.asarray(ctx.warm_start)
    label = manifest["args"].get("view", "adam") if is_policy else "adam"

    end = None
    if "episode" in wanted:
        if is_policy:
            print("episode.gif")
            end = episode_gif(args.run, manifest, ctx, objective, warm, out, args)[-1]
        else:
            print("  episode.gif is for policy runs - Adam has no rule to replay")
    if "training" in wanted:
        print("training.gif")
        training_gif(args.run, ctx, warm, out, args, label)
    if "before_after" in wanted:
        print("before_after.png")
        before_after(args.run, ctx, objective, warm, out, args, end)
    if "curves" in wanted:
        print("curves.png")
        curves([args.run, *args.against], out)
    print(f"written to {out}")


if __name__ == "__main__":
    main()
