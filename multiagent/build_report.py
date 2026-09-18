"""Rebuilds report.html from report_template.html and the runs under multiagent/runs/.

    python -m multiagent.build_report              # reuse cached measurements where present
    python -m multiagent.build_report --recompute  # redo them (after new or changed runs)

Everything the report shows is read from run directories, never typed in. Three measurements
aren't a single run's output, so they are computed here and cached in `multiagent/runs/q3/`:

    rescore.json     every saved run re-legalized by the final (portfolio) legalizer: the median
                     of its last 5 snapshots, and its last snapshot followed by `swap.py`
    jitter.json      random 1-cell jitter + legalize, 5 seeds per chip, under the original and
                     the final legalizer - the no-intelligence control
    pull_steps.json  the untrained pull rule on adaptec1 at several episode lengths

The runs themselves come from the commands in README.md; the directory names below are the ones
those experiments wrote. A missing run fails loudly rather than leaving a hole in a chart.
"""
import argparse
import base64
import json
import pathlib
import re
import statistics

from placax import _device  # noqa: F401  must precede jax imports

import jax
import jax.numpy as jnp
import numpy as np

HERE = pathlib.Path(__file__).parent
RUNS = HERE / "runs"
CACHE = RUNS / "q3"
CHIPS = {"adaptec1": "core", "bigblue1": "core", "ariane133": "die"}
_envs = {}


def env(chip: str):
    """`(ctx, objective, warm_hpwl)` for a chip, built once."""
    from multiagent import context, objective as objective_mod
    if chip not in _envs:
        ctx = context.build(f"benchmarks/{chip}", canvas=CHIPS[chip])
        objective = objective_mod.make(ctx)
        warm = objective_mod.report(ctx, objective, ctx.warm_start)["real_hpwl_snapped"]
        _envs[chip] = (ctx, objective, warm)
    return _envs[chip]


def summary(path: pathlib.Path) -> dict:
    if not (path / "summary.json").exists():
        raise SystemExit(f"missing run: {path} (see README.md for the command that makes it)")
    return json.loads((path / "summary.json").read_text())


def evals(run: pathlib.Path) -> list[dict]:
    lines = [json.loads(line) for line in (run / "log.jsonl").read_text().splitlines() if line.strip()]
    return [line for line in lines if "eval_hpwl_improvement" in line]


# ----------------------------------------------------------------------------------------------
# Cached measurements


def legalize_and_score(chip, raw, with_swaps=False) -> dict:
    from multiagent import legalize, objective as objective_mod
    from multiagent.swap import swap_descent
    ctx, objective, warm = env(chip)
    placed, rep = legalize.repair_and_report(ctx, objective, jnp.asarray(raw))
    out = {"improvement": 1 - rep["repaired_real_hpwl_snapped"] / warm,
           "move": rep["repair_mean_displacement_cells"], "spread": rep["repair_used_spread"]}
    if with_swaps and rep["repaired_is_legal"]:
        swapped, _ = swap_descent(ctx, np.asarray(placed, dtype=np.float32))
        after = objective_mod.report(ctx, objective, jnp.asarray(swapped))
        out["plus_swaps"] = 1 - after["real_hpwl_snapped"] / warm
    return out


def pull_final(chip: str, steps: int, k: int = 4) -> np.ndarray:
    """The pull rule's raw end placement - the same rule as pull.py, through the same rollout."""
    from multiagent import moves
    from multiagent.pull import partner_weights
    ctx, objective, _ = env(chip)
    weights = partner_weights(ctx, k)
    total = weights.sum(axis=1, keepdims=True)
    half = ctx.sizes_grid / 2

    def act(positions, _parts, _progress, _key):
        centers = positions + half
        pull = (weights @ centers) / jnp.maximum(total, 1e-9) - centers
        distance = jnp.linalg.norm(pull, axis=1, keepdims=True)
        step = pull / jnp.maximum(distance, 1e-9) * jnp.minimum(distance, 1.0)
        return jnp.where(total > 0, step, 0.0).astype(positions.dtype)

    final, _ = jax.jit(lambda: moves.rollout(ctx.warm_start, jax.random.PRNGKey(0), act,
                                             objective, ctx.lo, ctx.hi, steps, steps))()
    return np.asarray(final)


def rescore() -> dict:
    runs = {}
    for sweep in ("sweep", "sweep2"):
        for run in sorted((RUNS / sweep).iterdir()):
            snaps = sorted((run / "snapshots").glob("iter_*.npy")) if (run / "snapshots").is_dir() else []
            if not (run / "manifest.json").exists() or not snaps:
                continue
            args = json.loads((run / "manifest.json").read_text())["args"]
            chip = pathlib.Path(args["benchmark_dir"]).name
            tail = snaps[-5:]
            scored = [legalize_and_score(chip, np.load(s), with_swaps=(s == tail[-1])) for s in tail]
            runs[f"{sweep}/{run.name}"] = {
                "chip": chip, "median": statistics.median(x["improvement"] for x in scored),
                "last_plus_swaps": scored[-1].get("plus_swaps"),
            }
            print(f"  rescored {sweep}/{run.name}: {runs[f'{sweep}/{run.name}']['median']:+.1%}", flush=True)
    pull = {chip: legalize_and_score(chip, pull_final(chip, 32), with_swaps=True) for chip in CHIPS}
    return {"runs": runs, "pull": pull}


def jitter() -> dict:
    from multiagent import legalize
    out = {}
    for chip in CHIPS:
        ctx, objective, warm = env(chip)
        res = {"old": [], "new": [], "old_move": [], "new_move": []}
        for seed in range(5):
            rng = np.random.default_rng(seed)
            raw = jnp.clip(ctx.warm_start + rng.standard_normal(ctx.warm_start.shape).astype(np.float32),
                           ctx.lo, ctx.hi)
            for key, steps in (("old", 0), ("new", legalize.DEFAULT_SPREAD_STEPS)):
                _, rep = legalize.repair_and_report(ctx, objective, raw, spread_steps=steps)
                res[key].append(1 - rep["repaired_real_hpwl_snapped"] / warm)
                res[key + "_move"].append(rep["repair_mean_displacement_cells"])
        out[chip] = res
    return out


def pull_steps() -> dict:
    return {label: {str(steps): legalize_and_score("adaptec1", pull_final("adaptec1", steps, k))["improvement"]
                    for steps in (8, 16, 32, 64, 128)}
            for label, k in (("k4", 4), ("all", 0))}


def cached(name: str, compute, recompute: bool) -> dict:
    path = CACHE / name
    if recompute or not path.exists():
        print(f"computing {name} ...", flush=True)
        CACHE.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(compute(), indent=1))
    return json.loads(path.read_text())


# ----------------------------------------------------------------------------------------------
# The report's data


def collect(recompute: bool) -> dict:
    rs = cached("rescore.json", rescore, recompute)
    jit = cached("jitter.json", jitter, recompute)
    steps = cached("pull_steps.json", pull_steps, recompute)
    runs = rs["runs"]

    def transfer(cfg, seed, chip, swap=False):
        path = CACHE / f"xfer-sweep2-{cfg}-s{seed}-{chip}"
        return summary(path / "swap")["hpwl_improvement"] if swap else summary(path)["hpwl_improvement"]

    # Sweep 1 (seven configurations, no ariane), as logged.
    sweep1 = {}
    for run in sorted((RUNS / "sweep").glob("*-s[0-9]")):
        if run.name.startswith("xfer-") or not (run / "log.jsonl").exists():
            continue
        tail = evals(run)[-5:]
        sweep1.setdefault(re.sub(r"-s\d$", "", run.name), []).append({
            "final": runs[f"sweep/{run.name}"]["median"],
            "bigblue1": summary(RUNS / "sweep" / f"xfer-{run.name}")["hpwl_improvement"],
            "overlap": statistics.median(e["eval_overlap_ratio"] for e in tail),
        })

    # Sweep 2 (nine configurations), transfers re-run under the final legalizer.
    sweep2 = {}
    for run in sorted((RUNS / "sweep2").glob("*-s[0-9]")):
        if not (run / "log.jsonl").exists():
            continue
        cfg, seed = re.match(r"(.+)-s(\d)$", run.name).groups()
        sweep2.setdefault(cfg, []).append({
            "final": runs[f"sweep2/{run.name}"]["median"],
            "bigblue1": transfer(cfg, seed, "bigblue1"), "ariane133": transfer(cfg, seed, "ariane133"),
        })

    adam = lambda key: {"final": runs[key]["median"]}
    baselines = {
        "adam_ov0": {"adaptec1": adam("sweep/adam-ov0-adaptec1"), "bigblue1": adam("sweep/adam-ov0-bigblue1"),
                     "ariane133": adam("sweep2/adam-ov0-ariane133")},
        "adam_ov3": {"adaptec1": adam("sweep/adam-ov3-adaptec1"), "bigblue1": adam("sweep/adam-ov3-bigblue1"),
                     "ariane133": adam("sweep2/adam-ov3-ariane133")},
    }

    def curve(run):
        return [[e["moves"], e["eval_wl_norm"], e["eval_overlap_ratio"], e["eval_hpwl_improvement"]]
                for e in evals(run)]

    curves = {"policy_ov0": curve(RUNS / "sweep/m1-s0"), "policy_ov3": curve(RUNS / "sweep2/m1all-s0"),
              "adam_ov0": curve(RUNS / "sweep/adam-ov0-adaptec1"), "adam_ov3": curve(RUNS / "sweep/adam-ov3-adaptec1")}

    pull = {"adaptec1": {"k4": steps["k4"], "all": steps["all"]},
            **{chip: {"k4": {"32": rs["pull"][chip]["improvement"]}} for chip in ("bigblue1", "ariane133")}}
    pull["adaptec1"]["k4"]["32"] = rs["pull"]["adaptec1"]["improvement"]

    swap = {chip: summary(CACHE / f"swap-{chip}")["hpwl_improvement"] for chip in CHIPS}
    q3 = {
        "jitter": jit["adaptec1"]["new"], "swap": swap["adaptec1"],
        "pull": {"alone": rs["pull"]["adaptec1"]["improvement"], "plus": rs["pull"]["adaptec1"]["plus_swaps"]},
        "adam3": {"alone": runs["sweep/adam-ov3-adaptec1"]["median"], "plus": runs["sweep/adam-ov3-adaptec1"]["last_plus_swaps"]},
        "adam0": {"alone": runs["sweep/adam-ov0-adaptec1"]["median"], "plus": runs["sweep/adam-ov0-adaptec1"]["last_plus_swaps"]},
        **{v: {"alone": [runs[f"sweep2/{v}-s{i}"]["median"] for i in range(3)],
               "plus": [runs[f"sweep2/{v}-s{i}"]["last_plus_swaps"] for i in range(3)]} for v in ("m0", "m1", "m1all")},
    }
    unseen = {chip: {
        "swap": swap[chip], "jitter": [min(jit[chip]["new"]), max(jit[chip]["new"])],
        **{name: {"alone": [transfer(cfg, i, chip) for i in range(3)],
                  "plus": [transfer(cfg, i, chip, swap=True) for i in range(3)]}
           for name, cfg in (("m1", "m1"), ("async", "async0.5"))},
    } for chip in ("bigblue1", "ariane133")}

    W = RUNS / "swarmswap"
    imp = lambda name: summary(W / name)["hpwl_improvement"]
    swarm = {
        "central": swap,
        "at_once": {c: imp(f"exact-{c}-k0") for c in CHIPS},
        "resolve": {c: imp(f"exact-{c}-k0-resolve") for c in CHIPS},
        "resolve_k8": {c: imp(f"exact-{c}-k8-resolve") for c in CHIPS},
        **{v: {c: [imp(f"learned-{v}-s{s}-{c}-k0") for s in range(3)] for c in CHIPS} for v in ("m0", "m1", "m1all")},
        "two_chips": {"adaptec1": None, "bigblue1": None,
                      "ariane133": [imp(f"learned2-{v}-s{s}-ariane133") for v in ("m1", "m1all") for s in range(3)]},
        "nudge_swap": {c: imp(f"cycles-{c}-c1-nudgefirst") for c in CHIPS},
    }
    return {"sweep1": sweep1, "sweep2": sweep2, "baselines": baselines, "curves": curves,
            "pull": pull, "jitter": jit, "q3": q3, "unseen": unseen, "swarm": swarm}


MEDIA = {
    "IMG_OV0": ("sweep/m1-s0/viz/before_after.png", "image/png"),
    "IMG_OV3": ("sweep2/m1all-s0/viz/before_after.png", "image/png"),
    "GIF_OV3": ("sweep2/m1all-s0/viz/episode.gif", "image/gif"),
    "GIF_SW_ARIANE": ("swarmswap/gifs/ariane133-resolve.gif", "image/gif"),
    "GIF_SW_A1_BAD": ("swarmswap/gifs/adaptec1-all-at-once.gif", "image/gif"),
    "GIF_SW_A1_GOOD": ("swarmswap/gifs/adaptec1-resolve.gif", "image/gif"),
}


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--recompute", action="store_true",
                        help="Redo rescore.json, jitter.json and pull_steps.json.")
    parser.add_argument("--out", type=pathlib.Path, default=HERE / "report.html")
    args = parser.parse_args(argv)

    page = (HERE / "report_template.html").read_text()
    page = page.replace("{{DATA}}", json.dumps(collect(args.recompute), separators=(",", ":")))
    for key, (path, mime) in MEDIA.items():
        media = RUNS / path
        if not media.exists():
            raise SystemExit(f"missing media: {media} (python -m multiagent.visualize / "
                             f"python -m multiagent.swarm_swap run --gif=...)")
        page = page.replace("{{" + key + "}}", f"data:{mime};base64," + base64.b64encode(media.read_bytes()).decode())
    leftover = re.findall(r"\{\{[A-Z0-9_]+\}\}", page)
    if leftover:
        raise SystemExit(f"template placeholders with no value: {leftover}")
    args.out.write_text(page)
    print(f"wrote {args.out} ({len(page) // 1024} KB)")


if __name__ == "__main__":
    main()
