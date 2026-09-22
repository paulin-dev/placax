"""Builds paper.pdf: figures, tables and quoted numbers from the runs, then LaTeX.

    python -m multiagent.paper.build              # reuse cached measurements (runs/q3/*.json)
    python -m multiagent.paper.build --recompute  # redo them after new or changed runs
    python -m multiagent.paper.build --no-pdf     # figures and tables only

Nothing in the paper is typed in by hand. `results.collect()` reads every run; this script turns
that into
    figures/*.pdf    vector figures, one style
    tables/*.tex     booktabs tables
    numbers.tex      \\newcommand for every number the prose quotes
and compiles paper.tex with Tectonic (a self-contained LaTeX engine; ~/.local/bin/tectonic, or
set TECTONIC). The animations the PDF shows as key frames are copied to media/ for
SUPPLEMENTARY.md.
"""
import argparse
import json
import os
import pathlib
import shutil
import statistics
import subprocess

from placax import _device  # noqa: F401  must precede jax imports

import jax.numpy as jnp
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.collections import LineCollection, PatchCollection  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402

from multiagent import results  # noqa: E402

HERE = pathlib.Path(__file__).parent
RUNS = results.RUNS
CHIPS = ("adaptec1", "bigblue1", "ariane133")
TEXTWIDTH = 5.5  # inches, NeurIPS-style single column

COLOR = {"agents": "#2a78d6", "adam": "#eb6834", "pull": "#1baf7a", "swap": "#8e5c84",
         "neutral": "#6f777b", "overlap": "#e34948", "block": "#c9d3da", "light": "#e8ecee"}

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["STIX Two Text", "STIXGeneral", "DejaVu Serif"],
    "mathtext.fontset": "stix", "font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7, "axes.linewidth": 0.6,
    "xtick.major.width": 0.6, "ytick.major.width": 0.6, "xtick.major.size": 2.5, "ytick.major.size": 2.5,
    "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False,
    "pdf.fonttype": 42, "savefig.bbox": "tight", "savefig.pad_inches": 0.02, "lines.linewidth": 1.2,
})

mean = statistics.mean


def sd(values):
    return statistics.stdev(values) if len(values) > 1 else 0.0


def pct(v, sign=True):
    return f"{'+' if sign and v >= 0 else ''}{100 * v:.1f}".replace("-", "−")


def tex_pct(v):
    """A percentage for LaTeX, with a real minus sign."""
    return f"${'+' if v >= 0 else '-'}{abs(100 * v):.1f}$\\%"


def percent_axis(ax, axis="y"):
    fmt = matplotlib.ticker.FuncFormatter(lambda v, _: f"{100 * v:.0f}%".replace("-", "−"))
    (ax.yaxis if axis == "y" else ax.xaxis).set_major_formatter(fmt)


def label(ax, text):
    ax.set_title(text, loc="left", fontweight="bold", pad=3)


# ----------------------------------------------------------------------------------------------
# Placements


def draw_placement(ax, ctx, positions, warm=None, highlight=(), pairs=()):
    positions = np.asarray(positions)
    sizes = np.asarray(ctx.sizes_grid)
    canvas = np.asarray(ctx.canvas)
    from multiagent.visualize import overlapping
    clash = overlapping(positions, sizes)
    ax.set_xlim(-2, canvas[0] + 2)
    ax.set_ylim(-2, canvas[1] + 2)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.add_patch(Rectangle((0, 0), canvas[0], canvas[1], facecolor="white", edgecolor="black", linewidth=0.6))
    if warm is not None:
        ax.add_collection(PatchCollection([Rectangle(xy, w, h) for xy, (w, h) in zip(np.asarray(warm), sizes)],
                                          facecolor="none", edgecolor="#9aa3a8", linewidth=0.25, linestyle=(0, (2, 2))))
    faces = [COLOR["overlap"] if c else COLOR["block"] for c in clash]
    for m in highlight:
        faces[m] = COLOR["swap"]
    ax.add_collection(PatchCollection([Rectangle(xy, w, h) for xy, (w, h) in zip(positions, sizes)],
                                      facecolor=faces, edgecolor="#3c4448", linewidth=0.2, alpha=0.9))
    if pairs:
        centers = positions + sizes / 2
        ax.add_collection(LineCollection([[centers[i], centers[j]] for i, j in pairs],
                                         colors="#2b1633", linewidths=0.8))


def fig_placements(out):
    """Greedy start, the agents' end state and its legalization, without and with the penalty."""
    from multiagent import legalize, objective as objective_mod
    from multiagent.visualize import build_context, policy_episode
    fig, axes = plt.subplots(2, 3, figsize=(TEXTWIDTH, 3.75))
    rows = [(RUNS / "sweep/m1-s0", "no overlap penalty ($\\lambda_o=0$)"),
            (RUNS / "sweep2/m1all-s0", "overlap penalty $\\lambda_o=3$")]
    for r, (run, name) in enumerate(rows):
        manifest = json.loads((run / "manifest.json").read_text())
        ctx, objective = build_context(manifest)
        warm = np.asarray(ctx.warm_start)
        end = policy_episode(run, manifest, ctx, objective, "best")[-1]
        legal, _ = legalize.repair_and_report(ctx, objective, jnp.asarray(end))
        warm_hpwl = objective_mod.report(ctx, objective, ctx.warm_start)["real_hpwl_snapped"]
        for c, (pos, title) in enumerate([(warm, "greedy start"), (end, "agents, after 32 steps"),
                                          (np.asarray(legal, dtype=np.float32), "legalized")]):
            m = objective_mod.report(ctx, objective, jnp.asarray(pos))
            draw_placement(axes[r, c], ctx, pos, warm=None if c == 0 else warm)
            if c == 1:
                detail = f"{100 * m['overlap_ratio']:.0f}% of macro area overlapping"
            elif c == 2:
                detail = f"improvement {pct(1 - m['real_hpwl_snapped'] / warm_hpwl)}%"
            else:
                detail = "legal, improvement 0%"
            axes[r, c].set_title(f"{title}\n{detail}", fontsize=7, pad=2)
        axes[r, 0].text(-0.06, 0.5, name, transform=axes[r, 0].transAxes, rotation=90,
                        va="center", ha="right", fontsize=7.5)
    fig.subplots_adjust(wspace=0.05, hspace=0.28)
    fig.savefig(out)
    plt.close(fig)


def fig_swarm_frames(out, frames_out):
    """Key frames of the exact swap swarm on ariane133 (the PDF's stand-in for the GIF)."""
    from multiagent import objective as objective_mod, swarm_swap
    ctx, objective, warm_hpwl = results.env("ariane133")
    start = np.asarray(jnp.round(ctx.warm_start), dtype=np.float32)
    exact = swarm_swap.ExactGain(ctx)
    run = swarm_swap.swarm(ctx, start, exact, k=0, max_rounds=200, exact=exact, resolve=True)
    last = len(run["path"]) - 1
    picks = [0, 3, 12, last]
    fig, axes = plt.subplots(1, 4, figsize=(TEXTWIDTH, 1.75))
    for ax, r in zip(axes, picks):
        pos = run["path"][r]
        pairs = run["swapped"][r]
        hpwl = objective_mod.report(ctx, objective, jnp.asarray(pos))["real_hpwl_snapped"]
        draw_placement(ax, ctx, pos, highlight=sorted({m for p in pairs for m in p}), pairs=pairs)
        swaps = f"{len(pairs)} swap{'s' if len(pairs) != 1 else ''}"
        ax.set_title(("greedy start" if r == 0 else f"round {r}" + (" (final)" if r == last else "")
                      + f": {swaps}") + f"\nimprovement {pct(1 - hpwl / warm_hpwl)}%", fontsize=7, pad=2)
    fig.subplots_adjust(wspace=0.04)
    fig.savefig(out)
    plt.close(fig)
    frames_out.write_text(json.dumps({"rounds": last, "swaps": sum(run["swaps_per_round"])}))


# ----------------------------------------------------------------------------------------------
# Charts


def fig_dynamics(data, out):
    cv = data["curves"]
    series = [("policy_ov0", "no overlap penalty ($\\lambda_o=0$)", COLOR["agents"], "-"),
              ("policy_ov3", "overlap penalty $\\lambda_o=3$", COLOR["agents"], "--")]
    fig, (a, b) = plt.subplots(1, 2, figsize=(TEXTWIDTH, 1.85))
    for key, name, color, style in series:
        x = [r[0] for r in cv[key]]
        a.plot(x, [r[2] for r in cv[key]], style, color=color, label=name)
        b.plot(x, [r[3] for r in cv[key]], style, color=color)
    label(a, "(a) overlap handed to the legalizer")
    label(b, "(b) HPWL improvement after legalization")
    for ax in (a, b):
        ax.set_xlabel("simultaneous moves during training")
        percent_axis(ax)
        ax.set_xlim(0, 3200)
        ax.set_xlabel("simultaneous moves during training")
    b.axhline(0, color="black", linewidth=0.5)
    a.legend(loc="upper left", ncol=1)
    fig.subplots_adjust(wspace=0.3)
    fig.savefig(out)
    plt.close(fig)


def fig_jitter(data, out):
    """The no-intelligence control under both legalizers (5 jitter seeds per design)."""
    fig, ax = plt.subplots(figsize=(TEXTWIDTH * 0.6, 1.8))
    jit = data["jitter"]
    for i, chip in enumerate(CHIPS):
        for j, (key, name, color) in enumerate((("old", "repair only", COLOR["neutral"]),
                                                ("new", "spread, then repair (final)", COLOR["agents"]))):
            x = i + (j - 0.5) * 0.3
            ax.scatter(np.full(5, x) + np.linspace(-0.05, 0.05, 5), jit[chip][key], s=9, color=color,
                       zorder=3, label=name if i == 0 else None)
            ax.plot([x - 0.1, x + 0.1], [mean(jit[chip][key])] * 2, color="black", linewidth=1)
    ax.set_xticks(range(3), CHIPS)
    ax.axhline(0, color="black", linewidth=0.5)
    percent_axis(ax)
    ax.set_ylabel("improvement")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.14), ncol=2)
    fig.savefig(out)
    plt.close(fig)


def fig_sight(data, out):
    s2, base = data["sweep2"], data["baselines"]
    views = [("m0", "m0: itself"), ("m1", "m1: + 4 partners"), ("m1all", "m1all: + all partners")]
    fig, axes = plt.subplots(1, 3, figsize=(TEXTWIDTH, 1.55), sharey=True)
    for ax, chip in zip(axes, CHIPS):
        key = "final" if chip == "adaptec1" else chip
        if chip != "adaptec1":
            lo, hi = min(data["jitter"][chip]["new"]), max(data["jitter"][chip]["new"])
            ax.axvspan(lo, hi, color=COLOR["light"], zorder=0, label="random-jitter range")
        ax.axvline(base["adam_ov3"][chip]["final"], color=COLOR["adam"], linestyle="--", linewidth=0.9,
                   label="Adam, $\\lambda_o=3$ (on this chip)")
        ax.axvline(0, color="black", linewidth=0.5)
        for y, (view, _) in enumerate(views):
            vals = [r[key] for r in s2[view]]
            ax.scatter(vals, [y] * 3, s=11, color=COLOR["agents"], zorder=3)
            ax.plot([mean(vals)] * 2, [y - 0.25, y + 0.25], color="black", linewidth=1.2)
        ax.set_yticks(range(3), [v[1] for v in views])
        ax.set_ylim(-0.6, 2.6)
        ax.invert_yaxis()
        percent_axis(ax, "x")
        ax.set_title(f"{chip} ({'training design' if chip == 'adaptec1' else 'unseen'})", fontsize=7.5, pad=3)
        ax.set_xlim(-0.13, 0.15)
    axes[1].legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2)
    fig.subplots_adjust(wspace=0.08)
    fig.savefig(out)
    plt.close(fig)


def fig_q3(data, out):
    q = data["q3"]
    rows = [("random jitter (control)", q["jitter"], None, COLOR["neutral"]),
            ("swap search only", None, q["swap"], COLOR["neutral"]),
            ("pull rule", q["pull"]["alone"], q["pull"]["plus"], COLOR["pull"]),
            ("Adam, $\\lambda_o=3$", q["adam3"]["alone"], q["adam3"]["plus"], COLOR["adam"]),
            ("Adam, $\\lambda_o=0$", q["adam0"]["alone"], q["adam0"]["plus"], COLOR["adam"]),
            ("agents m0, $\\lambda_o=3$", q["m0"]["alone"], q["m0"]["plus"], COLOR["agents"]),
            ("agents m1, $\\lambda_o=3$", q["m1"]["alone"], q["m1"]["plus"], COLOR["agents"]),
            ("agents m1all, $\\lambda_o=3$", q["m1all"]["alone"], q["m1all"]["plus"], COLOR["agents"])]
    fig, ax = plt.subplots(figsize=(TEXTWIDTH * 0.72, 2.0))
    for y, (name, alone, plus, color) in enumerate(rows):
        a = None if alone is None else mean(np.atleast_1d(alone))
        p = None if plus is None else mean(np.atleast_1d(plus))
        if a is not None and p is not None:
            ax.annotate("", xy=(p, y), xytext=(a, y), arrowprops=dict(arrowstyle="-|>", color=color, lw=0.9,
                                                                      shrinkA=3, shrinkB=3, mutation_scale=7))
        if a is not None:
            ax.scatter([a], [y], s=22, facecolor="white", edgecolor=color, linewidth=1.1, zorder=3)
        if p is not None:
            ax.scatter([p], [y], s=22, color=color, zorder=3)
            ax.text(p + 0.008, y, pct(p) + "%", va="center", fontsize=6.5)
    ax.set_yticks(range(len(rows)), [r[0] for r in rows])
    ax.invert_yaxis()
    ax.axvline(0, color="black", linewidth=0.5)
    percent_axis(ax, "x")
    ax.set_xlim(-0.05, 0.31)
    ax.set_xlabel("HPWL improvement over the greedy start (adaptec1)")
    ax.scatter([], [], s=22, facecolor="white", edgecolor="black", label="method alone")
    ax.scatter([], [], s=22, color="black", label="then swap search")
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2)
    fig.savefig(out)
    plt.close(fig)


def fig_swarm_trace(out):
    """HPWL per swarm round: all agreed swaps at once vs wired swaps wait, against the central search."""
    fig, axes = plt.subplots(1, 3, figsize=(TEXTWIDTH, 1.7))
    for ax, chip in zip(axes, CHIPS):
        for name, key, color, style in (("all swaps at once", f"exact-{chip}-k0", COLOR["neutral"], "--"),
                                        ("wired swaps wait", f"exact-{chip}-k0-resolve", COLOR["swap"], "-")):
            s = results.summary(RUNS / "swarmswap" / key)
            traj = 1 - np.asarray(s["hpwl_trajectory"]) / s["warm_hpwl"]
            ax.plot(range(len(traj)), traj, style, color=color, label=name, marker="o", markersize=1.8)
        central = results.summary(RUNS / "q3" / f"swap-{chip}")["hpwl_improvement"]
        ax.axhline(central, color="black", linewidth=0.8, linestyle=":", label="central swap search")
        ax.axhline(0, color="black", linewidth=0.5)
        percent_axis(ax)
        ax.set_title(chip, fontsize=7.5, pad=3)
        ax.set_xlabel("round")
    handles, names = axes[0].get_legend_handles_labels()
    fig.legend(handles, names, loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=3)
    fig.subplots_adjust(wspace=0.35)
    fig.savefig(out)
    plt.close(fig)


# ----------------------------------------------------------------------------------------------
# Tables and quoted numbers


def ms(values, bold=False):
    """mean ± std over seeds, as LaTeX."""
    values = list(np.atleast_1d(values))
    m = mean(values)
    body = "0.0" if abs(m) < 5e-4 else f"{'+' if m >= 0 else '-'}{abs(100 * m):.1f}"
    if len(values) > 1:
        body += f"\\,\\pm\\,{100 * sd(values):.1f}"
    return f"$\\mathbf{{{body}}}$" if bold else f"${body}$"


def table_main(data):
    s2, base, w, jit = data["sweep2"], data["baselines"], data["swarm"], data["jitter"]
    agents = {c: [r["final" if c == "adaptec1" else c] for r in s2["m1all"]] for c in CHIPS}
    rows = [
        ("\\emph{No learning}", None),
        ("Random jitter + legalize (control)", {c: jit[c]["new"] for c in CHIPS}),
        ("Pull toward partners", {c: data["pull"][c]["k4"]["32"] for c in CHIPS}),
        ("Swap search (central)", {c: w["central"][c] for c in CHIPS}),
        ("\\emph{Direct optimization}", None),
        ("Adam, $\\lambda_o=0$", {c: base["adam_ov0"][c]["final"] for c in CHIPS}),
        ("Adam, $\\lambda_o=3$", {c: base["adam_ov3"][c]["final"] for c in CHIPS}),
        ("\\emph{Decentralized agents (ours)}", None),
        ("Nudging agents, m1all, $\\lambda_o=3$$^\\dagger$", agents),
        ("Swap swarm, exact judge", {c: w["resolve"][c] for c in CHIPS}),
        ("Nudging agents, then swap swarm$^{\\dagger\\ast}$", {c: w["nudge_swap"][c] for c in CHIPS}),
    ]
    best = {c: max(mean(np.atleast_1d(v[c])) for _, v in rows if v) for c in CHIPS}
    lines = ["\\begin{tabular}{@{}lccc@{}}", "\\toprule",
             "Method & adaptec1 & bigblue1 & ariane133 \\\\",
             " & \\small (training) & \\small (unseen) & \\small (unseen) \\\\", "\\midrule"]
    for name, vals in rows:
        if vals is None:
            lines.append(f"\\multicolumn{{4}}{{@{{}}l}}{{{name}}} \\\\")
            continue
        cells = [ms(vals[c], bold=abs(mean(np.atleast_1d(vals[c])) - best[c]) < 1e-9) for c in CHIPS]
        lines.append(f"\\quad {name} & " + " & ".join(cells) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    return "\n".join(lines)


def table_ablation(data):
    s1, s2 = data["sweep1"], data["sweep2"]

    def row(name, entries, has_ariane=True):
        home = ms([e["final"] for e in entries])
        ov = f"${100 * mean(e['overlap'] for e in entries):.0f}$\\%"
        bb = ms([e["bigblue1"] for e in entries])
        ar = ms([e["ariane133"] for e in entries]) if has_ariane else "--"
        return f"{name} & {home} & {ov} & {bb} & {ar} \\\\"

    lines = ["\\begin{tabular}{@{}lcccc@{}}", "\\toprule",
             "Variant (m1all unless noted) & adaptec1 & overlap$^\\ddagger$ & bigblue1 & ariane133 \\\\",
             "\\midrule", "\\multicolumn{5}{@{}l}{\\emph{Overlap penalty and step schedule}} \\\\",
             row("\\quad m1, $\\lambda_o=0$", s1["m1"], False),
             row("\\quad $\\lambda_o=0$", s1["m1all"], False),
             row("\\quad $\\lambda_o=3$", s1["m1all-ov3"], False),
             row("\\quad $\\lambda_o=10$", s1["m1all-ov10"], False),
             row("\\quad steps $16\\to1$, $\\lambda_o=0$", s1["m1all-sched"], False),
             row("\\quad steps $16\\to1$, $\\lambda_o=10$", s1["m1all-sched-ov10"], False),
             "\\midrule", "\\multicolumn{5}{@{}l}{\\emph{Swarm rules and generalization ($\\lambda_o=3$)}} \\\\",
             row("\\quad none (reference)", s2["m1all"]),
             row("\\quad alignment $a=0.5$", s2["align0.5"]),
             row("\\quad alignment $a=0.9$", s2["align0.9"]),
             row("\\quad random half of macros move", s2["async0.5"]),
             row("\\quad noisy starts ($\\sigma=4$ cells)", s2["noise4"]),
             row("\\quad noisy starts + trained on bigblue1$^\\S$", s2["noise4-2chips"]),
             "\\bottomrule", "\\end{tabular}"]
    return "\n".join(lines)


def table_swarm(data):
    w = data["swarm"]

    def cells(key):
        out = []
        for c in CHIPS:
            v = w[key][c]
            out.append("--" if v is None else ms(v))
        return " & ".join(out)

    lines = ["\\begin{tabular}{@{}llccc@{}}", "\\toprule",
             "Judge & Variant & adaptec1 & bigblue1 & ariane133 \\\\", "\\midrule",
             f"--- & central swap search (reference) & {cells('central')} \\\\", "\\midrule",
             f"exact & all agreed swaps at once & {cells('at_once')} \\\\",
             f"exact & wired swaps wait (Alg.~\\ref{{alg:swarm}}) & {cells('resolve')} \\\\",
             f"exact & wired swaps wait, 8 nearest candidates & {cells('resolve_k8')} \\\\", "\\midrule",
             f"learned & view m0 & {cells('m0')} \\\\",
             f"learned & view m1 & {cells('m1')} \\\\",
             f"learned & view m1all & {cells('m1all')} \\\\",
             f"learned & m1, m1all, also trained on bigblue1 & {cells('two_chips')} \\\\", "\\midrule",
             f"exact & after the agents' nudges & {cells('nudge_swap')} \\\\",
             "\\bottomrule", "\\end{tabular}"]
    return "\n".join(lines)


def table_benchmarks():
    lines = ["\\begin{tabular}{@{}llccl@{}}", "\\toprule",
             "Design & Source & Macros placed & Canvas covered & Macro shapes \\\\", "\\midrule"]
    info = {"adaptec1": ("ISPD 2005~\\citep{nam2005ispd}", "varied"),
            "bigblue1": ("ISPD 2005~\\citep{nam2005ispd}", "thin, varied"),
            "ariane133": ("Ariane RISC-V~\\citep{zaruba2019ariane,cheng2023assessment}", "128 identical SRAMs")}
    for chip in CHIPS:
        ctx, _, _ = results.env(chip)
        sizes = np.asarray(ctx.sizes_grid)
        canvas = np.asarray(ctx.canvas)
        util = (sizes[:, 0] * sizes[:, 1]).sum() / (canvas[0] * canvas[1])
        lines.append(f"{chip} & {info[chip][0]} & {ctx.n_macros} & ${100 * util:.1f}$\\% & {info[chip][1]} \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    return "\n".join(lines)


def numbers(data, frames):
    """Every number the prose quotes, as a LaTeX macro."""
    s1, s2, base, q, w, jit, u = (data[k] for k in ("sweep1", "sweep2", "baselines", "q3", "swarm", "jitter", "unseen"))
    home = lambda cfg: mean(r["final"] for r in s2[cfg])
    trans = lambda cfg, chip: mean(r[chip] for r in s2[cfg])
    two_chips_ariane = w["two_chips"]["ariane133"]
    n = {
        "FirstRunBest": max(r["final"] for r in s1["m1"]),
        "FirstRunMedian": mean(r["final"] for r in s1["m1"]),
        "PenaltyThree": mean(r["final"] for r in s1["m1all-ov3"]),
        "PenaltyThreeSd": sd([r["final"] for r in s1["m1all-ov3"]]),
        "NoPenaltySpread": max(r["final"] for r in s1["m1"]) - min(r["final"] for r in s1["m1"]),
        "AgentsHome": home("m1all"),
        "AgentsHomeMzero": home("m0"),
        "AgentsHomeMone": home("m1"),
        "TransferMzeroBigblue": trans("m0", "bigblue1"),
        "TransferMoneBigblue": trans("m1", "bigblue1"),
        "TransferAsyncBigblue": trans("async0.5", "bigblue1"),
        "TransferBestAriane": max(trans(cfg, "ariane133") for cfg in s2),
        "TransferWorstAriane": min(trans(cfg, "ariane133") for cfg in s2),
        "AlignHome": home("align0.9"),
        "JitterBigblueMax": max(jit["bigblue1"]["new"]),
        "JitterBigblueMin": min(jit["bigblue1"]["new"]),
        "JitterArianeOld": mean(jit["ariane133"]["old"]),
        "JitterArianeNew": mean(jit["ariane133"]["new"]),
        "AdamZeroHome": base["adam_ov0"]["adaptec1"]["final"],
        "AdamThreeHome": base["adam_ov3"]["adaptec1"]["final"],
        "AdamZeroAriane": base["adam_ov0"]["ariane133"]["final"],
        "AdamThreeBigblue": base["adam_ov3"]["bigblue1"]["final"],
        "SwapHome": w["central"]["adaptec1"], "SwapBigblue": w["central"]["bigblue1"], "SwapAriane": w["central"]["ariane133"],
        "SwarmHome": w["resolve"]["adaptec1"], "SwarmBigblue": w["resolve"]["bigblue1"], "SwarmAriane": w["resolve"]["ariane133"],
        "SwarmAtOnceHome": w["at_once"]["adaptec1"],
        "SwarmKeightAriane": w["resolve_k8"]["ariane133"],
        "NudgeSwapHome": w["nudge_swap"]["adaptec1"],
        "AgentsPlusSwaps": mean(q["m1"]["plus"]),
        "AdamPlusSwapsBest": max(q["adam3"]["plus"], q["adam0"]["plus"]),
        "LearnedTwoChipsAriane": mean(two_chips_ariane),
        "PullHome": q["pull"]["alone"],
    }
    lines = [f"\\newcommand{{\\n{k}}}{{{tex_pct(v)}}}" for k, v in n.items()]
    lines.append(f"\\newcommand{{\\nPenaltyThreeSdPts}}{{${100 * n['PenaltyThreeSd']:.1f}$}}")
    lines.append(f"\\newcommand{{\\nNoPenaltySpreadPts}}{{${100 * n['NoPenaltySpread']:.0f}$}}")
    move_old = mean(jit["ariane133"]["old_move"])
    move_new = mean(jit["ariane133"]["new_move"])
    lines.append(f"\\newcommand{{\\nMoveOldAriane}}{{${move_old:.0f}$}}")
    lines.append(f"\\newcommand{{\\nMoveNewAriane}}{{${move_new:.0f}$}}")
    adam_log = [json.loads(line) for line in (RUNS / "sweep2/adam-ov0-ariane133/log.jsonl").read_text().splitlines()]
    last = [line for line in adam_log if "eval_real_hpwl_snapped" in line][-1]
    warm = json.loads((RUNS / "sweep2/adam-ov0-ariane133/manifest.json").read_text())["warm_start"]["real_hpwl_snapped"]
    lines.append(f"\\newcommand{{\\nAdamRawAriane}}{{{tex_pct(1 - last['eval_real_hpwl_snapped'] / warm)}}}")
    scorer = json.loads((RUNS / "swarmswap/net-m1all-s0/scorer.json").read_text())
    lines.append(f"\\newcommand{{\\nScorerSamples}}{{${scorer['samples']:,}$}}".replace(",", "{,}"))
    lines.append(f"\\newcommand{{\\nSwarmRoundsAriane}}{{{frames['rounds']}}}")
    lines.append(f"\\newcommand{{\\nSwarmSwapsAriane}}{{{frames['swaps']}}}")
    return "\n".join(lines) + "\n"


# ----------------------------------------------------------------------------------------------


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--recompute", action="store_true")
    parser.add_argument("--no-pdf", action="store_true")
    args = parser.parse_args(argv)

    data = results.collect(args.recompute)
    figs, tabs = HERE / "figures", HERE / "tables"
    figs.mkdir(exist_ok=True)
    tabs.mkdir(exist_ok=True)
    print("figures ...", flush=True)
    fig_placements(figs / "placements.pdf")
    fig_dynamics(data, figs / "dynamics.pdf")
    fig_jitter(data, figs / "jitter.pdf")
    fig_sight(data, figs / "sight.pdf")
    fig_q3(data, figs / "q3.pdf")
    fig_swarm_trace(figs / "swarm_trace.pdf")
    fig_swarm_frames(figs / "swarm_frames.pdf", tabs / "swarm_frames.json")
    print("tables ...", flush=True)
    (tabs / "main.tex").write_text(table_main(data))
    (tabs / "ablation.tex").write_text(table_ablation(data))
    (tabs / "swarm.tex").write_text(table_swarm(data))
    (tabs / "benchmarks.tex").write_text(table_benchmarks())
    (HERE / "numbers.tex").write_text(numbers(data, json.loads((tabs / "swarm_frames.json").read_text())))

    media = HERE / "media"
    media.mkdir(exist_ok=True)
    for name in ("ariane133-resolve", "ariane133-k8", "bigblue1-resolve", "adaptec1-all-at-once",
                 "adaptec1-resolve", "adaptec1-nudge-then-swap"):
        shutil.copy(RUNS / "swarmswap/gifs" / f"{name}.gif", media / f"{name}.gif")
    shutil.copy(RUNS / "sweep2/m1all-s0/viz/episode.gif", media / "agents-episode.gif")

    if not args.no_pdf:
        tectonic = os.environ.get("TECTONIC") or shutil.which("tectonic") or str(pathlib.Path.home() / ".local/bin/tectonic")
        subprocess.run([tectonic, "-X", "compile", "paper.tex"], cwd=HERE, check=True)
        print(f"wrote {HERE / 'paper.pdf'}")


if __name__ == "__main__":
    main()
