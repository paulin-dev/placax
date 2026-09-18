"""The finished runs as one table - what goes in the paper, and what says whether it is ready.

    python -m multiagent.compare multiagent/runs/*

Reads each run directory's `summary.json` and `manifest.json` and prints one row per run, sorted by
the number that matters: the best LEGAL placement's real HPWL, after repair. Every row also carries
what it cost (moves, wall clock) and how much the repair had to move, because a repair that
relocates half the design has replaced the method's answer with its own.

It refuses to put two runs in the same table quietly if they were not run on the same design and
budget - the same rule `assert_comparable` enforces in the core, for the same reason.
"""
import argparse
import json
import pathlib


def load(run_dir: pathlib.Path) -> dict | None:
    """One run's manifest and summary, or None if it never got far enough to have both."""
    manifest_path, summary_path = run_dir / "manifest.json", run_dir / "summary.json"
    if not manifest_path.exists():
        return None
    manifest = json.loads(manifest_path.read_text())
    summary = json.loads(summary_path.read_text()) if summary_path.exists() else None
    return {"dir": run_dir, "manifest": manifest, "summary": summary}


def describe(run: dict) -> str:
    """The method, as it should appear in a results table."""
    args = run["manifest"]["args"]
    if run["manifest"]["method"] == "adam_on_positions":
        return f"adam (lr={args['lr']})"
    return f"policy {args['view']} (h={args['horizon']}, step={args['max_step']})"


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("runs", nargs="+", type=pathlib.Path)
    args = parser.parse_args(argv)

    runs = [run for run in (load(path) for path in args.runs if path.is_dir()) if run]
    if not runs:
        raise SystemExit("no run directories with a manifest.json among those paths")

    # The environment axes that have to agree for a shared table to mean anything.
    def environment(run: dict) -> tuple:
        args = run["manifest"]["args"]
        return (run["manifest"]["netlist_digest"], args["grid"], args["macro_budget"],
                args["canvas"], args["density_weight"], args["target_density"],
                args["gamma_cells"])

    environments = {environment(run) for run in runs}
    warm = runs[0]["manifest"]["warm_start"]["real_hpwl_snapped"]

    print(f"\ndesign: {runs[0]['manifest']['args']['benchmark_dir']}  "
          f"macros: {runs[0]['manifest']['n_macros']}  "
          f"grid: {runs[0]['manifest']['args']['grid']}\n")
    print(f"{'method':<34}{'best legal HPWL':>17}{'vs greedy':>11}{'overlap':>9}"
          f"{'moves':>9}{'repair':>9}{'secs':>8}")
    print("-" * 97)
    print(f"{'greedy_wiremask (the warm start)':<34}{warm:>17,.0f}{'':>11}"
          f"{'0.00%':>9}{0:>9}{'-':>9}{'-':>8}")

    def sort_key(run: dict) -> float:
        best = (run["summary"] or {}).get("best_legal")
        return best["real_hpwl_snapped"] if best else float("inf")

    for run in sorted(runs, key=sort_key):
        best = (run["summary"] or {}).get("best_legal")
        if not best:
            print(f"{describe(run):<34}{'no legal placement':>17}"
                  f"{'':>11}{'':>9}{'':>9}{'':>9}{'':>8}")
            continue
        summary = run["summary"]
        moves = best["iteration"] * (run["manifest"]["args"].get("steps", 1)
                                     if run["manifest"]["method"] != "adam_on_positions" else 1)
        print(f"{describe(run):<34}{best['real_hpwl_snapped']:>17,.0f}"
              f"{1.0 - best['real_hpwl_snapped'] / warm:>+10.2%}"
              f"{best['overlap_ratio']:>9.2%}{moves:>9,}"
              f"{best.get('repair_mean_displacement_cells', 0.0):>8.1f}c"
              f"{summary['wall_clock_s']:>8.0f}")

    print("\n'vs greedy' is against the warm start every run started from; positive is better.")
    print("'moves' is all-macro moves up to the best checkpoint - one policy iteration is --steps "
          "of them,\nand one Adam step is exactly one, so the two columns are comparable.")
    print("'repair' is the mean distance the legalizer had to move a macro, in grid cells.")
    if len(environments) > 1:
        print(f"\nWARNING: these {len(runs)} runs span {len(environments)} DIFFERENT environments "
              f"(design, grid, budget, canvas or objective weights differ). The rows above are not "
              f"comparable - check each manifest before using this table.")


if __name__ == "__main__":
    main()
