"""Production pipeline: loads a trained checkpoint (no training happens here), places every macro via
the RL policy, then hands off to DREAMPlace to place every remaining standard cell - and then, when the
run's config names a validator, converts the finished design to DEF/LEF and measures real PPA.

That last step is why placax/netlist/def_export.py exists. OpenROAD reads no Bookshelf, so the
benchmarks that ship here could not reach a validator at all: the box was configured, hashed and
documented, and no design in the repository could be pushed through it."""
import argparse
import dataclasses
import json
import pathlib
import subprocess
import sys

from placax import _device  # noqa: F401  must precede jax imports
from placax.core import reset
from placax.log import Log
from placax_agents.experiment.build import build
from placax_agents.experiment.config import PhysicalSpec, Spec
from placax_agents.experiment.export import write_placement
from placax_agents.experiment.physical import place_cells_and_measure
from placax_agents.experiment import presets as run_layout
from placax_agents.experiment.presets import OUTPUT_SUBDIRS, find_run_dir
from placax_agents.experiment.run import write_manifest
from scripts.presets import config_for
from placax_agents.ops.evaluate import evaluate
from placax_agents.ops.inference import is_bare_checkpoint, load_policy_variables
from placax_agents.policy.scale import to_grid_units
from placax_viz.placement import save_full_placement_image, save_placement_image, save_placement_with_nets_image

import numpy as np
from jax import random

DEFAULT_DOCKER_DREAMPLACE_ROOT = pathlib.Path("placax_tools/dreamplace/DREAMPlace")
"""Where --use_docker clones DREAMPlace to if --dreamplace_root isn't given - gitignored, since it's a
full external repo + compiled build output, never ours to track."""


def _parse_args(argv: list[str]):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark_dir", type=pathlib.Path, default=pathlib.Path("benchmarks/adaptec1"))
    parser.add_argument(
        "--preset", choices=sorted(OUTPUT_SUBDIRS), default="maskplace",
        help="Which setup to rebuild before loading the checkpoint when no --config is given - "
             "must match what the checkpoint was actually trained with (scripts/run_maskplace.py's "
             "checkpoints need --preset=maskplace, the default; scripts/run_training.py's need "
             "--preset=training). Prefer --config, which removes the guessing entirely.",
    )
    parser.add_argument(
        "--checkpoint", type=pathlib.Path, default=None,
        help="Defaults to runs/<benchmark>-<preset>/best_checkpoint.bin (or its newest seed*/) if "
             "it exists, else "
             ".../checkpoint.bin.",
    )
    parser.add_argument(
        "--macro_budget", type=str, default="all",
        help='Defaults to "all" (every macro placed - the production default: neither shipped preset\'s '
             "network has any architectural dependence on macro count, so a checkpoint trained with any "
             "budget still loads and places every macro with no shape mismatch and no retraining - "
             "confirmed end-to-end for --preset=maskplace). Pass an integer to match a specific training "
             "budget instead, e.g. for a fast/partial preview. Ignored by --preset=training (always all).",
    )
    parser.add_argument(
        "--output_dir", type=pathlib.Path, default=None,
        help="Defaults to <the checkpoint's run directory>/pipeline when the checkpoint lives under "
             "runs/, else runs/<benchmark>-<preset>/pipeline"
             "pipeline for --preset=maskplace, output/pipeline for --preset=training.",
    )
    parser.add_argument(
        "--dreamplace_root", type=pathlib.Path, default=None,
        help="Path to a DREAMPlace checkout (containing dreamplace/Placer.py, or install/dreamplace/"
             "Placer.py once built). If omitted entirely (no --use_docker either), the pipeline stops "
             f"after macro placement. With --use_docker, defaults to {DEFAULT_DOCKER_DREAMPLACE_ROOT} "
             "and is cloned/built automatically if not already there.",
    )
    parser.add_argument(
        "--use_docker", action="store_true",
        help="Run the external tools from their official Docker images instead of local installs: "
             "DREAMPlace from limbo018/dreamplace:cuda (cloned + built into --dreamplace_root on "
             "first use) and OpenROAD from the pinned openroad/orfs image. Neither toolchain has to "
             "exist on the host.",
    )
    parser.add_argument("--gpu", action="store_true", help="Run DREAMPlace on GPU.")
    parser.add_argument("--validator", default=None,
                        help="Measure real PPA with this validator, as 'name' or "
                             "'name:key=value,...' - e.g. 'openroad' or 'openroad:route=global'. "
                             "Written into the run's config (so into its manifest and hash) rather "
                             "than applied on the side, which is what makes the resulting ppa.json "
                             "attributable. Overrides whatever validator the config names.")
    parser.add_argument("--physical", type=pathlib.Path, default=None,
                        help="A physical stack as JSON ({cell_placer, validator}), written into the "
                             "run's config - e.g. benchmarks/ariane133-orfs/physical.json from "
                             "scripts/make_orfs_benchmark.py.")
    parser.add_argument("--openroad_binary", default="openroad",
                        help="OpenROAD executable when not using Docker (default: %(default)s). A "
                             "property of THIS MACHINE, so deliberately not part of the config.")
    parser.add_argument("--target_density", type=float, default=1.0)
    parser.add_argument(
        "--python_executable", type=str, default="python",
        help="Local-checkout mode only: python interpreter DREAMPlace itself should run under "
             "(often a separate env from this one). Ignored with --use_docker.",
    )
    parser.add_argument(
        "--dreamplace_extra_config", type=str, default=None,
        help="A JSON object string overriding/adding any DREAMPlace config field (e.g. "
             '\'{"num_bins_x": 1024, "random_seed": 42}\') - forwarded to DREAMPlaceCellPlacer\'s own '
             "extra_config, which already accepts arbitrary overrides; this just exposes that from the CLI.",
    )
    parser.add_argument(
        "--viz_resolution", type=int, default=1024,
        help="Bin resolution for the post-DREAMPlace full_placement.png cell-density raster; default 1024.",
    )
    parser.add_argument(
        "--nets_sample_fraction", type=float, default=1.0,
        help="Randomly keep only this fraction of macro-to-macro nets in macros_with_nets.png (MaskPlace's "
             "own convention for a denser netlist - \"For clarity, we only show 1%% wires\"); default 1.0 "
             "(show every net).",
    )
    parser.add_argument("--nets_seed", type=int, default=0, help="Seed for --nets_sample_fraction's subsample.")
    parser.add_argument(
        "--config", type=pathlib.Path, default=None,
        help="A run's manifest.json (or a bare ExperimentConfig JSON). STRONGLY PREFERRED over "
             "--preset: it rebuilds the exact environment the checkpoint was trained in - reward, "
             "observation, action mask, macro budget AND initial placement - instead of asking you "
             "to hand-match a preset name to a checkpoint. --preset and --macro_budget are ignored "
             "when this is given.",
    )
    args = parser.parse_args(argv[1:])
    args.macro_budget = None if args.macro_budget.lower() == "all" else int(args.macro_budget)
    if args.dreamplace_root is None and args.use_docker:
        args.dreamplace_root = DEFAULT_DOCKER_DREAMPLACE_ROOT
    if args.dreamplace_root is not None:
        # Docker bind mounts (-v) need an absolute host path, not one resolved relative to whatever
        # directory `docker` itself happens to run from.
        args.dreamplace_root = args.dreamplace_root.resolve()
    args.dreamplace_extra_config = (
        json.loads(args.dreamplace_extra_config) if args.dreamplace_extra_config else {}
    )
    # Returned whole rather than as a positional tuple: this used to be fifteen values unpacked by
    # position, which broke every caller and test the moment a flag was added in the middle.
    return args


def _resolve_checkpoint(
    benchmark_dir: pathlib.Path, preset: str, checkpoint_arg: pathlib.Path | None
) -> tuple[pathlib.Path, bool]:
    """Returns (checkpoint_path, bare): bare=True for a bare-weights bundle (the production default - no
    optimizer/RNG state), detected from the file's own contents (is_bare_checkpoint) so this works
    whatever the file is actually called, not just the conventional best_checkpoint.bin/checkpoint.bin
    names. Only the DEFAULT --checkpoint path (when none is given) uses that naming convention, to pick
    which of the two conventional files to default to, under the given preset's own default output
    run directory (runs/<benchmark>-<preset> - see placax_agents/experiment/presets.py)."""
    run_dir = find_run_dir(preset, benchmark_dir)
    checkpoint_path = checkpoint_arg or (run_dir / "best_checkpoint.bin")
    if checkpoint_arg is None and not checkpoint_path.exists():
        checkpoint_path = run_dir / "checkpoint.bin"
    if not checkpoint_path.exists():
        return checkpoint_path, True  # doesn't exist yet; bare is just a harmless default, caller errors next
    return checkpoint_path, is_bare_checkpoint(checkpoint_path)


def _with_physical_stack(config, validator: str | None, target_density: float, place_cells: bool,
                         physical_path: pathlib.Path | None = None):
    """`config` with the tools this pipeline is about to run written into it.

    Both go INTO the config rather than beside it - they are part of the environment, so the
    manifest written next to the outputs has to name them. `physical_path` replaces the whole
    stack with a design's own (`scripts/make_orfs_benchmark.py` writes one); `validator` then
    still overrides its validator. A cell placer the config already names is kept; otherwise
    DREAMPlace, the one this pipeline has always used.
    """
    physical = config.environment.physical
    if physical_path is not None:
        physical = PhysicalSpec.from_dict(json.loads(pathlib.Path(physical_path).read_text()))
        Log.info(f"physical stack from {physical_path}")
    if validator is not None:
        physical = dataclasses.replace(physical, validator=Spec.parse(validator))
        Log.info(f"validator {validator!r} written into this run's config")
    if place_cells and physical.cell_placer is None:
        physical = dataclasses.replace(
            physical, cell_placer=Spec("dreamplace", {"target_density": target_density})
        )
    elif place_cells:
        Log.info(f"cell placer {physical.cell_placer.name!r}, from this run's config")
    return dataclasses.replace(
        config, environment=dataclasses.replace(config.environment, physical=physical)
    )


def _machine(args, benchmark_dir: pathlib.Path, output_dir: pathlib.Path) -> dict:
    """Where this host's tools live and how to run them - never part of the config.

    Docker mode: the DREAMPlace container sees dreamplace_root plus the benchmark (nodes/nets/
    wts/scl) and the output (pl/aux/config/result), each at its own host path, so the absolute
    paths written into the .aux resolve unchanged inside the container. Two mounts, not their
    common parent: for an output outside the repository that parent is `/`, which Docker refuses.
    """
    from placax_tools.openroad.docker import host_mounts

    output_dir.mkdir(parents=True, exist_ok=True)
    return {
        "dreamplace_root": args.dreamplace_root, "gpu": args.gpu, "use_docker": args.use_docker,
        "python_executable": args.python_executable,
        "extra_mounts": tuple(host_mounts([benchmark_dir, output_dir])),
        "extra_config": args.dreamplace_extra_config, "openroad_binary": args.openroad_binary,
    }


def _default_output_dir(checkpoint_path: pathlib.Path, preset: str,
                        benchmark_dir: pathlib.Path) -> pathlib.Path:
    """Beside the run that produced the checkpoint, so a pipeline's results sit with their weights.

    Only for a checkpoint under `runs/`; one kept anywhere else (an old download, a benchmark
    directory) must not pull generated output in beside it.
    """
    runs = run_layout.RUNS_DIR.resolve()
    run_dir = checkpoint_path.resolve().parent
    if run_dir.is_relative_to(runs):
        return run_dir / "pipeline"
    return run_layout.run_root(preset, benchmark_dir) / "pipeline"


def _print_ppa(ppa, full_hpwl: float, output_dir: pathlib.Path) -> None:
    """The measured numbers, with a dash for anything that did not run - never a guess."""
    def show(value, unit=""):
        return "-" if value is None else f"{value:,.2f}{unit}"

    print()
    print(f"real PPA ({ppa.validator}, {ppa.tool_version}) -> {output_dir / 'ppa.json'}")
    print(f"  design area        {show(ppa.design_area, ' um^2')}")
    print(f"  utilization        {show(ppa.utilization_pct, ' %')}")
    if ppa.legalized is not None:
        print(f"  final legalization {'OpenROAD detailed placement' if ppa.legalized else 'FAILED'}"
              f" - moved cells up to {show(ppa.legalization_max_displacement, ' um')} "
              f"(mean {show(ppa.legalization_mean_displacement, ' um')}), HPWL "
              f"{show(ppa.hpwl_before_legalization, '')} -> {show(ppa.hpwl, '')}")
    print(f"  placement legal    {'-' if ppa.placement_legal is None else ppa.placement_legal}")
    for rule, count in ppa.placement_violations.items():
        print(f"    {rule} check failed on {count:,}")
    own = f"this script's own, on the same design: {full_hpwl:,.2f}"
    if ppa.legalized is not None:
        own = (f"before legalization OpenROAD measured {show(ppa.hpwl_before_legalization, '')}, "
               f"this script {full_hpwl:,.2f}")
    print(f"  full-design HPWL   {show(ppa.hpwl, ' um')}   ({own})")
    print(f"  worst slack        {show(ppa.timing_slack, ' ns')}")
    print(f"  routed wirelength  {show(ppa.routed_wirelength, ' um')}")
    print(f"  DRC violations     {'-' if ppa.drc_violations is None else ppa.drc_violations}")
    for note in ppa.notes:
        print(f"  not measured: {note}")


def main() -> None:
    Log.configure()
    args = _parse_args(sys.argv)
    benchmark_dir = args.benchmark_dir
    preset, checkpoint_arg = args.preset, args.checkpoint
    macro_budget, output_dir_arg = args.macro_budget, args.output_dir
    dreamplace_root = args.dreamplace_root
    viz_resolution = args.viz_resolution
    nets_sample_fraction, nets_seed = args.nets_sample_fraction, args.nets_seed
    config_path = args.config

    # Resolve to absolute paths up front: every path written into the DREAMPlace config/.aux below must
    # stay valid inside the Docker container too, which runs with a different cwd (/DREAMPlace) than this
    # process - a relative path here would silently resolve against the WRONG directory in --use_docker mode.
    benchmark_dir = benchmark_dir.resolve()
    if output_dir_arg is not None:
        output_dir_arg = output_dir_arg.resolve()
    if not list(benchmark_dir.glob("*.aux")) and not list(benchmark_dir.glob("*.def")):
        Log.error(f"'{benchmark_dir}' is neither a Bookshelf (.aux) nor a DEF design - the full "
                  f"flow needs the standard cells, which a clustered protobuf does not carry.")
        sys.exit(1)

    # Rebuild the environment from a CONFIG where one was given, so this pipeline runs the
    # checkpoint in the environment it was trained in rather than in whatever a preset name
    # happens to resolve to today. Hand-matching a --preset string to a checkpoint is exactly the
    # error class ExperimentConfig removed upstream, and it survived down here far too long.
    config = config_for(config_path, preset, benchmark_dir, macro_budget)
    config = _with_physical_stack(
        config, args.validator, args.target_density,
        place_cells=args.dreamplace_root is not None, physical_path=args.physical,
    )
    cell_placer = config.environment.physical.cell_placer

    checkpoint_path, bare = _resolve_checkpoint(benchmark_dir, preset, checkpoint_arg)
    if not checkpoint_path.exists():
        Log.error(f"'{checkpoint_path}' not found - train first (--preset={preset} expects a checkpoint "
                   f"matching that preset's own training script).")
        sys.exit(1)

    output_dir = output_dir_arg or _default_output_dir(checkpoint_path, preset, benchmark_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Resolve the config into the same benchmark/policy/observation/mask/warm start the
    # checkpoint was trained with. One build() call, the same one every training run goes through.
    Log.info(f"loading {benchmark_dir} ...")
    built = build(config)
    benchmark, policy, state_fn = built.benchmark, built.policy, built.state_fn
    # The outputs below are attributable now: a manifest sits beside them naming the config, its
    # hashes and this machine.
    manifest_path = write_manifest(output_dir, built.config)
    Log.info(f"manifest -> {manifest_path}  run={built.config.full_hash()}")

    # 2. Load the trained weights - inference only, nothing here ever trains.
    obs0 = state_fn(reset(benchmark.params, built.initial_positions), benchmark.params,
                    benchmark.sizes_array)
    variables_template = policy.init(random.PRNGKey(0), obs0)
    optimizer = None if bare else built.optimizer
    variables = load_policy_variables(variables_template, checkpoint_path, bare=bare, optimizer=optimizer)
    Log.info(f"loaded weights from {checkpoint_path} ({'bare' if bare else 'full'} checkpoint)")

    # 3. One greedy rollout, placing every macro the environment left to the agent. The warm-start
    # prefix and the macro count come from the config: replaying a warm-started checkpoint from an
    # empty canvas, as this script used to, is a different environment and a different result.
    positions, orientations, hpwl_value = evaluate(
        variables, policy.apply, benchmark.params, benchmark.sizes_array, benchmark.cell_size,
        benchmark.padded_pin_idx, benchmark.padded_pin_offset, benchmark.valid_mask,
        state_fn, built.extra_illegal_fn, built.initial_positions, built.n_placed,
    )
    Log.info(f"placed {positions.shape[0]} macros ({built.n_placed} pre-placed by the "
             f"environment's initial placement), real_hpwl={float(hpwl_value):.2f}")

    # 4. Render the macro-only placement, with and without net connections.
    grid_sizes = to_grid_units(benchmark.sizes_array, benchmark.cell_size)
    macros_png = output_dir / "macros_placed.png"
    save_placement_image(
        positions, grid_sizes, benchmark.params.grid_x, benchmark.params.effective_grid_y, macros_png
    )
    Log.info(f"wrote {macros_png}")

    nets_png = output_dir / "macros_with_nets.png"
    save_placement_with_nets_image(
        positions, benchmark.sizes_array, benchmark.params.grid_x, benchmark.params.effective_grid_y,
        benchmark.padded_pin_idx, benchmark.padded_pin_offset, benchmark.valid_mask, benchmark.cell_size,
        nets_png, sample_fraction=nets_sample_fraction, seed=nets_seed,
    )
    Log.info(f"wrote {nets_png}")

    # 5. Without a cell placer there is nothing more to do than write the macros out.
    if cell_placer is None or (cell_placer.name == "dreamplace" and dreamplace_root is None):
        exported = write_placement(built, positions, output_dir)
        print()
        print(f"macro placement done - pass --dreamplace_root=<path to a DREAMPlace checkout> (or "
              f"--use_docker) to also place standard cells from {exported.path}.")
        print("real PPA needs the standard cells placed first, so validation is skipped too.")
        return

    # 6-8. The configured physical stack on this placement: the macros written out, every
    # standard cell placed around them, the full-design HPWL, and - with a validator - the
    # converted design measured into ppa.json. The same function a comparison runs per agent
    # (experiment.physical.place_cells_and_measure), so the two can never finish a placement
    # two different ways.
    try:
        design = place_cells_and_measure(
            built, positions, output_dir, _machine(args, benchmark_dir, output_dir), orientations,
        )
    except (FileNotFoundError, subprocess.CalledProcessError) as exc:
        detail = getattr(exc, "stderr", None) or exc
        Log.error(f"the physical flow failed: {detail}\nMacro-only outputs above are still valid.")
        sys.exit(1)
    Log.info(f"cell placer wrote {design.result_pl}")
    Log.info(f"full-design HPWL = {design.full_hpwl:.2f}")
    if design.ppa is None:
        Log.info("no validator in this run's config, so no PPA was measured - pass "
                 "--validator=openroad (or name one in EnvironmentSpec.physical) to close the flow.")

    # 9. Render the full result: every macro AND every cell.
    names = list(design.positions)
    pos = np.array([design.positions[name] for name in names])
    sz = np.array([design.sizes[name] for name in names])
    macro_mask = np.array([name in design.macro_names for name in names])
    full_png = output_dir / "full_placement.png"
    save_full_placement_image(
        pos[macro_mask], sz[macro_mask], pos[~macro_mask], sz[~macro_mask],
        float((pos[:, 0] + sz[:, 0]).max()), float((pos[:, 1] + sz[:, 1]).max()), full_png,
        resolution=viz_resolution,
    )
    Log.info(f"wrote {full_png}")

    print()
    print("pipeline complete")
    print(f"  real_hpwl(macros only, {benchmark.params.n_macros} macros, {len(benchmark.nets)} macro-macro nets) "
          f"= {float(hpwl_value):.2f}")
    print(f"  full_hpwl(macros + cells, {len(names)} nodes) = {design.full_hpwl:.2f}")
    print(f"full placement ({len(names)} nodes: {int(macro_mask.sum())} macros, "
          f"{int((~macro_mask).sum())} cells) written to {design.result_pl}")
    if design.ppa is not None:
        _print_ppa(design.ppa, design.full_hpwl, output_dir)

if __name__ == "__main__":
    main()
