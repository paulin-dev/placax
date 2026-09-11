"""Production pipeline: loads a trained checkpoint (no training happens here), places every macro via
the RL policy, then hands off to DREAMPlace to place every remaining standard cell. OpenROAD validation
is intentionally not wired up yet - placax_tools/openroad/validator.py is ready for it once this design
has real LEF/DEF (our Bookshelf-only benchmarks don't)."""
import argparse
import json
import os
import pathlib
import subprocess
import sys

from placax import _device  # noqa: F401  must precede jax imports
from placax.core import reset
from placax.log import Log
from placax.extras.mst import hpwl_wirelength
from placax.netlist.bookshelf import parse_all_node_sizes, parse_nets, parse_pl_positions
from placax_agents.experiment.build import build
from placax_agents.experiment.export import write_placement
from placax_agents.experiment.presets import OUTPUT_SUBDIRS, find_run_dir
from placax_agents.experiment.run import write_manifest
from scripts.presets import config_for
from placax_agents.ops.evaluate import evaluate
from placax_agents.ops.inference import is_bare_checkpoint, load_policy_variables
from placax_agents.policy.scale import to_grid_units
from placax_tools.cell_placer import CellPlacer
from placax_tools.dreamplace.cell_placer import DREAMPlaceCellPlacer
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
        help="Defaults to <benchmark_dir>/output_<preset>/best_checkpoint.bin if it exists, else "
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
        help="Defaults to <benchmark_dir>/<preset's own output subdir>/pipeline, e.g. output_maskplace/"
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
        help="Run DREAMPlace via its official Docker image (limbo018/dreamplace:cuda) instead of a "
             "local checkout - no matching GCC/Boost/Bison/Flex/CMake/PyTorch toolchain needed on the "
             "host. Clones + builds DREAMPlace into --dreamplace_root automatically on first use.",
    )
    parser.add_argument("--gpu", action="store_true", help="Run DREAMPlace on GPU.")
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
    subdir (e.g. output_maskplace, output - see placax_agents/experiment/presets.py)."""
    run_dir = find_run_dir(preset, benchmark_dir)
    checkpoint_path = checkpoint_arg or (run_dir / "best_checkpoint.bin")
    if checkpoint_arg is None and not checkpoint_path.exists():
        checkpoint_path = run_dir / "checkpoint.bin"
    if not checkpoint_path.exists():
        return checkpoint_path, True  # doesn't exist yet; bare is just a harmless default, caller errors next
    return checkpoint_path, is_bare_checkpoint(checkpoint_path)


def _build_cell_placer(
    dreamplace_root: pathlib.Path,
    gpu: bool,
    target_density: float,
    python_executable: str,
    use_docker: bool,
    extra_mounts: tuple[pathlib.Path, ...],
    extra_config: dict | None = None,
    config=None,
) -> CellPlacer:
    """The CellPlacer this run uses: the one its CONFIG names, or DREAMPlace from the flags.

    A config that names a cell placer in `EnvironmentSpec.physical` has that choice hashed and
    written into its manifest, and this script writes a manifest beside its outputs - so it has to
    honor the recorded choice rather than construct DREAMPlace regardless, which would put one
    tool's numbers under another tool's name. `build_physical` makes the same split the rest of the
    physical flow makes: WHICH tool comes from the config, WHERE it is installed comes from this
    machine's flags, and only the second belongs on a command line.

    Falls back to DREAMPlace when the config names none, which is every shipped preset - a proxy
    run has no physical stack, and this pipeline still has to place cells.
    """
    machine = {
        "dreamplace_root": dreamplace_root, "gpu": gpu, "use_docker": use_docker,
        "python_executable": python_executable, "extra_mounts": extra_mounts,
        "extra_config": extra_config, "target_density": target_density,
    }
    if config is not None and config.environment.physical.cell_placer is not None:
        from placax_agents.experiment.build import build_physical

        Log.info(
            f"cell placer {config.environment.physical.cell_placer.name!r}, from this run's config"
        )
        placer, _validator = build_physical(config, machine)
        return placer
    return DREAMPlaceCellPlacer(
        dreamplace_root=dreamplace_root, gpu=gpu, target_density=target_density,
        python_executable=python_executable, use_docker=use_docker, extra_mounts=extra_mounts,
        extra_config=extra_config,
    )


def main() -> None:
    Log.configure()
    args = _parse_args(sys.argv)
    benchmark_dir = args.benchmark_dir
    preset, checkpoint_arg = args.preset, args.checkpoint
    macro_budget, output_dir_arg = args.macro_budget, args.output_dir
    dreamplace_root, use_docker, gpu = args.dreamplace_root, args.use_docker, args.gpu
    target_density, python_executable = args.target_density, args.python_executable
    dreamplace_extra_config = args.dreamplace_extra_config
    viz_resolution = args.viz_resolution
    nets_sample_fraction, nets_seed = args.nets_sample_fraction, args.nets_seed
    config_path = args.config

    # Resolve to absolute paths up front: every path written into the DREAMPlace config/.aux below must
    # stay valid inside the Docker container too, which runs with a different cwd (/DREAMPlace) than this
    # process - a relative path here would silently resolve against the WRONG directory in --use_docker mode.
    benchmark_dir = benchmark_dir.resolve()
    if output_dir_arg is not None:
        output_dir_arg = output_dir_arg.resolve()
    aux_candidates = list(benchmark_dir.glob("*.aux"))
    if not aux_candidates:
        Log.error(f"'{benchmark_dir}' has no .aux file - this pipeline only handles Bookshelf benchmarks for now.")
        sys.exit(1)
    design_name = aux_candidates[0].stem

    # Rebuild the environment from a CONFIG where one was given, so this pipeline runs the
    # checkpoint in the environment it was trained in rather than in whatever a preset name
    # happens to resolve to today. Hand-matching a --preset string to a checkpoint is exactly the
    # error class ExperimentConfig removed upstream, and it survived down here far too long.
    default_subdir = OUTPUT_SUBDIRS[preset]
    config = config_for(config_path, preset, benchmark_dir, macro_budget)

    checkpoint_path, bare = _resolve_checkpoint(benchmark_dir, preset, checkpoint_arg)
    if not checkpoint_path.exists():
        Log.error(f"'{checkpoint_path}' not found - train first (--preset={preset} expects a checkpoint "
                   f"matching that preset's own training script).")
        sys.exit(1)

    output_dir = output_dir_arg or (benchmark_dir / default_subdir / "pipeline")
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
    positions, hpwl_value = evaluate(
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

    # 5. Write the macro placement into the design's own format. DREAMPlace runs as an external
    # subprocess reading files from disk (docs/JAX_Placement_Environment_Spec.md section 5.7's
    # deliberate choice), so this is required, not optional. The writing itself lives in
    # placax_agents.experiment.export, shared with the physical evaluation, so the file measured
    # by a PPA run and the file written here can never be two different notions of "the placement".
    exported = write_placement(built, positions, output_dir)
    new_aux_path = exported.path

    if dreamplace_root is None:
        print()
        print(f"macro placement done - pass --dreamplace_root=<path to a DREAMPlace checkout> (or "
              f"--use_docker) to also place standard cells from {new_aux_path}.")
        print("OpenROAD validation is not run by this script yet (placax_tools/openroad/validator.py "
              "is ready for it once this design has real LEF/DEF).")
        return

    # 6. Hand off to a CellPlacer to place every remaining standard cell around the now-fixed
    # macros - the one this run's config names, or DREAMPlace when it names none. The call site
    # depends only on CellPlacer.place_bookshelf()'s generic contract (placax_tools/cell_placer.py),
    # so a different Bookshelf-capable placer is a registry entry and a config change.
    # Docker mode: the container only sees dreamplace_root (mounted at /DREAMPlace) plus whatever we
    # explicitly mount below - benchmark_dir (nodes/nets/wts/scl) and output_dir (pl/aux/config/result),
    # each at their own identical host path, so the absolute paths already written into new_aux_path
    # and the DREAMPlace config resolve unchanged inside the container too.
    common_mount_root = pathlib.Path(os.path.commonpath([benchmark_dir.resolve(), output_dir.resolve()]))
    cell_placer = _build_cell_placer(
        dreamplace_root, gpu, target_density, python_executable, use_docker, (common_mount_root,),
        extra_config=dreamplace_extra_config, config=built.config,
    )
    try:
        result_pl = cell_placer.place_bookshelf(new_aux_path, output_dir)
    except (FileNotFoundError, subprocess.CalledProcessError) as exc:
        Log.error(f"DREAMPlace cell placement failed ({exc}); macro-only outputs above are still valid.")
        sys.exit(1)
    Log.info(f"DREAMPlace wrote {result_pl}")

    # 7. Render the full result: every macro AND every cell, from DREAMPlace's own output file.
    all_sizes = parse_all_node_sizes(benchmark_dir / f"{design_name}.nodes")
    all_positions = parse_pl_positions(result_pl)
    names = [name for name in all_positions if name in all_sizes]
    pos = np.array([all_positions[name][:2] for name in names])
    sz = np.array([all_sizes[name] for name in names])
    die_width = float((pos[:, 0] + sz[:, 0]).max())
    die_height = float((pos[:, 1] + sz[:, 1]).max())
    macro_mask = np.array([name in benchmark.name_to_idx for name in names])

    full_png = output_dir / "full_placement.png"
    save_full_placement_image(
        pos[macro_mask], sz[macro_mask], pos[~macro_mask], sz[~macro_mask], die_width, die_height, full_png,
        resolution=viz_resolution,
    )
    Log.info(f"wrote {full_png}")

    # 8. Full-design HPWL (macros AND cells) from the actual final placement - distinct from real_hpwl
    # above, which only ever covers macro-to-macro nets (the RL reward's own scope). Plain Python
    # (hpwl_wirelength), not the JAX/padded-array hpwl() used in training - that form pads every net to
    # the netlist's max degree, which blows up in memory on a full netlist's high-fanout nets (clock/reset).
    full_centers = {name: (pos[i, 0] + sz[i, 0] / 2.0, pos[i, 1] + sz[i, 1] / 2.0) for i, name in enumerate(names)}
    full_nets = parse_nets(benchmark_dir / f"{design_name}.nets", set(all_sizes))
    full_hpwl = hpwl_wirelength(full_centers, full_nets)
    Log.info(f"full-design HPWL ({len(full_nets)} nets, macros + cells) = {full_hpwl:.2f}")

    print()
    print("pipeline complete")
    print(f"  real_hpwl(macros only, {benchmark.params.n_macros} macros, {len(benchmark.nets)} macro-macro nets) "
          f"= {float(hpwl_value):.2f}")
    print(f"  full_hpwl(macros + cells, {len(names)} nodes, {len(full_nets)} nets) = {full_hpwl:.2f}")
    print(f"full placement ({len(names)} nodes: {int(macro_mask.sum())} macros, "
          f"{int((~macro_mask).sum())} cells) written to {result_pl}")
    print("(routed wirelength / DRC / timing need OpenROAD - not run by this script yet)")


if __name__ == "__main__":
    main()
