
<h1 align="center">
    <picture>
        <source media="(prefers-color-scheme: dark)" srcset="assets/logo_dark.svg">
        <source media="(prefers-color-scheme: light)" srcset="assets/logo_light.svg">
        <img alt="placax" src="assets/logo_light.svg" width="300">
    </picture><br>
    <b>A fast, shared JAX environment for chip macro placement research</b>
</h1>

<p align="center">
  <a href="https://pypi.python.org/pypi/placax"><img src="https://img.shields.io/pypi/pyversions/placax.svg?style=flat" /></a>
  <a href= "https://badge.fury.io/py/placax"><img src="https://badge.fury.io/py/placax.svg" /></a>
  <a href= "https://github.com/paulin-dev/placax/blob/master/LICENSE.md"><img src="https://img.shields.io/badge/license-Apache2.0-blue.svg" /></a>
</p>

This is the detailled description



GitHub desc: A shared, differentiable JAX environment for chip macro placement


## Architecture

<img src="assets/placax_architecture.svg" alt="placax architecture" width="100%">

## Getting Started

```sh
pip install placax
```

## Training (MaskPlace pipeline)

```sh
python scripts/run_maskplace.py --benchmark_dir=benchmarks/adaptec1 --n_iterations=300 --n_episodes=10 --eval_every=5 --placement_images --patience=10
```

- `--benchmark_dir`: path to a downloaded benchmark (see `scripts/download_benchmarks.py`); default `benchmarks/adaptec1`.
- `--n_iterations`: target TOTAL buffered-PPO update cycle to train to, not an additional count - resuming from a checkpoint runs only the remainder needed to reach it (zero further iterations if already past it); default `100`.
- `--macro_budget`: place only the N most important macros (MaskPlace's `--pnm`); default `128`, MaskPlace's own value. Pass `all` to place every macro in the netlist instead - not yet verified to fit in memory or train well at that scale.
- `--n_episodes`: episodes collected per PPO update; default `10`, MaskPlace's own value. To find the largest value your GPU actually supports, use `scripts/subprocess_search.py` *separately first* (see below) rather than picking a number blind.
- `--log_every`: print a progress line to the console every this many iterations; default `1` (every iteration).
- `--eval_every`: compute real HPWL (a full extra greedy rollout, so not cheap) every this many iterations; default `10`.
- `--placement_images`: also write a placement snapshot PNG (`<benchmark_dir>/output_maskplace/placements/<iteration>.png` by default) - reuses the eval rollout that `--eval_every` already schedules, so it's free of extra rollouts; how many of those eval iterations actually get an image kept follows `--log_every`, not `--eval_every` (see `--placement_images_dir` to change where they're written).
- `--no_checkpoint`: don't read or write checkpoint.bin - a quick, disposable run that won't resume from or leave behind any state.
- `--patience`: stop early once real HPWL (per `--eval_every`) hasn't beaten its best value for this many consecutive evals; default `0` (disabled, always run the full `--n_iterations`).

Every iteration is appended to `<benchmark_dir>/output_maskplace/training_log.jsonl` regardless of `--log_every`, so the full history survives even if the console only shows a fraction of it.

Run `python scripts/run_maskplace.py --help` for the full flag list. The script auto-resumes from its checkpoint on re-run, so it's safe to stop and restart with the same flags.

### Finding the largest `--n_episodes` (or `--n_envs`) your hardware supports

`scripts/subprocess_search.py` sweeps a named flag across a list of candidate values, running the target script's own ordinary CLI once per value and stopping at the first one that doesn't fit - it deliberately imports nothing from placax/jax itself, so it can probe accurately without competing with its own subprocesses for GPU memory (see that module's docstring for why that matters). No cooperation is required from the target script:

```sh
python -m scripts.subprocess_search scripts.run_maskplace '--n_episodes=[1,2,4,8,10]' \
    --benchmark_dir=benchmarks/adaptec1 --macro_budget=128 --eval_every=1 --n_iterations=4 --no_checkpoint
```

Prints `RESULT=<largest value that worked>` - pass that as `--n_episodes` to the real training run. The same tool works for `scripts/run_training.py`'s `--n_envs`, or any other script following the same plain-CLI convention.

`--eval_every=1`/`--n_iterations=4` here deliberately don't match the real run's `--eval_every=10`: the eval rollout is a separately-compiled executable with its own memory footprint, so a probe needs to cross at least one eval boundary (plus a couple more iterations, to catch ordinary GPU allocator fragmentation drift) to be representative - forcing it every iteration reaches that footprint by iteration 1 instead of iteration 10, so 4 iterations suffice instead of 13. Trade-off: JAX's allocator can behave slightly differently depending on the exact iteration pattern (eval every iteration vs. only every 10th), so this is a faster but marginally less exact stand-in for the real run's precise allocator history.

## Production pipeline (inference)

Loads an already-trained checkpoint (no training happens here) and runs the full production flow: the RL
policy places every macro, then [DREAMPlace](https://github.com/limbo018/DREAMPlace) places every
remaining standard cell around them. OpenROAD validation isn't wired up yet.

```sh
python -m scripts.run_pipeline --benchmark_dir=benchmarks/adaptec1 --checkpoint=benchmarks/adaptec1/output_maskplace/best_checkpoint.bin --use_docker
```

- `--benchmark_dir`: path to a downloaded Bookshelf benchmark (only format supported so far); default `benchmarks/adaptec1`.
- `--preset`: which benchmark/policy/state_fn/reward setup to rebuild before loading the checkpoint - must match what it was actually trained with; default `maskplace` (`scripts/run_maskplace.py`'s own setup). `training` uses `scripts/run_training.py`'s plain CNN setup instead. See `scripts/presets.py` to register a custom one - this pipeline isn't tied to MaskPlace specifically.
- `--checkpoint`: bare-weights or full training-state checkpoint to load (auto-detected from its contents, not its filename); defaults to `<benchmark_dir>/<preset's own output subdir>/best_checkpoint.bin` if it exists, else `.../checkpoint.bin`.
- `--macro_budget`: default `all` - every macro placed, the production default. Neither shipped preset's network has any architectural dependence on macro count, so a checkpoint trained with any budget (e.g. MaskPlace's own default of 128) still loads and places every macro with no shape mismatch and no retraining. Pass an integer instead to match a specific training budget, e.g. for a fast/partial preview.
- `--output_dir`: where every output (placement PNGs, the DREAMPlace `.pl`/`.aux`/config, its result) is written; defaults to `<benchmark_dir>/<preset's own output subdir>/pipeline`.
- `--use_docker`: run DREAMPlace via its official Docker image (`limbo018/dreamplace:cuda`) instead of a local checkout - no matching GCC/Boost/Bison/Flex/CMake/PyTorch toolchain needed on the host. Clones and builds DREAMPlace into `--dreamplace_root` automatically on first use (a few minutes, one time only).
- `--dreamplace_root`: path to a DREAMPlace checkout. Defaults to `placax_tools/dreamplace/DREAMPlace` with `--use_docker` (auto-cloned/built there); if omitted with no `--use_docker` either, the pipeline stops after macro placement and just writes the Bookshelf files DREAMPlace would need.
- `--gpu`: run DREAMPlace on GPU.
- `--target_density`: DREAMPlace's target placement density; default `1.0`.
- `--python_executable`: local-checkout mode only - the Python interpreter DREAMPlace itself should run under (often a separate env from this one); ignored with `--use_docker`.
- `--dreamplace_extra_config`: a JSON object string overriding/adding any DREAMPlace config field, e.g. `--dreamplace_extra_config='{"num_bins_x": 1024, "random_seed": 42}'`.
- `--viz_resolution`: bin resolution for `full_placement.png`'s cell-density raster; default `1024`.
- `--nets_sample_fraction` / `--nets_seed`: randomly keep only this fraction of macro-to-macro nets in `macros_with_nets.png` (MaskPlace's own convention for a denser netlist); default `1.0` (every net).

Writes `macros_placed.png` and `macros_with_nets.png` (macro-only, always) plus, once DREAMPlace succeeds, `full_placement.png` (every macro and every cell) and the placed design itself at `<output_dir>/<design_name>/<design_name>.gp.pl`. Prints two HPWL numbers: `real_hpwl` (macro-to-macro nets only, the RL reward's own scope) and `full_hpwl` (every net, macros and cells, from the actual final placement) - both are geometric (half-perimeter) proxies, not a routed wirelength; that needs OpenROAD.

Run `python -m scripts.run_pipeline --help` for the full flag list.