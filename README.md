
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

## Experiments

Every run is described by one `ExperimentConfig` and executed by one shared loop, so two runs can
actually be compared. A config is split in two:

- **`environment`** - benchmark, grid, macro order/budget, reward, observation, action mask, and
  the compute budget. This must be *identical* between two runs for their results to mean
  anything side by side.
- **`agent`** - policy, optimizer, algorithm, loop shape. This is the thing *under test*.

That split is enforced mechanically rather than by careful reading:

```python
from placax_agents.experiment import Budget, assert_comparable, presets, run_experiment

budget = Budget(env_steps=5_000_000)
a = presets.maskplace("benchmarks/adaptec1", budget=budget)
b = presets.training("benchmarks/adaptec1", budget=budget)

assert_comparable(a, b)   # raises, listing every axis on which they differ
```

Two runs are comparable if and only if their `environment_hash()` matches; `full_hash()`
additionally identifies the agent and seed. Both are written into every run's outputs, so a
results file is attributable to the configuration that produced it without any external
bookkeeping.

Each run writes, into its output directory:

| file | contents |
|---|---|
| `manifest.json` | the full config, both hashes, and a machine fingerprint (git SHA, library versions, backend, device), written **before** training so a crashed run is still attributable |
| `training_log.jsonl` | one line per iteration, each carrying `full_hash`, `env_steps`, `episodes`, `wall_clock_s`, `loss` and `real_hpwl` |
| `state.json` | budget spend, so a resumed run continues the same budget instead of starting a fresh one |
| `checkpoint.bin` | resumable training state |
| `best_checkpoint.bin` | bare weights plus the `real_hpwl` that earned them |

Re-run a recorded configuration exactly with `--config=<path to a manifest's config>`; every flag
describing the setup is then ignored in favor of the recorded one.

### Compute budget

`--n_iterations` is not a comparable unit: one iteration is a single episode in the baseline loop
and ten episodes plus ten minibatch epochs in the MaskPlace loop. Budget in **env steps** instead
(one env step = one macro placed), which every agent family pays in identically:

```sh
python -m scripts.run_maskplace --benchmark_dir=benchmarks/adaptec1 --env_steps=5000000
```

The loop refuses to *start* an iteration that would exceed the cap, so two different loop shapes
given one `--env_steps` budget both finish at or below it rather than overshooting by a whole
iteration each. `--wall_clock_s` is also available and accumulates across resumes, but measures
the hardware as much as the method. Budgets may be combined; the first to bind stops the run, and
which one it was is logged.

### Comparing agents

`scripts/compare_agents.py` is the point of all of the above: several agents, one environment, one
budget, one scoring path.

```sh
python -m scripts.compare_agents --benchmark_dir=benchmarks/adaptec1 \
    --env_steps=500000 --agents=greedy_wiremask,random_search,ppo --seeds=3
```

It asserts the environment hash matches across every run *before* anything starts, so it refuses
to produce a table rather than producing a misleading one, and it computes each agent's HPWL
itself - an agent hands over a placement, never a score.

Three agents ship. `ppo` is the learner; the other two are baselines the project previously had
none of, which is why "better than X" could not be stated even against a trivial reference:

- **`random_search`** - uniformly-random legal placements, keeping the best. The honest compute
  floor. A method that does not clearly beat compute-matched random search has not demonstrated
  anything, and almost nothing in this literature reports it.
- **`greedy_wiremask`** - each macro at the legal cell that adds least wirelength. The classical
  strong baseline, and the one that says how much of a learned policy's score comes from learning
  rather than from the wiremask observation it was handed.

Adding a fourth is one entry in `AGENTS` (`placax_agents/experiment/build.py`) and a class with
three methods (`placax_agents/agents/base.py`); it inherits the shared evaluation, checkpointing,
budgeting and logging automatically.

### Reproducibility

**JAX on GPU is not run-to-run deterministic in this project.** Two identical runs (same seed,
same process, same machine) diverge - measured at ~1e-16 by the second PPO iteration and ~1e-10 by
the fifth. It is localized to the backward pass: forward passes and rollouts reproduce exactly,
but repeated calls to the same jitted `jax.grad(ppo_loss)` return convolution gradients differing
by up to 7.5e-9. Neither `--xla_gpu_deterministic_ops=true` nor
`--xla_gpu_exclude_nondeterministic_ops=true` removes it. The CPU backend *is* bit-exact, and CPU
and GPU disagree with each other, so results are not comparable across backends either.

What follows from that:

- A single GPU run is not a reproducible result. Vary `--seed` and report across seeds.
- Every run's `manifest.json` records the backend and device, so two results are never compared
  across different hardware by accident.
- `PLACAX_DETERMINISTIC=1` forces the CPU backend, the only configuration that actually delivers
  bit-exactness. The checkpoint-resume tests assert bit-exactness there and, elsewhere, assert
  resume is within the backend's own measured noise floor.

See `placax/reproducibility.py` for the measurement and `tests/determinism.py` for how the noise
floor is established.

## Training (MaskPlace pipeline)

```sh
python -m scripts.run_maskplace --benchmark_dir=benchmarks/adaptec1 --n_iterations=300 --n_episodes=10 --eval_every=5 --placement_images --patience=10
```

- `--benchmark_dir`: path to a downloaded benchmark (see `scripts/download_benchmarks.py`); default `benchmarks/adaptec1`.
- `--seed`: RNG seed for policy init and rollout sampling; default `42`, MaskPlace's own. Vary it to reproduce MaskPlace's own mean±std-across-seeds reporting - which, per the note above, is the only defensible way to report a GPU result.
- `--n_iterations`: budget in TOTAL training iterations, not an additional count - resuming at or past it runs zero further iterations. Default `100`, applied only when no other budget flag is given.
- `--env_steps` / `--wall_clock_s`: the other two budget dimensions; see **Compute budget** above.
- `--macro_budget`: place only the N most important macros (MaskPlace's `--pnm`); default `all`, matching the paper, which places every macro by RL. PPO2.py's own `--pnm` default is `128`.
- `--n_episodes`: episodes collected per PPO update; default `10`, MaskPlace's own value. To find the largest value your GPU supports, use `scripts/subprocess_search.py` *separately first* rather than picking a number blind.
- `--entropy_coef`: entropy bonus coefficient; default `0.0`, MaskPlace's own value.
- `--regularity_weight` / `--regularity_mode`: weight and shape of EXPlace's regularity (periphery) reward term, normalized to [0, 1] per macro; default `0.0` (off - pure MaskPlace) and `corner`. Run `scripts/measure_reward_terms.py` on the benchmark to size the weight against the HPWL term's actual magnitude.
- `--init_from`: warm-start weights from a bare-variables checkpoint (typically a previous run's `best_checkpoint.bin`) instead of random init; training then starts fresh at iteration 0 with a new optimizer state and budget.
- `--log_every`: print a progress line every this many iterations; default `1`.
- `--eval_every`: compute real HPWL (a full extra greedy rollout, so not cheap) every this many iterations; default `10`.
- `--placement_images` / `--placement_images_dir`: also write a placement snapshot PNG on every `--eval_every` iteration, reusing that iteration's already-scheduled eval rollout, so it costs no extra rollout.
- `--patience`: stop early once real HPWL hasn't beaten its best for this many consecutive evals; default `0` (disabled).
- `--config`: run a recorded `ExperimentConfig` JSON exactly, ignoring the setup flags above.
- `--output_dir`: where the manifest, log and checkpoints go; default `<benchmark_dir>/output_maskplace`.
- `--no_checkpoint`: run entirely in memory - no manifest, checkpoint or log written.

Run `python -m scripts.run_maskplace --help` for the full flag list. The script auto-resumes from
its checkpoint on re-run, so it's safe to stop and restart with the same flags.

`scripts/run_training.py` is the plain-CNN baseline and takes the same budget, seed, output and
run-control flags, plus `--n_envs` for the vmapped parallel loop. It is deliberately a weak
baseline rather than a tuned competitor - it exists so a change to a single axis can be measured
against something simple, which is only meaningful now that both can be pinned to one environment
and budget.

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

## Physical validation (real PPA)

Every number above is a geometric half-perimeter proxy. `scripts/validate_design.py` runs the
actual physical flow on a macro-placed DEF - standard-cell placement, then area, utilization and
timing from a real signoff tool:

```sh
python -m scripts.validate_design --def_path=placed.def --lef=tech.lef --lef=cells.lef \
    --use_docker --liberty=cells.lib --clock_period_ns=2.0
```

Timing needs both `--liberty` and `--clock_period_ns`; without them area and utilization still
come back and `timing_slack` reports as a dash - not computed, never guessed.
`placax_tools/pipeline.py`'s `place_and_validate` names neither DREAMPlace nor OpenROAD, so
substituting RePlAce, AutoDMP or another signoff tool is a change at the call site and nowhere
else.

**Not yet verified end to end.** No DEF/LEF design ships with this repo and OpenROAD is not a
dependency, so the real binary has never been driven through this path - the TCL generation,
output parsing and the composition all have tests, but someone with a real design should run it
before trusting the numbers. The Bookshelf benchmarks under `benchmarks/` cannot reach it at all,
since they carry no LEF/DEF; that is why validation went unwired for so long.
