# Placax reference

A map of the repository and the pieces you touch when running research. For *why* things are
shaped this way see `docs/JAX_Placement_Environment_Spec.md`; this is the what and where.

## The one thing to know

Every run is one `ExperimentConfig`, resolved by `build()`, executed by `run_experiment()`.
The config splits in two: **`environment`** must be identical between two runs for their numbers
to be comparable, **`agent`** is the thing under test. `assert_comparable()` checks that
mechanically instead of by reading two scripts.

```python
from placax_agents.experiment import Budget, assert_comparable, presets, run_experiment

budget = Budget(env_steps=200_000)
a = presets.training("benchmarks/adaptec1", budget=budget)
b = presets.maskplace("benchmarks/adaptec1", budget=budget)
assert_comparable(a, b)               # raises, listing every axis on which they differ
run_experiment(a, output_dir=pathlib.Path("runs/adaptec1-training"))
```

## Layout

| Directory | Tier | What lives there |
|---|---|---|
| `placax/` | environment library | the `reset`/`step` kernel, netlist parsing, pure-JAX rewards and masks |
| `placax_agents/` | training loops + experiment layer | agents, policies, PPO, and everything under `experiment/` |
| `placax_tools/` | external tools | `CellPlacer`/`Validator` ABCs, DREAMPlace and OpenROAD wrappers |
| `placax_viz/` | plotting | placement images, masks, curves, rollout animation |
| `scripts/` | entry points | what changes per experiment |
| `benchmarks/` | designs | adaptec1, bigblue1 (Bookshelf), ariane133 (protobuf) |
| `runs/` | output | gitignored; one directory per run |
| `docs/`, `tests/` | | the spec and decision records; the suite |

### Files worth knowing by name

| Path | What |
|---|---|
| `placax/core.py` | `reset` / `step` / `replay` — the one kernel every agent drives |
| `placax/action_space.py` | what an action IS, what it changes, when the episode ends |
| `placax/types.py` | `EnvState`, `EnvParams`, `RewardFn` |
| `placax/extras/rewards.py` | `hpwl`, `wiremask`, `smoothed_wirelength` |
| `placax/extras/legality.py` | overlap / out-of-bounds / completeness of a finished placement |
| `placax/extras/orientation.py` | macro orientation as a transform on the geometry inputs |
| `placax/netlist/rows.py` | placement rows and the core area, from `.scl` or DEF `ROW` |
| `placax_agents/benchmark.py` | `Benchmark.load()` — netlist to ready-to-train bundle |
| `placax_agents/experiment/config.py` | `ExperimentConfig` and the comparison levels |
| `placax_agents/experiment/registry.py` | every swappable component, by name |
| `placax_agents/experiment/build.py` | config to live objects; `AGENTS`, `LOOPS` |
| `placax_agents/experiment/run.py` | `run_experiment()` — the one loop, and `score()` |
| `placax_agents/experiment/export.py` | a run's placement back into `.pl`/DEF |
| `placax_agents/experiment/physical.py` | the configured cell placer + validator |

## What you can swap

**The registries are open. The keys below are what ships, not what is allowed.** The project's
rule is that anything a different team might do differently is a parameter, not a hard-coded
call — so a reward, state, policy, agent or action space of your own is an ordinary Python
function. Two routes:

**Register a name** — when a config should be able to select it, and survive a results file.

```python
from placax_agents.experiment.registry import register

def my_reward(grid, weight: float = 2.0):          # your own function, in your own module
    return functools.partial(make_scaled_hpwl_reward, dense=True, reward_scale=weight)

register("reward", "my_reward", my_reward)
# ...then Spec("my_reward", {"weight": 3.0}) anywhere a shipped reward would go.
```

It is then a component like any other: hashed, its own kwarg defaults completed into that hash,
JSON round-tripping, usable by every script. Use `register` rather than writing into the dict —
it clears the memoized defaults, and a name hashed before it existed would otherwise keep hashing
against an empty default set. `registered("reward")` lists what is available.

*Caveat worth knowing:* a config records the **name**. Two configs both saying `my_reward`
compare as comparable, and whether they used the same function depends on code the manifest's
`git_revision` may not cover. Version a custom component with the same care as the result.

**Hand over the object** — for a one-off that never needs a name.

```python
benchmark = Benchmark.load(directory, grid=224, make_reward_fn=my_factory)
built = build(config, benchmark=benchmark)                    # or...
built = dataclasses.replace(built, agent=MyAgent(benchmark))   # ...swap any resolved piece
run_experiment(config, output_dir, built=built)
```

Everything below is named in a config as `Spec("<key>", {...kwargs})` and is **hashed**.

| Slot | Registry | Shipped keys |
|---|---|---|
| `benchmark.order` | `ORDERS` | `alphabetical`, `area_desc`, `connectivity`, `connectivity_maskplace` |
| `reward` | `REWARDS` | `hpwl`, `maskplace`, `smoothed`, `hpwl_congestion` |
| `state` | `STATES` | `canvas`, `wiremask` |
| `action_mask` | `MASKS` | `wiremask_quality` |
| `action_space` | `ACTION_SPACES` | `discrete_grid`, `oriented_grid`, `perturbation` |
| `initial_placement` | `INITS` | `empty`, `greedy_wiremask_prefix` |
| `legalization` | `LEGALIZERS` | `row_snap` |
| `physical.cell_placer` | `CELL_PLACERS` | `dreamplace` |
| `physical.validator` | `VALIDATORS` | `openroad` |
| `agent.policy` | `POLICIES` | `cnn`, `mlp`, `wiremask_cnn`, `resnet_coarse_fine` |
| `agent.optimizer` | `OPTIMIZERS` | `adam`, `maskplace_split` |
| `agent.algorithm` | `AGENTS` | `ppo`, `greedy_wiremask`, `random_search`, `genetic`, `local_search` |
| `agent.loop` | `LOOPS` | `sequential`, `parallel`, `buffered` (PPO only) |

Plus two non-registry environment fields: `benchmark.canvas` (`die` or `core`) and
`benchmark.macro_budget`. `agent.algorithm` for `ppo` also takes `value_loss` (`mse`, `huber`).

**Which agent drives which action space.** `local_search` requires `perturbation`; `genetic`
drives `discrete_grid` or `oriented_grid`; everything else is `discrete_grid` only. A mismatch is
refused at `build()` — a policy emitting `(grid_x, grid_y)` logits has nowhere to put a macro
index.

## Classes

| Class | Where | Use it to |
|---|---|---|
| `ExperimentConfig` | `experiment/config.py` | describe a whole run; `.write()`/`.read()` it |
| `EnvironmentSpec` / `AgentSpec` / `Spec` | same | the two halves, and one named component |
| `BenchmarkSpec` | same | design, grid, macro budget, canvas, order |
| `Budget` | `experiment/budget.py` | cap `env_steps` / `iterations` / `wall_clock_s` |
| `BuiltExperiment` | `experiment/build.py` | what a config resolves to: `.benchmark`, `.agent`, `.action_space`, `.state_fn` |
| `Benchmark` | `benchmark.py` | the loaded netlist: `sizes_array`, `padded_pin_idx`, `cell_size`, `params`, `rows` |
| `Agent` (protocol) | `agents/base.py` | `init` / `update` / `best_positions` / `converged` — all an agent must provide |
| `ActionSpace` (protocol) | `action_space.py` | `reset` / `apply` / `done` / `target` / `episode_length` |
| `EnvState`, `EnvParams` | `types.py` | `positions`, `step`, `orientations`; grid and macro count |
| `PlacementRows` | `netlist/rows.py` | the core rectangle, row pitch, site width; `snap()`, `is_legal()` |
| `PPAResult` | `placax_tools/validator.py` | area, utilization, slack, routed wirelength, vias, DRC |

`registry.register(slot, name, builder)` / `registry.registered(slot)` are the functions for
adding your own; `registry.SLOTS` is every slot that accepts one.

## Comparison levels

`assert_comparable(*configs, level=...)`. The first four are increasingly strict; `protocol` is
orthogonal — the environment with the *design removed*, which is what a multi-design suite shares.

| Level | Asserts |
|---|---|
| `benchmark` | same design, grid, order, budget |
| `task` | + reward, warm start, constraints, physical stack, compute budget |
| `environment` | + the observation. **The default**, and the right level for comparing agents |
| `full` | + the agent and the seed — one exact run |
| `protocol` | everything except which netlist. For `--benchmark_dirs` suites |

## Run outputs

One directory per run under `runs/`. A comparison writes one subdirectory per `(agent, seed)` —
`ppo-seed0`, `genetic-seed1`, … — each a complete run, plus `results.json` at the top.

| File | What |
|---|---|
| `manifest.json` | the full config, its hashes, metric definitions, machine fingerprint. Written at start |
| `training_log.jsonl` | one line per iteration: spend, loss, and on evaluated iterations the score *and its legality* |
| `state.json` | budget spend, so a resumed run continues the same budget |
| `checkpoint.bin` | agent state + RNG key + iteration; resume is automatic |
| `best_checkpoint.bin` | bare weights plus the `real_hpwl` that earned them (agents with weights, when `--eval_every > 0`) |
| `results.json` | a comparison as data: per-design tables, hashes, spend, legality |
| `ppa.json` | a physical measurement, carrying the run's `full_hash` |

## Entry points

| Script | Does |
|---|---|
| `scripts/run_training.py` | trains the plain-CNN preset |
| `scripts/run_maskplace.py` | trains the MaskPlace-equivalent preset |
| `scripts/compare_agents.py` | several agents, one environment, one budget, one table (`--benchmark_dirs` for a suite) |
| `scripts/run_pipeline.py` | a trained checkpoint to macro placement to DREAMPlace |
| `scripts/validate_design.py` | the physical flow on a macro-placed DEF |
| `scripts/place_once.py` | one greedy rollout from a checkpoint |
| `scripts/visualize.py` | curves, placement images, observation channels, rollout GIF |
| `scripts/subprocess_search.py` | largest `--n_episodes`/`--n_envs` this machine fits |
| `scripts/measure_reward_terms.py` | relative magnitudes of a composite reward's terms |
| `scripts/download_benchmarks.py` | fetches benchmark suites |

Scripts that reload a checkpoint take `--config=<a run's manifest.json>`. Prefer it: a preset name
is a guess at the environment the checkpoint was trained in, a manifest is that environment.

## Adding something

`register(slot, name, builder)` from your own code — nothing in the library needs editing, and
every script that consumes configs gains it at once. What a builder must be:

| Slot | Builder signature | Returns |
|---|---|---|
| `reward` | `(grid, **kwargs)` | `(pin_idx, pin_offset, valid_mask, sizes, cell_size) -> RewardFn` |
| `state` | `(benchmark, **kwargs)` | `(state, params, sizes_array) -> obs dict` with `canvas` and `current_macro_size` |
| `action_mask` | `(benchmark, **kwargs)` | `obs -> (grid_x, grid_y) bool` |
| `action_space` | `(benchmark, **kwargs)` | an object with the `ActionSpace` methods |
| `initial_placement` | `(benchmark, **kwargs)` | `key -> (n_macros, 2) positions or None` |
| `legalization` | `(benchmark, **kwargs)` | `(placement, macro_sizes) -> placement` |
| `policy` | `(benchmark, **kwargs)` | a Flax module whose `.apply` returns `(logits, value)` |
| `order` | `(design_name, **kwargs)` | `(macro_sizes, nets) -> names in placement order` |
| `algorithm` | `(config, benchmark, env)` | `(agent, episodes_per_iteration, extra fields)` |

An agent needs only the four `Agent` methods (`init`, `update`, `best_positions`, `converged`);
it inherits evaluation, checkpointing, budgeting, logging and the shared scoring path. Take
`env.as_kwargs()` wholesale — an agent that drops the action mask is running a different
experiment while its config claims otherwise.

## Gotchas

- **`real_hpwl` is macro-to-macro only.** The parser drops nets with fewer than two macros, so it
  is not full-netlist HPWL and is not comparable to a paper that reports one.
- **A wirelength without its legality is not a result.** Overlapping macros have shorter wires;
  the mask has a relaxation valve that fires rather than deadlock. Every score reports overlap.
- **GPU runs are not bit-exact.** Report across seeds. `PLACAX_DETERMINISTIC=1` forces CPU.
- **`canvas` and `legalization` default to the historical behaviour** (`die`, none), which places
  macros off the design's real rows. `canvas="core"` + `legalization="row_snap"` is the
  physically realizable pair.
- **`local_search` needs a dense reward** — a terminal-only reward gives it no per-step signal.
  Watch `acceptance_rate` in the log.
- **`env_steps` is sample-matched, not compute-matched.** `gradient_steps` is reported separately.
- `JAX_ENABLE_X64=1` is set deliberately (`placax/_device.py`) and roughly doubles memory.
