# Placax: A Flexible, Shared JAX Environment for Chip Macro Placement Research (v4)

**Project name: Placax.** Reconciled specification — merges the tested project spec with a broader literature-compatibility check, an integration decision on DREAMPlace, and the three-tier code organization (`placax` / `placax_agents` / per-project scripts).

**v4 changes:** Sections 4, 4.1, 4.2, 5, and 7 are rewritten to match the shipped code exactly — real file paths, real function signatures, no more illustrative pseudocode that drifted from the implementation. Where the vision described here goes further than what's built (SHAC/ACO/GA training loops, offline pretraining, the cell-placer/validator tool integration), that's now called out explicitly as **not yet implemented** rather than left ambiguous. Section 1's "no part hard-coded" claim, which was aspirational in v3, is now actually true for every axis inside the kernel's reach (reward, placement order, lookahead, action masking, observation, policy, value loss) — see Section 5.

---

## 1. Executive summary

Build a **fast, general-purpose, JAX-based environment for chip macro placement**, designed so that no part of it — training algorithm, reward function, initial-placement strategy, cell-placement/validation tools — is hard-coded. Every one of those is a swappable input, following one pattern throughout: pass a function in, the environment calls it, nothing about the core changes when that function changes.

**The core is deliberately minimal: one `reset()`/`step()` kernel, not several.** PPO, SHAC, ant colony optimization, and genetic algorithms all drive the identical loop — the only thing that varies is which function decides the next action. Population methods (GA) don't need a separate batch-evaluation entry point either: `vmap` the same episode-replay function across a population of pre-committed action sequences. This was tested end-to-end with four structurally different agents against the same kernel, with zero changes to `reset()` or `step()` between them.

This environment enables a real comparison — SHAC-style analytic policy-gradient training (enabled by the environment's differentiability) against standard PPO — and, because the environment is genuinely general, tests other axes the same way: reward formulations, algorithm families, initial-placement strategies.

The project is **not** "build the first chip-placement environment" — those exist, one per paper, each bespoke and disposable. It **is** "build the first shared, fast, general one," closer to what Gymnax/Brax/Jumanji did for JAX-based RL in other domains — none of which cover chip placement, the confirmed gap this fills.

---

## 2. Background and motivation

### 2.1 The gap

Every existing RL placement method — AlphaChip/Circuit Training, MaskPlace, ChiPFormer, DeepTH, EfficientPlace — ships its own one-off training environment, tightly bolted to that paper's specific agent. No shared, general environment exists for the field to train against, unlike mature RL sub-fields (robotics, game-playing) that converged on shared standards years ago. Confirmed via landscape review, two independent novelty audits, and a fresh recheck: every JAX RL environment found (Pgx, Brax, JaxMARL, Gymnax, Jumanji) covers a different domain. None cover chip placement.

### 2.2 Why macro placement specifically, honestly

Macro placement is neither the biggest time-cost in a real chip project (verification dominates, 60–70%+ of total effort) nor the single biggest lever on final quality (RTL/architecture decisions set a ceiling nothing downstream exceeds). What it has instead is unusually high **RL-tractability**: a bounded decision space (hundreds of macros, not millions of cells), a clean measurable objective (HPWL, congestion), structure that repeats across designs. That's why it produced the field's most-cited result (AlphaChip, Nature 2021) and remains well-scoped for focused effort.

**General principle** (confirmed via NVIDIA's NVCell, which uses RL successfully for standard-cell layout but only by staying small in scope): RL succeeds when decision scope per episode is small and bounded, and gets replaced by classical/generative methods once scope becomes global. This is why standard-cell placement (DREAMPlace, analytical) and routing (moving toward diffusion) aren't primarily RL territory, while macro placement remains a good fit.

### 2.3 Feasibility grounding

Not a bet on unproven infrastructure. DREAMPlace, the field's standard standard-cell placer, is already fully differentiable with respect to placement coordinates — that's its core mechanism, and the direct ancestor of this project's differentiable HPWL/congestion reward. Separately, JAX RL environments in other domains have repeatedly demonstrated large speedups over PyTorch training loops (one comparable multi-agent JAX library: ~14x faster wall-clock, up to several thousand-fold when vectorized).

---

## 3. Design philosophy: flexibility by construction

**The mistake, recorded because it's easy to repeat elsewhere:** an early sketch computed reward directly inside `step()` — hard-coded. This would force any team wanting a different reward (a large, legitimate part of this field — LaMPlace and EIM exist specifically to test different rewards) to fork the environment rather than plug something in, defeating the point of a shared environment.

**The fix, applied everywhere:** pass the swappable thing in as an argument. Verified with running code on two axes:

1. **Reward function.** `step(state, action, reward_fn)`, not `reward_fn` baked in. Tested: same agent, same mechanics, different reward (-4.0 vs. -3.76) purely from which `reward_fn` was passed, with neither reward touching `reset()` or state-transition logic.
2. **External tool calls.** `place_and_validate(macro_positions, netlist, cellplace_fn, validate_fn)`, not hard-coded DREAMPlace/OpenROAD calls. Tested with stand-in functions for two different tool pairs, without touching the environment or agent.

**The rule for every future decision:** if a piece of logic could plausibly be done differently by a different team, it's a parameter, not a hard-coded call.

**What's swappable, as shipped today:** the algorithm/policy (any Flax module matching `AlgorithmFn`), the state representation (`StateFn`), the reward/cost function including its density — sparse/terminal or dense/per-step (`RewardFn`), the macro placement order (`OrderFn`), the action-legality/quality masking, the critic's value loss, the netlist format, the cell placer, the validator. See Section 5 for the concrete function/module for each. **Not yet implemented, still design/research-plan (Section 12):** algorithm backends beyond PPO (SHAC, ACO, GA), the offline-pretraining data regime, a learned warm-start strategy.

---

## 4. Architecture

```
Yosys / a Bookshelf|DEF|protobuf netlist (fixed, upstream, untouched)
        |
        v
Benchmark.load(benchmark_dir, order_fn=..., make_reward_fn=...)   (placax_agents/benchmark.py)
        |  parses the netlist, picks placement order, builds padded JAX arrays, builds reward_fn
        v
EnvState + EnvParams  (two separate flax.struct.dataclass pytrees — Gymnax convention)
        |
        v
+--------------------------------------------------+
|   reset() / step()  —  the ONE shared kernel      |
|   placax/core.py — differentiable, jit + vmap       |
|                                                      |
|   passed in as swappable arguments, not hard-coded: |
|     reward_fn     (RewardFn — sparse or dense)      |
|     initial_positions  (optional warm start,        |
|                          reset()'s own argument)     |
|   (policy_apply_fn/state_fn/order_fn compose one     |
|    level up, in placax_agents — Section 4.2)         |
+--------------------------------------------------+
        |
        v
   Placed macro positions
        |
        v
+- - - - - - - - - - - - -+   dashed = swappable,
|   Cell Placer            |   external, not reimplemented
|   default: DREAMPlace    |   placax_tools/dreamplace/cell_placer.py
+- - - - - - - - - - - - -+
        |
        v
+- - - - - - - - - - - - -+
|   Validator               |
|   default: OpenROAD       |   placax_tools/openroad/validator.py
+- - - - - - - - - - - - -+
        |
        v
   True PPA (checked occasionally, not every training step)
```

**Why one kernel, not several**: `reset()`/`step()` know nothing about policies, rewards' internal shape, or how many episodes run in parallel — `jax.vmap` over `collect_rollout` is what turns "one episode" into "n_envs episodes at once" (`placax_agents/training/loops/parallel_train.py`), with zero change to the kernel. The same reasoning is why a population method (GA) or a pheromone-table method (ACO) would layer on top of the identical kernel without touching it — `step()` doesn't know or care whether the action it receives came from a policy network's sample, a pre-committed genome entry, or a pheromone draw. **As shipped, only the PPO loop (sequential and parallel) is built** (`placax_agents/training/loops/`); SHAC/ACO/GA are the research plan (Section 12), not yet implemented — the claim here is that the kernel's shape doesn't block them, not that they exist yet.

### 4.1 Core interface contract

The real signatures, from `placax/core.py` and `placax/types.py`:

```python
from placax.core import reset, step
from placax.types import EnvParams, EnvState, RewardFn

params = EnvParams(grid=64, grid_y=None, n_macros=543)   # grid_y=None -> square canvas
state = reset(params, initial_positions=None)             # optional warm start (Section 5.3)
new_state, reward, done = step(state, action, reward_fn, params)
```

`EnvState`/`EnvParams` are `flax.struct.dataclass` pytrees, not raw dicts — `EnvParams.grid`/`grid_y`/`n_macros` are marked `pytree_node=False` (static shapes, not traced values), matching Gymnax's convention referenced below. `reward_fn` matches `RewardFn = Callable[[old_positions, new_positions, old_placed, new_placed], scalar]` — `step()` calls it **every step, unconditionally**; whether reward is sparse (fires once, at done) or dense (fires every step) is entirely `reward_fn`'s own choice, not a `step()` code path (see Section 5.2). This diverges from the toy `step(state, action, reward_fn)` (3 args, reward gated on `done` inside `step()` itself) shown in earlier drafts of this document and still used for the small illustrative agents in Section 7.4 below — those predate this generalization and are kept as-is for their pedagogical value, not as a second real contract.

### 4.2 Distribution: three tiers, not one, and not a framework

**Mindset: a functional interface, not a framework.** No base class to subclass, no plugin system. "Custom" means a plain Python function/Flax module matching the same input/output shape as a built-in one — e.g. `Benchmark.init_policy(policy, key)` accepts "any flax nn.Module whose `.apply` matches `AlgorithmFn`," not just the library's own `CNNActorCritic`.

An earlier version of this design described two tiers (library + per-project scripts). A third, middle tier turned out to be worth naming explicitly, because "the PPO loop" is itself reusable across projects — its control-flow skeleton (rollout, GAE, clipped loss, minibatch update) doesn't depend on what environment it's driving, only on the standard `reset`/`step`/`reward`/`done` shape. Collapsing that loop into every per-project script would mean copy-pasting the same logic repeatedly; folding it into the environment library would violate Section 4's "one shared kernel" principle, since the loop is algorithm-specific in a way `reset`/`step` are not.

**Three tiers, each with a named precedent already in the JAX RL ecosystem:**

1. **`placax`** — the environment library. Pip-installable, versioned, rarely touched. Contains only the kernel (`reset`/`step`), netlist parsing/ordering, and pure-JAX defaults (HPWL/wiremask, legality/quality masking). **Precedent: Gymnax.**
2. **`placax_agents`** — reusable algorithm loops (currently: PPO, sequential and parallel — Section 4). Each loop is a readable, forkable function — not a black-box class — because this project's actual research (Section 12) means modifying these loops directly (e.g. building the SHAC variant by copying the PPO loop's rollout logic and swapping in SHAC's backprop-through-time update). Hiding the loop behind a sealed abstraction would work against that goal. **Precedent: PureJaxRL** — published as clean, single-file, fork-and-modify implementations for exactly this reason, not as an installable black-box API.
3. **Per-project scripts** (`scripts/run_training.py` today; a per-experiment `my_research/` directory as usage grows) — what actually changes between experiments. Picks a reward function, a network, hyperparameters, and calls the relevant `placax_agents` loop (or forks it). Never modifies `placax`.

**Concrete file structure, as shipped (verified against the repo, not aspirational):**

```
placax/                          # Tier 1 — the environment library (≈ Gymnax)
    core.py                        reset() / step() — the kernel (Section 4.1)
    types.py                       EnvState, EnvParams, RewardFn, OrderFn, SizeMap, Nets, PinOffsets
    _device.py                     GPU/CPU fallback (Section 6.3) — imported before jax, everywhere
    netlist/
        __init__.py                  load_netlist() — detects format, dispatches (Bookshelf/DEF/protobuf)
        bookshelf.py, def_reader.py, def_writer.py, lef.py, protobuf_reader.py
        padding.py                   build_padded_arrays()/build_macro_net_index() — order_fn plugs in here
        order.py                     OrderFn implementations: alphabetical_order (default),
                                      area_desc_order, connectivity_order (Section 5.1b)
    extras/
        rewards.py                   hpwl(), wiremask(), make_hpwl_reward(padded_pin_idx, ..., dense=)
        masks.py                     occupancy_mask, boundary_mask, quality_mask, lookahead_illegal_masks
        render.py                    render() — boolean canvas from placed macro footprints
        mst.py                       Steiner-tree/RSMT cost (Prim's algorithm) — an alternative cost metric

placax_agents/                   # Tier 2 — reusable, forkable training loops (≈ PureJaxRL)
    benchmark.py                    Benchmark.load()/.init_policy() — netlist -> ready-to-train bundle
    types.py                        AlgorithmFn, StateFn — the two swappable-axis contracts (Section 5.1)
    policy/
        observation.py                observation(), lookahead_sizes(), make_wiremask_observation()
        action.py                     legal_action_logits() (extra_illegal=...), sample_action(), action_log_prob()
        scale.py                      grid-cell <-> real-unit conversion
        architectures/
            cnn.py                      CNNActorCritic
            wiremask_cnn.py             WiremaskCNNActorCritic (pairs with make_wiremask_observation)
    training/
        reward.py                     make_scaled_hpwl_reward(..., dense=) — the grid-unit RewardFn factory
        rollout.py                    collect_rollout() — one episode as a lax.scan
        algorithm/
            config.py                   PPOConfig (gamma, lam, clip_eps, value_coef, entropy_coef, value_loss_fn)
            gae.py                      compute_gae()
            loss.py                     ppo_loss(), mse_value_loss(), huber_value_loss()
            normalize.py, optimizer_step.py, running_stats.py
        loops/
            train.py                    train_sequential() — one episode, one update, repeated
            parallel_train.py           train_parallel() — n_envs episodes via vmap, one averaged update
            run.py, common.py
    ops/
        evaluate.py, checkpoint.py, resumable_train.py, autotune.py, n_envs.py

placax_tools/                    # Cell placer / validator wrappers (Section 5.4-5.5)
    dreamplace/cell_placer.py
    openroad/validator.py

benchmarks/                      # adaptec1, bigblue1 (Bookshelf), ariane133 (protobuf) — Section 10
scripts/                         # download_benchmarks.py, run_training.py, compare_sequential_vs_parallel.py
```

**Not yet implemented** (design intent from earlier sections, not present in this tree): `placax_agents/training/algorithm/{shac,aco,ga}.py`-style loops for algorithms beyond PPO, an `offline.py`-style pretraining loop (Section 5.6), a learned `init_fn` warm-start strategy. The interface reasoning in Section 4/8 for why they'd fit still stands; they just haven't been built.

**Rule for where new code goes:** if it could plausibly run unmodified against a *different* placement benchmark or a *different* algorithm, it belongs in Tier 1. If it's specific to one algorithm but reusable across projects that use that algorithm, Tier 2. If it's specific to one experiment, Tier 3 — and Tier 3 is expected to be the tier that changes on every commit.

---

## 5. The swappable components

### 5.1 The algorithm backend and the state representation — two independent axes, not one

Originally treated as one swappable slot (`agent_fn`), these are two separate axes, matching `placax_agents/types.py`'s `AlgorithmFn`/`StateFn` split. Welding them into a single function means you can't cleanly ask "does switching to a GNN state help PPO specifically" without risking that whatever else changed alongside the state encoding also affected the result — exactly the kind of confound most placement papers don't isolate (Section 1).

**Algorithm backend** (`AlgorithmFn = Callable[..., tuple[action_logits, value]]`, e.g. `CNNActorCritic.apply` in `policy/architectures/cnn.py`): as shipped, **PPO is the only algorithm implemented** (Section 4.2) — the interface argument that ACO/GA/other agents fit the same kernel without modification is design reasoning from the research plan (Section 12), not something with running code behind it in this codebase today. `Benchmark.init_policy(policy, key)` accepts any Flax module whose `.apply` matches `AlgorithmFn`, not just `CNNActorCritic` — `WiremaskCNNActorCritic` (`policy/architectures/wiremask_cnn.py`, adds a wiremask input channel) is the second one shipped, proof the slot is genuinely pluggable rather than sized to fit one network.

**State representation** (`StateFn = Callable[..., dict]`, e.g. `observation()` in `policy/observation.py`): a pure function applied *before* the policy sees anything, called by the Tier 2 training loop (`collect_rollout`, `train_sequential`/`train_parallel`), not by the kernel. Two keys are the only ones the training/eval loops themselves require — `"canvas"` and `"current_macro_size"` — everything else is convention, not requirement. The shipped `observation()` also exposes `positions`/`sizes_array`/`placed_mask`/`step` (for non-image policies, e.g. a GNN) and `lookahead_sizes` (an `(lookahead, 2)` array of upcoming macro sizes, `lookahead=1` by default — see `lookahead_sizes()`, static-shaped under jit). `make_wiremask_observation(...)` wraps any base `StateFn`, adding a `"wiremask"` key computed from `extras.rewards.wiremask()` — a decorator pattern for composing observation features rather than one fixed shape:

```python
from placax_agents.policy.observation import observation, make_wiremask_observation

state_fn = make_wiremask_observation(
    padded_pin_idx, padded_pin_offset, valid_mask,
    macro_net_idx, macro_net_offset, macro_net_valid,
    base_state_fn=observation,
)
# state_fn(state, params, sizes_array) -> {..., "wiremask": (grid_x, grid_y) float}
```

### 5.1b The macro placement order — a third axis, added after v3

Not present in the original design (v3 treated placement order as fixed/alphabetical, implicitly). `placax/netlist/order.py`'s `OrderFn = Callable[[macro_sizes, nets], list[str]]` decides which row index each macro gets — and therefore what order the RL episode places them in — as a plug-in to `build_padded_arrays(..., order_fn=...)` / `Benchmark.load(..., order_fn=...)`:

- `alphabetical_order` — deterministic, no opinion on quality. The default, matching the historical (pre-v4) behavior.
- `area_desc_order` — largest footprint first, a common placement heuristic.
- `connectivity_order` — breadth-first by shared nets, starting from the highest-degree macro; generalizes the topology-based ordering used by MaskPlace (Section 8) to any netlist, not specific to that one paper's implementation.

### 5.2 The reward / cost function, and its density

`RewardFn = Callable[[old_positions, new_positions, old_placed, new_placed], scalar]` (Section 4.1) — called every step by `step()`, unconditionally. In this project's usage, "reward function" and "cost function" are used interchangeably — the distinction (cost as a pure metric vs. reward as a weighted composition of costs) can matter once you're combining multiple objectives, but isn't load-bearing for the interface itself.

**Sparse vs. dense is a `reward_fn` choice, not a `step()` code path.** `extras.rewards.make_hpwl_reward(padded_pin_idx, padded_pin_offset, valid_mask, dense=False)` and `training.reward.make_scaled_hpwl_reward(..., dense=False)` (the grid-unit-aware wrapper actually plugged into training) both take a `dense` flag:

- `dense=False` (default): 0 every step, `-HPWL(final)` once every macro is placed — the historical sparse/terminal reward.
- `dense=True`: `-(HPWL(placed-so-far after) - HPWL(placed-so-far before))` every step — a dense, per-action reward, the shape MaskPlace's reward uses (Section 8). Its episode sum telescopes to the exact same total as `dense=False` (HPWL of an empty placement is 0) — sparse and dense are two credit-assignment choices over the same underlying quantity, not two different reward definitions.

Formulations to compare: HPWL-only, HPWL+congestion, a learned predictor (LaMPlace/EIM-style), alternative proxies (Euclidean wirelength, RUDY-based congestion, `extras/mst.py`'s Steiner-tree/RSMT cost as an alternative to bounding-box HPWL). Only HPWL (sparse and dense) is implemented today; the others remain research-plan items (Section 12).

**Critical implementation detail (tested, caught a real bug):** real netlists are hypergraphs — nets connect 2 to dozens of pins — but `vmap` needs uniform shape. Pad every net's pin list to the longest net's length, carry an explicit boolean mask. A naive unmasked version silently gave 30.0 instead of the correct 20.0 on a 4-cell test case, no error raised. See Section 7.2 for the corrected code. `hpwl()` additionally takes an optional `placed_mask` (defaults to "every macro placed") so it can be evaluated on a **partial** assignment, not just a finished one — that's what makes the dense reward above possible without a second, separate implementation.

**Action-space shaping composes the same way.** `legal_action_logits(logits, occupied, params, macro_size, extra_illegal=None)` OR's in any extra `(grid_x, grid_y)` bool cutoff on top of bare legality (overlap/out-of-bounds) — e.g. `extras.masks.quality_mask(scores, max_score)`, a one-line generic cutoff over any per-cell score (wiremask, congestion, density...), for algorithms that restrict actions beyond legality the way MaskPlace does (Section 8).

### 5.3 The initial-placement / warm-start strategy

**Motivation:** TILOS's independent assessment found Circuit Training leans heavily on the initial placement it receives from a commercial physical-synthesis tool — removing it measurably worsened routed wirelength. The starting point may matter as much as the algorithm refining it.

**Shipped today:** `reset(params, initial_positions=None)` accepts a prefix of already-decided positions (the rest left at the `-1` sentinel) and resumes `state.step` from wherever that prefix ends — a warm start in the literal sense (some macros arrive pre-placed), though not yet paired with a heuristic that *generates* good initial positions.

**Not yet implemented:** an `init_fn(netlist, key) -> initial_positions` layer with actual heuristics behind it — a fast classical method (simulated annealing, spectral/quadratic placement), output from an existing analytical placer (coarse DREAMPlace pass treating macros as large cells), or a learned model trained to predict good starting positions. Tests whether a free, open-source warm start can recover the benefit Circuit Training gets from a commercial one — flagged but never rigorously answered in earlier scoping, and still open.

### 5.4 The cell placer

`(macro_positions, netlist) -> full_placement`, called after the agent finishes placing macros. Wrapper: `placax_tools/dreamplace/cell_placer.py`. **Default: DREAMPlace** — free, open-source, GPU-accelerated, field standard. Any tool implementing the same interface substitutes: RePlAce, AutoDMP, a commercial placer.

### 5.5 The validator

`full_placement -> true_PPA_metrics`, called occasionally (not every training step). Wrapper: `placax_tools/openroad/validator.py`. **Default: OpenROAD-flow-scripts** — the only broadly-accessible full RTL-to-GDSII flow for labs without commercial signoff licenses.

**PPA-truth-checking** (does the fast proxy still predict real PPA) is in scope. **Functional/logical verification** (does the chip work — simulation, formal methods, 60–70%+ of a real project's time) is a separate process, not touched at all.

### 5.6 Data regime — online vs. offline (reopened; was previously excluded; still not implemented)

**This axis was explicitly out of scope earlier in this document** (old Section 8/14), on the reasoning that offline/generative training doesn't fit the single-kernel interface without real extra machinery. Reopened because the four-axis decomposition (algorithm × state representation × reward × regime) is treated as a research contribution in its own right — excluding one axis by construction undercuts that claim.

`regime_fn` lives entirely in `placax_agents`, alongside `algorithm_fn` implementations, not in the kernel:

- **Online, from-scratch** (current default): `algorithm_fn` starts untrained, learns purely from live `step()`/`reward_fn` calls. What `train_ppo`/`train_shac`/`train_aco`/`train_ga` already do.
- **Offline-pretrained, then fine-tuned** (ChiPFormer-style): a new pretraining loop trains a model on a fixed dataset of prior placements — sequence modeling / behavior cloning, no live `reward_fn` calls, no environment interaction at all during this phase. The resulting model is then used as (or fine-tuned into) an `algorithm_fn`, at which point it drives `step()` exactly like any other agent.

```python
# placax_agents/offline.py — new file, does not touch placax/core.py
def pretrain_offline(dataset, model_init, loss_fn, n_epochs):
    model = model_init
    for epoch in range(n_epochs):
        for batch in dataset:                 # (state, expert_action) or full trajectories
            loss = loss_fn(model, batch)       # e.g. decision-transformer sequence loss
            model = update_weights(model, loss)
    return model                                # usable directly as algorithm_fn from here
```

**Real cost of reopening this, stated plainly:** collecting or sourcing an offline dataset of expert placements is nontrivial — either running enough episodes of an existing method (DREAMPlace, MaskPlace) to log trajectories, or finding a published dataset. This wasn't required by any of the four already-tested agents. Budget for it accordingly; it's a genuinely new data-engineering task, not just a new training loop.

### 5.7 Why DREAMPlace stays external, not reimplemented

DREAMPlace is never in the training loop's hot path — it's called occasionally (Section 5.5's "not every training step" applies here too), not thousands of times per experiment. That removes most of the performance case for a JAX reimplementation, since the actual bottleneck during training is the fast in-JAX proxy reward, not cell placement.

**Three tiers of integration, in order of cost:**

1. **External subprocess call (current default).** Simple, correct given DREAMPlace runs occasionally, negligible overhead relative to thousands of JAX-native training episodes.
2. **In-process call via DREAMPlace's native PyTorch API**, tensors handed between JAX and PyTorch directly on GPU via `jax.dlpack`/`torch.dlpack` rather than round-tripping through `.pl` files. Removes real, measurable file I/O overhead without touching DREAMPlace's algorithm. Worth doing if profiling shows subprocess/file overhead is actually significant — not assumed up front.
3. **Native JAX reimplementation of DREAMPlace's algorithm.** Only justified if the research question becomes true end-to-end differentiability — gradients flowing from cell placement back into the macro-placement policy. That's reproducing a mature, multi-paper, heavily-optimized system's numerical core (electrostatics-based density model, GPU kernels), a project comparable in scope to this entire environment. **Not undertaken as part of this project's current scope.**

---

## 6. Technical requirements and constraints

### 6.1 Why JAX

**Against PyTorch:** `torch.func` now offers `vmap`/`grad`-like transforms, added later because PyTorch's original autograd doesn't compose with `vmap` natively, and still carries restrictions JAX's transforms (built this way from the start) don't share. Measured: one comparison found XLA beating PyTorch ~13x at moderate parallel-env counts before convergence at full GPU saturation; a widely-cited full-PPO-pipeline comparison found ~4000x end-to-end.

**Against Warp/DiffTaichi:** genuinely real, proven differentiable-simulation frameworks (same SHAC/AHAC lineage) — JAX has no monopoly on the technique. What they lack is the surrounding RL ecosystem (no Gymnax/PureJaxRL equivalent).

**Against custom C++/CUDA:** highest performance ceiling, wrong trade for this scope — no automatic differentiation for free.

### 6.2 `jax.numpy`, not NumPy, for anything JIT-compiled

Tested: a plain-NumPy reward function fails both `jax.grad()` and `jax.jit()` with `TracerArrayConversionError`. Matters differently per agent: hard requirement for SHAC (no gradient, no SHAC); for PPO/ACO/GA, `jax.jit()` is still needed to preserve the entire performance rationale for building this in JAX at all.

### 6.3 GPU setup with automatic CPU fallback

```bash
pip install -U "jax[cuda12]"
```

Recent JAX doesn't reliably fall back to CPU if the CUDA plugin is installed but no GPU is visible — raises `RuntimeError` instead. Fix, verified: detect GPU independently via `nvidia-smi` and set `JAX_PLATFORMS` before JAX is ever imported:

```python
import os, subprocess

def _gpu_available():
    try:
        subprocess.run(["nvidia-smi"], stdout=subprocess.DEVNULL,
                       stderr=subprocess.DEVNULL, check=True)
        return True
    except (FileNotFoundError, subprocess.CalledProcessError):
        return False

if not _gpu_available():
    os.environ["JAX_PLATFORMS"] = "cpu"   # must be set before jax is imported anywhere

import jax   # safe now
```

Must run before any `import jax`, including transitively. Nothing else changes between CPU/GPU — `jit`/`vmap`/`grad` behave identically, just slower on CPU. CPU suffices for correctness validation (Section 8); only the final speed comparison needs GPU.

### 6.4 Distribution

Not `.env` files — wrong model for algorithmic logic. Match Gymnax/Brax/Jumanji: publish as an installable Python package. `pip install`, `from <package> import reset, step`, write `reward_fn`/`agent_fn`/`cellplace_fn`/`validate_fn` per documented signatures, call the provided loop with functions passed as arguments. No forking, no subclassing for the common case.

---

## 7. Concrete code patterns

All snippets below are the real, shipped API (verified against the repo at v4) — not illustrative pseudocode. Toy sizes (4 macros, 4×4 grid) are for readability; real usage scales to real benchmarks (Section 9) unchanged.

### 7.1 The core kernel

```python
from placax.core import reset, step
from placax.types import EnvParams
import jax.numpy as jnp

params = EnvParams(grid=4, n_macros=4)

def terminal_reward(old_positions, new_positions, old_placed, new_placed):
    """Sparse: 0 every step, real reward only once every macro is placed."""
    return jnp.where(new_placed.all(), -new_positions.sum().astype(jnp.float32), 0.0)

state = reset(params)                                       # positions all -1 (unplaced), step=0
action = jnp.array([1, 2])
new_state, reward, done = step(state, action, terminal_reward, params)
```

`step()` calls `reward_fn` **every step, unconditionally** — `terminal_reward` above chooses to return 0 until `new_placed.all()`, but that's the reward function's decision, not something `step()` special-cases (Section 4.1, Section 5.2).

### 7.2 Differentiable HPWL, correctly handling variable-arity nets and partial assignments

```python
# placax/extras/rewards.py (abridged)
def hpwl(positions, padded_pin_idx, padded_pin_offset, valid_mask, placed_mask=None):
    if placed_mask is None:
        placed_mask = jnp.ones(positions.shape[0], dtype=bool)
    pin_xy = positions[padded_pin_idx].astype(jnp.float32) + padded_pin_offset
    counted = valid_mask & placed_mask[padded_pin_idx]        # padding AND not-yet-placed both excluded
    lo = jnp.where(counted[..., None], pin_xy,  _BIG).min(axis=1)
    hi = jnp.where(counted[..., None], pin_xy, -_BIG).max(axis=1)
    net_has_pins = counted.any(axis=1)
    return jnp.where(net_has_pins[:, None], hi - lo, 0.0).sum()
```

`jax.grad(hpwl)` gives an exact wirelength gradient directly. **Why masking matters (tested):** a naive unmasked version gave 30.0 vs. the correct 20.0 on a 4-cell hand-calculated example, silently, no error. Validate against a known-correct external number (this repo's tests check against MaskPlace's own real adaptec1 data — `tests/test_hpwl_real_benchmark.py`) before trusting any HPWL implementation. The optional `placed_mask` (added post-v3) lets `hpwl()` score a **partial** placement, not just a finished one — the mechanism the dense reward below is built on, not a separate implementation of it.

### 7.3 Reward as a swappable argument — sparse and dense from the same building block

```python
from placax.extras.rewards import make_hpwl_reward
from placax_agents.training.reward import make_scaled_hpwl_reward   # grid-unit-aware wrapper

# Sparse (default) — matches the historical/terminal-only reward:
sparse_reward_fn = make_hpwl_reward(padded_pin_idx, padded_pin_offset, valid_mask)

# Dense — non-zero every step, episode sum telescopes to the same total as sparse:
dense_reward_fn = make_hpwl_reward(padded_pin_idx, padded_pin_offset, valid_mask, dense=True)

# The one actually plugged into training (grid positions -> real units first):
reward_fn = make_scaled_hpwl_reward(
    padded_pin_idx, padded_pin_offset, valid_mask, sizes_array, cell_size, dense=False
)
```

```python
# placax_agents/training/rollout.py (abridged) — one episode as a lax.scan
def collect_rollout(key, variables, policy_apply_fn, params, reward_fn, sizes_array, cell_size, state_fn=observation):
    def scan_step(state, step_key):
        obs = state_fn(state, params, sizes_array)                              # swappable independently
        logits, value = policy_apply_fn(variables, obs)                          # swappable independently
        macro_size = to_grid_units(obs["current_macro_size"], cell_size)
        masked_logits = legal_action_logits(logits, obs["canvas"], params, macro_size)
        action = sample_action(step_key, masked_logits)
        new_state, reward, done = step(state, action, reward_fn, params)         # swappable independently
        return new_state, {"obs": obs, "action": action, "reward": reward, "done": done, ...}
    return jax.lax.scan(scan_step, reset(params), jax.random.split(key, params.n_macros))
```

Policy, state representation, and reward all vary independently through this one function — no kernel change for any of them.

### 7.4 Design sketch: agents beyond PPO (not implemented — Section 4.2/12)

The four-agent argument from earlier drafts of this document — that ACO/GA/greedy/random all fit the same kernel — is still believed true, but **no `placax_agents` loop for anything but PPO exists in the repo today.** The snippets below are kept as a design sketch, written against the pre-v4 simplified 3-arg `step(state, action, reward_fn)` for brevity; the real kernel takes 4 args and a 4-arg `reward_fn` (Section 7.1). Building any of these for real means porting to that real signature, not just plugging the sketch in as-is.

```python
def agent_random(state, key):
    return random.randint(key, (2,), 0, GRID)

def agent_greedy(state, key):
    idx = state["step"]
    if idx == 0:
        return jnp.array([GRID // 2, GRID // 2])
    prev = state["positions"][idx - 1]
    nudge = random.randint(key, (2,), -1, 2)
    return jnp.clip(prev + nudge, 0, GRID - 1)

pheromone = np.ones((N_MACROS, GRID * GRID))

def agent_aco(state, key):
    idx = state["step"]
    probs = np.array(pheromone[idx]) / pheromone[idx].sum()
    pos_idx = np.random.choice(GRID * GRID, p=probs)
    return jnp.array([pos_idx // GRID, pos_idx % GRID])
# after each generation: pheromone *= EVAPORATION_RATE; reinforce best ants' trails

def run_genome(genome, reward_fn):
    state = reset()
    for i in range(N_MACROS):
        state, reward, done = step(state, genome[i], reward_fn)
    return reward
# GA population fitness: jax.vmap(run_genome)(population, reward_fn) — no new entry point needed
```

### 7.5 Pluggable external tools

Shipped as small ABCs (`placax_tools/cell_placer.py`, `placax_tools/validator.py`), not bare functions — same substitution principle, more structure:

```python
from placax_tools.cell_placer import CellPlacer     # ABC: .place(def_path, lef_paths, output_dir) -> DEF path
from placax_tools.validator import Validator, PPAResult  # ABC: .validate(def_path, lef_paths, output_dir) -> PPAResult
from placax_tools.dreamplace.cell_placer import DREAMPlaceCellPlacer  # default CellPlacer
from placax_tools.openroad.validator import OpenROADValidator          # default Validator

def place_and_validate(def_path, lef_paths, output_dir, cell_placer: CellPlacer, validator: Validator) -> PPAResult:
    placed_def = cell_placer.place(def_path, lef_paths, output_dir)
    return validator.validate(placed_def, lef_paths, output_dir)

# A different team's tool substitutes by implementing the same ABC — e.g. an
# `AutoDMPCellPlacer(CellPlacer)` — with zero change to place_and_validate above.
```

---

## 8. Literature compatibility check

Checked the single-kernel design against 16 real tools/papers beyond the four directly tested agents — to confirm the interface generalizes rather than accidentally fitting only what was tried.

| Tool | Category | Fits via `step()`/`reset()`? |
|---|---|---|
| MaskPlace | CNN + PPO, sequential | Yes — and as of v4, its specific *mechanism* (not just "a CNN+PPO agent fits") composes from shipped pieces: `dense=True` reward (Section 5.2), `connectivity_order` (Section 5.1b), `make_wiremask_observation` + `WiremaskCNNActorCritic` + `quality_mask` for wirelength-guided action masking (Section 5.1/5.2). See the companion comparison doc for what still doesn't match (network capacity, exact hyperparameters) even with all of that wired up. |
| AlphaChip / Circuit Training | GNN + PPO, sequential | Yes — `agent_fn` |
| EfficientPlace | RL, sequential | Yes — `agent_fn` |
| DeepTH | GNN + policy gradient, sequential | Yes — `agent_fn` |
| Ant colony (generic) | Pheromone-guided sequential | Yes — tested |
| Genetic algorithm (generic) | Population of genomes | Yes — tested, via `vmap(run_genome)` |
| WireMask-BBO / BBOPlace-Bench | Black-box optimization | Yes — same pattern as GA |
| DREAMPlace | Analytical/gradient, all macros at once | **No** — kept external (Section 5.6) |
| RePlAce | Electrostatics analytical (DREAMPlace's basis) | **No** — external, via cell-placer interface |
| AutoDMP | Bayesian optimization wrapping DREAMPlace | **No** — external, nested tool calls |
| ChiPFormer | Offline RL / decision transformer, trained on fixed data | **In scope via `regime_fn`** (Section 5.6) — `pretrain_offline`, then `step()`-based fine-tuning |
| ChipDiffusion | Diffusion, all macros at once | **Partial** — pretraining fits `regime_fn`; inference (all macros simultaneously) still doesn't fit `step()`, same gap as DREAMPlace |
| FlowPlace | Flow matching, same shape as ChipDiffusion | **Partial** — same as ChipDiffusion |
| OpenROAD | Full RTL-to-GDS flow | N/A — instantiates the swappable Cell placer + Validator |
| OpenSTA | Static timing analysis | N/A — extends the Validator |
| Yosys | RTL synthesis | N/A — fixed upstream, produces the netlist |

**Takeaway:** every sequential/population-based agent in the literature fits the single kernel without modification. Simultaneous-optimization methods (DREAMPlace-family) still don't, and stay explicitly out of scope (Section 5.7). Offline/generative training (ChiPFormer, and the pretraining half of ChipDiffusion/FlowPlace) is now in scope via `regime_fn` (Section 5.6) — reversing the earlier exclusion.

---

## 9. Precedent to build on, not reinvent

- **Gymnax** — the `reset`/`step` convention, including explicit PRNG key threading. Directly the model for Tier 1 (`placax`, Section 4.2).
- **Brax, Jumanji, Pgx, JaxMARL** — closest precedent for a standard, reusable, GPU-vectorized RL environment, each a different domain. None cover chip placement.
- **PureJaxRL** — ready-made, JAX-native PPO baselines, published as forkable single files rather than a locked API. Directly the model for Tier 2 (`placax_agents`, Section 4.2).
- **DREAMPlace** — existence proof that a placement objective can be differentiable at all; its electrostatics formulation is the direct ancestor of this project's differentiable reward.
- **SHAC / AHAC** — the training-method lineage under test, from differentiable-simulation robotics. AHAC's horizon-truncation is directly reusable for the "stiffness" problem (unstable gradients near macro overlap, analogous to contact-rich robotics dynamics).

---

## 10. Data and benchmarks

| Benchmark | Format | Notes |
|---|---|---|
| ISPD 2005 (adaptec1-4, bigblue1-4) | Bookshelf | Most common placement-only benchmark. |
| MCNC (ami33, ami49) | Bookshelf | Older, smaller, quick tests. |
| Ariane (RISC-V CPU) | RTL → netlist | Most-used realistic testcase across RL placement papers; macros are mainly SRAM arrays. |
| ChiPBench's 20 designs | RTL, full open-source flow | Built to measure true final PPA, not just proxy metrics. |

MaskPlace's repo bundles `adaptec1` and `ariane` directly; DREAMPlace includes a download script for ISPD suites.

---

## 11. Validation plan and milestones

**Critical ordering: correctness before speed, speed before training, training before comparison.**

1. **Differentiable HPWL matches ground truth** — compare against a real DREAMPlace run or MaskPlace's published table. If these don't match, nothing built afterward can be trusted.
2. **Environment runs correctly with a trivial agent** on a real benchmark.
3. **Standard PPO baseline trains successfully** — the number to beat.
4. **SHAC-style training**, same environment, same benchmark, matched compute — the core comparison.
5. **Additional agents (ACO, GA) and reward functions** — demonstrating genuine generality.
6. **Occasional validation against the real tool pipeline** (DREAMPlace + OpenROAD).

Reference numbers (MaskPlace's published results):

| Benchmark | HPWL (×10⁵) | Overlap |
|---|---|---|
| adaptec1 | 6.57 | 0% |
| adaptec2 | 79.98 | 0% |
| bigblue1 | 2.42 | 0% |
| ariane | 14.86 | 1.94% |

Exact matches aren't the goal — training randomness means results differ — but landing in the same range, near 0% overlap, confirms the setup works.

---

## 12. Research plan enabled

- **Core experiment:** PPO vs. SHAC, matched compute, multiple seeds — is analytic policy-gradient training viable for placement, or does gradient "stiffness" near overlap dominate?
- **Reward comparisons:** HPWL vs. HPWL+congestion vs. a learned predictor, agent held fixed.
- **Algorithm-family comparisons:** PPO/SHAC vs. ACO vs. GA, reward and benchmark held fixed — does BBOPlace-Bench's finding (evolutionary/BBO beats RL on several benchmarks) extend to ACO/GA specifically?
- **State-representation comparisons:** raw coordinates vs. image (CNN) vs. graph (GNN), algorithm held fixed — isolates what most papers change simultaneously with the algorithm, per Section 1's framing.
- **Regime comparisons:** online from-scratch vs. offline-pretrained-then-fine-tuned, algorithm and reward held fixed — tests whether ChiPFormer's offline advantage holds once it's not confounded with everything else ChiPFormer also changes.
- **Initial-placement comparisons:** does a free warm start recover Circuit Training's commercial-tool advantage?
- **Seed-variance/legality auditing:** with this environment's speed, a proper 20+-seed, compute-matched variance study — currently missing from the field for any placement method.

---

## 13. Expected outcomes

1. **The environment itself** — reusable, open, fast — useful independent of which training method wins.
2. **If SHAC beats PPO:** a concrete, adoptable sample-efficiency/quality improvement over the field's standard.
3. **If SHAC underperforms:** an honest negative result — evidence the technique doesn't transfer cleanly from robotics, unpublished either way, saving others from re-testing blind.

---

## 14. Explicitly out of scope

- **Functional/logical verification** — separate process, different tools, 60–70%+ of real project time. Not touched.
- **Standard-cell placement as a primary RL target** — decision-scope mismatch; DREAMPlace (analytical) remains right for this stage.
- **Architecture/RTL-level design decisions** — highest quality-leverage stage, but a creative human process, not a clean sequential-decision problem.
- **Reward-design as the sole contribution** — LaMPlace/EIM already published here; this project treats reward testing as one use of a general environment, not its headline claim.
- **Reimplementing DREAMPlace's algorithm natively in JAX** (Section 5.6) — not justified unless the research question becomes true end-to-end differentiable joint optimization, a separate project in scope.
- **Simultaneous, all-macros-at-once optimization as an in-house agent** (ChipDiffusion/FlowPlace-style inference, DREAMPlace-style joint optimization) — still doesn't fit `step()`'s per-macro structure. Offline *pretraining* for these (Section 5.6) is in scope; simultaneous *inference* is not.

---

## 15. Timeline

| Phase | Milestone |
|---|---|
| Weeks 1-3 | Minimal JAX environment (grid actions with masking, differentiable HPWL/congestion), validated against an existing placer on small benchmarks. |
| Weeks 3-5 | Standard PPO baseline trained inside the environment — the number to beat. |
| Weeks 5-8+ | SHAC-style training, same environment; head-to-head comparison at matched compute, multiple seeds. |
| Ongoing, low-cost once above exists | Additional agents (ACO, GA), reward functions, initial-placement comparisons, seed-variance/legality audits — cheap experiments on the same infrastructure. |

---

## 16. Publication potential

Realistic target: an applied or workshop track at an ML-for-EDA-adjacent venue (DAC, ICCAD, MLCAD, DATE workshops), contingent on open-sourcing the environment, rigorous comparisons (multiple seeds, matched compute — a standard this field frequently skips), and honest reporting regardless of outcome. Comparable JAX-environment papers for other domains provide a realistic template: infrastructure plus one clean empirical comparison.

---

## 17. Glossary

- **PPO (Proximal Policy Optimization):** model-free RL, treats the environment as a black box, learns from sampled trial-and-error. The field's proven default for placement.
- **SHAC (Short-Horizon Actor-Critic):** differentiable-simulation training, backpropagates an exact gradient through a short window of steps, blended with a learned value function. Requires a differentiable environment.
- **Reward function / cost function:** the scalar objective a placement is scored against. Used interchangeably in this project.
- **HPWL:** half-perimeter wirelength — bounding box of a net's pins, summed as width+height. Cheapest common wirelength proxy.
- **Differentiable:** has an exact gradient computable via `jax.grad`, not just sampled/estimated.
- **JIT:** `jax.jit`, compiles Python/JAX code to fast machine code — most of JAX's speed advantage; requires avoiding data-dependent, shape-changing control flow.
- **vmap:** automatic vectorization across a batch dimension, no manual rewrite needed.
- **Ant Colony Optimization (ACO):** metaheuristic where "ants" probabilistically construct solutions via a pheromone table, reinforced on good trails, evaporating over time.
- **Genetic Algorithm (GA):** metaheuristic maintaining a population of candidate solutions ("individuals"/genomes), improved via selection, crossover, mutation.
- **Dependency injection:** the pattern this project's flexibility rests on — pass a function in as a parameter, rather than hard-coding a specific implementation.
- **RewardFn** (added v4): `(old_positions, new_positions, old_placed, new_placed) -> scalar`, called by `step()` every step. Sparse (fires once, at done) and dense (fires every step) reward are both just `RewardFn` implementations, distinguished by a `dense` flag on the factories that build them (Section 5.2), not by different code in `step()`.
- **OrderFn** (added v4): `(macro_sizes, nets) -> macro names in placement order` — decides which row index (and therefore which point in the episode) each macro gets. Plugs into `build_padded_arrays`/`Benchmark.load` (Section 5.1b).
- **StateFn / AlgorithmFn** (`placax_agents/types.py`): the two independent swappable-axis contracts from Section 5.1 — `StateFn` builds an observation dict from `EnvState`, `AlgorithmFn` turns that dict into `(action_logits, value)`.
- **Wiremask:** per-candidate-cell HPWL increase from placing the current macro there — `extras.rewards.wiremask()`. The guidance signal MaskPlace's policy conditions on and its action masking restricts to (Section 8); exposed as an optional observation channel via `make_wiremask_observation`, not baked into the default `observation()`.
