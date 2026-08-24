# Placax: A Flexible, Shared JAX Environment for Chip Macro Placement Research (v3)

**Project name: Placax.** Reconciled specification — merges the tested project spec with a broader literature-compatibility check, an integration decision on DREAMPlace, and the three-tier code organization (`placax` / `placax_agents` / per-project scripts).

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

**What's swappable:** the algorithm backend (PPO, SHAC, evolutionary, ACO, GA, or anything satisfying the interface), the state representation, the reward/cost function, the data regime (online from-scratch or offline-pretrained), the initial-placement/warm-start strategy, the cell placer, the validator.

---

## 4. Architecture

```
Yosys (fixed, upstream, untouched)
        |
        v
EnvState + EnvParams  (two separate pytrees — Gymnax convention)
        |
        v
+--------------------------------------------------+
|   reset() / step()  —  the ONE shared kernel      |
|   differentiable, jit + vmap                       |
|                                                      |
|   passed in as swappable arguments, not hard-coded: |
|     agent_fn      (PPO / SHAC / ACO / GA / other)   |
|     reward_fn                                       |
|     init_fn        (optional warm start)            |
+--------------------------------------------------+
        |
        v
   Placed macro positions
        |
        v
+- - - - - - - - - - - - -+   dashed = swappable,
|   Cell Placer            |   external, not reimplemented
|   default: DREAMPlace    |   (see Section 5.6)
+- - - - - - - - - - - - -+
        |
        v
+- - - - - - - - - - - - -+
|   Validator               |
|   default: OpenROAD       |
+- - - - - - - - - - - - -+
        |
        v
   True PPA (checked occasionally, not every training step)
```

**Why one kernel, not several**: an earlier exploration of this design considered separate entry points for sequential builders, population methods, and direct gradient-based optimizers. Testing showed this was unnecessary — GA's "population" is just `vmap` over the same per-episode function with a pre-committed action sequence (its genome) instead of an adaptive per-step decision; ACO's pheromone table plays the same structural role as PPO's policy network, just updated differently between episodes. One kernel, varied only by what's passed in, covers all four tested agents.

### 4.1 Core interface contract

Following Gymnax's convention (explicit PRNG key threading, since JAX has no hidden global random state like NumPy):

```python
obs, state = env.reset(key, params)
obs, state, reward, done, info = env.step(key, state, action, params)
```

Simplified form used in this project's tested code (Section 7):

```python
state = reset()
state, reward, done = step(state, action, reward_fn)
```

The real build should target the fuller Gymnax-style signature; the simplified form is what every snippet below actually uses.

### 4.2 Distribution: three tiers, not one, and not a framework

**Mindset: a functional interface, not a framework.** No base class to subclass, no plugin system. "Custom" means a plain Python function matching the same input/output shape as a built-in one. Proven with running code: a library-shipped `reward_hpwl` and a fully custom `my_custom_reward` were both passed into the identical `step()` with zero special-casing.

An earlier version of this design described two tiers (library + per-project scripts). A third, middle tier turned out to be worth naming explicitly, because "the PPO loop" is itself reusable across projects — its control-flow skeleton (rollout, GAE, clipped loss, minibatch update) doesn't depend on what environment it's driving, only on the standard `reset`/`step`/`reward`/`done` shape. Collapsing that loop into every per-project script would mean copy-pasting the same ~40 lines repeatedly; folding it into the environment library would violate Section 4's "one shared kernel" principle, since the loop is algorithm-specific in a way `reset`/`step` are not.

**Three tiers, each with a named precedent already in the JAX RL ecosystem:**

1. **`placax`** — the environment library. Pip-installable, versioned, rarely touched. Contains only the kernel (`reset`/`step`, legality/masking), plus built-in defaults (one HPWL reward, DREAMPlace/OpenROAD wrappers). **Precedent: Gymnax.**
2. **`placax_agents`** — reusable algorithm loops, one file per algorithm (`train_ppo`, `train_shac`, `train_aco`, `train_ga`). Each is a readable, forkable function — not a black-box class — because this project's actual research (Section 12) means modifying these loops directly (e.g. building the SHAC variant by copying `train_ppo`'s rollout logic and swapping in SHAC's backprop-through-time update). Hiding the loop behind a sealed abstraction would work against that goal. **Precedent: PureJaxRL** — published as clean, single-file, fork-and-modify implementations for exactly this reason, not as an installable black-box API.
3. **Per-project scripts** (`my_research/train.py`) — what actually changes between experiments. Picks a reward function, a network, hyperparameters, and calls the relevant `placax_agents` loop (or forks it). Never modifies `placax`.

**Tested with running code (Section 3):** a library-shipped `reward_hpwl` and a fully custom `my_custom_reward` were both passed into the identical `step()` with zero special-casing — confirming Tier 1's interface doesn't care which tier a function came from.

**Concrete file structure:**

```
placax/                     # Tier 1 — the environment library (≈ Gymnax)
    __init__.py                       exposes reset, step, reward_hpwl, EnvState, EnvParams
    core.py                           reset() / step() — the validated kernel
    types.py                          EnvState, EnvParams (flax.struct.dataclass, Section 4.1)
    rewards.py                        reward_hpwl, reward_congestion_aware — built-in defaults
    netlist.py                        Bookshelf/DEF parsing -> padded, masked arrays (Section 5.2)
    init_placements.py                init_fn implementations — warm-start strategies (Section 5.3)
    tools.py                          cellplace_with_dreamplace, validate_with_openroad (Section 5.4-5.6)

placax_agents/                  # Tier 2 — reusable, forkable training loops (≈ PureJaxRL)
    ppo.py                             train_ppo(reward_fn, policy_init_fn, ...)
    shac.py                            train_shac(reward_fn, policy_init_fn, ...)
    aco.py                             train_aco(reward_fn, n_ants, n_generations, ...)
    ga.py                              train_ga(reward_fn, pop_size, n_generations, ...)
    networks.py                        example CNN / GNN policy builders — swappable, not fixed

my_research/                       # Tier 3 — per-project scripts, changes constantly
    train_ppo_adaptec1.py
    train_shac_ariane.py
    compare_reward_formulations.py
    warm_start_ablation.py
```

**Rule for where new code goes:** if it could plausibly run unmodified against a *different* placement benchmark or a *different* algorithm, it belongs in Tier 1. If it's specific to one algorithm but reusable across projects that use that algorithm, Tier 2. If it's specific to one experiment, Tier 3 — and Tier 3 is expected to be the tier that changes on every commit.

---

## 5. The swappable components

### 5.1 The algorithm backend and the state representation — two independent axes, not one

Originally treated as one swappable slot (`agent_fn`), these are two separate axes. Welding them into a single function means you can't cleanly ask "does switching to a GNN state help PPO specifically" without risking that whatever else changed alongside the state encoding also affected the result — exactly the kind of confound most placement papers don't isolate (Section 1).

**Algorithm backend** (`algorithm_fn(observation, key) -> action`): any function matching the interface. Four tested, all against the identical kernel:

- **Random** — trivial baseline.
- **Greedy heuristic** — no learning, no parameters, still plugs in.
- **Ant Colony Optimization** — a pheromone table plays the role a neural network plays in PPO; "learning" happens between episodes via evaporation + reinforcement, not gradient descent. Tested: best-per-generation reward improved from -3/-4 to -1/0 over 15 generations.
- **Genetic Algorithm** — an individual is a fixed sequence of actions decided all at once (its genome), not an adaptive per-step decision. Playback uses the identical `step()` loop. Tested: best fitness improved from -2.0 to -1.0 over 15 generations via selection and crossover.

Not yet tested but fit the same interface with no kernel changes: SAC, tree-search-guided (MCTS-style) agents.

**State representation** (`state_fn(state) -> observation`): a pure function applied *before* `algorithm_fn` sees anything, called by the Tier 2 training loop, not by the kernel. Raw coordinates, a rendered image (wire-mask/position-mask style), or GNN-ready graph features (node/edge tensors from the netlist) are all valid `state_fn` outputs — swapping this independently of the algorithm is exactly what factoring it out enables.

```python
# Tier 2 (placax_agents) composition — the kernel itself is untouched
def rollout_episode(algorithm_fn, state_fn, reward_fn, key):
    state = reset()
    for t in range(N_MACROS):
        key, subkey = random.split(key)
        obs = state_fn(state)                          # swappable independently
        action = algorithm_fn(obs, subkey)              # swappable independently
        state, reward, done = step(state, action, reward_fn)
    return reward, state["positions"]
```

None of the four tested algorithms required any change to `reset()`, `step()`, or each other — and none will be required to add `state_fn` as an independent parameter either, since it's a Tier 2 composition detail.

### 5.2 The reward / cost function

Any function `positions -> scalar`. In this project's usage, "reward function" and "cost function" are used interchangeably — the distinction (cost as a pure metric vs. reward as a weighted composition of costs) can matter once you're combining multiple objectives, but isn't load-bearing for the interface itself.

Formulations to compare: HPWL-only, HPWL+congestion, a learned predictor (LaMPlace/EIM-style), alternative proxies (Euclidean wirelength, RUDY-based congestion).

**Critical implementation detail (tested, caught a real bug):** real netlists are hypergraphs — nets connect 2 to dozens of pins — but `vmap` needs uniform shape. Pad every net's pin list to the longest net's length, carry an explicit boolean mask. A naive unmasked version silently gave 30.0 instead of the correct 20.0 on a 4-cell test case, no error raised. See Section 7.2 for the corrected code.

### 5.3 The initial-placement / warm-start strategy

**Motivation:** TILOS's independent assessment found Circuit Training leans heavily on the initial placement it receives from a commercial physical-synthesis tool — removing it measurably worsened routed wirelength. The starting point may matter as much as the algorithm refining it.

**Design:** optional `init_fn(netlist, key) -> initial_state`, called before the agent's first action. Candidates: no warm start (current field standard), a fast classical heuristic (simulated annealing, spectral/quadratic placement), output from an existing analytical placer (coarse DREAMPlace pass treating macros as large cells), a learned model trained to predict good starting positions.

Tests whether a free, open-source warm start can recover the benefit Circuit Training gets from a commercial one — flagged but never rigorously answered in earlier scoping.

### 5.4 The cell placer

`(macro_positions, netlist) -> full_placement`, called after the agent finishes placing macros. **Default: DREAMPlace** — free, open-source, GPU-accelerated, field standard. Any tool implementing the same interface substitutes: RePlAce, AutoDMP, a commercial placer.

### 5.5 The validator

`full_placement -> true_PPA_metrics`, called occasionally (not every training step). **Default: OpenROAD-flow-scripts** — the only broadly-accessible full RTL-to-GDSII flow for labs without commercial signoff licenses.

**PPA-truth-checking** (does the fast proxy still predict real PPA) is in scope. **Functional/logical verification** (does the chip work — simulation, formal methods, 60–70%+ of a real project's time) is a separate process, not touched at all.

### 5.6 Data regime — online vs. offline (reopened; was previously excluded)

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

## 7. Concrete code patterns (tested)

Simplified for a small illustrative example (4 macros, 4×4 grid); real implementation scales to real benchmarks (Section 9).

### 7.1 The core kernel

```python
import jax.numpy as jnp
from jax import random

GRID = 4
N_MACROS = 4
NETLIST = jnp.array([[0, 1], [1, 2], [2, 3]])

def reset():
    return {"positions": jnp.full((N_MACROS, 2), -1), "step": 0}

def step(state, action, reward_fn):
    idx = state["step"]
    positions = state["positions"].at[idx].set(action)
    new_state = {"positions": positions, "step": idx + 1}
    done = new_state["step"] == N_MACROS
    reward = reward_fn(positions) if done else 0.0
    return new_state, reward, done
```

### 7.2 Differentiable reward, correctly handling variable-arity nets

```python
import jax.numpy as jnp
from jax import grad, vmap, jit

def hpwl(positions, padded_pin_idx, valid_mask):
    pin_xy = positions[padded_pin_idx]
    big = jnp.finfo(pin_xy.dtype).max
    lo = jnp.where(valid_mask[:, None], pin_xy,  big).min(axis=0)
    hi = jnp.where(valid_mask[:, None], pin_xy, -big).max(axis=0)
    return (hi - lo).sum()

total_hpwl = jit(lambda pos: vmap(hpwl, in_axes=(None, 0, 0))(pos, all_nets, all_masks).sum())
wirelength_gradient = grad(total_hpwl)
```

**Why masking matters (tested):** naive unmasked version gave 30.0 vs. correct 20.0 on a 4-cell hand-calculated example, silently, no error. Validate against a known-correct external number (DREAMPlace's own reported HPWL) before trusting this, every time.

### 7.3 Reward as a swappable argument

```python
def reward_hpwl(positions):
    return -hpwl(positions)

def reward_congestion_aware(positions):
    spread = jnp.var(positions.astype(jnp.float32))
    return -hpwl(positions) + 0.5 * spread

def run_episode(agent_fn, reward_fn, key):
    state = reset()
    for _ in range(N_MACROS):
        key, subkey = random.split(key)
        action = agent_fn(state, subkey)
        state, reward, done = step(state, action, reward_fn)
    return reward, state["positions"]
```

### 7.4 Four agents, same interface (all tested)

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

```python
def cellplace_with_dreamplace(macro_positions, netlist):
    ...  # writes .pl, shells out to DREAMPlace, reads result back

def cellplace_with_replace(macro_positions, netlist):
    ...  # different team's tool, same interface

def validate_with_openroad(full_placement):
    ...  # shells out to OpenROAD-flow-scripts

def place_and_validate(macro_positions, netlist, cellplace_fn, validate_fn):
    full_placement = cellplace_fn(macro_positions, netlist)
    return validate_fn(full_placement)
```

---

## 8. Literature compatibility check

Checked the single-kernel design against 16 real tools/papers beyond the four directly tested agents — to confirm the interface generalizes rather than accidentally fitting only what was tried.

| Tool | Category | Fits via `step()`/`reset()`? |
|---|---|---|
| MaskPlace | CNN + PPO, sequential | Yes — `agent_fn` |
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
