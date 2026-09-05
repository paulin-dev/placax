# Porting EXPlace and FlowPlace onto placax

Notes from reading [lamda-bbo/EXPlace](https://github.com/lamda-bbo/EXPlace) (ICLR'26) and
[lamda-bbo/flowplace](https://github.com/lamda-bbo/flowplace) (DAC'26) against the current
placax env. Both are from the same group (Nanjing University / Huawei Noah's Ark).

## Verdict

**EXPlace is a small delta on placax — a few days of work, mostly new mask functions.**
It is the same paradigm placax already implements (sequential macro-by-macro placement on a
grid canvas, per-candidate cost maps, PPO on a CNN). placax already has the single most
expensive piece of it.

**FlowPlace is a different paradigm and a large delta.** `reset()`/`step()`/action masking are
all unused by it. But two placax pieces drop straight in, and one of them is load-bearing.

---

# Part 1 — EXPlace

## Why it's close

EXPlace's env (`src/place_env/place_env.py`, 847 lines) is structurally the same as placax's:

| EXPlace | placax | Status |
|---|---|---|
| `get_wire_mask` | `extras/rewards.py::wiremask` | **already done**, and placax's is better — segment-reduced over unique net buckets, vmapped over the grid |
| `get_position_mask` | `extras/masks.py::occupancy_mask \| boundary_mask` | **already done** |
| `__draw_canvas` | `extras/render.py::render` | **already done** |
| lookahead on next macro | `lookahead_wiremasks`, `lookahead_sizes` | **already done** |
| PPO + actor/critic CNN | `training/algorithm/`, `policy/architectures/` | **already done** |
| `get_regularity_mask` | — | missing |
| `get_hierarchy_mask` | — | missing |
| `get_dataflow_mask` | — | missing |
| `get_port_mask` | — | missing |

Every EXPlace mask is a plain `(grid, grid)` float cost map for the macro about to be placed,
and the reward is a weighted sum of those maps read at the chosen cell:

```python
costs = {k: self.masks[k][x, y] for k in self.reward_max}
reward = sum(-self.trade_off_coeff[k] * costs_norm[k] for k in costs)
```

Default weights (`config/iccad.yaml`): `used_masks: [reg, hier, df, wire, port, pos]`,
`trade_off_coeff: [0.45, 0.2, 0.15, 0.15, 0.05]`. Note **regularity carries the largest
weight (0.45), roughly 3× the wirelength term.** That is the paper's central claim in one
line: HPWL alone is the wrong proxy, and pushing macros to the periphery matters more.

## The four missing masks

### `regularity_mask` — easiest, and the highest-weighted
Pure function of macro size and grid extent. No episode state at all:

```python
x1 = where((r >= 1) & (r <= end_x), r - start_x + 1, 0)   # r = arange(grid_x)
x2 = where((r >= 1) & (r <= end_x), end_x - r + 1, 0)
x_row = coef_x * minimum(x1, x2)                           # distance-to-nearest-edge in x
mask = x_row[:, None] + y_col[None, :]                     # corner_regular
# or minimum(x_mask, y_mask) for edge_regular
```

Because it depends only on `(size_x, size_y)`, it can be precomputed once per distinct macro
size at benchmark-load time and indexed — no per-step cost whatsoever. Start here; it is
~15 lines and carries 0.45 of the reward.

### `hierarchy_mask` — structurally identical to the existing wiremask
Bounding-box area increment of the macro's cluster. Needs static cluster assignments plus a
running per-cluster bbox over placed members — the exact shape of `_wiremask_baseline`:
reduce to `(lo, hi, has_members)` per cluster once per step, then broadcast `min`/`max`
against all candidate cells. Reuse that code pattern directly.

Cluster assignment itself is upstream work — EXPlace ships it as preprocessed data from
Google Drive rather than computing it in-repo.

### `dataflow_mask` — needs a `dataflow_mat`
Flow-weighted distance from the candidate cell to every placed macro:
`sum_b flow[a,b] * ||center_a(x,y) - center_b||`. In JAX that's a
`(grid_x, grid_y, n_macros)` broadcast — at grid=224, n=128 that is ~25 MB in fp32, fine.
The real work is building `dataflow_mat` (register-hop-distance-weighted macro connectivity),
which EXPlace also ships preprocessed. `prune_dataflow_mat(keep_ratio=0.1)` keeps only the
top 10% of connections, so it is sparse in practice.

### `port_mask` — cheapest to write
Overlap area between the candidate macro rectangle and I/O keepout rectangles. Same broadcast
shape as `occupancy_mask` but returns area rather than a bool. Static rectangles → fully
vectorized. Lowest weight (0.05); do it last.

## Three friction points

**1. `RewardFn` can't see the masks.** The signature is

```python
RewardFn = Callable[[old_positions, new_positions, old_placed, new_placed], jax.Array]
```

but EXPlace's reward needs `mask_k[action]`, and the observation has *already computed* those
maps. Recomputing inside `reward_fn` doubles the cost of the expensive one.

Mitigation: the `wire` term is free — `wiremask[x, y]` **is** the dense-HPWL delta, so
`make_hpwl_reward(dense=True)` already produces exactly that term. Only the four new terms
need a lookup, and three of the four are cheap. Still, widening `RewardFn` to take the
pre-step observation is the clean fix, at the cost of a breaking change to `placax/types.py`.

**2. Per-mask reward normalization is stateful across steps.** With `use_reward_scaling`,
EXPlace tracks running per-episode `reward_min`/`reward_max` per mask and normalizes each
cost into `[0,1]` before the weighted sum. placax's `RewardFn` is pure and has nowhere to keep
that. Either add a `reward_stats` field to `EnvState` (it is a `flax.struct.dataclass`, so
this threads through `scan` fine) or normalize per-step by the mask's own max — which is what
EXPlace's `__mask_normalization` already does for the *observation* path, so the two would
at least be consistent. Not equivalent to the paper, so worth an ablation.

**3. Prototype / regulator mode.** With `regulator_flag`, EXPlace subtracts `mask[proto_x,
proto_y]` — the cost at the macro's position in a DREAMPlace warm-start — so the reward
measures *improvement over an analytical placement* rather than absolute quality. This is the
macro-regulator idea and it is a large part of why these methods beat from-scratch RL.

**placax already supports this**: `reset(params, initial_positions=...)` takes a warm start
and `step()` resumes from it. What is missing is the DREAMPlace call to produce the prototype
and a `prototype_canvas` observation channel.

## Also worth lifting

- **`corner_flag`** (`find_all_corners` / `find_nearest_corner`): snaps the sampled action to
  the nearest contact point against a placed macro or the boundary. A pure action
  post-process — belongs next to `make_wiremask_quality_illegal` in `policy/action.py`.
- **Observation channels**: EXPlace's state is `canvas + prototype_canvas + normalized masks +
  next-macro masks + size`. `make_wiremask_observation` already stacks lookahead wiremasks;
  generalize it to stack all mask types, and grow `wiremask_cnn.py`'s input channels.

## Suggested order

1. `regularity_mask` + weighted-sum reward (0.45 weight, ~15 lines, no new data)
2. Prototype warm-start from DREAMPlace + regulator-mode subtraction
3. `port_mask`
4. `hierarchy_mask` (needs clustering)
5. `dataflow_mask` (needs `dataflow_mat`)

Steps 1–2 are testable on adaptec1 without any preprocessed data from their Drive link.
Steps 4–5 require building the clustering and dataflow extraction that EXPlace ships as
opaque `processed_data/`.

---

# Part 2 — FlowPlace

## Why it's far

FlowPlace does not place macros one at a time. All macros hold continuous coordinates in
`[-1,1]²` and move *simultaneously* along a learned velocity field, integrated with an
Euler/Heun ODE sampler. There is no episode, no action, no legality masking during generation.
So `core.py`'s `reset`/`step`, `policy/action.py`, and the whole PPO stack are unused.

Architecture is `AttGNN` over a PyG graph (`networks/gnn.py`, 602 lines) with sinusoidal time
encoding — placax has only CNNs in `policy/architectures/`.

## What drops straight in

**`hpwl()` is already a differentiable guidance potential.** FlowPlace hand-writes a PyG
`MessagePassing` module to get `hpwl_guidance_potential(x, cond, ...)` and its gradient.
placax's `extras/rewards.py::hpwl` computes exactly that, already in fp32 over
`positions[padded_pin_idx] + padded_pin_offset` — so `jax.grad(hpwl)` gives the guidance
gradient for free.

One caveat: call it with `cell_size=None`. The `_quantize` path does `jnp.round(...)`, whose
gradient is zero everywhere — it would silently kill the guidance signal.

**placax's sequential env is the right tool for FlowPlace's final legalizer.** This is the
non-obvious one. `hard_constraint_legalizer.py` takes the ODE's raw continuous output and
re-places macros greedily — largest-area-first, over a 32×32 candidate grid, scoring each
candidate by

```python
w_legality * overlap + w_hpwl * hpwl + w_dist * displacement   # w_legality = 1e6
```

That is a sequential, grid-scored, mask-filtered placement pass — i.e. precisely what
placax's `step()` + `occupancy_mask` + `wiremask` already do, only placax's version is
jitted and vmapped where theirs is a Python loop. Their `candidate_distance_score` and
`regularity_cost_for_macro` are ~10 lines each and compose with placax's existing
`quality_mask` cost-composition idiom.

## What you would have to build

- **Continuous-position support.** placax positions are integer grid indices with a `-1`
  sentinel for unplaced. Flow matching needs float coordinates throughout, so this is a second
  representation alongside the existing one, not a modification of it.
- **A differentiable overlap potential.** `legality_potential_tile` uses a softmax-smoothed
  pairwise overlap with an annealed sharpness (`legality_softmax_factor` ramps from min to max
  over sampling). placax's `occupancy_mask` is a hard boolean prefix-sum — correct for
  masking, useless for gradients. This is a genuinely new function.
- **A GNN backbone** in Flax.
- **The synthetic data generator.** Worth being precise here, because it is easy to guess wrong:
  FlowPlace does **not** train on real benchmarks. `data-gen/` samples entirely synthetic
  netlists — macro sizes, aspect ratios, terminal counts and edges all drawn from configured
  distributions — then places them with a legality mask and a **boundary-weighted sampling
  prior** (`weight_mode="dist_to_boundary"`, weight `∝ grid/(dist_to_nearest_edge + 1)²`).
  That periphery prior is the *only* domain knowledge injected, and it is the same intuition
  as EXPlace's regularity mask. Real benchmarks are used at inference only.

  Practical consequence: **you need no benchmark data to train it**, and placax's legality
  masks plus `render` cover the placement half of the generator. The random netlist sampler is
  new code. Also note placax's padded pin arrays are a *better* fit for JAX than PyG's
  `edge_index` — you would not port their graph format.

- **DREAMPlace in the loop** for standard-cell placement and HPWL scoring, same as EXPlace.

---

# Recommendation

Do EXPlace first, and specifically do the regularity mask and the DREAMPlace warm-start first.
Those two changes are small, need no preprocessed data, and target the two things the paper
identifies as why HPWL-only RL underperforms: no periphery prior, and starting from scratch
instead of refining an analytical solution.

FlowPlace is the more interesting long-term target for a JAX env — one-shot generation in
seconds versus hours of PPO rollout, and jitted ODE sampling is a natural fit — but it is a
parallel codepath, not an evolution of the current one. If you go there, the legalizer is the
piece to build first, since it is reusable under either paradigm and placax is already most
of the way to it.
