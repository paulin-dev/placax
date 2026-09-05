# How the RL Agent Learns — A Guided Tour of `placax`'s MaskPlace Brain

This document explains, from first principles and grounded in this repo's actual code, how
reinforcement learning (RL) works, what our MaskPlace agent's "brain" physically is, how it makes a
placement decision, how it learns from experience, and where that brain lives on disk between runs.

---

## 1. The RL loop, in general

Reinforcement learning has one core loop: an **agent** looks at a **state**, picks an **action**, the
**environment** reacts and hands back a **reward** and a **new state**. Repeat until the episode ends.
Over many episodes, the agent adjusts its own decision-making so that future rewards get bigger.

```mermaid
flowchart LR
    A["Agent<br/>(the policy network)"] -- "action a_t" --> E["Environment<br/>(the placement grid)"]
    E -- "new state s_t+1<br/>reward r_t" --> A
    A -. "learns from many\n(state, action, reward) tuples" .-> A
```

Three ingredients make this concrete for a given problem:

| Ingredient | Generic RL | Our MaskPlace agent |
|---|---|---|
| **State** | Whatever the agent can observe | Canvas so far + wiremask heatmap + legality mask + next-macro preview |
| **Action** | A choice from some action space | Pick one grid cell `(x, y)` to place the current macro |
| **Reward** | A scalar signal, higher = better | Negative HPWL delta (wirelength got shorter = positive reward) |

One **episode** = placing every macro of a netlist, one at a time, until none are left
(`placax/core.py::reset`/`step`). One **iteration** of training = collecting a batch of such episodes,
then updating the network's weights once from that batch.

---

## 2. What is "the brain," physically?

The agent's entire decision-making ability lives in a big pile of **numbers**: the weights of a
neural network. There is no symbolic rule like "if a macro is large, place it in the corner" written
anywhere — every behavior the agent has is the *emergent result* of millions of floating-point numbers
that got nudged, gradient step by gradient step, until they happen to produce good placements.

Concretely, in this codebase the brain is a Python/JAX object called `variables`: a nested dictionary
of arrays (a "pytree" in JAX terminology). It has exactly two top-level parts:

```mermaid
flowchart TD
    V["variables<br/>(the whole brain)"] --> P["params<br/>(TRAINABLE weights —<br/>these are what gradient descent changes)"]
    V --> B["batch_stats<br/>(BatchNorm running statistics —<br/>along for the ride, not trained by the optimizer)"]
    P --> P1["_FineBranch_0<br/>~121 numbers"]
    P --> P2["resnet_backbone<br/>~11,689,512 numbers"]
    P --> P3["_CoarseBranch_0<br/>~403,747 numbers"]
    P --> P4["Conv_0 (merge)<br/>~3 numbers"]
    P --> P5["critic_step_embed / critic_hidden1/2 / critic_value<br/>~139,457 numbers"]
```

(Those parameter counts are real, measured directly from this project's own policy —
`ResNetCoarseFineActorCritic`, `placax_agents/policy/architectures/resnet_cnn.py`.) The whole thing is
about **12.2 million numbers**. The overwhelming majority of them (95%+) belong to the ResNet-18
backbone alone.

**"Learning" = changing these numbers a little bit, repeatedly, in a direction that makes rewarded
actions more likely and unrewarded ones less likely.** Nothing else about the agent changes between a
freshly initialized run and a fully trained one — same code, same architecture, same shapes. Only the
*values* inside `variables["params"]` differ.

---

## 3. The network architecture: two branches, one decision

The policy is `ResNetCoarseFineActorCritic`. It looks at the current state from two angles at once —
a *fine*, purely local view, and a *coarse*, wide-context view — and merges them into one logit per
grid cell (how attractive that cell is for the current macro). A separate small head estimates the
*value* of the current state (used only during training, to judge whether an action outperformed
expectations).

```mermaid
flowchart TD
    subgraph Inputs
        CV["canvas<br/>(where macros already are)"]
        WM["wiremask (current + next)<br/>(cost heatmap per cell)"]
        PM["posmask (current + next)<br/>(illegal-cell mask)"]
    end

    subgraph FineBranch["Fine branch — _FineBranch (MyCNN)"]
        F1["1x1 Conv, 8ch + ReLU"] --> F2["1x1 Conv, 8ch + ReLU"] --> F3["1x1 Conv, 1ch<br/>(no receptive-field growth —\nonly looks at one cell at a time)"]
    end

    subgraph CoarseBranch["Coarse branch — _CoarseBranch"]
        direction TB
        RB["ResNet-18 backbone<br/>(ImageNet-pretrained,\n8 residual blocks,\n11.7M of the 12.2M total weights)"] --> POOL["global average pool"]
        POOL --> SEED["Dense -> small (7x7x16) seed"]
        SEED --> UP["5 learned deconv (upsample) layers"]
        UP --> RESIZE["resize to grid_x x grid_y"]
    end

    CV & WM --> RB
    WM & PM --> F1

    F3 --> MERGE["Conv 1x1<br/>concat(fine, coarse) -> 1 channel<br/>= action_logits"]
    RESIZE --> MERGE

    MERGE --> LOGITS["action_logits<br/>one number per grid cell:\nhow good is this cell?"]

    subgraph Critic["Critic head"]
        C1["small MLP / pooled features"] --> VAL["value<br/>(a single number:\nhow good is this state overall?)"]
    end

    F3 -.-> C1
    RESIZE -.-> C1

    style RB fill:#3b5,stroke:#333
    style LOGITS fill:#e8a33d,stroke:#333
    style VAL fill:#4a9,stroke:#333
```

Why two branches? The **fine** branch reacts to local detail (is *this specific* cell physically free
right now), while the **coarse** branch — via the ResNet — gives the network a sense of the *whole
canvas's* shape and congestion, the way a human glances at the full floorplan before deciding where a
big macro should roughly go. This mirrors MaskPlace's own two-branch design (`MyCNN` + `MyCNNCoarse`
in the original paper's `PPO2.py`).

---

## 4. From a raw state to one placement decision

```mermaid
sequenceDiagram
    participant Env as Environment (placax.core)
    participant Obs as observation()
    participant Net as ResNetCoarseFineActorCritic
    participant Act as legal_action_logits / sample_action

    Env->>Obs: current EnvState (positions so far, which macro is next)
    Obs->>Obs: build canvas, wiremask, posmask, lookahead arrays
    Obs->>Net: obs dict
    Net->>Net: forward pass through both branches (section 3)
    Net-->>Act: action_logits (grid_x, grid_y), value
    Act->>Act: set illegal cells (occupied, out of bounds,<br/>too-far-from-target-wiremask) to -inf
    Act->>Act: softmax -> probabilities over legal cells only
    alt training rollout
        Act->>Env: sample one action from that distribution
    else evaluation / real inference
        Act->>Env: take the single highest-probability action (argmax)
    end
    Env->>Env: place the macro there, compute HPWL-based reward,<br/>advance to the next macro
```

Two files do the heavy lifting here: `placax_agents/policy/observation.py` (state → network input) and
`placax_agents/policy/action.py` (`legal_action_logits`, `sample_action`, `action_log_prob`). Sampling
(used during training rollouts, via `collect_rollout` in `placax_agents/training/rollout.py`) injects
exploration — the agent sometimes tries a *not-currently-favorite* cell, which is exactly how it can
ever discover better strategies. Argmax (used in `placax_agents/ops/evaluate.py::evaluate`, and in
production inference) is fully deterministic: same weights, same state, always the same placement.

---

## 5. How the numbers actually get updated: one PPO training iteration

This is the "learning" step. It happens once per training **iteration**, after collecting a batch of
episodes.

```mermaid
flowchart TD
    S1["1. Collect a batch of episodes<br/>(collect_buffer, vmapped over n_episodes)<br/>— sample actions, record (obs, action, reward, log_prob, value)"]
    S2["2. Compute advantages & returns<br/>(compute_gae)<br/>— 'was this action better or worse<br/>than the critic expected?'"]
    S3["3. Run ppo_epochs passes over minibatches<br/>(_run_epochs / _minibatch_update)"]
    S4["4. For each minibatch:<br/>compute the PPO clipped loss (ppo_loss)<br/>= policy loss + value_coef*value loss - entropy_coef*entropy"]
    S5["5. Backpropagate:<br/>jax.value_and_grad -> one gradient<br/>per weight in variables['params']"]
    S6["6. Adam optimizer turns each gradient<br/>into a small weight update<br/>(clipped to max_grad_norm)"]
    S7["7. New, slightly different variables['params']<br/>-> the brain, updated"]

    S1 --> S2 --> S3 --> S4 --> S5 --> S6 --> S7
    S7 -. "next iteration, repeat" .-> S1
```

The **PPO clipped objective** is what keeps this stable: it increases the probability of actions whose
advantage was positive (better than the critic expected) and decreases it for negative-advantage
actions, but *clips* how far any single update can push a probability ratio, so one noisy batch can't
swing the weights wildly. `entropy_coef` adds a small bonus for keeping the action distribution spread
out (more exploration, less premature overconfidence); `gamma`/`lam` control how rewards get discounted
and smoothed across a whole episode (`placax_agents/training/algorithm/config.py::PPOConfig`,
`gae.py`, `loss.py`).

**Only `variables["params"]` gets trained.** `variables["batch_stats"]` (the ResNet's BatchNorm
statistics) is deliberately excluded from the optimizer — it's tagged as a "frozen collection" in
`placax_agents/training/algorithm/optimizer_step.py::apply_gradient_update` and simply passed through
each step unchanged by gradient descent (it does still get *recomputed live* on every forward pass, per
the September 2026 BatchNorm fix — but that recomputation isn't "learning," it's just normalization
math, not weights being nudged by a gradient).

---

## 6. Where the brain lives between runs: checkpoints

In memory, `variables` is just a Python object that vanishes the moment the process exits. To survive
across runs (or hand a trained policy to a completely different script), it has to be written to disk.

```mermaid
flowchart LR
    subgraph Memory["In memory, during training"]
        V["variables (dict of jax.Arrays)<br/>+ opt_state (Adam's own momentum/variance)<br/>+ running_stats (return normalizer)<br/>+ key (RNG state)<br/>+ iteration (int)"]
    end

    V -- "save_checkpoint()<br/>flax.serialization.to_bytes<br/>(msgpack encoding)" --> F1["checkpoint.bin<br/>on disk<br/>(full resumable state)"]
    F1 -- "load_checkpoint()<br/>flax.serialization.from_bytes<br/>+ dtype-cast to the template" --> V2["variables restored<br/>(bit-identical weights)"]

    V -. "at a NEW best real_hpwl only" .-> F2["best_checkpoint.bin<br/>(variables + real_hpwl ONLY —<br/>no optimizer/RNG state,<br/>meant for pure inference)"]
```

Two files, two purposes (`placax_agents/ops/checkpoint.py`, `placax_agents/training/loops/common.py`):

- **`checkpoint.bin`** — the *full* resumable training state: weights, the Adam optimizer's own
  internal momentum, the RNG key, the iteration counter. Written every single iteration
  (`checkpoint_every_n`), so a crash never loses more than one iteration of progress. This is what
  `--n_iterations` resumption reads.
- **`best_checkpoint.bin`** — just the weights (`variables`) plus the `real_hpwl` they achieved, saved
  only when a new best evaluation score is reached. This is the file meant for **production inference**
  (`scripts/run_pipeline.py`, `scripts/place_once.py`): load these weights, run a single greedy
  (argmax) rollout, done — no optimizer, no training loop, no further learning involved at all.

The file itself is a flat binary blob (msgpack-encoded arrays) — there is nothing "readable" in it by
eye, but structurally it is exactly the `variables` pytree from section 2, byte for byte.

---

## 7. Putting it all together: the full lifecycle

```mermaid
flowchart TD
    subgraph Train["Training (scripts/run_maskplace.py)"]
        T0["policy.init(key, obs0)<br/>-> random weights<br/>(ResNet part starts from<br/>real ImageNet weights, not random)"]
        T1["repeat: collect episodes -> PPO update<br/>(section 5)"]
        T2["every iteration: save checkpoint.bin"]
        T3["every --eval_every iterations:<br/>greedy eval -> real_hpwl<br/>if it's a new best -> save best_checkpoint.bin"]
        T0 --> T1 --> T2 --> T1
        T1 --> T3 --> T1
    end

    T2 -. "crash / stop, later resume\n(open_train_state reads checkpoint.bin)" .-> T1
    T3 --> BC["best_checkpoint.bin"]

    subgraph Infer["Inference / production (scripts/run_pipeline.py)"]
        I0["load benchmark, build the SAME policy shape"]
        I1["load_checkpoint(bare template, best_checkpoint.bin)"]
        I2["ONE greedy (argmax) rollout —<br/>no sampling, no gradients, no learning"]
        I3["macro placements -> DREAMPlace places\nremaining standard cells"]
        I0 --> I1 --> I2 --> I3
    end

    BC --> I1
```

The key mental model: **training is the only phase where the brain changes.** Everything downstream
of a saved checkpoint — evaluation, visualization, the production pipeline — loads those exact numbers
and just *runs the forward pass once per macro*. There's no learning happening at inference time, only
at training time; the network's *behavior* (how live BatchNorm statistics get computed per forward
pass) is identical in both phases, but the *weights* are completely frozen outside of training.

---

## Glossary

| Term | Meaning here |
|---|---|
| **Policy** | The network that outputs `action_logits` — decides *what to do*. |
| **Value function / critic** | The (smaller) network that outputs `value` — estimates *how good the current state is*, used only to compute advantages during training. |
| **Weights / parameters** | The actual numbers (`variables["params"]`) that gradient descent adjusts. This *is* the brain. |
| **`batch_stats`** | BatchNorm's per-channel running mean/variance inside the ResNet backbone — not trained by the optimizer, recomputed live each forward pass. |
| **Logit** | A raw, pre-softmax score. Higher logit = more attractive action, before converting to a probability. |
| **Advantage** | "How much better was this action than what the critic expected?" — the signal PPO actually optimizes. |
| **Episode** | One full placement of every macro in the netlist, start to finish. |
| **Iteration** | One PPO update: collect a batch of episodes, then update the weights once. |
| **Checkpoint** | A file on disk holding a snapshot of `variables` (and, for the full one, the rest of the resumable training state). |
| **Rollout** | The act of running the policy through an episode, either by sampling (training) or by argmax (evaluation/inference). |
