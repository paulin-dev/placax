"""One training loop, driven entirely by an ExperimentConfig, that records what it did.

This replaces the two hand-rolled loops the training scripts each carried. They differed in ways
nobody chose deliberately - one had early stopping and a best-checkpoint, the other had snapshots;
one took a --seed, the other hardcoded PRNGKey(0); neither wrote down a single thing about its own
configuration. Every run now goes through here, whatever the agent, so every run gets the same
treatment and leaves the same evidence behind:

  manifest.json      the full config, its hashes, the metric definitions and the machine
                     fingerprint, written at start
  training_log.jsonl one line per iteration, each carrying the run's hash, its budget spend, and
                     - on every evaluated iteration - the placement's LEGALITY alongside its
                     score, because a wirelength number for an overlapping placement is not a
                     result
  state.json         budget spend, so a resumed run continues the same budget
  checkpoint.bin     the agent's own state plus the RNG key and iteration count
  best_checkpoint.bin  bare weights plus the real_hpwl that earned them (agents that have weights)

The loop talks only to the Agent protocol, and scores placements itself. That second part matters
as much as the first: an agent returns a PLACEMENT and the runner computes its HPWL, so
"PPO beat random search" can never come down to two agents using different scoring conventions.
"""
import json
import pathlib
import time

from placax.log import Log  # must precede jax imports
from placax.core import replay
from placax.extras.legality import jitted_legality
from placax.extras.orientation import effective_sizes, oriented_pin_offsets
from placax.extras.rewards import hpwl
from placax.reproducibility import describe_determinism, fingerprint
from placax_agents.agents.ppo import is_ppo_state
from placax_agents.experiment.budget import BudgetTracker, BudgetUse
from placax_agents.experiment.build import BuiltExperiment, build
from placax_agents.experiment.config import ExperimentConfig
from placax_agents.ops.checkpoint import load_checkpoint, save_checkpoint
from placax_agents.policy.scale import to_grid_units, to_real_centers
from placax_agents.training.algorithm.running_stats import init_running_stats

import jax.numpy as jnp
from jax import random

METRICS = {
    "real_hpwl": "Half-perimeter wirelength of the final placement, in real design units, over "
                 "MACRO-TO-MACRO nets only - the netlist parser drops every net with fewer than "
                 "two macros, so this is not full-netlist HPWL and is not directly comparable to "
                 "a paper that reports one. Computed by the runner from the placement the agent "
                 "handed over, identically for every agent.",
    "reward_return": "Sum of the run's CONFIGURED reward over the episode, obtained by replaying "
                     "the final placement through placax.core.step. This is what the agent was "
                     "actually asked to optimize; real_hpwl is fixed across configs so a table "
                     "stays readable when the reward changes.",
    "overlap_ratio": "Fraction of total macro area covered by more than one macro. Should be 0; "
                     "anything else means the action mask's relaxation valve fired and the "
                     "placement is not physically realizable.",
    "out_of_bounds_ratio": "Fraction of total macro area falling outside the canvas.",
    "gradient_steps": "Parameter updates applied so far. Zero for a non-learning method - the "
                      "compute env_steps deliberately does not price.",
}
"""What each logged metric actually means, written into every manifest.

A number is only comparable if its definition travels with it, and `real_hpwl`'s definition in
particular is easy to get wrong from the outside: it is macro-to-macro only.
"""

MANIFEST_NAME = "manifest.json"
LOG_NAME = "training_log.jsonl"
STATE_NAME = "state.json"
CHECKPOINT_NAME = "checkpoint.bin"
BEST_CHECKPOINT_NAME = "best_checkpoint.bin"


def write_manifest(output_dir: pathlib.Path, config: ExperimentConfig) -> pathlib.Path:
    """Records the config and the machine alongside the results, before any of them exist.

    Written at start rather than at the end so a crashed or killed run still leaves behind an
    attributable record of what it was trying to do.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / MANIFEST_NAME
    path.write_text(json.dumps({
        "config": config.to_dict(),
        "metrics": METRICS,
        "fingerprint": fingerprint(),
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }, indent=2, sort_keys=True) + "\n")
    return path


def _bundle(agent_state, key, iteration: int) -> dict:
    """The resumable training state: whatever the agent carries, plus the runner's own two things."""
    return {"agent_state": agent_state, "iteration": jnp.array(iteration), "key": key}


def _load_bundle(template: dict, path: pathlib.Path) -> dict:
    """Reads a checkpoint, migrating the pre-Agent layout if that is what's on disk.

    Checkpoints written before agents existed stored variables/opt_state/running_stats at the top
    level rather than under "agent_state". Those are real in-flight training runs, so they are
    migrated rather than rejected - a refactor should not cost someone a partially-trained model.
    """
    try:
        return load_checkpoint(template, path)
    except (KeyError, ValueError) as exc:
        agent_state = template["agent_state"]
        if not is_ppo_state(agent_state):
            raise
        legacy_template = {
            "variables": agent_state["variables"], "opt_state": agent_state["opt_state"],
            "running_stats": agent_state.get("running_stats", init_running_stats()),
            "iteration": template["iteration"], "key": template["key"],
        }
        try:
            legacy = load_checkpoint(legacy_template, path)
        except (KeyError, ValueError):
            raise exc
        Log.info(f"  migrated a pre-agent checkpoint layout from {path}")
        return _bundle(
            {"variables": legacy["variables"], "opt_state": legacy["opt_state"],
             "running_stats": legacy["running_stats"]},
            legacy["key"], int(legacy["iteration"]),
        )


def _read_budget_use(output_dir: pathlib.Path | None, fallback_iterations: int) -> BudgetUse:
    """Budget spent by earlier invocations of this run.

    Falls back to the checkpoint's own iteration count when state.json is absent, which is what
    happens for a run started before budgets existed: iterations are recovered, and env_steps and
    wall-clock restart from zero. Better than refusing to resume.
    """
    if output_dir is None:
        return BudgetUse()
    path = output_dir / STATE_NAME
    if not path.exists():
        return BudgetUse(iterations=fallback_iterations)
    return BudgetUse.from_dict(json.loads(path.read_text())["budget_use"])


def _write_budget_use(output_dir: pathlib.Path | None, config: ExperimentConfig, use: BudgetUse) -> None:
    if output_dir is None:
        return
    (output_dir / STATE_NAME).write_text(json.dumps({
        "budget_use": use.to_dict(),
        "environment_hash": config.environment_hash(),
        "full_hash": config.full_hash(),
    }, indent=2, sort_keys=True) + "\n")


def _append_log(log_path: pathlib.Path | None, entry: dict) -> None:
    if log_path is None:
        return
    with open(log_path, "a") as f:
        f.write(json.dumps(entry) + "\n")


def _read_best_real_hpwl(variables_template, path: pathlib.Path | None) -> float:
    """The real_hpwl bundled alongside path's weights, or +inf if there isn't one yet."""
    if path is None or not path.exists() or variables_template is None:
        return float("inf")
    template = {"variables": variables_template, "real_hpwl": jnp.array(0.0)}
    return float(load_checkpoint(template, path)["real_hpwl"])


def score_placement(benchmark, positions, orientations=None) -> float:
    """Real HPWL of a placement, computed by the runner so every agent is measured identically.

    Orientation enters as a transform on the geometry rather than as a new argument to `hpwl`: a
    turned macro occupies its height by its width, and its pins rotate about its center with it.
    Both are the identity when nothing is oriented, so an un-oriented placement scores exactly
    what it always did. See placax/extras/orientation.py.
    """
    sizes = effective_sizes(benchmark.sizes_array, orientations)
    centers = to_real_centers(positions, sizes, benchmark.cell_size)
    offsets = oriented_pin_offsets(
        benchmark.padded_pin_offset, benchmark.padded_pin_idx, orientations
    )
    return float(hpwl(centers, benchmark.padded_pin_idx, offsets, benchmark.valid_mask))


def best_orientations(agent, state):
    """The agent's chosen orientations, or None for one that does not choose them.

    Read through `getattr` rather than added to the Agent protocol: an agent driving a space
    without orientation has nothing to say here, and requiring every agent to declare that would
    be ceremony. See placax/extras/orientation.py.
    """
    getter = getattr(agent, "best_orientations", None)
    return getter(state) if getter is not None else None


def score(benchmark, positions, n_placed: int = 0, orientations=None) -> dict:
    """Everything the runner measures about one placement, computed identically for every agent.

    Three things rather than one, because a wirelength number alone can hide two different
    problems. `real_hpwl` is the fixed cross-config metric. `reward_return` is what the agent was
    actually asked to optimize - replayed through the same `step()` a policy drives, so swapping
    the reward moves the number every agent is judged by. The legality fields say whether the
    placement is physically realizable at all: overlap makes wires shorter, so an illegal
    placement looks like a better result unless legality is reported beside the score.
    """
    grid_sizes = to_grid_units(
        effective_sizes(benchmark.sizes_array, orientations), benchmark.cell_size
    )
    measured = jitted_legality(positions, grid_sizes, benchmark.params).to_dict()
    return {
        "real_hpwl": score_placement(benchmark, positions, orientations),
        "reward_return": float(
            replay(positions, benchmark.reward_fn, benchmark.params, n_placed)
        ),
        **measured,
    }


def run_experiment(
    config: ExperimentConfig,
    output_dir: pathlib.Path | None,
    built: BuiltExperiment | None = None,
    eval_every: int = 10,
    log_every: int = 1,
    patience: int = 0,
    placement_images_dir: pathlib.Path | None = None,
    init_from: pathlib.Path | None = None,
    resume: bool = True,
):
    """Runs `config` until its budget is exhausted, returning (agent_state, log).

    output_dir=None runs entirely in memory: no manifest, no checkpoints, no log file - for
    throwaway probe runs. Everything else about the run is identical.
    """
    built = built if built is not None else build(config)
    # build() resolves the netlist digest, so the config it carries identifies the DESIGN and not
    # just the path it was loaded from. Everything recorded from here on uses that one.
    config = built.config
    benchmark, agent = built.benchmark, built.agent

    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        manifest_path = write_manifest(output_dir, config)
        checkpoint_path = output_dir / CHECKPOINT_NAME
        best_checkpoint_path = output_dir / BEST_CHECKPOINT_NAME
        log_path = output_dir / LOG_NAME
        Log.info(f"manifest -> {manifest_path}")
    else:
        checkpoint_path = best_checkpoint_path = log_path = None

    Log.info(f"experiment {config.name!r}  env={config.environment_hash()}  run={config.full_hash()}")
    Log.info(f"  {describe_determinism()}")

    # 1. The agent's starting state, optionally warm-started from a previous run's best weights.
    key = random.PRNGKey(config.seed)
    key, init_key = random.split(key)
    agent_state = agent.init(init_key)
    if init_from is not None:
        if not is_ppo_state(agent_state):
            raise ValueError(
                f"--init_from loads policy weights, which agent {agent.name!r} does not have"
            )
        agent_state = dict(agent_state)
        agent_state["variables"] = load_checkpoint(
            {"variables": agent_state["variables"], "real_hpwl": jnp.array(0.0)}, init_from
        )["variables"]
        Log.info(f"  warm-starting weights from {init_from}")
        resume = False  # these weights are the starting point; don't overwrite them from a checkpoint

    # 2. Resume agent state and budget spend together, so both describe the same run.
    start_iteration = 0
    if resume and checkpoint_path is not None and checkpoint_path.exists():
        restored = _load_bundle(_bundle(agent_state, key, 0), checkpoint_path)
        agent_state, key, start_iteration = (
            restored["agent_state"], restored["key"], int(restored["iteration"])
        )
    prior_use = _read_budget_use(output_dir if resume else None, start_iteration)
    tracker = BudgetTracker(config.environment.budget, prior=prior_use)
    if prior_use.iterations:
        Log.info(f"  resumed at {tracker.describe()}")

    variables_template = agent_state["variables"] if is_ppo_state(agent_state) else None
    best_real_hpwl = _read_best_real_hpwl(variables_template, best_checkpoint_path)
    if best_real_hpwl < float("inf"):
        Log.info(f"  best real_hpwl so far: {best_real_hpwl:.1f}")
    # Reset per invocation: a resumed best_checkpoint doesn't carry how many evals without
    # improvement preceded it, so patience counts only evals in this invocation.
    evals_without_improvement = 0

    grid_sizes = to_grid_units(benchmark.sizes_array, benchmark.cell_size)
    log: list[dict] = []
    stop_reason = tracker.exhausted()
    if stop_reason is not None:
        Log.info(f"  budget already exhausted ({stop_reason}); nothing to do")
    elif not tracker.can_afford(built.episodes_per_iteration, built.steps_per_episode):
        # A budget that cannot pay for a single iteration is a configuration mistake, not a
        # finished run, so say so rather than exiting successfully having trained nothing.
        raise ValueError(
            f"budget of {config.environment.budget.env_steps:,} env steps cannot afford one "
            f"iteration of this agent, which costs {built.env_steps_per_iteration:,} "
            f"({built.episodes_per_iteration} episodes x {built.steps_per_episode} macros). "
            f"Raise the budget to a multiple of that, or lower the agent's episodes per "
            f"iteration."
        )

    while stop_reason is None and tracker.can_afford(
        built.episodes_per_iteration, built.steps_per_episode
    ):
        # 3. One agent update, whatever that means for this agent.
        key, step_key = random.split(key)
        agent_state, result = agent.update(step_key, agent_state)
        use = tracker.record_iteration(
            result.episodes, built.steps_per_episode, result.gradient_steps
        )

        # 4. Periodic eval: the agent hands over its best placement, the runner scores it.
        #    Charged to the budget - an eval places every remaining macro, which is exactly as
        #    much environment work as a training episode. Leaving it free meant two runs on one
        #    env_step budget did different amounts of work if their --eval_every differed.
        measured = None
        if eval_every > 0 and use.iterations % eval_every == 0:
            positions = agent.best_positions(agent_state)
            orientations = best_orientations(agent, agent_state)
            use = tracker.record_evaluation(built.steps_per_episode)
            measured = score(benchmark, positions, built.n_placed, orientations)
            real_hpwl = measured["real_hpwl"]
            if not measured["is_legal"]:
                Log.warning(
                    f"  iteration {use.iterations}: placement is NOT legal "
                    f"(overlap {measured['overlap_ratio']:.1%}, out of bounds "
                    f"{measured['out_of_bounds_ratio']:.1%}, {measured['n_unplaced']} unplaced) - "
                    f"its wirelength is not a realizable result"
                )
            if placement_images_dir is not None:
                from placax_viz.placement import save_placement_image

                placement_images_dir.mkdir(parents=True, exist_ok=True)
                save_placement_image(
                    positions,
                    to_grid_units(effective_sizes(benchmark.sizes_array, orientations),
                                  benchmark.cell_size),
                    benchmark.params.grid_x, benchmark.params.effective_grid_y,
                    placement_images_dir / f"{use.iterations}.png",
                )
            if real_hpwl < best_real_hpwl:
                best_real_hpwl, evals_without_improvement = real_hpwl, 0
                if best_checkpoint_path is not None and is_ppo_state(agent_state):
                    save_checkpoint(
                        {"variables": agent_state["variables"],
                         "real_hpwl": jnp.array(best_real_hpwl)},
                        best_checkpoint_path,
                    )
                Log.info(f"  new best real_hpwl={best_real_hpwl:.1f} at iteration {use.iterations}")
            else:
                evals_without_improvement += 1

        # 5. Every iteration is logged with the spend and the run hash that produced it, so a
        #    log line is attributable to a configuration without any external bookkeeping.
        entry = {
            "iteration": use.iterations,
            "env_steps": use.env_steps,
            "eval_env_steps": use.eval_env_steps,
            "episodes": use.episodes,
            "gradient_steps": use.gradient_steps,
            "wall_clock_s": round(use.wall_clock_s, 3),
            "loss": result.loss,
            "real_hpwl": None,
            "full_hash": config.full_hash(),
            **(measured or {}),
            **result.metrics,
        }
        log.append(entry)
        _append_log(log_path, entry)
        if log_every > 0 and use.iterations % log_every == 0:
            loss_str = f"{result.loss:>10.4f}" if result.loss is not None else "         -"
            hpwl_str = f"{entry['real_hpwl']:.1f}" if entry["real_hpwl"] is not None else "-"
            Log.info(f"{tracker.describe()}  loss={loss_str}  real_hpwl={hpwl_str}")

        # 6. Checkpoint every iteration so a crash never costs more than one.
        if checkpoint_path is not None:
            save_checkpoint(_bundle(agent_state, key, use.iterations), checkpoint_path)
        _write_budget_use(output_dir, config, use)

        # 7. Stop on budget, on a stalled real_hpwl, or because the agent says it is done.
        stop_reason = tracker.exhausted()
        if stop_reason is None and patience > 0 and evals_without_improvement >= patience:
            stop_reason = "patience"
        if stop_reason is None and agent.converged(agent_state):
            # A deterministic heuristic cannot produce a different placement next iteration.
            # Spending the rest of the budget replaying it would write thousands of identical
            # log lines and checkpoints; the budget still records what was actually used.
            stop_reason = "converged"

    # The loop also exits when the next iteration wouldn't fit, which is the env_step cap binding
    # one iteration earlier than `exhausted()` would report it.
    stop_reason = stop_reason or "env_steps"
    final_use = tracker.use
    Log.info(
        f"stopped: {stop_reason} - {final_use.iterations} iterations, "
        f"{final_use.env_steps:,} env steps, {final_use.wall_clock_s:.0f}s"
    )
    _write_budget_use(output_dir, config, final_use)
    return agent_state, log
