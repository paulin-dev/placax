"""Runs the MaskPlace-equivalent pipeline end to end, using only placax's
existing pluggable pieces (not a MaskPlace reimplementation):

  - connectivity_order        macro placement order (MaskPlace's topology order)
  - macro_budget              place only the N most important macros (--pnm)
  - dense=True reward         per-step HPWL delta, not a terminal-only reward
  - make_wiremask_observation wiremask + 2-macro lookahead as observation channels
  - make_wiremask_quality_illegal   wirelength-guided action masking (soft_coefficient)
  - ResNetCoarseFineActorCritic(critic_style="step_embedding")  two-branch network,
                               critic shares zero parameters with the actor
  - maskplace_ppo_config       its GAE/entropy/value-loss choices
  - maskplace_optimizer        independently-clipped, independently-stepped actor/critic Adam
  - train_buffered            its buffer-collect + minibatch-epoch PPO update procedure

Usage: python scripts/run_maskplace.py --benchmark_dir=benchmarks/adaptec1 --n_iterations=300 \\
           --n_episodes=10   (run --help for all flags, including macro_budget)

To find the largest n_episodes your hardware actually supports, use scripts/subprocess_search.py
*separately first* (not a flag of this script - see that module's docstring for why: this
process's own GPU memory reservation, just from starting up, would otherwise compete with the
disposable subprocesses being probed). It needs no special cooperation from this script - just
run it with its own ordinary CLI, once per candidate:

    python -m scripts.subprocess_search scripts.run_maskplace '--n_episodes=[1,2,4,8,10]' \\
        --benchmark_dir=benchmarks/adaptec1 --macro_budget=128 --eval_every=1 --n_iterations=4 --no_checkpoint

--eval_every=1/--n_iterations=4 there deliberately don't match a real run's --eval_every=10: the
eval rollout is a separately-compiled executable with its own memory footprint, so a probe needs
to cross at least one eval boundary (plus a couple more iterations, to catch ordinary GPU
allocator fragmentation drift) to be representative - forcing it every iteration reaches that
footprint by iteration 1 instead of iteration 10, so 4 iterations suffice instead of 13.
"""
import argparse
import pathlib
import sys

from placax import _device  # noqa: F401  must precede jax imports
from placax.core import reset
from placax.log import Log
from placax.netlist.order import connectivity_order
from placax.netlist.padding import build_macro_net_index
from placax_agents.benchmark import Benchmark
from placax_agents.ops.resumable_train import _append_log_entry, _maybe_evaluate
from placax_agents.policy.action import make_wiremask_quality_illegal
from placax_agents.policy.architectures.resnet_cnn import (
    ResNetCoarseFineActorCritic,
    build_pretrained_resnet_backbone,
    build_untrained_resnet_backbone,
)
from placax_agents.policy.observation import make_wiremask_observation
from placax_agents.training.algorithm.config import PPOConfig
from placax_agents.training.algorithm.loss import huber_value_loss
from placax_agents.training.algorithm.split_optimizer import make_grouped_optimizer
from placax_agents.training.loops.buffered_train import buffered_train_step
from placax_agents.training.loops.common import checkpoint_every_n, open_train_state, save_train_state
from placax_agents.training.reward import make_scaled_hpwl_reward

import optax
from jax import random

WIREMASK_MARGIN = 1.0  # MaskPlace's own --soft_coefficient default

MASKPLACE_REWARD_DIVISOR = 200.0
"""MaskPlace's own constant divisor on its (grid-unit) reward, applied per step in PPO2.py."""

MASKPLACE_MACRO_BUDGET = 128
"""MaskPlace's own --pnm default: place only its 128 most important macros, not the whole netlist."""

MASKPLACE_N_EPISODES = 10
"""MaskPlace's own buffer size: 10 episodes collected (in parallel, via vmap) per PPO update."""

MASKPLACE_LEARNING_RATE = 2.5e-3
"""MaskPlace's own --lr default (PPO2.py)."""

MASKPLACE_MAX_GRAD_NORM = 0.5
"""MaskPlace's own PPO.max_grad_norm; clipped per-network, not jointly."""


def maskplace_ppo_config() -> PPOConfig:
    """PPOConfig matching MaskPlace's own PPO2.py defaults (gamma=0.95, no GAE smoothing, no entropy bonus)."""
    return PPOConfig(gamma=0.95, lam=1.0, clip_eps=0.2, entropy_coef=0.0, value_loss_fn=huber_value_loss)


def maskplace_optimizer(
    learning_rate: float = MASKPLACE_LEARNING_RATE,
    max_grad_norm: float = MASKPLACE_MAX_GRAD_NORM,
    critic_param_prefix: str = "critic_",
    value_coef: float = PPOConfig().value_coef,
) -> optax.GradientTransformation:
    """Separately-clipped Adam for actor and critic, matching MaskPlace's two-independent-backward-pass setup.

    Requires critic params named under critic_param_prefix and disjoint from the actor's
    (true for critic_style="step_embedding", not the shared-trunk "canvas" style).

    value_coef must match whatever value_coef the training loss uses: ppo_loss differentiates
    policy_loss + value_coef*value_loss as one scalar, so the critic's gradient arrives here
    already pre-scaled by value_coef. Left uncompensated, clipping that scaled gradient to
    max_grad_norm silently turns the critic's real clip threshold into max_grad_norm/value_coef
    instead of the intended max_grad_norm; dividing it back out here makes both networks clip
    against their own true, unweighted gradient norm - matching MaskPlace's two fully independent
    backward passes (each with its own untouched clip_grad_norm_(0.5), no value_coef involved).
    """
    actor_chain = optax.chain(optax.clip_by_global_norm(max_grad_norm), optax.adam(learning_rate))
    critic_chain = optax.chain(optax.clip_by_global_norm(max_grad_norm / value_coef), optax.adam(learning_rate))
    return make_grouped_optimizer(critic_chain, actor_chain, critic_param_prefix)


def _parse_args(argv: list[str]) -> tuple[pathlib.Path, int, int | None, str, int, int]:
    """(benchmark_dir, n_iterations, macro_budget, n_episodes_arg, log_every, eval_every,
    no_checkpoint) from named --flag=value CLI args; macro_budget is None if --macro_budget=all
    was given; n_episodes_arg is an int-as-string (auto-detection is a separate step - see
    scripts/subprocess_search.py and this module's own docstring for why it isn't a flag of this
    script)."""
    parser = argparse.ArgumentParser(description="Run the MaskPlace-equivalent pipeline end to end.")
    parser.add_argument(
        "--benchmark_dir", type=pathlib.Path, default=pathlib.Path("benchmarks/adaptec1"),
        help="Path to a downloaded benchmark directory (default: benchmarks/adaptec1).",
    )
    parser.add_argument(
        "--n_iterations", type=int, default=100,
        help="Number of buffered-PPO update cycles to run (default: 100).",
    )
    parser.add_argument(
        "--macro_budget", type=str, default=str(MASKPLACE_MACRO_BUDGET),
        help='Place only the N most important macros, MaskPlace\'s --pnm (default: %(default)s, '
             'MaskPlace\'s own value); pass "all" to place every macro in the netlist instead.',
    )
    parser.add_argument(
        "--n_episodes", type=str, default=str(MASKPLACE_N_EPISODES),
        help='Episodes collected per PPO update (default: %(default)s, MaskPlace\'s own value). To find '
             'the largest value your hardware supports, run scripts/subprocess_search.py separately '
             "first (see this module's own docstring for an example) and pass its result here - "
             'auto-detection deliberately isn\'t a flag of this script, since this process reserving '
             "its own GPU memory just by starting would compete with the very subprocesses being probed.",
    )
    parser.add_argument(
        "--log_every", type=int, default=1,
        help="Print a progress line to the console every this many iterations (default: 1, every iteration).",
    )
    parser.add_argument(
        "--eval_every", type=int, default=10,
        help="Compute real HPWL (a full extra greedy rollout) every this many iterations (default: 10).",
    )
    parser.add_argument(
        "--no_checkpoint", action="store_true",
        help="Don't read or write checkpoint.bin - useful for a quick, disposable run (e.g. "
             "when probing via scripts/subprocess_search.py) that shouldn't resume from or "
             "leave behind any state.",
    )
    args = parser.parse_args(argv[1:])
    macro_budget = None if args.macro_budget.lower() == "all" else int(args.macro_budget)
    return (
        args.benchmark_dir, args.n_iterations, macro_budget, args.n_episodes, args.log_every,
        args.eval_every, args.no_checkpoint,
    )


def _maskplace_reward_fn(padded_pin_idx, padded_pin_offset, valid_mask, sizes_array, cell_size):
    """MaskPlace's own reward magnitude: its reward is accumulated in grid-cell units (pins rounded
    to the nearest cell) then divided by 200 before being buffered (PPO2.py's `reward / 200.0`), not
    the raw real-unit HPWL delta placax computes by default. Dividing by cell_size here converts our
    real-unit delta back to that grid-unit-equivalent scale before applying the same /200; without
    this, clip_eps/entropy_coef/max_grad_norm - all tuned against MaskPlace's own reward magnitude -
    would be operating on rewards ~1000x too large for this benchmark's cell_size."""
    reward_scale = 1.0 / (cell_size * MASKPLACE_REWARD_DIVISOR)
    return make_scaled_hpwl_reward(
        padded_pin_idx, padded_pin_offset, valid_mask, sizes_array, cell_size,
        dense=True, reward_scale=reward_scale,
    )


def _load_benchmark(benchmark_dir: pathlib.Path, macro_budget: int | None) -> Benchmark:
    """Connectivity order, macro budget, dense reward - loaded once, shared everywhere below."""
    return Benchmark.load(
        benchmark_dir,
        grid=224,  # MaskPlace's own grid resolution
        order_fn=connectivity_order,
        macro_budget=macro_budget,
        make_reward_fn=_maskplace_reward_fn,
    )


def _build_state_fn(benchmark: Benchmark):
    """Builds the wiremask + 2-macro-lookahead observation function, MaskPlace's own channel set."""
    # 1. Precompute, once, which nets touch each macro - the wiremask
    #    observation needs this lookup on every step, so building it here
    #    (outside the per-step hot path) avoids redoing the work per call.
    macro_net_idx, macro_net_offset, macro_net_valid = build_macro_net_index(
        benchmark.padded_pin_idx, benchmark.padded_pin_offset, benchmark.valid_mask,
        n_macros=benchmark.params.n_macros,
    )
    # 2. Build the actual observation function, closing over that index.
    return make_wiremask_observation(
        benchmark.padded_pin_idx, benchmark.padded_pin_offset, benchmark.valid_mask,
        macro_net_idx, macro_net_offset, macro_net_valid, cell_size=benchmark.cell_size, lookahead=2,
    )


def _resnet_backbone():
    """Uses real ImageNet weights if placax[resnet] is installed, otherwise an offline same-shape stand-in."""
    try:
        # Prefer real pretrained weights when the optional dependency is available.
        import flaxmodels  # noqa: F401
    except ImportError:
        # Fall back to an untrained backbone with matching shapes, so training
        # still works offline - just tell the user clearly which path was taken.
        Log.info("flaxmodels not installed (pip install placax[resnet]) - using an "
                  "offline, untrained ResNet backbone instead of real ImageNet weights.")
        return build_untrained_resnet_backbone()
    return build_pretrained_resnet_backbone()


def _build_policy(benchmark: Benchmark) -> ResNetCoarseFineActorCritic:
    """MaskPlace's own network shape: fine + coarse-ResNet branches, step-embedding critic."""
    return ResNetCoarseFineActorCritic(
        resnet_backbone=_resnet_backbone(), params=benchmark.params, cell_size=benchmark.cell_size,
        critic_style="step_embedding",
    )


def _train_and_eval_loop(
    key, variables, policy, benchmark: Benchmark, optimizer, ppo_config, state_fn,
    extra_illegal_fn, checkpoint_path: pathlib.Path | None, n_iterations: int, n_episodes: int,
    log_every: int, eval_every: int, log_path: pathlib.Path | None,
):
    """Runs n_iterations of buffered-PPO training one iteration at a time, resuming from
    checkpoint_path if it exists. Real HPWL is computed every eval_every iterations (a full extra
    greedy rollout, so not cheap); a progress line is printed every log_every iterations; every
    iteration is appended to log_path as JSONL regardless of log_every."""
    # Resume from checkpoint_path if it exists (read once here, not once per iteration below).
    variables, opt_state, running_stats, key, start_iteration = open_train_state(
        variables, key, optimizer, checkpoint_path
    )

    log = []
    for i in range(n_iterations):
        current_iteration = start_iteration + i + 1

        # 1. One buffered-PPO update: collect n_episodes fresh episodes, train on them.
        #    ppo_epochs=10, batch_size=64 are train_buffered's own defaults, matching MaskPlace.
        key, buffer_key = random.split(key)
        variables, opt_state, running_stats, loss = buffered_train_step(
            buffer_key, variables, opt_state, running_stats, optimizer, policy.apply,
            benchmark.params, benchmark.reward_fn, benchmark.sizes_array, benchmark.cell_size,
            n_episodes, ppo_epochs=10, batch_size=64, state_fn=state_fn, ppo_config=ppo_config,
            extra_illegal_fn=extra_illegal_fn,
        )

        # 2. Real-HPWL eval (only every eval_every iterations) + log entry (always, to
        #    log_path; console only every log_every iterations).
        real_hpwl = _maybe_evaluate(
            current_iteration, eval_every, variables, policy.apply, benchmark.params,
            benchmark.sizes_array, benchmark.cell_size, benchmark.padded_pin_idx,
            benchmark.padded_pin_offset, benchmark.valid_mask, state_fn, extra_illegal_fn,
        )
        _append_log_entry(log, log_path, current_iteration, loss, real_hpwl)
        if current_iteration % log_every == 0:
            hpwl_str = f"{real_hpwl:.1f}" if real_hpwl is not None else "-"
            Log.info(f"iter {current_iteration:>6}/{n_iterations}  loss={loss:>10.4f}  real_hpwl={hpwl_str}")

        # 3. Checkpoint every iteration (episodes are expensive to recollect on this hardware,
        #    so a crash should never lose more than one iteration of progress).
        checkpoint_every_n(checkpoint_path, 1, current_iteration, variables, opt_state, running_stats, key)

    # Always checkpoint at the end too (in case n_iterations was 0) - unless checkpoint_path is
    # None, meaning the caller (e.g. the memory-fitting probe) doesn't want anything written.
    if checkpoint_path is not None:
        save_train_state(checkpoint_path, variables, opt_state, running_stats, key, start_iteration + n_iterations)
    return variables


def main() -> None:
    """CLI entry point: wires up the MaskPlace-equivalent pipeline and runs/resumes training."""
    Log.configure()

    # 1. Parse CLI args and make sure the requested benchmark actually exists.
    benchmark_dir, n_iterations, macro_budget, n_episodes_arg, log_every, eval_every, no_checkpoint = _parse_args(
        sys.argv
    )
    if not benchmark_dir.exists():
        Log.error(f"'{benchmark_dir}' not found - run scripts/download_benchmarks.py first.")
        sys.exit(1)

    # 2. n_episodes must be an explicit value here - no in-process "auto" (see this module's own
    #    docstring, and scripts/subprocess_search.py's, for why): this process reserves its own
    #    GPU memory just by starting (JAX's default allocator preallocates a large fraction of
    #    the device as one arena on its first backend touch, which happens at import time via
    #    placax's own package init), so it can never probe disposable subprocess candidates
    #    without competing with them for the same physical device. Auto-detection has to be a
    #    genuinely separate, earlier invocation - one that never imports jax/placax at all - not
    #    something this script can do to itself no matter how its own internals are reordered.
    try:
        n_episodes = int(n_episodes_arg)
    except ValueError:
        Log.error(
            f"--n_episodes must be an integer, got {n_episodes_arg!r}. To find the largest value "
            "your hardware supports, run this first (separately, not as part of this script):\n\n"
            f"  python -m scripts.subprocess_search scripts.run_maskplace '--n_episodes=[1,2,4,8,{MASKPLACE_N_EPISODES}]' "
            f"--benchmark_dir={benchmark_dir} --macro_budget={macro_budget} "
            "--eval_every=1 --n_iterations=4 --no_checkpoint\n\n"
            "then pass the RESULT= value it prints as --n_episodes here."
        )
        sys.exit(1)

    # 3. Load the netlist with MaskPlace's own ordering/reward choices.
    Log.info(f"loading {benchmark_dir} (connectivity order, macro_budget={macro_budget}, dense reward) ...")
    benchmark = _load_benchmark(benchmark_dir, macro_budget)
    Log.info(f"  {len(benchmark.macro_sizes)} macros, {len(benchmark.nets)} nets, cell_size={benchmark.cell_size:.2f}")

    # 4. Build the observation function, illegal-action mask, and policy network.
    state_fn = _build_state_fn(benchmark)
    extra_illegal_fn = make_wiremask_quality_illegal(margin=WIREMASK_MARGIN)
    policy = _build_policy(benchmark)

    # 5. Initialize the policy's parameters using one dummy observation, so
    #    Flax can infer every layer's shape from real input.
    key = random.PRNGKey(0)
    key, init_key = random.split(key)
    obs0 = state_fn(reset(benchmark.params), benchmark.params, benchmark.sizes_array)
    variables = policy.init(init_key, obs0)

    # 6. Set up the checkpoint location; if one already exists, training below will resume from
    #    it instead of starting over - unless --no_checkpoint was given, meaning don't read or
    #    write any state at all (a quick, disposable run, e.g. a scripts/subprocess_search.py probe).
    if no_checkpoint:
        checkpoint_path = log_path = None
    else:
        output_dir = benchmark_dir / "output_maskplace"
        output_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_path = output_dir / "checkpoint.bin"
        log_path = output_dir / "training_log.jsonl"
        resuming = checkpoint_path.exists()
        Log.info(f"{'resuming from' if resuming else 'starting fresh, will save to'} {checkpoint_path}")

    # 7. Build the PPO config/optimizer matching MaskPlace's own hyperparameters.
    ppo_config = maskplace_ppo_config()
    optimizer = maskplace_optimizer(value_coef=ppo_config.value_coef)  # separately-clipped actor/critic Adam

    Log.info(
        f"running {n_iterations} more buffered-PPO iterations "
        f"({n_episodes} episodes/buffer, 10 minibatch epochs, batch 64, "
        f"independent actor/critic optimizers) ..."
    )

    # 8. Run the actual training/eval loop.
    variables = _train_and_eval_loop(
        key, variables, policy, benchmark, optimizer, ppo_config, state_fn, extra_illegal_fn,
        checkpoint_path, n_iterations, n_episodes=n_episodes,
        log_every=log_every, eval_every=eval_every, log_path=log_path,
    )

    if checkpoint_path is not None:
        print()
        print(f"checkpoint saved to {checkpoint_path} - re-run this script to continue training.")
        print(f"full history saved to {log_path}")


if __name__ == "__main__":
    main()
