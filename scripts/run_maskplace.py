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
           --n_episodes=auto   (run --help for all flags, including macro_budget)
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
from placax_agents.ops.autotune import find_max_via_subprocess, is_oom
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

_MODULE = "scripts.run_maskplace"
_OOM_MARKER = "PLACAX_PROBE_OOM"

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


def _parse_args(argv: list[str]) -> tuple[pathlib.Path, int, int | None, str, int, int, int]:
    """(benchmark_dir, n_iterations, macro_budget, n_episodes_arg, log_every, eval_every,
    max_episodes) from named --flag=value CLI args; macro_budget is None if --macro_budget=all
    was given; n_episodes_arg is either an int-as-string or the literal "auto" (see
    NEpisodesDetector)."""
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
        help='Episodes collected per PPO update (default: %(default)s, MaskPlace\'s own value); '
             'pass "auto" to auto-detect the largest that fits on this hardware (see NEpisodesDetector).',
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
        "--max_episodes", type=int, default=MASKPLACE_N_EPISODES,
        help='Only used with --n_episodes=auto: the largest n_episodes the search is allowed to try '
             '(default: %(default)s, MaskPlace\'s own value - going higher deviates from what the '
             'paper\'s hyperparameters were tuned against, so this script keeps that as the default '
             "ceiling rather than searching purely by hardware capability). Raise this explicitly if "
             "you deliberately want a bigger buffer than MaskPlace's own and have the GPU memory for it.",
    )
    args = parser.parse_args(argv[1:])
    macro_budget = None if args.macro_budget.lower() == "all" else int(args.macro_budget)
    return (
        args.benchmark_dir, args.n_iterations, macro_budget, args.n_episodes, args.log_every,
        args.eval_every, args.max_episodes,
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
        macro_net_idx, macro_net_offset, macro_net_valid, lookahead=2,
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


def _probe_buffer_fits(benchmark_dir: str, macro_budget: str, n_episodes: int, eval_every: int) -> None:
    """Subprocess entry point (see NEpisodesDetector): runs this exact pipeline (_train_and_eval_loop,
    the same function the real run uses) for just over one eval_every window, reporting the outcome
    via exit code. Running only a single iteration would systematically overestimate what n_episodes
    actually fits: a GPU allocator running this close to its ceiling can fit fine for a while and then
    fail several iterations later purely from ordinary allocator fragmentation drift, a real effect a
    one-shot probe can't see at all - this needs to actually run long enough to cross at least one eval
    (a separate compiled executable, with its own memory footprint on top of training's) plus a few
    more iterations after it, not just prove the very first iteration doesn't immediately fail."""
    budget = None if macro_budget == "None" else int(macro_budget)
    benchmark = _load_benchmark(pathlib.Path(benchmark_dir), budget)
    state_fn = _build_state_fn(benchmark)
    extra_illegal_fn = make_wiremask_quality_illegal(margin=WIREMASK_MARGIN)
    policy = _build_policy(benchmark)
    obs0 = state_fn(reset(benchmark.params), benchmark.params, benchmark.sizes_array)
    variables = policy.init(random.PRNGKey(0), obs0)
    ppo_config = maskplace_ppo_config()
    optimizer = maskplace_optimizer(value_coef=ppo_config.value_coef)
    try:
        # checkpoint_path=None/log_path=None: exercise the exact real loop without touching disk.
        _train_and_eval_loop(
            random.PRNGKey(0), variables, policy, benchmark, optimizer, ppo_config, state_fn,
            extra_illegal_fn, checkpoint_path=None, n_iterations=eval_every + 3, n_episodes=n_episodes,
            log_every=eval_every + 3, eval_every=eval_every, log_path=None,
        )
    except Exception as e:
        if is_oom(e):  # doesn't fit - report it and let the parent see the exit code
            print(_OOM_MARKER)
            sys.exit(1)
        raise  # anything else is a real bug - propagate with its traceback


class NEpisodesDetector:
    """Auto-detects the largest n_episodes (buffer size) this hardware can run for the
    MaskPlace pipeline, by probing candidates for real in disposable subprocesses - same
    reasoning as ops.n_envs.NEnvsDetector: JAX's GPU memory accounting is unreliable after
    the first real allocation, and a crashing/hung candidate must not take down the caller.
    Entirely optional: only used if the CLI is given "auto" for n_episodes."""

    def __init__(
        self,
        benchmark_dir: pathlib.Path,
        macro_budget: int | None = None,
        # 32 is a generic "how far is worth probing" ceiling, not tied to MaskPlace's own
        # n_episodes default - this class just answers "how much fits on this hardware";
        # whether staying at MaskPlace's own buffer size matters is the caller's call (see
        # --max_episodes, which run_maskplace.py's own CLI defaults to MASKPLACE_N_EPISODES).
        max_candidate: int = 32,
        eval_every: int = 10,  # must match the real run's --eval_every for the probe to be representative
        # Generous: each probe subprocess compiles this ResNet-backed pipeline's shapes from cold (see
        # placax._device's JAX_COMPILATION_CACHE_DIR), so most of this budget goes to compile time, not
        # steady-state iteration time - too tight a timeout misreports a slow-to-compile n as infeasible.
        timeout_s: float = 900.0,
        verbose: bool = True,
    ):
        self.benchmark_dir = benchmark_dir
        self.macro_budget = macro_budget
        self.max_candidate = max_candidate
        self.eval_every = eval_every
        self.timeout_s = timeout_s
        self.verbose = verbose

    def detect(self) -> int:
        """Binary-searches the largest n_episodes (up to max_candidate) that actually fits."""
        # Deliberately inherit the parent's default JAX allocator settings (unlike
        # ops.n_envs's probe, which disables preallocation): this ResNet-heavy
        # pipeline's cuDNN/cuBLAS autotuning can spuriously OOM under a growing
        # allocator even when the real run - which uses the default preallocated
        # arena - fits fine. The probe must match what the real run actually does.
        return find_max_via_subprocess(
            _MODULE, ["--probe", str(self.benchmark_dir), str(self.macro_budget), str(self.eval_every)],
            max_candidate=self.max_candidate, timeout_s=self.timeout_s, oom_marker=_OOM_MARKER,
            verbose=self.verbose,
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
    benchmark_dir, n_iterations, macro_budget, n_episodes_arg, log_every, eval_every, max_episodes = _parse_args(
        sys.argv
    )
    if not benchmark_dir.exists():
        Log.error(f"'{benchmark_dir}' not found - run scripts/download_benchmarks.py first.")
        sys.exit(1)

    # 2. Resolve the buffer size FIRST, before this process touches JAX/the GPU at all: either
    #    MaskPlace's fixed 10, an explicit override, or - only if the CLI asked for "auto" - the
    #    largest that actually fits on this hardware (see NEpisodesDetector). This ordering
    #    matters, not just style - JAX's default GPU allocator preallocates a large fraction
    #    (75% by default, see placax._device) of the device as one arena on its FIRST allocation
    #    and never gives it back. If this process had already loaded the benchmark/built the
    #    policy (both real GPU ops) before probing, its own ~75% would still be reserved and
    #    alive while each disposable probe subprocess ALSO tries to preallocate its own ~75% of
    #    the same physical device - the two compete for one GPU's worth of memory, making every
    #    probe look far more memory-constrained than the real (single-process) training run ever
    #    will be, and can throttle n_episodes down for no reason related to actual training cost.
    #    Resolving n_episodes here, before this process's own first GPU allocation, means each
    #    probe subprocess runs against an otherwise-idle GPU, same as the real run.
    if n_episodes_arg.lower() == "auto":
        Log.info("auto-detecting n_episodes (probing candidates in disposable subprocesses) ...")
        n_episodes = NEpisodesDetector(
            benchmark_dir, macro_budget, max_candidate=max_episodes, eval_every=eval_every,
        ).detect()
        if n_episodes < 1:
            Log.error("not even n_episodes=1 fits on this hardware - try a smaller macro_budget.")
            sys.exit(1)
        Log.info(f"  -> n_episodes={n_episodes} (auto-detected, MaskPlace's own default is {MASKPLACE_N_EPISODES})")
    else:
        n_episodes = int(n_episodes_arg)

    # 3. Load the netlist with MaskPlace's own ordering/reward choices. This is this process's
    #    own first real GPU-touching JAX call - see step 2's comment for why it must come after
    #    n_episodes is resolved, not before.
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

    # 6. Set up the checkpoint location; if one already exists, training
    #    below will resume from it instead of starting over.
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

    print()
    print(f"checkpoint saved to {checkpoint_path} - re-run this script to continue training.")
    print(f"full history saved to {log_path}")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--probe":
        # argv: --probe benchmark_dir macro_budget eval_every n_episodes (n_episodes is appended
        # last by find_max_via_subprocess's try_fn, after NEpisodesDetector's own fixed_args).
        _probe_buffer_fits(sys.argv[2], sys.argv[3], int(sys.argv[5]), int(sys.argv[4]))
    else:
        main()
