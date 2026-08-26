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
from functools import partial

from placax import _device  # noqa: F401  must precede jax imports
from placax.core import reset
from placax.log import Log
from placax.netlist.order import connectivity_order
from placax.netlist.padding import build_macro_net_index
from placax_agents.benchmark import Benchmark
from placax_agents.ops.autotune import find_max_via_subprocess, is_oom
from placax_agents.ops.evaluate import evaluate
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
from placax_agents.training.loops.buffered_train import train_buffered
from placax_agents.training.reward import make_scaled_hpwl_reward

import optax
from jax import random

WIREMASK_MARGIN = 1.0  # MaskPlace's own --soft_coefficient default

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
) -> optax.GradientTransformation:
    """Separately-clipped Adam for actor and critic, matching MaskPlace's two-optimizer setup.

    Requires critic params named under critic_param_prefix and disjoint from the actor's
    (true for critic_style="step_embedding", not the shared-trunk "canvas" style).
    """
    per_network = optax.chain(optax.clip_by_global_norm(max_grad_norm), optax.adam(learning_rate))
    return make_grouped_optimizer(per_network, per_network, critic_param_prefix)


def _parse_args(argv: list[str]) -> tuple[pathlib.Path, int, int | None, str]:
    """(benchmark_dir, n_iterations, macro_budget, n_episodes_arg) from named --flag=value CLI args;
    macro_budget is None if --macro_budget=all was given; n_episodes_arg is either an int-as-string
    or the literal "auto" (see NEpisodesDetector)."""
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
    args = parser.parse_args(argv[1:])
    macro_budget = None if args.macro_budget.lower() == "all" else int(args.macro_budget)
    return args.benchmark_dir, args.n_iterations, macro_budget, args.n_episodes


def _load_benchmark(benchmark_dir: pathlib.Path, macro_budget: int | None) -> Benchmark:
    """Connectivity order, macro budget, dense reward - loaded once, shared everywhere below."""
    return Benchmark.load(
        benchmark_dir,
        grid=224,  # MaskPlace's own grid resolution
        order_fn=connectivity_order,
        macro_budget=macro_budget,
        make_reward_fn=partial(make_scaled_hpwl_reward, dense=True),
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


def _probe_buffer_fits(benchmark_dir: str, macro_budget: str, n_episodes: int) -> None:
    """Subprocess entry point (see NEpisodesDetector): attempts one n_episodes-sized buffered-PPO
    step of this exact pipeline, reporting the outcome via exit code."""
    budget = None if macro_budget == "None" else int(macro_budget)
    benchmark = _load_benchmark(pathlib.Path(benchmark_dir), budget)
    state_fn = _build_state_fn(benchmark)
    extra_illegal_fn = make_wiremask_quality_illegal(margin=WIREMASK_MARGIN)
    policy = _build_policy(benchmark)
    obs0 = state_fn(reset(benchmark.params), benchmark.params, benchmark.sizes_array)
    variables = policy.init(random.PRNGKey(0), obs0)
    try:
        # ppo_epochs=1 already hits the same peak memory as a real run (rollout
        # collection and one gradient minibatch dominate; more epochs just repeat it).
        train_buffered(
            random.PRNGKey(0), variables, policy.apply, benchmark.params, benchmark.reward_fn,
            benchmark.sizes_array, benchmark.cell_size, n_iterations=1, n_episodes=n_episodes,
            ppo_epochs=1, optimizer=maskplace_optimizer(), state_fn=state_fn,
            ppo_config=maskplace_ppo_config(), extra_illegal_fn=extra_illegal_fn,
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
        max_candidate: int = MASKPLACE_N_EPISODES,  # no point searching past MaskPlace's own value
        timeout_s: float = 180.0,
        verbose: bool = True,
    ):
        self.benchmark_dir = benchmark_dir
        self.macro_budget = macro_budget
        self.max_candidate = max_candidate
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
            _MODULE, ["--probe", str(self.benchmark_dir), str(self.macro_budget)],
            max_candidate=self.max_candidate, timeout_s=self.timeout_s, oom_marker=_OOM_MARKER,
            verbose=self.verbose,
        )


def _train_and_eval_loop(
    key, variables, policy, benchmark: Benchmark, optimizer, ppo_config, state_fn,
    extra_illegal_fn, checkpoint_path: pathlib.Path, n_iterations: int, eval_every: int, n_episodes: int,
):
    """Runs n_iterations of buffered-PPO training in eval_every-sized chunks, logging real HPWL after each."""
    done, remaining = 0, n_iterations
    while remaining > 0:
        # 1. Train for one chunk at a time (not all n_iterations at once) so
        #    we can check in on real progress every eval_every iterations.
        chunk = min(eval_every, remaining)
        variables, losses = train_buffered(
            key, variables, policy.apply, benchmark.params, benchmark.reward_fn,
            benchmark.sizes_array, benchmark.cell_size, n_iterations=chunk, n_episodes=n_episodes,
            optimizer=optimizer, state_fn=state_fn, ppo_config=ppo_config,
            extra_illegal_fn=extra_illegal_fn, checkpoint_path=checkpoint_path,
        )
        done += chunk
        remaining -= chunk

        # 2. Evaluate the current policy deterministically to get the actual
        #    HPWL it achieves - training loss alone doesn't tell us that.
        _positions, real_hpwl = evaluate(
            variables, policy.apply, benchmark.params, benchmark.sizes_array, benchmark.cell_size,
            benchmark.padded_pin_idx, benchmark.padded_pin_offset, benchmark.valid_mask,
            state_fn, extra_illegal_fn,
        )
        # 3. Report progress so a long training run isn't a silent black box.
        Log.info(f"iter {done:>5}/{n_iterations}  loss={losses[-1]:>10.4f}  real_hpwl={float(real_hpwl):.1f}")
    return variables


def main() -> None:
    """CLI entry point: wires up the MaskPlace-equivalent pipeline and runs/resumes training."""
    Log.configure()

    # 1. Parse CLI args and make sure the requested benchmark actually exists.
    benchmark_dir, n_iterations, macro_budget, n_episodes_arg = _parse_args(sys.argv)
    if not benchmark_dir.exists():
        Log.error(f"'{benchmark_dir}' not found - run scripts/download_benchmarks.py first.")
        sys.exit(1)

    # 2. Load the netlist with MaskPlace's own ordering/reward choices.
    Log.info(f"loading {benchmark_dir} (connectivity order, macro_budget={macro_budget}, dense reward) ...")
    benchmark = _load_benchmark(benchmark_dir, macro_budget)
    Log.info(f"  {len(benchmark.macro_sizes)} macros, {len(benchmark.nets)} nets, cell_size={benchmark.cell_size:.2f}")

    # 3. Build the observation function, illegal-action mask, and policy network.
    state_fn = _build_state_fn(benchmark)
    extra_illegal_fn = make_wiremask_quality_illegal(margin=WIREMASK_MARGIN)
    policy = _build_policy(benchmark)

    # 4. Initialize the policy's parameters using one dummy observation, so
    #    Flax can infer every layer's shape from real input.
    key = random.PRNGKey(0)
    key, init_key = random.split(key)
    obs0 = state_fn(reset(benchmark.params), benchmark.params, benchmark.sizes_array)
    variables = policy.init(init_key, obs0)

    # 5. Set up the checkpoint location; if one already exists, training
    #    below will resume from it instead of starting over.
    output_dir = benchmark_dir / "output_maskplace"
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_dir / "checkpoint.bin"
    resuming = checkpoint_path.exists()
    Log.info(f"{'resuming from' if resuming else 'starting fresh, will save to'} {checkpoint_path}")

    # 6. Build the PPO config/optimizer matching MaskPlace's own hyperparameters.
    ppo_config = maskplace_ppo_config()
    optimizer = maskplace_optimizer()  # separately-clipped, separately-stepped actor/critic Adam

    # 7. Resolve the buffer size: either MaskPlace's fixed 10, an explicit
    #    override, or - only if the CLI asked for "auto" - the largest that
    #    actually fits on this hardware (see NEpisodesDetector).
    if n_episodes_arg.lower() == "auto":
        Log.info("auto-detecting n_episodes (probing candidates in disposable subprocesses) ...")
        n_episodes = NEpisodesDetector(benchmark_dir, macro_budget).detect()
        if n_episodes < 1:
            Log.error("not even n_episodes=1 fits on this hardware - try a smaller macro_budget.")
            sys.exit(1)
        Log.info(f"  -> n_episodes={n_episodes} (auto-detected, MaskPlace's own default is {MASKPLACE_N_EPISODES})")
    else:
        n_episodes = int(n_episodes_arg)
    Log.info(
        f"running {n_iterations} more buffered-PPO iterations "
        f"({n_episodes} episodes/buffer, 10 minibatch epochs, batch 64, "
        f"independent actor/critic optimizers) ..."
    )

    # 8. Run the actual training/eval loop.
    variables = _train_and_eval_loop(
        key, variables, policy, benchmark, optimizer, ppo_config, state_fn, extra_illegal_fn,
        checkpoint_path, n_iterations, eval_every=10, n_episodes=n_episodes,
    )

    print()
    print(f"checkpoint saved to {checkpoint_path} - re-run this script to continue training.")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--probe":
        _probe_buffer_fits(sys.argv[2], sys.argv[3], int(sys.argv[4]))
    else:
        main()
