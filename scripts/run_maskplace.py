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

Usage: python scripts/run_maskplace.py [benchmark_dir] [n_iterations] [macro_budget]
"""
import pathlib
import sys
from functools import partial

from placax import _device  # noqa: F401  must precede jax imports
from placax.core import reset
from placax.log import Log
from placax.netlist.order import connectivity_order
from placax.netlist.padding import build_macro_net_index
from placax_agents.benchmark import Benchmark
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

MASKPLACE_LEARNING_RATE = 2.5e-3
"""MaskPlace's own --lr default (PPO2.py)."""

MASKPLACE_MAX_GRAD_NORM = 0.5
"""MaskPlace's own PPO.max_grad_norm; clipped per-network, not jointly."""


def maskplace_ppo_config() -> PPOConfig:
    """PPOConfig matching MaskPlace's own PPO2.py defaults: gamma=0.95,
    no GAE smoothing (lam=1.0), no entropy bonus, Huber value loss."""
    return PPOConfig(gamma=0.95, lam=1.0, clip_eps=0.2, entropy_coef=0.0, value_loss_fn=huber_value_loss)


def maskplace_optimizer(
    learning_rate: float = MASKPLACE_LEARNING_RATE,
    max_grad_norm: float = MASKPLACE_MAX_GRAD_NORM,
    critic_param_prefix: str = "critic_",
) -> optax.GradientTransformation:
    """Adam(learning_rate) for actor and critic separately, each with its
    own clip-by-global-norm(max_grad_norm); matches MaskPlace's two
    separate optimizers without needing a second backward pass. Requires
    critic params named under critic_param_prefix and disjoint from the
    actor's - true for critic_style="step_embedding", not "canvas"
    (shared trunk)."""
    per_network = optax.chain(optax.clip_by_global_norm(max_grad_norm), optax.adam(learning_rate))
    return make_grouped_optimizer(per_network, per_network, critic_param_prefix)


def _parse_args(argv: list[str]) -> tuple[pathlib.Path, int, int | None]:
    """(benchmark_dir, n_iterations, macro_budget) from sys.argv."""
    benchmark_dir = pathlib.Path(argv[1] if len(argv) > 1 else "benchmarks/adaptec1")
    n_iterations = int(argv[2]) if len(argv) > 2 else 100
    macro_budget = int(argv[3]) if len(argv) > 3 else None
    return benchmark_dir, n_iterations, macro_budget


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
    """wiremask + 2-macro lookahead - MaskPlace's own observation channel set."""
    macro_net_idx, macro_net_offset, macro_net_valid = build_macro_net_index(
        benchmark.padded_pin_idx, benchmark.padded_pin_offset, benchmark.valid_mask,
        n_macros=benchmark.params.n_macros,
    )
    return make_wiremask_observation(
        benchmark.padded_pin_idx, benchmark.padded_pin_offset, benchmark.valid_mask,
        macro_net_idx, macro_net_offset, macro_net_valid, lookahead=2,
    )


def _resnet_backbone():
    """Real ImageNet weights if placax[resnet] is installed; otherwise the
    offline, same-shape stand-in, with a clear note about which was used."""
    try:
        import flaxmodels  # noqa: F401
    except ImportError:
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
    extra_illegal_fn, checkpoint_path: pathlib.Path, n_iterations: int, eval_every: int,
):
    """Runs n_iterations of buffered-PPO training in eval_every-sized chunks, logging real HPWL after each."""
    done, remaining = 0, n_iterations
    while remaining > 0:
        chunk = min(eval_every, remaining)
        variables, losses = train_buffered(
            key, variables, policy.apply, benchmark.params, benchmark.reward_fn,
            benchmark.sizes_array, benchmark.cell_size, n_iterations=chunk,
            optimizer=optimizer, state_fn=state_fn, ppo_config=ppo_config,
            extra_illegal_fn=extra_illegal_fn, checkpoint_path=checkpoint_path,
        )
        done += chunk
        remaining -= chunk

        _positions, real_hpwl = evaluate(
            variables, policy.apply, benchmark.params, benchmark.sizes_array, benchmark.cell_size,
            benchmark.padded_pin_idx, benchmark.padded_pin_offset, benchmark.valid_mask,
            state_fn, extra_illegal_fn,
        )
        Log.info(f"iter {done:>5}/{n_iterations}  loss={losses[-1]:>10.4f}  real_hpwl={float(real_hpwl):.1f}")
    return variables


def main() -> None:
    Log.configure()

    benchmark_dir, n_iterations, macro_budget = _parse_args(sys.argv)
    if not benchmark_dir.exists():
        Log.error(f"'{benchmark_dir}' not found - run scripts/download_benchmarks.py first.")
        sys.exit(1)

    Log.info(f"loading {benchmark_dir} (connectivity order, macro_budget={macro_budget}, dense reward) ...")
    benchmark = _load_benchmark(benchmark_dir, macro_budget)
    Log.info(f"  {len(benchmark.macro_sizes)} macros, {len(benchmark.nets)} nets, cell_size={benchmark.cell_size:.2f}")

    state_fn = _build_state_fn(benchmark)
    extra_illegal_fn = make_wiremask_quality_illegal(margin=WIREMASK_MARGIN)
    policy = _build_policy(benchmark)

    key = random.PRNGKey(0)
    key, init_key = random.split(key)
    obs0 = state_fn(reset(benchmark.params), benchmark.params, benchmark.sizes_array)
    variables = policy.init(init_key, obs0)

    output_dir = benchmark_dir / "output_maskplace"
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_dir / "checkpoint.bin"
    resuming = checkpoint_path.exists()
    Log.info(f"{'resuming from' if resuming else 'starting fresh, will save to'} {checkpoint_path}")

    ppo_config = maskplace_ppo_config()
    optimizer = maskplace_optimizer()  # separately-clipped, separately-stepped actor/critic Adam
    Log.info(
        f"running {n_iterations} more buffered-PPO iterations "
        f"(MaskPlace's own procedure: 10 episodes/buffer, 10 minibatch epochs, batch 64, "
        f"independent actor/critic optimizers) ..."
    )

    variables = _train_and_eval_loop(
        key, variables, policy, benchmark, optimizer, ppo_config, state_fn, extra_illegal_fn,
        checkpoint_path, n_iterations, eval_every=10,
    )

    print()
    print(f"checkpoint saved to {checkpoint_path} - re-run this script to continue training.")


if __name__ == "__main__":
    main()
