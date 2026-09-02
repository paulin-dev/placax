"""Runs the MaskPlace-equivalent pipeline end to end, using only placax's existing pluggable pieces."""
import argparse
import pathlib
import sys

from placax import _device  # noqa: F401  must precede jax imports
from placax.core import reset
from placax.log import Log
from placax.netlist.order import connectivity_order_for
from placax.netlist.padding import build_macro_net_index
from placax_agents.benchmark import Benchmark
from placax_agents.ops.checkpoint import load_checkpoint, save_checkpoint
from placax_agents.ops.resumable_train import _append_log_entry, _evaluate, _save_placement_image
from placax_agents.policy.action import make_wiremask_quality_illegal
from placax_agents.policy.architectures.resnet_cnn import (
    ResNetCoarseFineActorCritic,
    build_pretrained_resnet_backbone,
    build_untrained_resnet_backbone,
)
from placax_agents.policy.observation import make_wiremask_observation
from placax_agents.policy.scale import to_grid_units
from placax_agents.training.algorithm.config import PPOConfig
from placax_agents.training.algorithm.loss import huber_value_loss
from placax_agents.training.algorithm.split_optimizer import make_grouped_optimizer
from placax_agents.training.loops.buffered_train import buffered_train_step
from placax_agents.training.loops.common import checkpoint_every_n, open_train_state
from placax_agents.training.reward import make_scaled_hpwl_reward

import jax.numpy as jnp
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


def maskplace_ppo_config(entropy_coef: float = 0.0) -> PPOConfig:
    """PPOConfig matching MaskPlace's own PPO2.py defaults: raw, unnormalized advantages/returns
    (PPO2.py's `advantage = (target_v - critic_net_output).detach()` and
    `value_loss = F.smooth_l1_loss(critic_output, target_v)` use neither) and, by default,
    entropy_coef=0.0 (no entropy bonus, also matching PPO2.py).

    Note: entropy_coef=0.0 here has a known consequence in this JAX/float32 implementation - with nothing
    bounding actor-logit magnitude (single fixed deterministic netlist -> consistent gradient direction
    every step -> Adam's updates accumulate roughly linearly), logits saturate completely (exact 0/1
    probabilities, exact zero gradient) by iteration ~4 in float32 (confirmed by direct checkpoint
    inspection), and again by iteration ~6 even with jax_enable_x64 (see placax/_device.py) - x64 only
    buys more headroom before hitting the same wall, not immunity from it. This literal-MaskPlace config
    is kept for faithful comparison; PPOConfig's own general default (entropy_coef=0.01, normalization ON)
    avoids the saturation and is what other configs in this codebase should keep using. Passing a nonzero
    entropy_coef here (normalization still OFF, e.g. via --entropy_coef) tests a narrower question: an
    entropy bonus only ever counteracts logit growth *before* saturation (its own gradient vanishes
    together with every other softmax-based gradient once probabilities actually hit exact 0/1) - it's
    unverified whether 0.01 is strong enough relative to unnormalized advantage/return scale to prevent
    saturation ever being reached at all.
    """
    return PPOConfig(
        gamma=0.95, lam=1.0, clip_eps=0.2, entropy_coef=entropy_coef, value_loss_fn=huber_value_loss,
        normalize_advantages=False, normalize_returns=False,
    )


def maskplace_optimizer(
    learning_rate: float = MASKPLACE_LEARNING_RATE,
    max_grad_norm: float = MASKPLACE_MAX_GRAD_NORM,
    critic_param_prefix: str = "critic_",
    value_coef: float = PPOConfig().value_coef,
) -> optax.GradientTransformation:
    """Separately-clipped Adam for actor and critic, matching MaskPlace's two-independent-backward-pass setup."""
    actor_chain = optax.chain(optax.clip_by_global_norm(max_grad_norm), optax.adam(learning_rate))
    critic_chain = optax.chain(optax.clip_by_global_norm(max_grad_norm / value_coef), optax.adam(learning_rate))
    return make_grouped_optimizer(critic_chain, actor_chain, critic_param_prefix)


def _parse_args(argv: list[str]) -> tuple[pathlib.Path, int, int | None, str, int, int, bool, bool, pathlib.Path | None, pathlib.Path | None, int, int, float]:
    """Parses named --flag=value CLI args into the run configuration tuple."""
    parser = argparse.ArgumentParser(description="Run the MaskPlace-equivalent pipeline end to end.")
    parser.add_argument(
        "--benchmark_dir", type=pathlib.Path, default=pathlib.Path("benchmarks/adaptec1"),
        help="Path to a downloaded benchmark directory (default: benchmarks/adaptec1).",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="RNG seed for policy init and rollout sampling (default: 42, MaskPlace's own --seed "
             "default). MaskPlace itself reports mean+-std across several seeds per benchmark, not a "
             "single run - vary this to reproduce that spread, or to check whether a given run's "
             "outcome is representative or just that seed's luck.",
    )
    parser.add_argument(
        "--n_iterations", type=int, default=100,
        help="Target TOTAL buffered-PPO update cycle to train to (default: 100), not an additional "
             "count - e.g. --n_iterations=300 resumed from iteration 100 trains 200 more to reach "
             "300 total; resuming at or past --n_iterations runs zero further iterations.",
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
    parser.add_argument(
        "--placement_images", action="store_true",
        help="Also write a placement snapshot PNG on every --eval_every iteration (default "
             "location: <output_dir>/placements/<iteration>.png) - reuses that iteration's "
             "already-scheduled eval rollout, so this adds no extra rollout, just one image write "
             "per eval.",
    )
    parser.add_argument(
        "--placement_images_dir", type=pathlib.Path, default=None,
        help="Where to write placement snapshots (implies --placement_images; default: "
             "<output_dir>/placements).",
    )
    parser.add_argument(
        "--init_from", type=pathlib.Path, default=None,
        help="Warm-start the policy's initial weights from a bare-variables checkpoint - most "
             "naturally <output_dir>/best_checkpoint.bin from a previous run (see "
             "_save_best_checkpoint's own docstring) - instead of the network's own random init. "
             "Takes priority over any existing checkpoint.bin: training starts fresh from these "
             "weights at iteration 0 with a freshly-initialized optimizer/RNG key, even if "
             "checkpoint.bin already holds a previous run's state - that state is overwritten "
             "going forward, not loaded from.",
    )
    parser.add_argument(
        "--patience", type=int, default=0,
        help="Stop early once real_hpwl (see --eval_every) hasn't beaten its best value for this "
             "many consecutive evals (default: 0, disabled - always run the full --n_iterations). "
             "MaskPlace's own stopping rule is a fixed reward threshold tuned to its own reward "
             "scaling/macro budget, which doesn't generalize across benchmarks/configs, so this "
             "uses best-real_hpwl patience instead.",
    )
    parser.add_argument(
        "--entropy_coef", type=float, default=0.0,
        help="Entropy bonus coefficient, passed straight to maskplace_ppo_config() (default: 0.0, "
             "MaskPlace's own value - see that function's docstring for the logit-saturation issue "
             "this causes and what a nonzero value here does and doesn't fix). Advantage/return "
             "normalization stay off regardless of this flag, matching MaskPlace's own PPO2.py.",
    )
    args = parser.parse_args(argv[1:])
    macro_budget = None if args.macro_budget.lower() == "all" else int(args.macro_budget)
    return (
        args.benchmark_dir, args.n_iterations, macro_budget, args.n_episodes, args.log_every,
        args.eval_every, args.no_checkpoint, args.placement_images, args.placement_images_dir,
        args.init_from, args.patience, args.seed, args.entropy_coef,
    )


def _maskplace_reward_fn(padded_pin_idx, padded_pin_offset, valid_mask, sizes_array, cell_size):
    """Converts real-unit HPWL delta to MaskPlace's own grid-unit reward magnitude (divided by 200)."""
    reward_scale = 1.0 / (cell_size * MASKPLACE_REWARD_DIVISOR)
    return make_scaled_hpwl_reward(
        padded_pin_idx, padded_pin_offset, valid_mask, sizes_array, cell_size,
        dense=True, reward_scale=reward_scale,
    )


def _maskplace_connectivity_weights(benchmark_name: str) -> tuple[float, float]:
    """(candidate_weight, degree_weight) for connectivity_order, per MaskPlace's own
    get_node_id_to_name_topology: it scales `candidates*W1 + node_net_num*W2 + area` differently for
    "ariane" (W1=30000, W2=1000) and "bigblue3" (W1=1, W2=100000) than everything else (W1=1, W2=1000).
    Matched by substring so directory names like "ariane133" still pick up the right override."""
    name = benchmark_name.lower()
    if "ariane" in name:
        return 30000.0, 1000.0
    if "bigblue3" in name:
        return 1.0, 100000.0
    return 1.0, 1000.0


def _load_benchmark(benchmark_dir: pathlib.Path, macro_budget: int | None) -> Benchmark:
    """Connectivity order, macro budget, dense reward - loaded once, shared everywhere below."""
    candidate_weight, degree_weight = _maskplace_connectivity_weights(benchmark_dir.name)
    return Benchmark.load(
        benchmark_dir,
        grid=224,  # MaskPlace's own grid resolution
        order_fn=connectivity_order_for(candidate_weight, degree_weight),
        macro_budget=macro_budget,
        make_reward_fn=_maskplace_reward_fn,
    )


def _build_state_fn(benchmark: Benchmark):
    """Builds the wiremask + 2-macro-lookahead observation function, MaskPlace's own channel set."""
    # 1. Precompute, once, which nets touch each macro, so it's not redone on every step.
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
        # Fall back to an untrained backbone with matching shapes, so training still works offline.
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


def _read_best_real_hpwl(variables_template, best_checkpoint_path: pathlib.Path | None) -> float:
    """The real_hpwl bundled alongside best_checkpoint_path's weights, or +inf if it doesn't exist yet."""
    if best_checkpoint_path is None or not best_checkpoint_path.exists():
        return float("inf")
    template = {"variables": variables_template, "real_hpwl": jnp.array(0.0)}
    return float(load_checkpoint(template, best_checkpoint_path)["real_hpwl"])


def _save_best_checkpoint(best_checkpoint_path: pathlib.Path, variables, real_hpwl: float) -> None:
    """Saves bare policy weights + the real_hpwl that earned them, deliberately without optimizer/RNG state."""
    save_checkpoint({"variables": variables, "real_hpwl": jnp.array(real_hpwl)}, best_checkpoint_path)


def _train_and_eval_loop(
    key, variables, policy, benchmark: Benchmark, optimizer, ppo_config, state_fn,
    extra_illegal_fn, checkpoint_path: pathlib.Path | None, n_iterations: int, n_episodes: int,
    log_every: int, eval_every: int, log_path: pathlib.Path | None,
    placement_images_dir: pathlib.Path | None = None, best_checkpoint_path: pathlib.Path | None = None,
    resume: bool = True, patience: int = 0,
):
    """Trains up to iteration n_iterations (the TOTAL target, not an additional count), resuming from
    checkpoint_path unless resume=False - resuming past n_iterations already runs zero further iterations,
    and resuming short of it runs only the remainder needed to reach it. Periodic eval/logging/checkpointing.
    Stops early if patience>0 and real_hpwl hasn't beaten its best in that many consecutive evals."""
    # Resume from checkpoint_path if it exists and resume=True (read once here, not per iteration).
    variables, opt_state, running_stats, key, start_iteration = open_train_state(
        variables, key, optimizer, checkpoint_path if resume else None
    )
    grid_sizes = to_grid_units(benchmark.sizes_array, benchmark.cell_size)
    total_iterations = n_iterations
    remaining_iterations = max(0, n_iterations - start_iteration)
    if start_iteration > 0:
        if remaining_iterations > 0:
            Log.info(f"  resumed at iteration {start_iteration}; running {remaining_iterations} more to reach {total_iterations}")
        else:
            Log.info(f"  resumed at iteration {start_iteration}, already >= --n_iterations={total_iterations}; nothing to do")
    best_real_hpwl = _read_best_real_hpwl(variables, best_checkpoint_path)
    if best_real_hpwl < float("inf"):
        Log.info(f"  best real_hpwl so far: {best_real_hpwl:.1f} -> {best_checkpoint_path}")
    # Resets each run: a resumed best_checkpoint doesn't carry over how many evals-without-
    # improvement preceded it, so patience is counted only against evals in this invocation.
    evals_without_improvement = 0

    log = []
    for i in range(remaining_iterations):
        current_iteration = start_iteration + i + 1

        # 1. One buffered-PPO update: collect n_episodes fresh episodes, train on them.
        key, buffer_key = random.split(key)
        variables, opt_state, running_stats, loss = buffered_train_step(
            buffer_key, variables, opt_state, running_stats, optimizer, policy.apply,
            benchmark.params, benchmark.reward_fn, benchmark.sizes_array, benchmark.cell_size,
            n_episodes, ppo_epochs=10, batch_size=64, state_fn=state_fn, ppo_config=ppo_config,
            extra_illegal_fn=extra_illegal_fn,
        )

        # 2. Real-HPWL eval (only every eval_every iterations) + log entry (console every log_every).
        real_hpwl = None
        if current_iteration % eval_every == 0:
            real_hpwl, eval_positions = _evaluate(
                variables, policy.apply, benchmark.params, benchmark.sizes_array, benchmark.cell_size,
                benchmark.padded_pin_idx, benchmark.padded_pin_offset, benchmark.valid_mask,
                state_fn, extra_illegal_fn,
            )
            if placement_images_dir is not None:
                _save_placement_image(
                    placement_images_dir, current_iteration, eval_positions, grid_sizes, benchmark.params
                )
            if real_hpwl < best_real_hpwl:
                best_real_hpwl = real_hpwl
                evals_without_improvement = 0
                if best_checkpoint_path is not None:
                    _save_best_checkpoint(best_checkpoint_path, variables, best_real_hpwl)
                    Log.info(f"  new best real_hpwl={best_real_hpwl:.1f} at iteration {current_iteration} -> {best_checkpoint_path}")
            else:
                evals_without_improvement += 1
        _append_log_entry(log, log_path, current_iteration, loss, real_hpwl)
        if current_iteration % log_every == 0:
            hpwl_str = f"{real_hpwl:.1f}" if real_hpwl is not None else "-"
            Log.info(f"iter {current_iteration:>6}/{total_iterations}  loss={loss:>10.4f}  real_hpwl={hpwl_str}")

        # 3. Checkpoint every iteration so a crash never loses more than one iteration of progress.
        checkpoint_every_n(checkpoint_path, 1, current_iteration, variables, opt_state, running_stats, key)

        # 4. Early stop once real_hpwl has stalled for `patience` consecutive evals.
        if patience > 0 and evals_without_improvement >= patience:
            Log.info(
                f"  stopping early at iteration {current_iteration}: real_hpwl hasn't beaten "
                f"{best_real_hpwl:.1f} in {evals_without_improvement} evals (--patience={patience})"
            )
            break

    return variables


def main() -> None:
    """CLI entry point: wires up the MaskPlace-equivalent pipeline and runs/resumes training."""
    Log.configure()

    # 1. Parse CLI args and make sure the requested benchmark actually exists.
    (
        benchmark_dir, n_iterations, macro_budget, n_episodes_arg, log_every, eval_every, no_checkpoint,
        want_placement_images, placement_images_dir_arg, init_from, patience, seed, entropy_coef,
    ) = _parse_args(sys.argv)
    if not benchmark_dir.exists():
        Log.error(f"'{benchmark_dir}' not found - run scripts/download_benchmarks.py first.")
        sys.exit(1)

    # 2. n_episodes must be an explicit value here - this process reserves its own GPU memory just
    #    by starting, so it can't probe disposable subprocess candidates without competing with them.
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
    extra_illegal_fn = make_wiremask_quality_illegal(margin=WIREMASK_MARGIN, cell_size=benchmark.cell_size)
    policy = _build_policy(benchmark)

    # 5. Initialize the policy's parameters using one dummy observation, so Flax can infer shapes.
    key = random.PRNGKey(seed)
    key, init_key = random.split(key)
    obs0 = state_fn(reset(benchmark.params), benchmark.params, benchmark.sizes_array)
    variables = policy.init(init_key, obs0)
    if init_from is not None:
        # Expects best_checkpoint.bin's own {"variables": ..., "real_hpwl": ...} schema.
        variables = load_checkpoint({"variables": variables, "real_hpwl": jnp.array(0.0)}, init_from)["variables"]
        Log.info(f"warm-starting initial weights from {init_from}")

    # 6. Set up the checkpoint location; training resumes from it unless --no_checkpoint or --init_from was given.
    resume = init_from is None
    if no_checkpoint:
        checkpoint_path = log_path = best_checkpoint_path = None
        placement_images_dir = placement_images_dir_arg  # only if explicitly given - no output_dir to default into
    else:
        output_dir = benchmark_dir / "output_maskplace"
        output_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_path = output_dir / "checkpoint.bin"
        log_path = output_dir / "training_log.jsonl"
        best_checkpoint_path = output_dir / "best_checkpoint.bin"
        resuming = resume and checkpoint_path.exists()
        Log.info(f"{'resuming from' if resuming else 'starting fresh, will save to'} {checkpoint_path}")
        if not resume and checkpoint_path.exists():
            Log.info(f"  note: {checkpoint_path} exists but --init_from was given, so it's overwritten fresh from iteration 0, not resumed")
        placement_images_dir = placement_images_dir_arg or (
            output_dir / "placements" if want_placement_images else None
        )
    if placement_images_dir is not None:
        # Gated by --eval_every alone (it reuses that iteration's already-scheduled eval rollout,
        # see _train_and_eval_loop below) - --log_every only controls the console line's cadence,
        # independently, and has no bearing on when a snapshot gets written.
        Log.info(f"writing a placement snapshot every {eval_every} iterations to {placement_images_dir}")

    # 7. Build the PPO config/optimizer matching MaskPlace's own hyperparameters.
    ppo_config = maskplace_ppo_config(entropy_coef=entropy_coef)
    optimizer = maskplace_optimizer(value_coef=ppo_config.value_coef)  # separately-clipped actor/critic Adam

    Log.info(
        f"training to iteration {n_iterations} "
        f"({n_episodes} episodes/buffer, 10 minibatch epochs, batch 64, "
        f"independent actor/critic optimizers, entropy_coef={entropy_coef}) ..."
    )

    # 8. Run the actual training/eval loop.
    variables = _train_and_eval_loop(
        key, variables, policy, benchmark, optimizer, ppo_config, state_fn, extra_illegal_fn,
        checkpoint_path, n_iterations, n_episodes=n_episodes,
        log_every=log_every, eval_every=eval_every, log_path=log_path,
        placement_images_dir=placement_images_dir, best_checkpoint_path=best_checkpoint_path,
        resume=resume, patience=patience,
    )

    if checkpoint_path is not None:
        print()
        print(f"checkpoint saved to {checkpoint_path} - re-run this script to continue training.")
        best_real_hpwl = _read_best_real_hpwl(variables, best_checkpoint_path)
        if best_real_hpwl < float("inf"):
            print(f"best real_hpwl={best_real_hpwl:.1f} saved to {best_checkpoint_path} "
                  f"(use --init_from={best_checkpoint_path} to warm-start a fresh run from it)")
        print(f"full history saved to {log_path}")


if __name__ == "__main__":
    main()
