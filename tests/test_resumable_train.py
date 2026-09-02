import pathlib

from placax.core import reset  # noqa: F401  must precede jax imports
from placax.extras.rewards import make_hpwl_reward  # noqa: F401
from placax.types import EnvParams  # noqa: F401
from placax_agents.ops.resumable_train import resumable_train  # noqa: F401
from placax_agents.policy.architectures.cnn import CNNActorCritic  # noqa: F401
from placax_agents.policy.observation import observation  # noqa: F401

import jax
import jax.numpy as jnp
from jax import random


def _toy_setup():
    params = EnvParams(grid=8, n_macros=4)
    # float32, matching placax/netlist/padding.py's real (deliberately float32) sizes_array/
    # padded_pin_offset - not left to default to float64 under JAX_ENABLE_X64 like a real benchmark
    # never would, since that silently hides any float32/float64 interaction real training hits.
    sizes_array = jnp.array([[2.0, 2.0], [1.0, 1.0], [2.0, 1.0], [1.0, 2.0]], dtype=jnp.float32)
    padded_pin_idx = jnp.array([[0, 1]])
    padded_pin_offset = jnp.zeros((1, 2, 2), dtype=jnp.float32)
    valid_mask = jnp.array([[True, True]])
    reward_fn = make_hpwl_reward(padded_pin_idx, padded_pin_offset, valid_mask)
    return params, sizes_array, reward_fn, padded_pin_idx, padded_pin_offset, valid_mask


def test_interrupted_run_exactly_matches_continuous_run(tmp_path: pathlib.Path) -> None:
    # The core correctness property: splitting a run across two resumed
    # calls must give bit-for-bit identical results to one continuous
    # call of the combined iteration count - both the final weights and
    # every per-iteration loss.
    params, sizes_array, reward_fn, ppi, ppo, vm = _toy_setup()
    policy = CNNActorCritic()
    obs0 = observation(reset(params), params, sizes_array)
    variables_init = policy.init(random.PRNGKey(0), obs0)

    path_interrupted = tmp_path / "interrupted.bin"
    _, log1 = resumable_train(
        path_interrupted, variables_init, random.PRNGKey(1), policy.apply, params, reward_fn,
        sizes_array, 1.0, n_iterations=3, padded_pin_idx=ppi, padded_pin_offset=ppo,
        valid_mask=vm, checkpoint_every=1,
    )
    v_interrupted, log2 = resumable_train(
        path_interrupted, variables_init, random.PRNGKey(1), policy.apply, params, reward_fn,
        sizes_array, 1.0, n_iterations=2, padded_pin_idx=ppi, padded_pin_offset=ppo,
        valid_mask=vm, checkpoint_every=1,
    )
    interrupted_losses = [e["loss"] for e in log1] + [e["loss"] for e in log2]

    path_continuous = tmp_path / "continuous.bin"
    v_continuous, log_continuous = resumable_train(
        path_continuous, variables_init, random.PRNGKey(1), policy.apply, params, reward_fn,
        sizes_array, 1.0, n_iterations=5, padded_pin_idx=ppi, padded_pin_offset=ppo,
        valid_mask=vm, checkpoint_every=1,
    )
    continuous_losses = [e["loss"] for e in log_continuous]

    assert interrupted_losses == continuous_losses
    leaves_interrupted = jax.tree_util.tree_leaves(v_interrupted)
    leaves_continuous = jax.tree_util.tree_leaves(v_continuous)
    assert all((a == b).all() for a, b in zip(leaves_interrupted, leaves_continuous))


def test_eval_every_gates_real_hpwl_computation(tmp_path: pathlib.Path) -> None:
    params, sizes_array, reward_fn, ppi, ppo, vm = _toy_setup()
    policy = CNNActorCritic()
    obs0 = observation(reset(params), params, sizes_array)
    variables_init = policy.init(random.PRNGKey(0), obs0)

    path = tmp_path / "eval_gate.bin"
    _, log = resumable_train(
        path, variables_init, random.PRNGKey(1), policy.apply, params, reward_fn,
        sizes_array, 1.0, n_iterations=4, padded_pin_idx=ppi, padded_pin_offset=ppo,
        valid_mask=vm, checkpoint_every=1, eval_every=2,
    )
    real_hpwl_present = [e["real_hpwl"] is not None for e in log]
    assert real_hpwl_present == [False, True, False, True]


def test_log_path_persists_full_history(tmp_path: pathlib.Path) -> None:
    params, sizes_array, reward_fn, ppi, ppo, vm = _toy_setup()
    policy = CNNActorCritic()
    obs0 = observation(reset(params), params, sizes_array)
    variables_init = policy.init(random.PRNGKey(0), obs0)

    path = tmp_path / "log_test.bin"
    log_path = tmp_path / "log.jsonl"
    resumable_train(
        path, variables_init, random.PRNGKey(1), policy.apply, params, reward_fn,
        sizes_array, 1.0, n_iterations=3, padded_pin_idx=ppi, padded_pin_offset=ppo,
        valid_mask=vm, checkpoint_every=1, log_path=log_path,
    )
    lines = log_path.read_text().strip().split("\n")
    assert len(lines) == 3

    import json

    parsed = [json.loads(line) for line in lines]
    assert [p["iteration"] for p in parsed] == [1, 2, 3]


def test_interrupted_parallel_run_exactly_matches_continuous_run(tmp_path: pathlib.Path) -> None:
    # The same critical property, now for the parallel path (n_envs>1):
    # splitting a run across two resumed calls must still give
    # bit-for-bit identical results to one continuous call.
    params, sizes_array, reward_fn, ppi, ppo, vm = _toy_setup()
    policy = CNNActorCritic()
    obs0 = observation(reset(params), params, sizes_array)
    variables_init = policy.init(random.PRNGKey(0), obs0)

    path_interrupted = tmp_path / "interrupted_par.bin"
    _, log1 = resumable_train(
        path_interrupted, variables_init, random.PRNGKey(1), policy.apply, params, reward_fn,
        sizes_array, 1.0, n_iterations=3, padded_pin_idx=ppi, padded_pin_offset=ppo,
        valid_mask=vm, checkpoint_every=1, eval_every=100, n_envs=4, mode="parallel",
    )
    v_interrupted, log2 = resumable_train(
        path_interrupted, variables_init, random.PRNGKey(1), policy.apply, params, reward_fn,
        sizes_array, 1.0, n_iterations=2, padded_pin_idx=ppi, padded_pin_offset=ppo,
        valid_mask=vm, checkpoint_every=1, eval_every=100, n_envs=4, mode="parallel",
    )
    interrupted_losses = [e["loss"] for e in log1] + [e["loss"] for e in log2]

    path_continuous = tmp_path / "continuous_par.bin"
    v_continuous, log_continuous = resumable_train(
        path_continuous, variables_init, random.PRNGKey(1), policy.apply, params, reward_fn,
        sizes_array, 1.0, n_iterations=5, padded_pin_idx=ppi, padded_pin_offset=ppo,
        valid_mask=vm, checkpoint_every=1, eval_every=100, n_envs=4, mode="parallel",
    )
    continuous_losses = [e["loss"] for e in log_continuous]

    assert interrupted_losses == continuous_losses
    leaves_interrupted = jax.tree_util.tree_leaves(v_interrupted)
    leaves_continuous = jax.tree_util.tree_leaves(v_continuous)
    assert all((a == b).all() for a, b in zip(leaves_interrupted, leaves_continuous))


def test_extra_illegal_fn_is_threaded_through(tmp_path: pathlib.Path) -> None:
    # Regression: resumable_train() used to silently drop extra_illegal_fn on the floor (never passed
    # to its own gradient-step or eval calls, even though train_sequential/train_parallel/evaluate all
    # already accept and honor it) - forcing MaskPlace's own action-quality masking (or any custom
    # ExtraIllegalFn) to bypass this public API entirely. A call-counting closure proves it's actually
    # invoked somewhere in the traced computation, not just accepted and ignored.
    params, sizes_array, reward_fn, ppi, ppo, vm = _toy_setup()
    policy = CNNActorCritic()
    obs0 = observation(reset(params), params, sizes_array)
    variables_init = policy.init(random.PRNGKey(0), obs0)

    calls = []

    def counting_illegal_fn(obs):
        calls.append(1)
        return jnp.zeros((params.grid, params.grid), dtype=bool)

    path = tmp_path / "extra_illegal.bin"
    resumable_train(
        path, variables_init, random.PRNGKey(1), policy.apply, params, reward_fn,
        sizes_array, 1.0, n_iterations=1, padded_pin_idx=ppi, padded_pin_offset=ppo,
        valid_mask=vm, checkpoint_every=1, eval_every=1, extra_illegal_fn=counting_illegal_fn,
    )
    assert len(calls) > 0


def test_snapshots_are_never_overwritten(tmp_path: pathlib.Path) -> None:
    params, sizes_array, reward_fn, ppi, ppo, vm = _toy_setup()
    policy = CNNActorCritic()
    obs0 = observation(reset(params), params, sizes_array)
    variables_init = policy.init(random.PRNGKey(0), obs0)

    path = tmp_path / "snap_ckpt.bin"
    snapshot_dir = tmp_path / "snapshots"
    resumable_train(
        path, variables_init, random.PRNGKey(1), policy.apply, params, reward_fn,
        sizes_array, 1.0, n_iterations=6, padded_pin_idx=ppi, padded_pin_offset=ppo,
        valid_mask=vm, checkpoint_every=1, snapshot_dir=snapshot_dir, snapshot_every=2,
    )
    snapshot_names = sorted(p.name for p in snapshot_dir.glob("*.bin"))
    assert snapshot_names == ["checkpoint_iter_2.bin", "checkpoint_iter_4.bin", "checkpoint_iter_6.bin"]
    assert path.exists()  # the main, overwritten "resume from here" file also still exists
