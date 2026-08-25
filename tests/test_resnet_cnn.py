import functools

from placax.core import reset  # noqa: F401  must precede jax imports
from placax.netlist.padding import build_macro_net_index
from placax.types import EnvParams
from placax_agents.policy.architectures.resnet_cnn import (
    ResNetCoarseFineActorCritic,
    build_pretrained_resnet_backbone,
    build_untrained_resnet_backbone,
    extract_resnet_backbone_weights,
    load_resnet_backbone_weights,
)
from placax_agents.ops.checkpoint import save_checkpoint
from placax_agents.policy.observation import make_wiremask_observation, observation

import jax
import jax.numpy as jnp
import optax
import pytest
from jax import random

try:
    import flaxmodels  # noqa: F401

    _HAS_FLAXMODELS = True
except ImportError:
    _HAS_FLAXMODELS = False

pytestmark = pytest.mark.skipif(not _HAS_FLAXMODELS, reason="optional placax[resnet] dependency not installed")


def _toy_obs():
    params = EnvParams(grid=8, n_macros=4)
    sizes_array = jnp.array([[2.0, 2.0], [1.0, 1.0], [2.0, 1.0], [1.0, 2.0]])
    padded_pin_idx = jnp.array([[0, 1], [1, 2]])
    padded_pin_offset = jnp.zeros((2, 2, 2))
    valid_mask = jnp.array([[True, True], [True, True]])
    macro_net_idx, macro_net_offset, macro_net_valid = build_macro_net_index(
        padded_pin_idx, padded_pin_offset, valid_mask, n_macros=4
    )
    state_fn = make_wiremask_observation(
        padded_pin_idx, padded_pin_offset, valid_mask, macro_net_idx, macro_net_offset, macro_net_valid,
        base_state_fn=functools.partial(observation, lookahead=2), lookahead=2,
    )
    obs = state_fn(reset(params), params, sizes_array)
    return params, obs


def test_resnet_actor_critic_output_shapes() -> None:
    params, obs = _toy_obs()
    policy = ResNetCoarseFineActorCritic(
        resnet_backbone=build_untrained_resnet_backbone(), params=params, cell_size=1.0
    )
    variables = policy.init(random.PRNGKey(0), obs)
    action_logits, value = policy.apply(variables, obs)
    assert action_logits.shape == (params.grid_x, params.effective_grid_y)
    assert value.shape == ()


def test_resnet_actor_critic_step_embedding_critic_variant() -> None:
    # MaskPlace's own critic shape: keyed purely on step, no canvas input.
    params, obs = _toy_obs()
    policy = ResNetCoarseFineActorCritic(
        resnet_backbone=build_untrained_resnet_backbone(), params=params, cell_size=1.0,
        critic_style="step_embedding", max_episode_macros=16,
    )
    variables = policy.init(random.PRNGKey(0), obs)
    action_logits, value = policy.apply(variables, obs)
    assert action_logits.shape == (params.grid_x, params.effective_grid_y)
    assert value.shape == ()


def test_step_embedding_critic_params_are_disjoint_from_the_actor() -> None:
    # The property split_optimizer's independence relies on: no top-level
    # param name both starts with "critic_" and contributes to
    # action_logits' computation graph.
    params, obs = _toy_obs()
    policy = ResNetCoarseFineActorCritic(
        resnet_backbone=build_untrained_resnet_backbone(), params=params, cell_size=1.0,
        critic_style="step_embedding", max_episode_macros=16,
    )
    variables = policy.init(random.PRNGKey(0), obs)
    top_level_names = set(variables["params"])
    critic_names = {name for name in top_level_names if name.startswith("critic_")}
    assert critic_names  # the critic really is named this way
    assert critic_names <= {"critic_step_embed", "critic_hidden", "critic_value"}
    assert "resnet_backbone" not in critic_names  # the actor's backbone isn't accidentally swept in


def test_maskplace_optimizer_updates_a_real_policys_variables() -> None:
    from placax_agents.training.algorithm.config import maskplace_optimizer

    params, obs = _toy_obs()
    policy = ResNetCoarseFineActorCritic(
        resnet_backbone=build_untrained_resnet_backbone(), params=params, cell_size=1.0,
        critic_style="step_embedding", max_episode_macros=16,
    )
    variables = policy.init(random.PRNGKey(0), obs)

    def toy_loss(v):
        action_logits, value = policy.apply(v, obs)
        return action_logits.sum() + value

    grads = jax.grad(toy_loss)(variables)
    optimizer = maskplace_optimizer(learning_rate=0.01)
    opt_state = optimizer.init(variables)
    updates, _new_state = optimizer.update(grads, opt_state, variables)
    new_variables = optax.apply_updates(variables, updates)

    old_leaves = jax.tree_util.tree_leaves(variables)
    new_leaves = jax.tree_util.tree_leaves(new_variables)
    assert any(not jnp.allclose(o, n) for o, n in zip(old_leaves, new_leaves))


def test_build_untrained_resnet_backbone_uses_no_pretrained_weights() -> None:
    # Regression guard: this specific helper must stay offline - it's
    # what this file's own tests, and CI, rely on to avoid a network call.
    backbone = build_untrained_resnet_backbone()
    assert backbone.pretrained is None


def test_build_pretrained_resnet_backbone_does_not_download_at_construction_time() -> None:
    # Constructing the module is pure dataclass storage - the download
    # (if any) only happens inside setup(), which only runs on
    # .init()/.apply(). Must stay fast and offline on its own.
    backbone = build_pretrained_resnet_backbone(ckpt_dir="/tmp/unused-ckpt-dir")
    assert backbone.pretrained == "imagenet"
    assert backbone.ckpt_dir == "/tmp/unused-ckpt-dir"


def test_load_resnet_backbone_weights_overlays_only_the_backbone_subtree(tmp_path) -> None:
    params, obs = _toy_obs()
    policy = ResNetCoarseFineActorCritic(
        resnet_backbone=build_untrained_resnet_backbone(), params=params, cell_size=1.0
    )
    variables_a = policy.init(random.PRNGKey(0), obs)
    variables_b = policy.init(random.PRNGKey(1), obs)  # a different seed -> different weights everywhere

    ckpt_path = tmp_path / "backbone.flax"
    save_checkpoint(extract_resnet_backbone_weights(variables_a), ckpt_path)

    # Overlaying variables_a's backbone onto variables_b's full variables:
    # the backbone subtree should now match variables_a's, everything else
    # (fine branch, merge conv, critic) should still match variables_b's.
    merged = load_resnet_backbone_weights(variables_b, ckpt_path)

    def all_close(a, b):
        return all(jnp.allclose(x, y) for x, y in zip(jax.tree_util.tree_leaves(a), jax.tree_util.tree_leaves(b)))

    assert all_close(merged["params"]["resnet_backbone"], variables_a["params"]["resnet_backbone"])
    assert not all_close(merged["params"]["resnet_backbone"], variables_b["params"]["resnet_backbone"])
    assert all_close(merged["params"]["_FineBranch_0"], variables_b["params"]["_FineBranch_0"])


def test_resnet_actor_critic_reacts_to_the_wiremask_channel() -> None:
    params, obs = _toy_obs()
    policy = ResNetCoarseFineActorCritic(
        resnet_backbone=build_untrained_resnet_backbone(), params=params, cell_size=1.0
    )
    variables = policy.init(random.PRNGKey(0), obs)

    flat_obs = dict(obs)
    flat_obs["lookahead_wiremasks"] = jnp.zeros_like(obs["lookahead_wiremasks"])
    logits_flat, _ = policy.apply(variables, flat_obs)

    peaked_obs = dict(obs)
    peaked_obs["lookahead_wiremasks"] = obs["lookahead_wiremasks"].at[0, 4, 4].set(10.0)
    logits_peaked, _ = policy.apply(variables, peaked_obs)

    assert not jnp.allclose(logits_flat, logits_peaked)
