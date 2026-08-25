from placax_agents.policy.architectures.cnn import CNNActorCritic  # noqa: F401  must precede jax imports
from placax_agents.policy.architectures.wiremask_cnn import WiremaskCNNActorCritic  # noqa: F401

import jax
import jax.numpy as jnp
from jax import random


def test_cnn_actor_critic_output_shapes() -> None:
    obs = {"canvas": jnp.zeros((8, 8), dtype=bool)}
    policy = CNNActorCritic()
    variables = policy.init(random.PRNGKey(0), obs)
    action_logits, value = policy.apply(variables, obs)
    assert action_logits.shape == (8, 8)
    assert value.shape == ()  # scalar


def test_cnn_actor_critic_hyperparameters_are_configurable() -> None:
    # Regression test: an earlier version hardcoded features=16 and
    # kernel_size=(3,3) directly in the module body - no way to build a
    # wider or deeper network without editing the class itself.
    obs = {"canvas": jnp.zeros((8, 8), dtype=bool)}
    wide_policy = CNNActorCritic(features=32, num_conv_layers=3)
    wide_variables = wide_policy.init(random.PRNGKey(0), obs)
    action_logits, value = wide_policy.apply(wide_variables, obs)
    assert action_logits.shape == (8, 8)
    assert value.shape == ()

    narrow_policy = CNNActorCritic(features=4, num_conv_layers=1)
    narrow_variables = narrow_policy.init(random.PRNGKey(0), obs)
    narrow_param_count = sum(x.size for x in jax.tree_util.tree_leaves(narrow_variables))
    wide_param_count = sum(x.size for x in jax.tree_util.tree_leaves(wide_variables))
    assert narrow_param_count < wide_param_count  # confirms the config actually changed the network


def test_wiremask_cnn_actor_critic_output_shapes() -> None:
    obs = {"canvas": jnp.zeros((8, 8), dtype=bool), "wiremask": jnp.zeros((8, 8))}
    policy = WiremaskCNNActorCritic()
    variables = policy.init(random.PRNGKey(0), obs)
    action_logits, value = policy.apply(variables, obs)
    assert action_logits.shape == (8, 8)
    assert value.shape == ()


def test_wiremask_cnn_actor_critic_reacts_to_the_wiremask_channel() -> None:
    # Regression guard: confirms wiremask is actually wired into the
    # trunk, not silently ignored - two different wiremasks must produce
    # different logits for the same canvas.
    policy = WiremaskCNNActorCritic()
    canvas = jnp.zeros((8, 8), dtype=bool)
    variables = policy.init(random.PRNGKey(0), {"canvas": canvas, "wiremask": jnp.zeros((8, 8))})

    flat_wiremask = jnp.zeros((8, 8))
    peaked_wiremask = jnp.zeros((8, 8)).at[4, 4].set(10.0)
    logits_flat, _ = policy.apply(variables, {"canvas": canvas, "wiremask": flat_wiremask})
    logits_peaked, _ = policy.apply(variables, {"canvas": canvas, "wiremask": peaked_wiremask})
    assert not jnp.allclose(logits_flat, logits_peaked)
