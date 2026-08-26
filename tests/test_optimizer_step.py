from placax_agents.training.algorithm.optimizer_step import apply_gradient_update  # noqa: F401  must precede jax
from placax_agents.training.algorithm.running_stats import init_running_stats  # noqa: F401

import jax
import jax.numpy as jnp
import optax


def test_apply_gradient_update_changes_params_and_returns_finite_loss() -> None:
    # Flax-shaped variables: "params" is trainable, "batch_stats" mimics a frozen collection.
    variables = {"params": {"w": jnp.array([1.0, 2.0, 3.0])}, "batch_stats": {"var": jnp.array([1.0, 1.0])}}
    optimizer = optax.adam(1e-2)
    opt_state = optimizer.init(variables["params"])
    running_stats = init_running_stats()

    def loss_fn(variables, advantages, returns):
        # a trivial, real loss - depends on params, advantages, and returns
        return jnp.sum(variables["params"]["w"] ** 2) + advantages.mean() * 0.0 + returns.mean() * 0.0

    advantages = jnp.array([1.0, -1.0, 2.0])
    returns = jnp.array([10.0, 20.0, 30.0])

    new_vars, new_opt_state, new_stats, loss = apply_gradient_update(
        variables, opt_state, running_stats, optimizer, loss_fn, advantages, returns
    )

    assert jnp.isfinite(loss)
    assert not jnp.allclose(new_vars["params"]["w"], variables["params"]["w"])  # params actually changed
    assert jnp.array_equal(new_vars["batch_stats"]["var"], variables["batch_stats"]["var"])  # frozen collection untouched
    assert float(new_stats.count) > float(running_stats.count)  # running stats updated
