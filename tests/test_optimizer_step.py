from placax_agents.training.algorithm.optimizer_step import apply_gradient_update  # noqa: F401  must precede jax
from placax_agents.training.algorithm.running_stats import init_running_stats  # noqa: F401

import jax
import jax.numpy as jnp
import optax


def test_apply_gradient_update_changes_params_and_returns_finite_loss() -> None:
    variables = {"w": jnp.array([1.0, 2.0, 3.0])}
    optimizer = optax.adam(1e-2)
    opt_state = optimizer.init(variables)
    running_stats = init_running_stats()

    def loss_fn(params, advantages, returns):
        # a trivial, real loss - depends on params, advantages, and returns
        return jnp.sum(params["w"] ** 2) + advantages.mean() * 0.0 + returns.mean() * 0.0

    advantages = jnp.array([1.0, -1.0, 2.0])
    returns = jnp.array([10.0, 20.0, 30.0])

    new_vars, new_opt_state, new_stats, loss = apply_gradient_update(
        variables, opt_state, running_stats, optimizer, loss_fn, advantages, returns
    )

    assert jnp.isfinite(loss)
    assert not jnp.allclose(new_vars["w"], variables["w"])  # params actually changed
    assert float(new_stats.count) > float(running_stats.count)  # running stats updated
