"""Generalized Advantage Estimation - turns a rollout's rewards and
values into advantages and returns, scanned backwards through the
trajectory (a genuine sequential recurrence)."""
import jax
import jax.numpy as jnp


def compute_gae(
    rewards: jax.Array,
    values: jax.Array,
    dones: jax.Array,
    next_value: jax.Array,
    gamma: float = 0.99,
    lam: float = 0.95,
) -> tuple[jax.Array, jax.Array]:
    """Returns (advantages, returns) for one trajectory. next_value is
    the bootstrap value after the last step - in practice irrelevant
    here since done=True on that step zeroes it."""
    def scan_fn(gae, transition):
        reward, value, next_val, done = transition
        delta = reward + gamma * next_val * (1 - done) - value
        gae = delta + gamma * lam * (1 - done) * gae
        return gae, gae

    next_values = jnp.concatenate([values[1:], next_value[None]])
    _, advantages = jax.lax.scan(
        scan_fn,
        jnp.array(0.0),
        (rewards, values, next_values, dones.astype(jnp.float32)),
        reverse=True,
    )
    returns = advantages + values
    return advantages, returns
