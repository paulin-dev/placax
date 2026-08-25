"""Generalized Advantage Estimation, scanned backwards through the trajectory."""
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
    """Returns (advantages, returns): delta_t = r_t + gamma*V(s_{t+1})*(1-done) - V(s_t),
    A_t = delta_t + gamma*lam*(1-done)*A_{t+1}, returns = advantages + values."""
    def scan_fn(gae, transition):
        # One backward GAE step; reverse=True below feeds transitions last-to-first.
        reward, value, next_val, done = transition
        delta = reward + gamma * next_val * (1 - done) - value
        gae = delta + gamma * lam * (1 - done) * gae
        return gae, gae

    next_values = jnp.concatenate([values[1:], next_value[None]])  # V(s_{t+1}) per step, shifted by one
    _, advantages = jax.lax.scan(
        scan_fn,
        jnp.array(0.0),
        (rewards, values, next_values, dones.astype(jnp.float32)),
        reverse=True,
    )
    returns = advantages + values
    return advantages, returns
