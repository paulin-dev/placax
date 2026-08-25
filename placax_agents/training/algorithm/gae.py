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
    """Computes GAE advantages and returns: delta_t = r_t + gamma*V(s_{t+1})*(1-done) - V(s_t),
    A_t = delta_t + gamma*lam*(1-done)*A_{t+1}, returns = advantages + values."""

    def scan_fn(gae, transition):
        # One backward step: how much better was this action than the value function expected,
        # plus a discounted, done-gated share of the advantage already computed for the next step.
        reward, value, next_val, done = transition
        delta = reward + gamma * next_val * (1 - done) - value
        gae = delta + gamma * lam * (1 - done) * gae
        return gae, gae

    # Shift values by one step to get V(s_{t+1}) aligned with each transition; the final
    # next_value (usually 0 for a finished episode) fills in after the last real step.
    next_values = jnp.concatenate([values[1:], next_value[None]])
    # Walk the trajectory backwards (reverse=True) since each advantage depends on the next one.
    _, advantages = jax.lax.scan(
        scan_fn,
        jnp.array(0.0),
        (rewards, values, next_values, dones.astype(jnp.float32)),
        reverse=True,
    )
    # Returns are what the value function is trained to predict: advantage plus its own baseline.
    returns = advantages + values
    return advantages, returns
