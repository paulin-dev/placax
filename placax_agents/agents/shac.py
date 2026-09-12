"""SHAC: the analytic-gradient agent this project was founded to compare against PPO.

`docs/JAX_Placement_Environment_Spec.md` §13 frames the whole project around one question - "if
SHAC beats PPO, a concrete improvement over the field's standard; if it underperforms, an honest
negative result" - and until now the agent could not be written, because three environment-level
pieces were missing. All three exist as of this module:

  1. **a wirelength every macro feels** - `REWARDS["smoothed"]`, measured at 100% gradient
     coverage for 1% fidelity (docs/Action_Space_Decision.md);
  2. **a legality that has a gradient** - `extras/density.py`, since masking is comparisons and
     `d(overlap)/d(position)` through a mask is identically zero;
  3. **a continuous action** - `ContinuousPlacement`, because `d(action)/d(parameters)` through a
     categorical over grid cells does not exist.

**What SHAC is.** Short-Horizon Actor-Critic (Xu et al., 2022): instead of estimating the policy
gradient from sampled returns the way PPO does, differentiate the objective *through the
simulator* for a short window of H steps, and bootstrap the rest of the episode with a learned
value function. The short window is the point - a long one is what makes backpropagation through
time in a physical simulator explode or vanish - so the horizon is the method's central
hyperparameter, not an implementation detail.

**How the window is implemented here.** One episode per iteration, one scan, with the gradient
path CUT at every H-th step (`jax.lax.stop_gradient` on the carried placement) and the value
function added at that boundary. That is truncated backpropagation through time, expressed as a
single differentiable rollout rather than as an outer loop over windows - so the whole update is
one `value_and_grad` and one optimizer step, and the horizon costs nothing but a boolean per step.

**Where the gradient actually flows, stated precisely.** Two paths run from an action to a later
reward, and only one of them survives:

  * *action -> placement -> every later reward.* Alive. `ContinuousPlacement.apply` writes a float
    into `positions`, and the differentiable reward reads those positions, so a macro placed at
    step 3 keeps receiving gradient from the wirelength and density of steps 4..H.
  * *action -> placement -> canvas -> the policy's next action.* Cut, because `render` is built
    from comparisons. This is why the shipped SHAC policy reads COORDINATES rather than the
    canvas (`policy/architectures/continuous.py`): a canvas-reading policy would silently lose
    the first path's partner and leave SHAC differentiating one step at a time while looking
    exactly the same from outside.

**The honest caveat about legality.** Placements under the discrete spaces are legal by
construction, because illegal cells are masked out. A continuous placement is only as legal as the
density penalty made it, so a SHAC run reports overlap like every other run and a row that is not
100% legal has not produced a result. That is not a defect of this agent; it is the trade the
method requires, and it is why `density_weight` is the first thing to tune on a new design.
"""
from placax.action_space import ContinuousPlacement  # must precede jax imports
from placax.core import reset, step
from placax_agents.agents.base import UpdateResult
from placax_agents.policy.architectures.continuous import mean_action, sample_continuous

import jax
import jax.numpy as jnp


class SHACAgent:
    """Differentiates the placement objective through a short window of the episode."""

    name = "shac"

    def __init__(self, benchmark, policy, optimizer, state_fn, horizon: int = 8,
                 gamma: float = 0.99, value_coef: float = 0.5, entropy_coef: float = 0.0,
                 extra_illegal_fn=None, initial_positions=None, n_placed: int = 0,
                 action_space=None):
        self.benchmark = benchmark
        self.policy = policy
        self.optimizer = optimizer
        self.state_fn = state_fn
        self.extra_illegal_fn = extra_illegal_fn
        self.initial_positions = initial_positions
        self.n_placed = n_placed
        self.action_space = action_space if action_space is not None else ContinuousPlacement()

        if horizon < 1:
            raise ValueError(f"SHAC's horizon is a window of steps to differentiate, got {horizon}")
        self.horizon = horizon
        """Steps backpropagated through before the value function takes over. The method's central
        hyperparameter: too short and it is a one-step gradient with a critic doing all the work,
        too long and the product of per-step Jacobians stops being usable."""

        self.gamma = gamma
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        """Entropy here is the Gaussian's own, `sum(log_std) + const` - a direct pull on the
        exploration width rather than a term over a categorical. Zero by default: the width is
        already learned, and SHAC's gradient is not the high-variance estimator entropy bonuses
        usually exist to stabilize."""

    # ------------------------------------------------------------------ Agent

    def init(self, key: jax.Array) -> dict:
        obs0 = self.state_fn(
            reset(self.benchmark.params, self.initial_positions, self.action_space),
            self.benchmark.params, self.benchmark.sizes_array,
        )
        variables = self.policy.init(key, obs0)
        return {"variables": variables, "opt_state": self.optimizer.init(variables["params"])}

    def update(self, key: jax.Array, state: dict) -> tuple[dict, UpdateResult]:
        variables, opt_state, loss, metrics = _shac_update(
            key, state["variables"], state["opt_state"], self.optimizer, self, self.policy.apply
        )
        return (
            {"variables": variables, "opt_state": opt_state},
            UpdateResult(episodes=1, loss=float(loss), gradient_steps=1,
                         metrics={name: float(value) for name, value in metrics.items()}),
        )

    def best_positions(self, state: dict) -> jax.Array:
        """A deterministic rollout - every macro at the policy's mean, no exploration noise."""
        positions, _value = _greedy_rollout(state["variables"], self, self.policy.apply)
        return positions

    def converged(self, _state: dict) -> bool:
        return False

    def bare_variables(self, state: dict):
        return state["variables"]


def _rollout(variables, agent: SHACAgent, apply_fn, key, stochastic: bool):
    """One episode, returning (final positions, per-step rewards, per-step values, mean log_std).

    Differentiable in `variables` up to the horizon boundaries: every `agent.horizon` steps the
    carried placement is passed through `stop_gradient`, which is what makes this a SHORT-horizon
    method rather than full backpropagation through a 543-step episode.
    """
    params = agent.benchmark.params
    length = agent.action_space.episode_length(params, agent.n_placed)
    start = reset(params, agent.initial_positions, agent.action_space)

    # Which steps end a window. Computed here as a static array rather than from the traced scan
    # index, so the horizon is visible in the compiled program instead of as arithmetic per step.
    boundaries = ((jnp.arange(length) + 1) % agent.horizon == 0).at[length - 1].set(True)

    def scan_step(carry, inputs):
        env_state = carry
        step_key, is_boundary = inputs
        obs = agent.state_fn(env_state, params, agent.benchmark.sizes_array)
        action_params, value = apply_fn(variables, obs)
        action = (
            sample_continuous(step_key, action_params) if stochastic
            else mean_action(action_params)
        )
        next_state, reward, _done = step(
            env_state, action, agent.benchmark.reward_fn, params, agent.action_space
        )
        # The window boundary: cut the path from here backwards, so the next window's gradient
        # starts fresh and the value function is what connects them.
        carried = jax.tree_util.tree_map(
            lambda leaf: jnp.where(is_boundary, jax.lax.stop_gradient(leaf), leaf), next_state
        )
        return carried, (reward, value, is_boundary, action_params["log_std"].mean())

    final_state, (rewards, values, flags, log_stds) = jax.lax.scan(
        scan_step, start, (jax.random.split(key, length), boundaries)
    )
    return final_state.positions, rewards, values, flags, log_stds


def _windowed_return(rewards, values, boundaries, gamma: float):
    """The SHAC objective: each window's discounted reward plus the value at its boundary.

    Walked backwards so a window's return is built from its own steps only - at a boundary the
    accumulator restarts from the bootstrap `V(s_H)` instead of carrying the next window's return
    backwards, which is exactly the truncation the `stop_gradient` above performs on the state.
    """

    def backward(carry, inputs):
        reward, value, is_boundary = inputs
        # At a boundary the tail is the critic's estimate; inside a window it is what follows.
        tail = jnp.where(is_boundary, value, carry)
        carry = reward + gamma * tail
        return carry, carry

    _final, returns = jax.lax.scan(
        backward, jnp.array(0.0), (rewards, values, boundaries), reverse=True
    )
    return returns


def _loss(variables, agent: SHACAgent, apply_fn, key):
    """Actor loss (negative windowed return) + critic loss, in one differentiable pass."""
    _positions, rewards, values, boundaries, log_stds = _rollout(
        variables, agent, apply_fn, key, stochastic=True
    )
    returns = _windowed_return(rewards, values, boundaries, agent.gamma)

    # The actor maximizes the windowed return, differentiated THROUGH the placement.
    actor_loss = -returns.mean()
    # The critic regresses onto the same returns, which are a target rather than a signal to
    # differentiate - hence the stop_gradient, or the critic's error would push the actor around.
    critic_loss = ((values - jax.lax.stop_gradient(returns)) ** 2).mean()
    entropy = log_stds.mean()

    loss = actor_loss + agent.value_coef * critic_loss - agent.entropy_coef * entropy
    return loss, {
        "actor_loss": actor_loss,
        "critic_loss": critic_loss,
        "episode_return": rewards.sum(),
        "mean_log_std": entropy,
    }


def _shac_update(key, variables, opt_state, optimizer, agent: SHACAgent, apply_fn):
    """One episode, one gradient step. Jitted at the call site below."""
    import optax

    (loss, metrics), grads = jax.value_and_grad(_loss, has_aux=True)(
        variables, agent, apply_fn, key
    )
    updates, opt_state = optimizer.update(grads["params"], opt_state, variables["params"])
    params = optax.apply_updates(variables["params"], updates)
    variables = {**variables, "params": params}
    metrics["grad_norm"] = optax.tree.norm(grads["params"])
    return variables, opt_state, loss, metrics


def _greedy_rollout(variables, agent: SHACAgent, apply_fn):
    """The placement the policy's means produce - no noise, and no gradient needed."""
    positions, _rewards, values, _flags, _log_stds = _rollout(
        variables, agent, apply_fn, jax.random.PRNGKey(0), stochastic=False
    )
    return positions, values


__all__ = ["SHACAgent"]
