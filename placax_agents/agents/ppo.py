"""PPO behind the Agent seam - the same three loop shapes, now one agent among several."""
from placax_agents.agents.base import UpdateResult  # must precede jax imports
from placax_agents.ops.evaluate import _jitted_evaluate
from placax_agents.training.algorithm.running_stats import init_running_stats

import jax


class PPOAgent:
    """Wraps a policy, an optimizer and one of the training loops into an Agent.

    Its state is the three things PPO carries between iterations - policy variables, optimizer
    state, and the running return statistics - as an ordinary dict, so the runner can checkpoint
    it without knowing any of that.
    """

    name = "ppo"

    def __init__(self, benchmark, policy, optimizer, step_fn, episodes_per_iteration: int,
                 state_fn, extra_illegal_fn=None):
        self.benchmark = benchmark
        self.policy = policy
        self.optimizer = optimizer
        self.state_fn = state_fn
        self.extra_illegal_fn = extra_illegal_fn
        self._step_fn = step_fn
        self._episodes = episodes_per_iteration

    def init(self, key: jax.Array) -> dict:
        from placax.core import reset

        obs0 = self.state_fn(reset(self.benchmark.params), self.benchmark.params,
                             self.benchmark.sizes_array)
        variables = self.policy.init(key, obs0)
        return {
            "variables": variables,
            # Shaped for params only, matching apply_gradient_update, which trains
            # variables["params"] and leaves any frozen collection alone.
            "opt_state": self.optimizer.init(variables["params"]),
            "running_stats": init_running_stats(),
        }

    def update(self, key: jax.Array, state: dict) -> tuple[dict, UpdateResult]:
        variables, opt_state, running_stats, loss = self._step_fn(
            key, state["variables"], state["opt_state"], state["running_stats"]
        )
        new_state = {
            "variables": variables, "opt_state": opt_state, "running_stats": running_stats,
        }
        return new_state, UpdateResult(episodes=self._episodes, loss=float(loss))

    def best_positions(self, state: dict) -> jax.Array:
        """A greedy (argmax over legal cells) rollout - PPO's own notion of its best placement.

        Only the positions are returned; the runner scores them, so every agent is measured by
        the same HPWL code rather than by whatever each one computes for itself.
        """
        positions, _hpwl = _jitted_evaluate(
            state["variables"], self.policy.apply, self.benchmark.params,
            self.benchmark.sizes_array, self.benchmark.cell_size, self.benchmark.padded_pin_idx,
            self.benchmark.padded_pin_offset, self.benchmark.valid_mask, self.state_fn,
            self.extra_illegal_fn,
        )
        return positions

    def bare_variables(self, state: dict):
        """Policy weights alone, for the best-checkpoint bundle scripts/run_pipeline.py reads."""
        return state["variables"]


def is_ppo_state(state) -> bool:
    """Whether an agent state carries policy variables, i.e. whether bare weights can be saved."""
    return isinstance(state, dict) and "variables" in state


__all__ = ["PPOAgent", "is_ppo_state"]
