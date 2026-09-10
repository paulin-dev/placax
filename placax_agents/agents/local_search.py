"""Hill-climbing / simulated annealing over macro moves - the first agent that is not constructive.

Every agent here so far builds a placement once, left to right, and never revisits a decision:
PPO, both baselines and the GA all choose where the *next* macro goes. Local search is the family
that does the opposite - it starts from a complete placement and improves it by moving macros that
are already down - and the constructive kernel had no action that could say that. This agent is
what makes `placax.action_space.Perturbation` more than an interface: it drives it end to end.

**It ranks moves by the run's configured reward, per step.** Under a perturbation space the
per-step reward IS the improvement - `reward_fn(before, after, ...)` on a complete placement is
exactly the change the move caused - so hill climbing has the signal it needs without inventing a
second objective. That is also the constraint worth stating plainly: **this agent needs a DENSE
reward.** A sparse (terminal-only) reward returns 0 for every non-final step, which leaves the
search with nothing to climb and degenerates it into a random walk. `acceptance_rate` is reported
every iteration so that failure is visible in the log rather than inferred from a bad result.

Legality is the environment's, as always, with one wrinkle the constructive agents never hit: to
ask whether macro *m* may move to a cell, *m* must first be lifted off the canvas, or it blocks
itself everywhere except where it already is. `_EnvironmentBound._illegal_for_macro` does that.
"""
from placax_agents.agents.base import UpdateResult  # must precede jax imports
from placax_agents.agents.environment_bound import _EnvironmentBound
from placax.core import reset, step

import jax
import jax.numpy as jnp


class LocalSearchAgent(_EnvironmentBound):
    """Proposes a move, keeps it if the reward improves (or, at temperature > 0, sometimes anyway)."""

    name = "local_search"

    def __init__(self, benchmark, temperature: float = 0.0, cooling: float = 0.95,
                 state_fn=None, extra_illegal_fn=None, initial_positions=None, n_placed: int = 0,
                 action_space=None):
        super().__init__(benchmark, state_fn, extra_illegal_fn, initial_positions, n_placed,
                         action_space)
        if initial_positions is None:
            raise ValueError(
                "local search improves an existing placement, so it needs one to start from. Set "
                "EnvironmentSpec.initial_placement to something that fills the canvas (e.g. "
                "'greedy_wiremask_prefix' with n_macros covering the whole design)."
            )
        self.temperature = temperature
        """0 is strict hill climbing: only improving moves are kept. Above 0, a worsening move is
        accepted with probability exp(delta / T), which is simulated annealing - the escape hatch
        from the local optimum a greedy pass lands in."""

        self.cooling = cooling

    def init(self, key: jax.Array) -> dict:
        """Start from the environment's initial placement, which is where every restart begins."""
        start = reset(self.benchmark.params, self.initial_positions, self.action_space)
        return {
            "positions": start.positions,
            "best_positions": start.positions,
            "best_return": jnp.array(0.0),
            "temperature": jnp.asarray(self.temperature, dtype=jnp.float32),
        }

    def update(self, key: jax.Array, state: dict) -> tuple[dict, UpdateResult]:
        """One episode of `n_moves` proposals, each accepted or rejected on the spot."""
        final, accepted, total = _anneal(key, state, self)
        improved = total > state["best_return"]
        return (
            {
                "positions": final,
                "best_positions": jnp.where(improved, final, state["best_positions"]),
                "best_return": jnp.where(improved, total, state["best_return"]),
                "temperature": state["temperature"] * self.cooling,
            },
            UpdateResult(
                episodes=1,
                loss=None,
                gradient_steps=0,
                metrics={
                    # Near zero means the search is stuck, or the reward is sparse and there is no
                    # per-step signal at all - the two failures worth telling apart in a log.
                    "acceptance_rate": float(accepted) / max(self._episode_length, 1),
                    "episode_improvement": float(total),
                    "temperature": float(state["temperature"]),
                },
            ),
        )

    def best_positions(self, state: dict) -> jax.Array:
        return state["best_positions"]


def _anneal(key: jax.Array, state: dict, agent: LocalSearchAgent):
    """`n_moves` proposals through the kernel, returning (positions, n_accepted, total reward)."""
    params = agent.benchmark.params
    start = reset(params, state["positions"], agent.action_space)

    def scan_step(carry, keys):
        current, accepted, total = carry
        propose_key, accept_key = keys
        action = _propose(propose_key, current, agent)

        # The move goes through the real step(), so the reward is the run's configured one,
        # computed by the same code a policy's move would be.
        candidate, reward, _done = step(
            current, action, agent.benchmark.reward_fn, params, agent.action_space
        )
        # Metropolis: always take an improvement; take a worsening move with probability
        # exp(delta / T), which is 0 at T = 0 and turns this into strict hill climbing.
        temperature = jnp.maximum(state["temperature"], 1e-8)
        threshold = jnp.exp(jnp.minimum(reward / temperature, 0.0))
        take = (reward > 0) | (jax.random.uniform(accept_key) < threshold)

        # Rejecting still costs a step: the environment work was done either way, and the budget
        # has to see it or a search that rejects everything looks free.
        positions = jnp.where(take, candidate.positions, current.positions)
        kept = current.replace(positions=positions, step=candidate.step)
        return (kept, accepted + take, total + jnp.where(take, reward, 0.0)), None

    keys = jax.random.split(key, agent._episode_length * 2).reshape(-1, 2, 2)
    (final, accepted, total), _ = jax.lax.scan(
        scan_step, (start, jnp.asarray(0), jnp.asarray(0.0)), keys
    )
    return final.positions, accepted, total


def _propose(key: jax.Array, state, agent: LocalSearchAgent) -> jax.Array:
    """A uniformly random macro, moved to a uniformly random cell that is legal without it."""
    macro_key, cell_key = jax.random.split(key)
    macro = jax.random.randint(macro_key, (), 0, agent.benchmark.params.n_macros)
    illegal = agent._illegal_for_macro(state, macro)
    # Uniform over legal cells, exactly as RandomSearchAgent samples - equal logits, -inf on the
    # illegal ones - so the two agents explore the same space and differ only in what they keep.
    logits = jnp.where(illegal, -jnp.inf, 0.0)
    flat_idx = jax.random.categorical(cell_key, logits.ravel())
    return jnp.array([macro, flat_idx // illegal.shape[1], flat_idx % illegal.shape[1]])


__all__ = ["LocalSearchAgent"]
