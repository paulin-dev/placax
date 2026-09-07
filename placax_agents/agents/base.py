"""The Agent seam: what every method must provide, and nothing more.

Before this, the training loops took a Flax `apply` function, Flax `variables` and an optax
`GradientTransformation` directly. That is a PPO shape, not an agent shape - a population method
has no optimizer state, a pheromone method has no parameters, and a pure heuristic has no state
at all. Any of them would have had to write its own loop from `collect_rollout` downward, and
would then have missed the shared evaluation, checkpointing, budgeting and logging - which is
exactly how per-paper one-off environments come about.

An Agent owns an opaque state pytree and answers three questions:

    init(key)            -> state          what do I start from?
    update(key, state)   -> state, result  spend some environment budget and improve
    best_positions(state) -> positions     your single best placement right now

The third one is load-bearing for comparability. The agent returns a PLACEMENT, not a score:
the runner computes HPWL itself, identically for every agent. If each agent reported its own
number, "PPO beat random search" could come down to two different scoring conventions, which is
precisely the class of error this project exists to rule out.

`update` also reports how many episodes it actually spent, rather than that being a static
property of the loop, so an agent whose per-iteration cost varies (a GA with a growing
population, an early-terminating search) is still charged correctly against the budget.
"""
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

import jax


@dataclass(frozen=True)
class UpdateResult:
    """What one agent update did, in units the runner can budget and log."""

    episodes: int
    """Full episodes' worth of environment interaction this update consumed. The runner
    multiplies by n_macros to charge env steps, so this must count real placements - a
    population of 32 candidate placements is 32 episodes, not one."""

    loss: float | None = None
    """The agent's own training signal, if it has one. None for methods that don't train."""

    metrics: dict[str, float] = field(default_factory=dict)
    """Anything else worth putting in the log - population diversity, acceptance rate, entropy."""


@runtime_checkable
class Agent(Protocol):
    """A method for choosing macro placements, and improving at it or not."""

    name: str

    def init(self, key: jax.Array) -> Any:
        """The agent's starting state: any pytree, checkpointed and restored by the runner."""
        ...

    def update(self, key: jax.Array, state: Any) -> tuple[Any, UpdateResult]:
        """One iteration of whatever this agent does. Returns the new state and what it cost."""
        ...

    def best_positions(self, state: Any) -> jax.Array:
        """This agent's single best placement given its current state, as (n_macros, 2) grid
        positions. The runner scores it; the agent must not."""
        ...
