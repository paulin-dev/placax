"""How an action changes a placement - the last thing in the kernel that was not swappable.

`step()` used to contain the whole decision in one line:

    positions = state.positions.at[state.step].set(action)

That fixes four things at once: one macro per step, in a pre-computed order, with the action being
an integer grid cell, and the episode ending after `n_macros` of them. It is a faithful
implementation of the sequential-constructive paradigm MaskPlace, EXPlace and ChiPFormer share -
and it was expressed as though it were the only paradigm, in a project whose stated purpose is to
be the shared environment several paradigms are compared in. See docs/Action_Space_Decision.md.

**What the generalization actually turned on.** Not the transition - that is three lines. The
problem is that `state.step` does double duty: it means *how many actions have been taken*
(termination, budgeting) and *which macro is next* (`sizes_array[state.step]`, the lookahead, the
wiremask's baseline). For a constructive space those coincide. For a perturbation space they do
not - every macro is already placed, and the action names the one being moved. So an ActionSpace
has to own the answer to "what is this step about", not merely "what does this action do".

`target()` is that answer, and it is deliberately allowed to be undefined: a perturbation space
returns -1, because at observation time nobody has chosen a macro yet. An observation that needs a
current macro is a constructive-only observation, and that is honest rather than papered over -
the same way a wiremask observation pairs with a policy that reads a wiremask.

**What is NOT here.** A continuous space for SHAC. That needs a differentiable density term to
express legality, which does not exist: legality is enforced by masking, and masks have no
gradient. `docs/Action_Space_Decision.md` measured the other half of that problem (a smoothed
wirelength gives 100% gradient coverage for 1% fidelity) and the density term is what remains. The
protocol is shaped so that adding it later is an implementation, not another kernel change.
"""
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from placax import _device  # noqa: F401  must run before any `import jax` below
from placax.types import EnvParams, EnvState

import jax
import jax.numpy as jnp

UNPLACED = -1
"""The sentinel in `positions` for a macro that has not been placed, and in `target()` for a step
that is not about any particular macro."""


@runtime_checkable
class ActionSpace(Protocol):
    """What an action is, what it does, and when the episode is over."""

    name: str

    constructive: bool
    """Whether a finished placement IS the sequence of actions that produced it.

    True for the spaces that append one macro per step: the placement can be re-driven through
    `step()` action by action, which is what `replay()` and every population method do. False for
    a space whose actions move macros already down - the same placement is reachable by countless
    different move sequences, so it carries no episode to replay, and scoring one means asking
    what it is worth relative to where the episode STARTED (see `experiment.run.episode_return`).
    Two different answers to "what is this placement's return", and the space is what decides
    which one applies."""

    def encode(self, positions: jax.Array, orientations: jax.Array | None) -> jax.Array:
        """The action sequence that produces `positions` (and `orientations`) under this space.

        The inverse of `apply`, and the piece `replay()` was missing: it used to feed the raw
        `(n_macros, 2)` position rows in as actions, which is right only for a space whose action
        IS a grid cell. Under `oriented_grid` a 2-wide row was read as a 3-wide action, and JAX's
        out-of-bounds clamping quietly turned the y coordinate into the orientation.

        Non-constructive spaces raise: there is no sequence to return.
        """
        ...

    def reset(self, params: EnvParams, initial_positions: jax.Array | None) -> EnvState:
        """The starting state, given an optional warm start."""
        ...

    def apply(self, state: EnvState, action: jax.Array, params: EnvParams) -> EnvState:
        """`state` with `action` applied. The one place a placement changes."""
        ...

    def done(self, state: EnvState, params: EnvParams) -> jax.Array:
        """Whether the episode has finished, given the state AFTER the action."""
        ...

    def target(self, state: EnvState, params: EnvParams) -> jax.Array:
        """Which macro this step concerns, or UNPLACED when the space has no such notion.

        Constructive spaces answer `state.step`; a perturbation space cannot answer at all, since
        the macro is part of the action the agent has not yet chosen.
        """
        ...

    def episode_length(self, params: EnvParams, n_placed: int) -> int:
        """How many actions one episode takes. A static shape: it sizes every rollout scan."""
        ...

    def random_action(self, key: jax.Array, params: EnvParams) -> jax.Array:
        """One uniformly random action of this space's own SHAPE.

        Here rather than in the kernel because the shape is the space's: `(x, y)` for a
        constructive space, `(macro, x, y)` for a perturbation one. Uniform over the space, NOT
        over the legal actions - legality is masked one level up, in `policy.action`, and a
        sampler that quietly applied it would put two different definitions of legal in the tree.
        """
        ...


@dataclass(frozen=True)
class DiscreteGridPlacement:
    """One macro per step, in array order, at an integer grid cell. The historical behaviour.

    Every result this project has produced used this, and it stays the default, so nothing that
    does not ask for a different space can observe that the seam was introduced at all.
    """

    name: str = "discrete_grid"

    constructive: bool = True

    def encode(self, positions: jax.Array, orientations: jax.Array | None = None) -> jax.Array:
        """The action IS the grid cell, so a placement is already its own action sequence."""
        return positions

    def reset(self, params: EnvParams, initial_positions: jax.Array | None) -> EnvState:
        if initial_positions is None:
            initial_positions = jnp.full((params.n_macros, 2), UNPLACED)
        # The warm start's prefix decides where step() resumes from.
        n_placed = (initial_positions[:, 0] >= 0).sum()
        return EnvState(positions=initial_positions, step=n_placed)

    def apply(self, state: EnvState, action: jax.Array, params: EnvParams) -> EnvState:
        positions = state.positions.at[state.step].set(action)
        return EnvState(positions=positions, step=state.step + 1)

    def done(self, state: EnvState, params: EnvParams) -> jax.Array:
        return state.step == params.n_macros

    def target(self, state: EnvState, params: EnvParams) -> jax.Array:
        return state.step

    def episode_length(self, params: EnvParams, n_placed: int) -> int:
        return params.n_macros - n_placed

    def random_action(self, key: jax.Array, params: EnvParams) -> jax.Array:
        # x and y drawn independently, so a non-square canvas is sampled over its real extent.
        x_key, y_key = jax.random.split(key)
        return jnp.array([
            jax.random.randint(x_key, (), 0, params.grid_x),
            jax.random.randint(y_key, (), 0, params.effective_grid_y),
        ])


@dataclass(frozen=True)
class Perturbation:
    """Move an already-placed macro: the action is `(macro, x, y)`.

    The shape local search needs - simulated annealing, most real GAs, and FlowPlace's hard
    legalizer all express themselves as "move macro 7 to cell (12, 30)", which the constructive
    kernel had no way to say. It is also what this project's own `row_snap` legalizer is: it moves
    committed macros, which is why it has to run at export rather than inside an episode.

    Every macro must already be placed before the first move, so this space is only meaningful
    with a warm start - `reset` refuses an empty canvas rather than silently perturbing sentinels.
    """

    n_moves: int = 64
    """Actions in one episode. Unlike the constructive space, this has nothing to do with the
    macro count: a perturbation episode is a search budget, and it is the environment's to fix so
    that two agents given the same config do the same amount of work."""

    name: str = "perturbation"

    constructive: bool = False
    """A placement here is not a sequence of appends - countless move sequences reach the same
    one - so it cannot be replayed, and its return is measured against the episode's start."""

    def encode(self, positions: jax.Array, orientations: jax.Array | None = None) -> jax.Array:
        raise ValueError(
            "a perturbation placement is not a sequence of actions: its macros were moved, not "
            "appended, and many different move sequences produce the same final placement. Score "
            "it against the placement the episode STARTED from instead - see "
            "placax_agents.experiment.run.episode_return."
        )

    def reset(self, params: EnvParams, initial_positions: jax.Array | None) -> EnvState:
        if initial_positions is None:
            raise ValueError(
                "the perturbation action space moves macros that are already placed, so it needs "
                "an initial placement - set EnvironmentSpec.initial_placement to something other "
                "than 'empty' (e.g. 'greedy_wiremask_prefix' covering every macro)."
            )
        return EnvState(positions=initial_positions, step=jnp.asarray(0))

    def apply(self, state: EnvState, action: jax.Array, params: EnvParams) -> EnvState:
        # action = (macro, x, y). Writing at an index the action chose, rather than at state.step,
        # is the whole difference from the constructive space.
        positions = state.positions.at[action[0]].set(action[1:3])
        return EnvState(positions=positions, step=state.step + 1)

    def done(self, state: EnvState, params: EnvParams) -> jax.Array:
        return state.step >= self.n_moves

    def target(self, state: EnvState, params: EnvParams) -> jax.Array:
        # Undefined by construction: the macro is part of an action not yet chosen. Returning a
        # sentinel rather than a plausible index keeps a constructive-only observation from
        # quietly producing nonsense here.
        return jnp.asarray(UNPLACED)

    def episode_length(self, params: EnvParams, n_placed: int) -> int:
        return self.n_moves

    def random_action(self, key: jax.Array, params: EnvParams) -> jax.Array:
        macro_key, x_key, y_key = jax.random.split(key, 3)
        return jnp.array([
            jax.random.randint(macro_key, (), 0, params.n_macros),
            jax.random.randint(x_key, (), 0, params.grid_x),
            jax.random.randint(y_key, (), 0, params.effective_grid_y),
        ])


@dataclass(frozen=True)
class OrientedGridPlacement:
    """Constructive, but the action is `(x, y, orientation)` - the agent chooses the turn too.

    The other half of "placement representation": a placement was a position and nothing else, so
    a tall SRAM could never be laid on its side. Real flows emit an orientation per instance, and
    every writer here preserved whatever the source file said, which made the axis look fixed at
    north when in fact it was simply absent.

    Rotations only - N, W, S, E - not the four mirrors; a mirror changes pin positions without
    changing the footprint, so it moves wirelength and not legality. See extras/orientation.py.
    """

    name: str = "oriented_grid"

    constructive: bool = True

    def encode(self, positions: jax.Array, orientations: jax.Array | None = None) -> jax.Array:
        """`(x, y)` plus the turn, so a replay reproduces the orientations as well as the cells.

        Without this, `replay()` fed 2-wide rows to `apply`, which reads `action[2]` - clamped by
        JAX to the y coordinate, so every macro was replayed at an orientation nobody chose, and
        the reward that came back described a placement that never existed.
        """
        if orientations is None:
            orientations = jnp.zeros((positions.shape[0],), dtype=jnp.int32)
        return jnp.concatenate([positions, orientations[:, None]], axis=1)

    def reset(self, params: EnvParams, initial_positions: jax.Array | None) -> EnvState:
        if initial_positions is None:
            initial_positions = jnp.full((params.n_macros, 2), UNPLACED)
        n_placed = (initial_positions[:, 0] >= 0).sum()
        # Materialized rather than left None: this space writes into it every step, and a state
        # whose orientations appear partway through would change pytree structure mid-scan.
        return EnvState(
            positions=initial_positions, step=n_placed,
            orientations=jnp.zeros((params.n_macros,), dtype=jnp.int32),
        )

    def apply(self, state: EnvState, action: jax.Array, params: EnvParams) -> EnvState:
        return EnvState(
            positions=state.positions.at[state.step].set(action[:2]),
            step=state.step + 1,
            # Cast explicitly: a weakly-typed action would otherwise widen the int32
            # orientation array and warn about a lossy scatter.
            orientations=state.orientations.at[state.step].set(action[2].astype(jnp.int32)),
        )

    def done(self, state: EnvState, params: EnvParams) -> jax.Array:
        return state.step == params.n_macros

    def target(self, state: EnvState, params: EnvParams) -> jax.Array:
        return state.step

    def episode_length(self, params: EnvParams, n_placed: int) -> int:
        return params.n_macros - n_placed

    def random_action(self, key: jax.Array, params: EnvParams) -> jax.Array:
        x_key, y_key, turn_key = jax.random.split(key, 3)
        return jnp.array([
            jax.random.randint(x_key, (), 0, params.grid_x),
            jax.random.randint(y_key, (), 0, params.effective_grid_y),
            jax.random.randint(turn_key, (), 0, 4),
        ])


DISCRETE_GRID = DiscreteGridPlacement()
"""The default everywhere, so the historical behaviour needs no argument to select."""


__all__ = ["ActionSpace", "DiscreteGridPlacement", "OrientedGridPlacement", "Perturbation",
           "DISCRETE_GRID", "UNPLACED"]
