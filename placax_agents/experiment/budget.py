"""The compute budget: the shared currency that makes two different agents comparable.

Before this existed, "how much compute did this run get" was expressed as `--n_iterations`, which
meant "one episode plus one gradient step" in one training script and "ten episodes plus ten
epochs of minibatch updates" in another. Two agents could not be given the same budget even in
principle, so no cross-agent comparison in this project meant anything.

The unit that fixes it is **env_steps**: one env step is one `placax.core.step()` call, i.e. one
macro placed. Every agent family pays in exactly that currency regardless of how it decides where
to put the macro - a PPO rollout, a GA genome replay, and an ACO ant walk that all place 543
macros have all spent 543 env steps. Wall-clock is recorded alongside it but is deliberately not
the primary unit: it measures the hardware as much as the method.

A budget may cap any combination of the three; the run stops at whichever binds first, and which
one it was is recorded, because "hit the step cap" and "ran out of time" are very different
results to read a metric under.
"""
import time
from dataclasses import dataclass, replace


@dataclass(frozen=True)
class Budget:
    """Caps on what one run may spend. None means that dimension is uncapped."""

    env_steps: int | None = None
    iterations: int | None = None
    wall_clock_s: float | None = None

    def __post_init__(self) -> None:
        if self.env_steps is None and self.iterations is None and self.wall_clock_s is None:
            raise ValueError(
                "a Budget must cap at least one of env_steps, iterations or wall_clock_s - "
                "an uncapped budget cannot be matched across agents, which is the point of it"
            )
        for field_name in ("env_steps", "iterations", "wall_clock_s"):
            value = getattr(self, field_name)
            if value is not None and value <= 0:
                raise ValueError(f"Budget.{field_name} must be positive, got {value}")

    def to_dict(self) -> dict:
        return {
            "env_steps": self.env_steps,
            "iterations": self.iterations,
            "wall_clock_s": self.wall_clock_s,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Budget":
        return cls(
            env_steps=data.get("env_steps"),
            iterations=data.get("iterations"),
            wall_clock_s=data.get("wall_clock_s"),
        )


@dataclass(frozen=True)
class BudgetUse:
    """What a run has spent so far. Checkpointed, so a resumed run continues the same budget
    rather than silently starting a fresh one."""

    iterations: int = 0
    episodes: int = 0
    env_steps: int = 0
    wall_clock_s: float = 0.0

    def to_dict(self) -> dict:
        return {
            "iterations": self.iterations,
            "episodes": self.episodes,
            "env_steps": self.env_steps,
            "wall_clock_s": self.wall_clock_s,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "BudgetUse":
        return cls(
            iterations=int(data.get("iterations", 0)),
            episodes=int(data.get("episodes", 0)),
            env_steps=int(data.get("env_steps", 0)),
            wall_clock_s=float(data.get("wall_clock_s", 0.0)),
        )


class BudgetTracker:
    """Accumulates spend against a Budget and says when (and why) the run should stop.

    Wall-clock accumulates ACROSS resumes: `prior` carries what earlier invocations already
    spent, so `--wall_clock_s=3600` means an hour of total training for this experiment, not an
    hour per time someone restarts it. That is the only reading under which a wall-clock budget
    is comparable between an agent that ran straight through and one that was interrupted.
    """

    def __init__(self, budget: Budget, prior: BudgetUse | None = None):
        self.budget = budget
        self._prior = prior or BudgetUse()
        self._committed = self._prior
        self._started_at = time.monotonic()

    @property
    def use(self) -> BudgetUse:
        """Spend so far, with wall-clock brought up to this instant."""
        return replace(
            self._committed,
            wall_clock_s=self._prior.wall_clock_s + (time.monotonic() - self._started_at),
        )

    def record_iteration(self, episodes: int, n_macros: int) -> BudgetUse:
        """Books one training iteration that collected `episodes` full episodes of `n_macros` each."""
        self._committed = replace(
            self._committed,
            iterations=self._committed.iterations + 1,
            episodes=self._committed.episodes + episodes,
            env_steps=self._committed.env_steps + episodes * n_macros,
        )
        return self.use

    def can_afford(self, episodes: int, n_macros: int) -> bool:
        """Whether one more iteration of this size fits inside the budget without exceeding it.

        Checked BEFORE running an iteration, because an iteration is atomic: stopping only once
        a cap has already been passed lets each loop shape overshoot by a different amount
        (a buffered loop collecting 10 episodes overshoots ten times as far as a sequential one),
        which is exactly the incomparability the budget exists to remove. Refusing to start means
        two loop shapes given one env_step budget both finish at or below it.

        Wall-clock is deliberately not predicted here - how long an iteration takes isn't known
        before running it - so that dimension still stops after the fact. It is the one budget
        dimension that was never exactly comparable anyway.
        """
        if self.budget.env_steps is None:
            return True
        return self.use.env_steps + episodes * n_macros <= self.budget.env_steps

    def exhausted(self) -> str | None:
        """The name of the first cap that has been reached, or None while budget remains."""
        use = self.use
        # Ordered most-meaningful-first so the reported reason is the one worth reading: a run
        # that hit its step cap did the work asked of it; one that hit the clock may not have.
        if self.budget.env_steps is not None and use.env_steps >= self.budget.env_steps:
            return "env_steps"
        if self.budget.iterations is not None and use.iterations >= self.budget.iterations:
            return "iterations"
        if self.budget.wall_clock_s is not None and use.wall_clock_s >= self.budget.wall_clock_s:
            return "wall_clock_s"
        return None

    def describe(self) -> str:
        """A one-line progress string for the console, showing spend against whatever is capped."""
        use = self.use
        parts = []
        if self.budget.iterations is not None:
            parts.append(f"iter {use.iterations}/{self.budget.iterations}")
        else:
            parts.append(f"iter {use.iterations}")
        if self.budget.env_steps is not None:
            parts.append(f"steps {use.env_steps:,}/{self.budget.env_steps:,}")
        else:
            parts.append(f"steps {use.env_steps:,}")
        if self.budget.wall_clock_s is not None:
            parts.append(f"{use.wall_clock_s:.0f}/{self.budget.wall_clock_s:.0f}s")
        else:
            parts.append(f"{use.wall_clock_s:.0f}s")
        return "  ".join(parts)
