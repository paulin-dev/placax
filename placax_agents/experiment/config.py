"""ExperimentConfig: the single, serializable description of everything a run depends on.

The problem this solves is the one the macro-placement literature is notorious for. Before this,
each training script in this repo hardcoded its own grid, macro order, reward, policy,
observation, action mask, optimizer, PPO hyperparameters, loop shape and seed - so the two
shipped setups differed on thirteen axes at once, and comparing their results measured the sum of
thirteen changes. Worse, no output file recorded ANY of them: a training log held only
`{iteration, loss, real_hpwl}`, which cannot be attributed back to the configuration that
produced it.

An ExperimentConfig is deliberately split in two:

  `environment` - benchmark, grid, macro order/budget, reward, observation, action mask, and the
      compute budget. This is what must be **identical** between two runs for their results to be
      comparable at all.

  `agent` - policy, optimizer, algorithm and loop. This is the thing **under test**, the one part
      that is supposed to differ between compared runs.

That split is what `environment_hash()` is for. Two runs are legitimately comparable if and only
if their environment hashes match, and that is a one-line assertion rather than a careful reading
of two scripts. `full_hash()` covers the agent and seed as well, identifying an exact run.

Every component is named by a registry key plus JSON-scalar kwargs rather than held as a live
Python object, because a config that cannot round-trip through JSON cannot be written into a
results file - and a configuration that is not written down is not reproducible, whatever the
code around it does.
"""
import hashlib
import json
import pathlib
from dataclasses import dataclass, field
from typing import Any

from placax_agents.experiment.budget import Budget


@dataclass(frozen=True)
class Spec:
    """One swappable component: a registry key plus its JSON-scalar arguments."""

    name: str
    kwargs: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {"name": self.name, "kwargs": dict(self.kwargs)}

    @classmethod
    def from_dict(cls, data: dict | str | None) -> "Spec | None":
        # A bare string is accepted as shorthand for a component that takes no arguments.
        if data is None:
            return None
        if isinstance(data, str):
            return cls(name=data)
        return cls(name=data["name"], kwargs=dict(data.get("kwargs", {})))


@dataclass(frozen=True)
class BenchmarkSpec:
    """Which netlist, at what canvas resolution, in what order, and how much of it."""

    benchmark_dir: str
    grid: int = 224
    macro_budget: int | None = None
    order: Spec = field(default_factory=lambda: Spec("alphabetical"))

    @property
    def path(self) -> pathlib.Path:
        return pathlib.Path(self.benchmark_dir)

    def to_dict(self) -> dict:
        return {
            # Stored as the plain string it was given, so the hash doesn't change just because
            # two machines mount the same benchmark at different absolute paths.
            "benchmark_dir": self.benchmark_dir,
            "grid": self.grid,
            "macro_budget": self.macro_budget,
            "order": self.order.to_dict(),
        }

    @classmethod
    def from_dict(cls, data: dict) -> "BenchmarkSpec":
        return cls(
            benchmark_dir=data["benchmark_dir"],
            grid=data.get("grid", 224),
            macro_budget=data.get("macro_budget"),
            order=Spec.from_dict(data.get("order", "alphabetical")),
        )


@dataclass(frozen=True)
class EnvironmentSpec:
    """Everything that must be identical across two runs for them to be comparable."""

    benchmark: BenchmarkSpec
    reward: Spec
    state: Spec
    budget: Budget
    action_mask: Spec | None = None

    def to_dict(self) -> dict:
        return {
            "benchmark": self.benchmark.to_dict(),
            "reward": self.reward.to_dict(),
            "state": self.state.to_dict(),
            "action_mask": self.action_mask.to_dict() if self.action_mask else None,
            "budget": self.budget.to_dict(),
        }

    @classmethod
    def from_dict(cls, data: dict) -> "EnvironmentSpec":
        return cls(
            benchmark=BenchmarkSpec.from_dict(data["benchmark"]),
            reward=Spec.from_dict(data["reward"]),
            state=Spec.from_dict(data["state"]),
            action_mask=Spec.from_dict(data.get("action_mask")),
            budget=Budget.from_dict(data["budget"]),
        )


@dataclass(frozen=True)
class AgentSpec:
    """The thing under test: how actions get chosen and how the choice is improved."""

    policy: Spec
    algorithm: Spec
    optimizer: Spec
    loop: Spec

    def to_dict(self) -> dict:
        return {
            "policy": self.policy.to_dict(),
            "algorithm": self.algorithm.to_dict(),
            "optimizer": self.optimizer.to_dict(),
            "loop": self.loop.to_dict(),
        }

    @classmethod
    def from_dict(cls, data: dict) -> "AgentSpec":
        return cls(
            policy=Spec.from_dict(data["policy"]),
            algorithm=Spec.from_dict(data["algorithm"]),
            optimizer=Spec.from_dict(data["optimizer"]),
            loop=Spec.from_dict(data["loop"]),
        )


def _canonical_json(data: dict) -> str:
    """Key-sorted, whitespace-free JSON, so an identical config always hashes identically."""
    return json.dumps(data, sort_keys=True, separators=(",", ":"))


def _hash(data: dict) -> str:
    return hashlib.sha256(_canonical_json(data).encode()).hexdigest()[:12]


@dataclass(frozen=True)
class ExperimentConfig:
    """One fully-specified experiment: the environment it runs in and the agent under test."""

    name: str
    environment: EnvironmentSpec
    agent: AgentSpec
    seed: int = 42

    def environment_hash(self) -> str:
        """Identifies the environment alone - benchmark, reward, observation, mask, budget.

        Two runs are comparable if and only if these match. Assert on it before putting two
        numbers in the same table; it is far more reliable than reading two scripts and
        believing they agree.
        """
        return _hash(self.environment.to_dict())

    def full_hash(self) -> str:
        """Identifies the exact run, agent and seed included."""
        return _hash({"environment": self.environment.to_dict(),
                      "agent": self.agent.to_dict(), "seed": self.seed})

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "seed": self.seed,
            "environment": self.environment.to_dict(),
            "agent": self.agent.to_dict(),
            # Derived, and written out so a results file carries them without anyone
            # needing to re-derive them or import this module to read it.
            "environment_hash": self.environment_hash(),
            "full_hash": self.full_hash(),
        }

    @classmethod
    def from_dict(cls, data: dict) -> "ExperimentConfig":
        return cls(
            name=data["name"],
            environment=EnvironmentSpec.from_dict(data["environment"]),
            agent=AgentSpec.from_dict(data["agent"]),
            seed=data.get("seed", 42),
        )

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, sort_keys=True)

    @classmethod
    def from_json(cls, text: str) -> "ExperimentConfig":
        return cls.from_dict(json.loads(text))

    def write(self, path: pathlib.Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.to_json() + "\n")

    @classmethod
    def read(cls, path: pathlib.Path) -> "ExperimentConfig":
        return cls.from_json(path.read_text())


def assert_comparable(*configs: ExperimentConfig) -> None:
    """Raises unless every config shares one environment - the precondition for comparing results.

    Use this wherever two runs' numbers are about to be put side by side. It is the mechanical
    form of "every experiment used exactly the same benchmark, reward, constraints, and compute
    budget", and it catches the case the literature keeps getting wrong: two methods compared
    under quietly different setups.
    """
    if len(configs) < 2:
        return
    reference, *rest = configs
    expected = reference.environment_hash()
    for other in rest:
        if other.environment_hash() != expected:
            differences = _describe_differences(reference.environment.to_dict(),
                                                other.environment.to_dict())
            raise ValueError(
                f"experiments {reference.name!r} and {other.name!r} do not share an environment "
                f"({expected} vs {other.environment_hash()}), so their results are not "
                f"comparable. Differences:\n  " + "\n  ".join(differences)
            )


def _describe_differences(a: dict, b: dict, prefix: str = "") -> list[str]:
    """Every leaf path where two nested dicts disagree, as readable `path: x != y` lines."""
    differences = []
    for key in sorted(set(a) | set(b)):
        left, right = a.get(key), b.get(key)
        path = f"{prefix}{key}"
        if isinstance(left, dict) and isinstance(right, dict):
            differences.extend(_describe_differences(left, right, prefix=f"{path}."))
        elif left != right:
            differences.append(f"{path}: {left!r} != {right!r}")
    return differences
