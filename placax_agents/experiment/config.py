"""ExperimentConfig: the single, serializable description of everything a run depends on.

The problem this solves is the one the macro-placement literature is notorious for. Before this,
each training script in this repo hardcoded its own grid, macro order, reward, policy,
observation, action mask, optimizer, PPO hyperparameters, loop shape and seed - so the two
shipped setups differed on thirteen axes at once, and comparing their results measured the sum of
thirteen changes. Worse, no output file recorded ANY of them: a training log held only
`{iteration, loss, real_hpwl}`, which cannot be attributed back to the configuration that
produced it.

An ExperimentConfig is deliberately split in two:

  `environment` - benchmark, grid, macro order/budget, initial placement, reward, observation,
      action mask, the physical (cell placer / validator) stack, and the compute budget. This is
      what must be **identical** between two runs for their results to be comparable at all.

  `agent` - policy, optimizer, algorithm and loop. This is the thing **under test**, the one part
      that is supposed to differ between compared runs.

That split is what the hashes are for. Two runs are legitimately comparable if and only if their
hashes match at the level the comparison claims, and that is a one-line assertion rather than a
careful reading of two scripts.

**Four hash levels, because "comparable" is not one question.** A single all-or-nothing hash
forces an experiment that deliberately varies one environment axis to abandon the mechanism
entirely, which is how the mechanism stops being used:

  `benchmark_hash()`   the design itself - netlist contents, grid, macro order and budget.
  `task_hash()`        + what is being optimized and under what rules: reward, initial placement,
                       action-legality constraints, physical stack, compute budget. Everything
                       except how the agent is allowed to *look* at it. This is the level a
                       state-representation comparison asserts (see docs §12: "raw coordinates
                       vs. image vs. graph, algorithm held fixed") - those runs differ in
                       `state` on purpose, so `environment_hash` would reject them and tell the
                       researcher nothing.
  `environment_hash()` + the observation. The default, and the right level for an agent
                       comparison: everything the agent did not choose.
  `full_hash()`        + the agent and the seed. Identifies an exact run.

Every component is named by a registry key plus JSON-scalar kwargs rather than held as a live
Python object, because a config that cannot round-trip through JSON cannot be written into a
results file - and a configuration that is not written down is not reproducible, whatever the
code around it does.

**A hash is taken over what a config MEANS, not over how completely it was spelled out.** Every
Spec's kwargs are completed with its builder's own defaults on the way into a hash (see
`defaults.py`), so `Spec("hpwl")` and `Spec("hpwl", {"dense": False, "reward_scale": 1.0})` -
one reward, one set of values - hash alike. Without that, `assert_comparable` rejected two
identical experiments written by two people, and a config read back from JSON could never match
one built from a preset. What gets *written down* is still exactly what the author wrote: the
completion happens in `identity()`, never in `to_dict()`, so a config round-trips unchanged.
"""
import hashlib
import json
import pathlib
from dataclasses import dataclass, field, replace
from typing import Any

from placax_agents.experiment.budget import Budget


@dataclass(frozen=True)
class Spec:
    """One swappable component: a registry key plus its JSON-scalar arguments."""

    name: str
    kwargs: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        """Exactly what was written down - the provenance half, unchanged by anything here."""
        return {"name": self.name, "kwargs": dict(self.kwargs)}

    def identity(self, slot: str) -> dict:
        """This component as a HASH sees it: its name, and its kwargs completed with defaults.

        `Spec("hpwl")` and `Spec("hpwl", {"dense": False, "reward_scale": 1.0})` are one reward
        with one set of values, so they must hash alike; completing both against the builder's
        own signature is what makes that true. `slot` says which registry to read the defaults
        from, since a Spec on its own does not know whether it is a reward or a policy.
        See placax_agents/experiment/defaults.py.
        """
        from placax_agents.experiment.defaults import complete_kwargs

        return {"name": self.name, "kwargs": complete_kwargs(slot, self.name, self.kwargs)}

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
    """Which netlist, at what canvas resolution, in what order, and how much of it.

    `netlist_digest` is what actually identifies the design. It is filled in by
    `build_benchmark()` from the parsed macros and nets (not from the files on disk, so an
    unrelated whitespace change doesn't invalidate a comparison, and a real edit to a macro
    size does). It starts as None on a config built from a preset and is resolved before the
    run's manifest is written.

    The path is deliberately NOT hashed. Hashing it gets both directions wrong: two machines
    mounting the same design at different paths look incomparable, while editing a netlist in
    place looks comparable. The design's directory name is hashed as a readable label, and the
    digest carries the actual invariant.
    """

    benchmark_dir: str
    grid: int = 224
    macro_budget: int | None = None
    order: Spec = field(default_factory=lambda: Spec("alphabetical"))
    netlist_digest: str | None = None

    @property
    def path(self) -> pathlib.Path:
        return pathlib.Path(self.benchmark_dir)

    @property
    def design(self) -> str:
        """The design's name - the directory's basename, independent of where it is mounted."""
        return self.path.name

    def with_digest(self, digest: str) -> "BenchmarkSpec":
        return replace(self, netlist_digest=digest)

    def identity(self) -> dict:
        """The part that decides whether two runs used the same design. Path excluded."""
        return {
            "design": self.design,
            "netlist_digest": self.netlist_digest,
            "grid": self.grid,
            "macro_budget": self.macro_budget,
            "order": self.order.identity("order"),
        }

    def to_dict(self) -> dict:
        # Provenance (where it was loaded from) plus what it actually was. Deliberately NOT
        # `**self.identity()`: an identity completes each Spec's defaults for hashing, while what
        # gets written down stays exactly what the author wrote, so a config round-trips through
        # JSON unchanged.
        return {
            "benchmark_dir": self.benchmark_dir,
            "design": self.design,
            "netlist_digest": self.netlist_digest,
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
            netlist_digest=data.get("netlist_digest"),
        )


@dataclass(frozen=True)
class PhysicalSpec:
    """The cell placer and validator a run's PPA numbers came from.

    Named here rather than left to a downstream script's CLI flags because a PPA number is only
    attributable if the tools that produced it are recorded next to it. Both default to None,
    which is the honest description of a run that only ever reported the geometric proxy: no
    physical flow was involved, and the config says so rather than implying a default.
    """

    cell_placer: Spec | None = None
    validator: Spec | None = None

    def to_dict(self) -> dict:
        return {
            "cell_placer": self.cell_placer.to_dict() if self.cell_placer else None,
            "validator": self.validator.to_dict() if self.validator else None,
        }

    @classmethod
    def from_dict(cls, data: dict | None) -> "PhysicalSpec":
        data = data or {}
        return cls(
            cell_placer=Spec.from_dict(data.get("cell_placer")),
            validator=Spec.from_dict(data.get("validator")),
        )


@dataclass(frozen=True)
class EnvironmentSpec:
    """Everything that must be identical across two runs for them to be comparable."""

    benchmark: BenchmarkSpec
    reward: Spec
    state: Spec
    budget: Budget
    action_mask: Spec | None = None
    initial_placement: Spec = field(default_factory=lambda: Spec("empty"))
    physical: PhysicalSpec = field(default_factory=PhysicalSpec)

    def task_identity(self) -> dict:
        """The design plus what is being optimized on it, under what rules, for how long.

        Everything except the observation - i.e. everything two runs that differ only in state
        representation still share, and must still be asserted to share.
        """
        return {
            "benchmark": self.benchmark.identity(),
            "reward": self.reward.identity("reward"),
            "initial_placement": self.initial_placement.identity("initial_placement"),
            "action_mask": self.action_mask.identity("action_mask") if self.action_mask else None,
            # The physical stack is deliberately NOT completed with its builders' defaults: those
            # signatures mix experiment settings (target_density, liberty) with this machine's
            # install paths (dreamplace_root, openroad_binary), and completing them would fold a
            # machine-specific value into the environment hash - the one thing build_physical's
            # config/machine split exists to prevent. See defaults.py.
            "physical": self.physical.to_dict(),
            "budget": self.budget.to_dict(),
        }

    def identity(self) -> dict:
        """The task plus the observation: everything the agent did not choose."""
        return {**self.task_identity(), "state": self.state.identity("state")}

    def to_dict(self) -> dict:
        return {
            "benchmark": self.benchmark.to_dict(),
            "reward": self.reward.to_dict(),
            "state": self.state.to_dict(),
            "action_mask": self.action_mask.to_dict() if self.action_mask else None,
            "initial_placement": self.initial_placement.to_dict(),
            "physical": self.physical.to_dict(),
            "budget": self.budget.to_dict(),
        }

    @classmethod
    def from_dict(cls, data: dict) -> "EnvironmentSpec":
        return cls(
            benchmark=BenchmarkSpec.from_dict(data["benchmark"]),
            reward=Spec.from_dict(data["reward"]),
            state=Spec.from_dict(data["state"]),
            action_mask=Spec.from_dict(data.get("action_mask")),
            initial_placement=Spec.from_dict(data.get("initial_placement", "empty")),
            physical=PhysicalSpec.from_dict(data.get("physical")),
            budget=Budget.from_dict(data["budget"]),
        )


@dataclass(frozen=True)
class AgentSpec:
    """The thing under test: how actions get chosen and how the choice is improved.

    Only `algorithm` is required. A gradient-free method has no policy network, no optimizer and
    no training loop, and forcing it to name three placeholders would make the config lie about
    what the method is - so those three are optional and simply absent for such agents.
    """

    algorithm: Spec
    policy: Spec | None = None
    optimizer: Spec | None = None
    loop: Spec | None = None

    def to_dict(self) -> dict:
        return {
            "algorithm": self.algorithm.to_dict(),
            "policy": self.policy.to_dict() if self.policy else None,
            "optimizer": self.optimizer.to_dict() if self.optimizer else None,
            "loop": self.loop.to_dict() if self.loop else None,
        }

    def identity(self) -> dict:
        """The agent as `full_hash` sees it, every Spec completed with its own defaults."""
        return {
            "algorithm": self.algorithm.identity("algorithm"),
            "policy": self.policy.identity("policy") if self.policy else None,
            "optimizer": self.optimizer.identity("optimizer") if self.optimizer else None,
            "loop": self.loop.identity("loop") if self.loop else None,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "AgentSpec":
        return cls(
            algorithm=Spec.from_dict(data["algorithm"]),
            policy=Spec.from_dict(data.get("policy")),
            optimizer=Spec.from_dict(data.get("optimizer")),
            loop=Spec.from_dict(data.get("loop")),
        )


def _canonical_json(data: dict) -> str:
    """Key-sorted, whitespace-free JSON, so an identical config always hashes identically."""
    return json.dumps(data, sort_keys=True, separators=(",", ":"))


def _hash(data: dict) -> str:
    return hashlib.sha256(_canonical_json(data).encode()).hexdigest()[:12]


COMPARISON_LEVELS = ("benchmark", "task", "environment", "full")
"""Increasingly strict definitions of "the same run setup" - see this module's docstring."""


def _unwrap_manifest(data: dict) -> dict:
    """The config inside `data`, whether that is a bare config or a run's whole manifest.

    A manifest is `{"config": ..., "metrics": ..., "fingerprint": ...}`; a bare config has
    "name" at the top. Detected by shape rather than by filename, so a caller can pass either
    without saying which it has.
    """
    if "name" not in data and isinstance(data.get("config"), dict):
        return data["config"]
    return data


@dataclass(frozen=True)
class ExperimentConfig:
    """One fully-specified experiment: the environment it runs in and the agent under test."""

    name: str
    environment: EnvironmentSpec
    agent: AgentSpec
    seed: int = 42

    # ------------------------------------------------------------------ hashes

    def benchmark_hash(self) -> str:
        """Identifies the design alone - netlist contents, grid, macro order and budget."""
        return _hash(self.environment.benchmark.identity())

    def task_hash(self) -> str:
        """Identifies what is being optimized, on what, under what rules, for how long.

        The level a state-representation comparison asserts: those runs differ in `state` by
        design, and this is the invariant they must still share.
        """
        return _hash(self.environment.task_identity())

    def environment_hash(self) -> str:
        """Identifies the environment - the task plus the observation.

        Two runs comparing two AGENTS are comparable if and only if these match. Assert on it
        before putting two numbers in the same table; it is far more reliable than reading two
        scripts and believing they agree.
        """
        return _hash(self.environment.identity())

    def full_hash(self) -> str:
        """Identifies the exact run, agent and seed included."""
        return _hash({"environment": self.environment.identity(),
                      "agent": self.agent.identity(), "seed": self.seed})

    def hash_at(self, level: str) -> str:
        """The hash for a named comparison level, so callers can parameterize over strictness."""
        if level not in COMPARISON_LEVELS:
            raise ValueError(
                f"unknown comparison level {level!r}; choose one of {', '.join(COMPARISON_LEVELS)}"
            )
        return {
            "benchmark": self.benchmark_hash, "task": self.task_hash,
            "environment": self.environment_hash, "full": self.full_hash,
        }[level]()

    # ------------------------------------------------------------ serialization

    def with_benchmark(self, benchmark: BenchmarkSpec) -> "ExperimentConfig":
        """This config with a different BenchmarkSpec - how a resolved digest gets folded in."""
        return replace(self, environment=replace(self.environment, benchmark=benchmark))

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "seed": self.seed,
            "environment": self.environment.to_dict(),
            "agent": self.agent.to_dict(),
            # Derived, and written out so a results file carries them without anyone
            # needing to re-derive them or import this module to read it.
            "benchmark_hash": self.benchmark_hash(),
            "task_hash": self.task_hash(),
            "environment_hash": self.environment_hash(),
            "full_hash": self.full_hash(),
        }

    @classmethod
    def from_dict(cls, data: dict) -> "ExperimentConfig":
        """Reads a config dict, or a whole manifest that carries one under "config".

        The manifest is the only file a run actually writes that holds a config, so it is the
        file someone will reach for when a script asks for one. Refusing it with `KeyError:
        'name'` made every documented `--config` path - including the attributable-PPA path in
        `scripts/validate_design.py` - unusable against real run output.
        """
        data = _unwrap_manifest(data)
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


_LEVEL_IDENTITY = {
    "benchmark": lambda config: config.environment.benchmark.identity(),
    "task": lambda config: config.environment.task_identity(),
    "environment": lambda config: config.environment.identity(),
    "full": lambda config: {"environment": config.environment.identity(),
                            "agent": config.agent.identity(), "seed": config.seed},
}


def _benchmark_identity(identity: dict) -> dict | None:
    """The benchmark identity nested inside a level identity, whatever level produced it.

    Each level wraps the one below it, so the benchmark sits at a different depth in each: the
    "benchmark" level IS one, "task"/"environment" nest it under `benchmark`, and "full" nests
    that whole environment under `environment` first. The wrappers are walked by name, in order,
    rather than guessed at: the previous version looked for a `benchmark` key at the top of a
    full identity, where the only keys are `environment`/`agent`/`seed`, so at that level the
    digest was neither found (the fallback below never fired) nor stripped - and comparing an
    unresolved config against a built one reported "different designs", the exact false alarm
    this machinery exists to prevent.
    """
    node = identity
    if "environment" in node:  # "full" wraps the environment identity
        node = node["environment"]
    if "benchmark" in node:  # "task" and "environment" wrap the benchmark identity
        node = node["benchmark"]
    return node if "netlist_digest" in node else None


def _digest_of(identity: dict) -> str | None:
    """The netlist digest inside a level identity, wherever that level nests it."""
    benchmark = _benchmark_identity(identity)
    return benchmark.get("netlist_digest") if benchmark is not None else None


def _strip_digest(identity: dict) -> dict:
    """The same identity with the netlist digest removed, for a comparison that can't use it."""
    import copy

    stripped = copy.deepcopy(identity)
    benchmark = _benchmark_identity(stripped)
    if benchmark is not None:
        benchmark.pop("netlist_digest", None)
    return stripped


def _comparison_views(configs, level: str) -> list[dict]:
    """Each config's identity at `level`, with digests dropped if any config lacks one.

    A config that has not been through `build()` has no netlist digest yet - it has not looked at
    the design, so it makes no claim about its contents. Comparing that None against a resolved
    config's real digest would report "different designs" for two configs that may well name the
    same one, which is a false alarm in the one mechanism that has to be trustworthy. So the
    digest is used only when EVERY config in the comparison carries one; otherwise the
    comparison falls back to what they all do assert - design name, grid, order and budget.
    """
    identity = _LEVEL_IDENTITY[level]
    identities = [identity(config) for config in configs]
    if any(_digest_of(one) is None for one in identities):
        return [_strip_digest(one) for one in identities]
    return identities


def assert_comparable(*configs: ExperimentConfig, level: str = "environment") -> None:
    """Raises unless every config matches at `level` - the precondition for comparing results.

    Use this wherever two runs' numbers are about to be put side by side. It is the mechanical
    form of "every experiment used exactly the same benchmark, reward, constraints, and compute
    budget", and it catches the case the literature keeps getting wrong: two methods compared
    under quietly different setups.

    `level` says which invariant the comparison actually claims. Default "environment" - two
    agents, everything else held fixed. Drop to "task" for a state-representation study, whose
    whole point is that the observation differs; "benchmark" only asserts the same design.
    Deliberately explicit: a looser level is a claim about what the comparison means, so it
    should be visible at the call site rather than inferred.
    """
    if level not in COMPARISON_LEVELS:
        raise ValueError(
            f"unknown comparison level {level!r}; choose one of {', '.join(COMPARISON_LEVELS)}"
        )
    if len(configs) < 2:
        return
    views = _comparison_views(configs, level)
    reference, reference_view = configs[0], views[0]
    for other, other_view in zip(configs[1:], views[1:]):
        if other_view != reference_view:
            differences = _describe_differences(reference_view, other_view)
            raise ValueError(
                f"experiments {reference.name!r} and {other.name!r} do not share a {level} "
                f"({reference.hash_at(level)} vs {other.hash_at(level)}), so their results are "
                f"not comparable. Differences:\n  " + "\n  ".join(differences)
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
