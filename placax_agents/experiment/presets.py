"""The shipped experiment configurations, as data.

These used to live as hardcoded choices scattered through two training scripts, which is why the
two differed on thirteen axes at once with nothing recording it. They are configs now, so the
differences between them are readable in one place, and any of them can be overridden from a CLI
flag or a JSON file without editing code.

Both presets reproduce their original script's behavior exactly - see
tests/test_experiment_presets.py, which pins every hyperparameter against the values the scripts
used, so this refactor cannot have silently changed what a run does.
"""
import pathlib

from placax_agents.experiment.budget import Budget
from placax_agents.experiment.config import (
    AgentSpec, BenchmarkSpec, EnvironmentSpec, ExperimentConfig, Spec,
)

MASKPLACE_GRID = 224
"""MaskPlace's own grid resolution."""

MASKPLACE_MACRO_BUDGET = 128
"""MaskPlace's own --pnm default: place only its 128 most important macros, not the whole netlist."""

MASKPLACE_N_EPISODES = 10
"""MaskPlace's own buffer size: 10 episodes collected (in parallel, via vmap) per PPO update."""

WIREMASK_MARGIN = 1.0
"""MaskPlace's own --soft_coefficient default."""


def maskplace(
    benchmark_dir: pathlib.Path | str,
    *,
    seed: int = 42,
    macro_budget: int | None = None,
    budget: Budget | None = None,
    n_episodes: int = MASKPLACE_N_EPISODES,
    entropy_coef: float = 0.0,
    regularity_weight: float = 0.0,
    regularity_mode: str = "corner",
) -> ExperimentConfig:
    """MaskPlace reproduced: 224 grid, connectivity order, dense HPWL/200 reward, wiremask
    observation and action mask, ResNet coarse/fine policy, buffered PPO.

    PPO's hyperparameters are PPO2.py's own: no entropy bonus and raw, unnormalized advantages
    and returns (its `advantage = (target_v - critic_output).detach()` and
    `smooth_l1_loss(critic_output, target_v)` use neither).

    History worth keeping: this config genuinely used to saturate - logits reaching exact 0/1
    probabilities by iteration ~4-6, confirmed by checkpoint inspection - and normalization was
    briefly forced on as a workaround. The root cause was unrelated to normalization: the ResNet
    backbone ran with a frozen, eval-mode BatchNorm, so as the surrounding conv weights fine-tuned
    nothing corrected the growing mismatch against stale statistics, compounding into a ~1e6-scale
    activation blowup through the backbone's 8 blocks within ~10-16 iterations. Fixed by switching
    the backbone to live train-mode statistics, matching PPO2.py, which never calls .eval() on its
    resnet either. With that fixed, these literal reference values are what runs.

    regularity_weight > 0 adds EXPlace's periphery term on top - the one EXPlace change that needs
    none of the preprocessed clustering/dataflow data it ships separately.
    """
    return ExperimentConfig(
        name=f"maskplace-{pathlib.Path(benchmark_dir).name}",
        seed=seed,
        environment=EnvironmentSpec(
            benchmark=BenchmarkSpec(
                benchmark_dir=str(benchmark_dir),
                grid=MASKPLACE_GRID,
                macro_budget=macro_budget,
                order=Spec("connectivity_maskplace"),
            ),
            reward=Spec("maskplace", {
                "regularity_weight": regularity_weight,
                "regularity_mode": regularity_mode,
            }),
            state=Spec("wiremask", {"lookahead": 2}),
            action_mask=Spec("wiremask_quality", {"margin": WIREMASK_MARGIN}),
            budget=budget or Budget(iterations=100),
        ),
        agent=AgentSpec(
            policy=Spec("resnet_coarse_fine", {"critic_style": "step_embedding"}),
            algorithm=Spec("ppo", {
                "gamma": 0.95, "lam": 1.0, "clip_eps": 0.2, "value_coef": 0.5,
                "entropy_coef": entropy_coef, "value_loss": "huber",
                "normalize_advantages": False, "normalize_returns": False,
            }),
            optimizer=Spec("maskplace_split", {"learning_rate": 2.5e-3, "max_grad_norm": 0.5}),
            loop=Spec("buffered", {
                "n_episodes": n_episodes, "ppo_epochs": 10, "batch_size": 64,
            }),
        ),
    )


def training(
    benchmark_dir: pathlib.Path | str,
    *,
    seed: int = 0,
    budget: Budget | None = None,
    n_envs: int = 1,
) -> ExperimentConfig:
    """The plain-CNN baseline: 64 grid, alphabetical order, sparse terminal -HPWL reward, canvas-
    only observation, no extra action mask, textbook PPO defaults, one episode per update.

    Deliberately a weak baseline rather than a tuned competitor - it exists so that a change to
    any single axis can be measured against something simple, which is only meaningful now that
    both presets can be pinned to the same environment and budget.
    """
    loop = Spec("sequential") if n_envs <= 1 else Spec("parallel", {"n_envs": n_envs})
    return ExperimentConfig(
        name=f"training-{pathlib.Path(benchmark_dir).name}",
        seed=seed,
        environment=EnvironmentSpec(
            benchmark=BenchmarkSpec(
                benchmark_dir=str(benchmark_dir),
                grid=64,
                macro_budget=None,
                order=Spec("alphabetical"),
            ),
            reward=Spec("hpwl", {"dense": False, "reward_scale": 1.0}),
            state=Spec("canvas", {"lookahead": 1}),
            action_mask=None,
            budget=budget or Budget(iterations=100),
        ),
        agent=AgentSpec(
            policy=Spec("cnn", {"features": 16, "num_conv_layers": 2}),
            algorithm=Spec("ppo", {
                "gamma": 0.99, "lam": 0.95, "clip_eps": 0.2, "value_coef": 0.5,
                "entropy_coef": 0.01, "value_loss": "mse",
                "normalize_advantages": True, "normalize_returns": True,
            }),
            optimizer=Spec("adam", {"learning_rate": 3e-4}),
            loop=loop,
        ),
    )


PRESETS = {"maskplace": maskplace, "training": training}
"""preset name -> builder(benchmark_dir, **overrides) -> ExperimentConfig."""

OUTPUT_SUBDIRS = {"maskplace": "output_maskplace", "training": "output"}
"""Where each preset's runs write by default, under the benchmark directory."""


def build_preset(name: str, benchmark_dir: pathlib.Path | str, **overrides) -> ExperimentConfig:
    """Looks up a preset by name, with a readable error listing what is registered."""
    if name not in PRESETS:
        raise KeyError(f"unknown preset {name!r}; registered: {', '.join(sorted(PRESETS))}")
    return PRESETS[name](benchmark_dir, **overrides)
