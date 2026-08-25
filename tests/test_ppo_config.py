from placax_agents.training.algorithm.config import PPOConfig


def test_ppo_config_defaults_match_prior_hardcoded_values() -> None:
    # Regression test: these were the values silently hardcoded inside
    # compute_gae/ppo_loss before this config existed - the defaults
    # here must match, or every existing training run's behavior would
    # silently change.
    config = PPOConfig()
    assert config.gamma == 0.99
    assert config.lam == 0.95
    assert config.clip_eps == 0.2
    assert config.value_coef == 0.5
    assert config.entropy_coef == 0.01


def test_ppo_config_is_hashable() -> None:
    # Required for use as a JAX static jit argument.
    assert hash(PPOConfig()) == hash(PPOConfig())
    assert hash(PPOConfig(clip_eps=0.3)) != hash(PPOConfig())


def test_ppo_config_overrides_work() -> None:
    config = PPOConfig(gamma=0.9, entropy_coef=0.1)
    assert config.gamma == 0.9
    assert config.entropy_coef == 0.1
    assert config.lam == 0.95  # untouched fields keep their default
