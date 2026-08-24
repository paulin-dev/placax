import pathlib

from placax.core import reset, step  # noqa: F401  must precede jax imports
from placax.netlist import load_netlist  # noqa: F401
from placax.netlist.padding import build_padded_arrays  # noqa: F401
from placax.extras.rewards import make_hpwl_reward  # noqa: F401
from placax.types import EnvParams  # noqa: F401
from placax_agents.policy.action import legal_action_logits, sample_action  # noqa: F401
from placax_agents.policy.observation import observation  # noqa: F401
from placax_agents.policy.architectures.cnn import CNNActorCritic  # noqa: F401
from placax_agents.policy.scale import compute_grid_scale, to_grid_units  # noqa: F401

import jax.numpy as jnp
import pytest
from jax import random

REAL_ADAPTEC1 = pathlib.Path("/home/claude/maskplace/maskplace/adaptec1")


@pytest.mark.skipif(not REAL_ADAPTEC1.exists(), reason="real adaptec1 benchmark not available")
def test_full_forward_pass_at_real_scale() -> None:
    """observation -> policy -> scale -> mask -> sample -> step, all
    together, on real adaptec1 (543 macros, real sizes) - the first
    time any of placax_agents runs against real data, not a toy example."""
    macro_sizes, nets = load_netlist(REAL_ADAPTEC1)
    _, sizes_array, padded_pin_idx, padded_pin_offset, valid_mask = build_padded_arrays(
        macro_sizes, nets
    )
    reward_fn = make_hpwl_reward(padded_pin_idx, padded_pin_offset, valid_mask)
    params = EnvParams(grid=64, n_macros=len(macro_sizes))
    state = reset(params)
    cell_size = compute_grid_scale(sizes_array, params.grid)

    policy = CNNActorCritic()
    key = random.PRNGKey(0)
    key, init_key = random.split(key)
    obs = observation(state, params, sizes_array)
    variables = policy.init(init_key, obs)

    logits, value = policy.apply(variables, obs)
    assert logits.shape == (params.grid, params.grid)
    assert value.shape == ()

    macro_size = to_grid_units(obs["current_macro_size"], cell_size)
    masked_logits = legal_action_logits(logits, obs["canvas"], params, macro_size)
    n_legal = int((masked_logits > -jnp.inf).sum())
    assert n_legal > 0  # the first macro should have somewhere legal to go

    key, action_key = random.split(key)
    action = sample_action(action_key, masked_logits)

    new_state, _reward, _done = step(state, action, reward_fn, params)
    assert new_state.step == 1
    assert new_state.positions[0].tolist() == action.tolist()
