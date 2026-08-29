from placax.extras.render import render  # noqa: F401  must precede jax imports
from placax.types import EnvParams  # noqa: F401
from placax_agents.policy.action import (  # noqa: F401
    legal_action_logits,
    make_wiremask_quality_illegal,
    sample_action,
)

import jax
import jax.numpy as jnp
from jax import random


def test_legal_action_logits_masks_occupied_and_out_of_bounds() -> None:
    params = EnvParams(grid=4, n_macros=2)
    positions = jnp.array([[0, 0], [-1, -1]])
    sizes_array = jnp.array([[1.0, 1.0], [1.0, 1.0]])
    occupied = render(positions, sizes_array, params.grid)
    logits = jnp.zeros((4, 4))

    masked = legal_action_logits(logits, occupied, params, macro_size=(1, 1))
    assert masked[0, 0] == -jnp.inf  # occupied
    assert masked[1, 1] == 0.0  # legal, unmasked


def test_legal_action_logits_masks_footprint_that_would_overflow() -> None:
    params = EnvParams(grid=4, n_macros=1)
    occupied = jnp.zeros((4, 4), dtype=bool)
    logits = jnp.zeros((4, 4))

    masked = legal_action_logits(logits, occupied, params, macro_size=(2, 2))
    assert masked[3, 3] == -jnp.inf  # a 2x2 macro can't start at the last row/col
    assert masked[0, 0] == 0.0  # fits fine


def test_legal_action_logits_protects_a_large_macros_full_footprint() -> None:
    # Regression test: an earlier version derived occupancy from
    # compute_occupied(), which only marks each placed macro's single
    # reference cell, not its real size - a 5x5 macro's own interior was
    # left completely unmasked, a real, confirmed overlap bug.
    params = EnvParams(grid=8, n_macros=2)
    positions = jnp.array([[1, 1], [-1, -1]])
    sizes_array = jnp.array([[5.0, 5.0], [1.0, 1.0]])  # macro 0 is a real 5x5, occupies [1:6,1:6]
    occupied = render(positions, sizes_array, params.grid)
    logits = jnp.zeros((8, 8))

    masked = legal_action_logits(logits, occupied, params, macro_size=(1, 1))
    assert masked[4, 4] == -jnp.inf  # inside macro 0's real footprint, not just its corner
    assert masked[7, 7] == 0.0  # genuinely outside the footprint, still legal


def test_legal_action_logits_falls_back_when_genuinely_no_room() -> None:
    # Regression test: a real 543-macro rollout on a 64x64 grid hit a
    # macro with zero legal cells anywhere - masking then produced
    # all -inf logits, and log_softmax(all -inf) is NaN. Masking must
    # be skipped entirely in this case, not applied and left broken.
    params = EnvParams(grid=4)
    occupied = jnp.ones((4, 4), dtype=bool)  # every cell occupied, genuinely no room
    logits = jnp.zeros((4, 4))

    masked = legal_action_logits(logits, occupied, params, macro_size=(1, 1))
    assert not (masked == -jnp.inf).any()
    assert jnp.isfinite(jax.nn.log_softmax(masked.ravel())).all()


def test_legal_action_logits_composes_with_an_extra_illegal_mask() -> None:
    # A wirelength-quality cutoff (or anything else) can restrict actions
    # beyond bare legality - generic, not tied to any one scoring signal.
    params = EnvParams(grid=4, n_macros=1)
    occupied = jnp.zeros((4, 4), dtype=bool)
    logits = jnp.zeros((4, 4))
    extra_illegal = jnp.zeros((4, 4), dtype=bool).at[2, 2].set(True)

    masked = legal_action_logits(logits, occupied, params, macro_size=(1, 1), extra_illegal=extra_illegal)
    assert masked[2, 2] == -jnp.inf  # illegal only via extra_illegal, not occupancy/boundary
    assert masked[0, 0] == 0.0


def test_legal_action_logits_extra_illegal_defaults_to_no_effect() -> None:
    params = EnvParams(grid=4, n_macros=1)
    occupied = jnp.zeros((4, 4), dtype=bool)
    logits = jnp.zeros((4, 4))
    assert (legal_action_logits(logits, occupied, params, (1, 1)) == 0.0).all()


def test_make_wiremask_quality_illegal_flags_cells_above_the_margin() -> None:
    wiremask = jnp.array([[0.0, 5.0], [2.0, 10.0]])
    obs = {"wiremask": wiremask, "canvas": jnp.zeros((2, 2), dtype=bool), "current_macro_size": jnp.array([1.0, 1.0])}
    extra_illegal_fn = make_wiremask_quality_illegal(margin=0.3, cell_size=1.0)
    illegal = extra_illegal_fn(obs)
    # Matches MaskPlace's own PPO2.py/place_env.py: normalized by its own max (10.0) to
    # [[0, 0.5], [0.2, 1.0]] before the margin cutoff, min 0.0 + margin 0.3 -> cutoff 0.3.
    assert illegal.tolist() == [[False, True], [False, True]]


def test_make_wiremask_quality_illegal_uses_a_custom_key() -> None:
    obs = {
        "score": jnp.array([[0.0, 1.0]]),
        "canvas": jnp.zeros((1, 2), dtype=bool),
        "current_macro_size": jnp.array([1.0, 1.0]),
    }
    extra_illegal_fn = make_wiremask_quality_illegal(margin=0.5, cell_size=1.0, wiremask_key="score")
    illegal = extra_illegal_fn(obs)
    assert illegal.tolist() == [[False, True]]


def test_make_wiremask_quality_illegal_composes_with_legal_action_logits() -> None:
    params = EnvParams(grid=2, n_macros=1)
    occupied = jnp.zeros((2, 2), dtype=bool)
    logits = jnp.zeros((2, 2))
    obs = {
        "wiremask": jnp.array([[0.0, 5.0], [0.0, 0.0]]),
        "canvas": occupied,
        "current_macro_size": jnp.array([1.0, 1.0]),
    }
    extra_illegal_fn = make_wiremask_quality_illegal(margin=0.5, cell_size=1.0)

    masked = legal_action_logits(logits, occupied, params, (1, 1), extra_illegal_fn(obs))
    assert masked[0, 1] == -jnp.inf  # illegal via the wiremask cutoff, not occupancy/boundary
    assert masked[0, 0] == 0.0


def test_make_wiremask_quality_illegal_excludes_occupied_cells_from_the_minimum() -> None:
    # Regression test for a confirmed discrepancy against MaskPlace's reference (PPO2.py's
    # Actor.forward): the wiremask's cheapest cell (0.0) sits exactly on an already-occupied
    # macro, cell (1, 1). The reference never lets an occupied cell win the minimum used to set
    # the quality threshold (it adds a large constant to occupied cells before taking `.min()`);
    # without that exclusion, the threshold comes out lower than it should and wrongly excludes
    # cells the reference would still accept.
    wiremask = jnp.array(
        [[5.0, 4.0, 3.0, 4.0, 5.0],
         [4.0, 0.0, 1.0, 2.0, 3.0],  # (1, 1) is occupied but has the lowest raw value, 0.0
         [5.0, 4.0, 3.0, 4.0, 5.0]]
    )
    canvas = jnp.zeros((3, 5), dtype=bool).at[1, 1].set(True)  # occupied at (1, 1) only
    obs = {"wiremask": wiremask, "canvas": canvas, "current_macro_size": jnp.array([1.0, 1.0])}
    extra_illegal_fn = make_wiremask_quality_illegal(margin=0.3, cell_size=1.0)

    illegal = extra_illegal_fn(obs)
    # scale = wiremask.max() = 5.0; the minimum used for the threshold must come from the
    # cheapest LEGAL cell, (1, 2) at normalized 0.2, not the occupied cell's 0.0 - giving
    # threshold = 0.2 + 0.3 = 0.5. Without the fix, the threshold would be 0.0 + 0.3 = 0.3
    # instead, wrongly flagging (1, 3) (normalized 0.4) as illegal even though 0.4 <= 0.5.
    assert illegal[1, 3].item() is False  # normalized 0.4: legal under the correct 0.5 threshold
    assert illegal[0, 0].item() is True  # normalized 1.0: illegal under either threshold
    assert illegal[1, 1].item() is False  # occupied cell itself: a separate occupancy check handles it


def test_sample_action_only_picks_legal_cells() -> None:
    logits = jnp.full((4, 4), -jnp.inf)
    logits = logits.at[2, 3].set(0.0)  # exactly one legal cell

    for seed in range(10):
        action = sample_action(random.PRNGKey(seed), logits)
        assert action.tolist() == [2, 3]
