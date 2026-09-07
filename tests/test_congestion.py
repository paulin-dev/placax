"""RUDY congestion: the second objective the reward comparison needs, and its properties.

docs §12 lists "HPWL vs. HPWL+congestion vs. a learned predictor, agent held fixed" as a headline
experiment. Only wirelength existed, so that comparison could not be run from a config at all -
the reward axis was swappable in principle and single-valued in practice.

These tests pin the properties that make the estimate meaningful rather than merely present: a
net's demand actually lands inside its own bounding box, spreading a net over more area lowers
its density, overflow only counts what exceeds capacity, and the combined reward really does
trade wirelength against congestion.
"""
import dataclasses
import pathlib

from placax.core import replay, reset  # noqa: F401  must precede jax imports
from placax.extras.congestion import (
    congestion_overflow, net_bounding_boxes, peak_congestion, rudy_density,
)
from placax.types import EnvParams
from placax_agents.experiment.budget import Budget
from placax_agents.experiment.build import build
from placax_agents.experiment.config import Spec
from placax_agents.experiment.presets import training
from placax_agents.training.reward import make_hpwl_congestion_reward, make_scaled_hpwl_reward

import jax.numpy as jnp
import pytest

PARAMS = EnvParams(grid=8, n_macros=2)


def _one_net(positions):
    """A single 2-pin net over two macros, in the padded-array form the reward code expects."""
    pin_idx = jnp.array([[0, 1]])
    pin_offset = jnp.zeros((1, 2, 2))
    valid = jnp.ones((1, 2), dtype=bool)
    return positions, pin_idx, pin_offset, valid


def test_demand_lands_inside_the_nets_own_bounding_box() -> None:
    # The whole estimate is "a net's wire is somewhere in its bounding box". If demand appears
    # outside it, the map is measuring something else.
    positions, pin_idx, pin_offset, valid = _one_net(jnp.array([[2.0, 2.0], [5.0, 4.0]]))
    density = rudy_density(positions, pin_idx, pin_offset, valid, PARAMS, cell_size=1.0)

    assert float(density[2:5, 2:4].sum()) > 0.0
    # Everything outside the box is untouched.
    outside = float(density.sum()) - float(density[2:5, 2:4].sum())
    assert outside == pytest.approx(0.0, abs=1e-6)


def test_spreading_one_net_further_lowers_its_density() -> None:
    # RUDY's actual claim: the same wire spread over more area congests less per bin.
    tight, pin_idx, pin_offset, valid = _one_net(jnp.array([[1.0, 1.0], [2.0, 2.0]]))
    spread = jnp.array([[0.0, 0.0], [7.0, 7.0]])

    tight_peak = float(peak_congestion(
        rudy_density(tight, pin_idx, pin_offset, valid, PARAMS, cell_size=1.0)))
    spread_peak = float(peak_congestion(
        rudy_density(spread, pin_idx, pin_offset, valid, PARAMS, cell_size=1.0)))
    assert spread_peak < tight_peak


def test_more_nets_over_one_region_congest_it_more() -> None:
    positions = jnp.array([[1.0, 1.0], [3.0, 3.0]])
    pin_offset = jnp.zeros((3, 2, 2))
    valid = jnp.ones((3, 2), dtype=bool)
    one = rudy_density(positions, jnp.array([[0, 1]]), pin_offset[:1], valid[:1], PARAMS, 1.0)
    three = rudy_density(
        positions, jnp.array([[0, 1], [0, 1], [0, 1]]), pin_offset, valid, PARAMS, 1.0
    )
    assert float(three.max()) == pytest.approx(3 * float(one.max()), rel=1e-5)


def test_a_net_with_fewer_than_two_placed_pins_demands_nothing() -> None:
    # A partial placement must not be charged for wire between a macro and a macro that isn't
    # anywhere yet - the same rule hpwl() follows for the same reason.
    positions, pin_idx, pin_offset, valid = _one_net(jnp.array([[2.0, 2.0], [5.0, 4.0]]))
    placed = jnp.array([True, False])
    density = rudy_density(positions, pin_idx, pin_offset, valid, PARAMS, 1.0, placed_mask=placed)
    assert float(density.sum()) == pytest.approx(0.0, abs=1e-6)

    _lo, _hi, has_box = net_bounding_boxes(positions, pin_idx, pin_offset, valid, placed)
    assert not bool(has_box[0])


def test_overflow_counts_only_what_exceeds_capacity() -> None:
    density = jnp.array([[0.5, 1.0], [1.5, 3.0]])
    assert float(congestion_overflow(density, capacity=1.0)) == pytest.approx(0.5 + 2.0)
    assert float(congestion_overflow(density, capacity=3.0)) == pytest.approx(0.0)


def test_the_combined_reward_actually_trades_wirelength_against_congestion() -> None:
    """With congestion_weight=0 it must reduce EXACTLY to the HPWL reward, and above 0 it must
    penalize a layout that congests - otherwise the axis is decorative."""
    positions = jnp.array([[1, 1], [2, 2]])
    pin_idx, pin_offset = jnp.array([[0, 1]]), jnp.zeros((1, 2, 2))
    valid = jnp.ones((1, 2), dtype=bool)
    sizes = jnp.array([[1.0, 1.0], [1.0, 1.0]])
    placed = jnp.array([True, True])
    unplaced = jnp.array([False, False])
    args = (jnp.zeros_like(positions) - 1, positions, unplaced, placed)

    bare = make_scaled_hpwl_reward(pin_idx, pin_offset, valid, sizes, 1.0)
    off = make_hpwl_congestion_reward(
        pin_idx, pin_offset, valid, sizes, 1.0, PARAMS, congestion_weight=0.0
    )
    assert float(off(*args)) == pytest.approx(float(bare(*args)), rel=1e-6)

    on = make_hpwl_congestion_reward(
        pin_idx, pin_offset, valid, sizes, 1.0, PARAMS, congestion_weight=100.0, capacity=0.0
    )
    # capacity=0 makes every bin of demand count as overflow, so the penalty must bite.
    assert float(on(*args)) < float(off(*args))


def test_the_dense_form_telescopes_to_the_sparse_one(tmp_path: pathlib.Path) -> None:
    """Sparse and dense must be two credit-assignment choices over one quantity, not two rewards.

    This is the property the HPWL reward already had and the one a new objective is easiest to
    get wrong, so it is checked by replaying a real placement through the kernel both ways.
    """
    benchmark_dir = tmp_path / "bench"
    benchmark_dir.mkdir()
    (benchmark_dir / "s.aux").write_text("RowBasedPlacement : s.nodes s.nets s.wts s.pl s.scl\n")
    (benchmark_dir / "s.nodes").write_text(
        "UCLA nodes 1.0\nNumNodes : 3\nNumTerminals : 3\n"
        "a 2 2 terminal\nb 2 2 terminal\nc 2 2 terminal\n"
    )
    (benchmark_dir / "s.nets").write_text(
        "UCLA nets 1.0\nNumNets : 2\nNumPins : 4\n"
        "NetDegree : 2 n0\n\ta I : 0.0 0.0\n\tb O : 0.0 0.0\n"
        "NetDegree : 2 n1\n\tb I : 0.0 0.0\n\tc O : 0.0 0.0\n"
    )

    def _config(dense: bool):
        config = training(benchmark_dir, budget=Budget(iterations=1))
        return dataclasses.replace(config, environment=dataclasses.replace(
            config.environment,
            benchmark=dataclasses.replace(config.environment.benchmark, grid=8),
            reward=Spec("hpwl_congestion", {"congestion_weight": 2.0, "dense": dense}),
        ))

    sparse_built = build(_config(dense=False))
    dense_built = build(_config(dense=True))
    positions = sparse_built.agent.best_positions(
        sparse_built.agent.init(__import__("jax").random.PRNGKey(0))
    )
    params = sparse_built.benchmark.params
    sparse_total = float(replay(positions, sparse_built.benchmark.reward_fn, params))
    dense_total = float(replay(positions, dense_built.benchmark.reward_fn, params))
    assert dense_total == pytest.approx(sparse_total, rel=1e-4)


def test_the_congestion_reward_is_selectable_from_a_config(tmp_path: pathlib.Path) -> None:
    benchmark_dir = tmp_path / "bench"
    benchmark_dir.mkdir()
    (benchmark_dir / "s.aux").write_text("RowBasedPlacement : s.nodes s.nets s.wts s.pl s.scl\n")
    (benchmark_dir / "s.nodes").write_text(
        "UCLA nodes 1.0\nNumNodes : 2\nNumTerminals : 2\na 2 2 terminal\nb 2 2 terminal\n"
    )
    (benchmark_dir / "s.nets").write_text(
        "UCLA nets 1.0\nNumNets : 1\nNumPins : 2\n"
        "NetDegree : 2 n0\n\ta I : 0.0 0.0\n\tb O : 0.0 0.0\n"
    )
    config = training(benchmark_dir, budget=Budget(iterations=1))
    with_congestion = dataclasses.replace(config, environment=dataclasses.replace(
        config.environment,
        benchmark=dataclasses.replace(config.environment.benchmark, grid=8),
        reward=Spec("hpwl_congestion", {"congestion_weight": 1.0}),
    ))
    # Swapping the objective is a different task, which is what makes the comparison meaningful.
    assert with_congestion.task_hash() != config.task_hash()
    assert build(with_congestion).benchmark.reward_fn is not None
