"""Smoke tests for the decentralized macro agents - the checks that would otherwise be a wasted day.

Deliberately small and fast: one real design (adaptec1) truncated to a handful of macros on a
coarse grid, so the whole file runs in seconds and can be re-run after every change. What is being
checked is the wiring, not the science:

  * the transition cannot produce an off-canvas placement,
  * the objective's gradient actually reaches (nearly) every macro - the thing the whole method
    depends on, and the thing that was measured rather than assumed for every other gradient
    decision in this project,
  * each view level is a strict extension of the one below it,
  * both entry points run end to end and write a readable run directory.
"""
import json
import pathlib

import pytest

from placax import _device  # noqa: F401  must precede jax imports

import jax
import jax.numpy as jnp
import numpy as np

from multiagent import adam, context, legalize, moves, neighbors, objective as objective_mod
from multiagent import train
from multiagent import view as view_mod
from multiagent.policy import SharedMacroPolicy, displacements

BENCHMARK = pathlib.Path("benchmarks/adaptec1")
GRID = 32
BUDGET = 8


@pytest.fixture(scope="module")
def ctx():
    if not BENCHMARK.is_dir():
        pytest.skip(f"{BENCHMARK} not present; run scripts/download_benchmarks.py")
    return context.build(BENCHMARK, grid=GRID, macro_budget=BUDGET, canvas="die", k_neighbors=3)


@pytest.fixture(scope="module")
def objective(ctx):
    return objective_mod.make(ctx)


def test_warm_start_is_a_legal_complete_placement(ctx):
    assert ctx.warm_start.shape == (BUDGET, 2)
    measured = ctx.legality_of(ctx.warm_start)
    assert measured["n_unplaced"] == 0
    assert measured["is_legal"], measured


def test_bounds_keep_every_macro_on_the_canvas(ctx):
    # The bound is on the macro's BODY, not its corner: corner + footprint must fit.
    assert jnp.all(ctx.hi + ctx.sizes_grid <= ctx.canvas + 1e-4)
    assert jnp.all(ctx.warm_start >= ctx.lo) and jnp.all(ctx.warm_start <= ctx.hi)


def test_connection_weights_are_symmetric_and_diagonal_free(ctx):
    weights = ctx.connection_weights
    assert np.allclose(weights, weights.T)
    assert np.allclose(np.diag(weights), 0.0)
    assert weights.max() > 0.0, "no macro pair shares a net; the m1 view would carry no signal"


def test_top_k_neighbours_are_sorted_strongest_first(ctx):
    picked = ctx.connection_weights[np.arange(ctx.n_macros)[:, None], np.asarray(ctx.neighbor_idx)]
    assert np.all(np.diff(picked, axis=1) <= 1e-6)
    assert np.all(np.asarray(ctx.neighbor_weight) <= 1.0 + 1e-6)
    # A slot flagged invalid must carry no weight, and vice versa.
    assert np.array_equal(np.asarray(ctx.neighbor_valid), picked > 0.0)


def test_top_k_handles_a_macro_with_no_connections():
    weights = np.zeros((3, 3), dtype=np.float32)
    weights[0, 1] = weights[1, 0] = 1.0
    idx, weight, valid = neighbors.top_k_neighbors(weights, k=2)
    assert idx.shape == (3, 2) and weight.shape == (3, 2)
    assert not valid[2].any(), "macro 2 is wired to nothing, so every slot must be invalid"


@pytest.mark.parametrize("level,expected", [("m0", 8), ("m1", 8 + 6 * 3), ("m2", 8 + 6 * 3 + 3)])
def test_view_feature_counts(ctx, objective, level, expected):
    view_fn, n_features = view_mod.make(ctx, level)
    assert n_features == expected
    obs = view_fn(ctx.warm_start, objective.parts(ctx.warm_start), jnp.float32(0.0))
    assert obs.shape == (ctx.n_macros, expected)
    assert jnp.all(jnp.isfinite(obs))


def test_all_partners_view_extends_m1_and_is_finite(ctx, objective):
    parts = objective.parts(ctx.warm_start)
    m1 = view_mod.make(ctx, "m1")[0](ctx.warm_start, parts, jnp.float32(0.0))
    view_fn, n_features = view_mod.make(ctx, "m1all")
    obs = view_fn(ctx.warm_start, parts, jnp.float32(0.0))
    assert obs.shape == (ctx.n_macros, n_features) == (ctx.n_macros, m1.shape[1] + 6)
    assert jnp.allclose(obs[:, : m1.shape[1]], m1)
    assert jnp.all(jnp.isfinite(obs))
    grads = jax.grad(lambda p: view_fn(p, parts, jnp.float32(0.0)).sum())(ctx.warm_start)
    assert jnp.all(jnp.isfinite(grads))


def test_overlap_term_is_zero_when_legal_and_positive_when_stacked(ctx):
    objective = objective_mod.make(ctx, overlap_weight=1.0)
    assert float(objective.parts(ctx.warm_start)["overlap_norm"]) < 1e-6
    stacked = jnp.zeros_like(ctx.warm_start)
    assert float(objective.parts(stacked)["overlap_norm"]) > 0.0
    grads = jax.grad(lambda p: objective.parts(p)["overlap_norm"])(ctx.warm_start + 0.3)
    assert jnp.all(jnp.isfinite(grads))


def test_step_bound_shrinks_from_start_to_end():
    from multiagent.policy import step_bound
    assert step_bound(jnp.float32(0.5), 1.0) == 1.0
    assert jnp.isclose(step_bound(jnp.float32(0.0), 1.0, 16.0), 16.0)
    assert jnp.isclose(step_bound(jnp.float32(1.0), 1.0, 16.0), 1.0)
    assert jnp.isclose(step_bound(jnp.float32(0.5), 1.0, 16.0), 4.0)


def test_each_view_extends_the_one_below_it(ctx, objective):
    parts = objective.parts(ctx.warm_start)
    m0 = view_mod.make(ctx, "m0")[0](ctx.warm_start, parts, jnp.float32(0.25))
    m1 = view_mod.make(ctx, "m1")[0](ctx.warm_start, parts, jnp.float32(0.25))
    m2 = view_mod.make(ctx, "m2")[0](ctx.warm_start, parts, jnp.float32(0.25))
    assert jnp.allclose(m1[:, : m0.shape[1]], m0)
    assert jnp.allclose(m2[:, : m1.shape[1]], m1)
    # m2's last three columns are the shared summary: every macro reads the same copy.
    assert jnp.allclose(m2[:, -3:], m2[0, -3:])


def test_moves_cannot_leave_the_canvas(ctx):
    # A displacement far larger than the canvas, in both directions.
    shoved = moves.apply_deltas(ctx.warm_start, jnp.full_like(ctx.warm_start, 1e3), ctx.lo, ctx.hi)
    pulled = moves.apply_deltas(ctx.warm_start, jnp.full_like(ctx.warm_start, -1e3), ctx.lo, ctx.hi)
    for placement in (shoved, pulled):
        assert jnp.all(placement >= ctx.lo) and jnp.all(placement <= ctx.hi)


def test_objective_gradient_reaches_nearly_every_macro(ctx, objective):
    """The measurement the whole method rests on - raw HPWL reaches 24% of macros, this must not."""
    gradient = jax.grad(objective.total)(ctx.warm_start)
    receiving = (jnp.abs(gradient).sum(axis=-1) > 0).sum()
    assert receiving >= ctx.n_macros - 1, f"only {receiving}/{ctx.n_macros} macros get a gradient"


def test_warm_start_normalizes_the_wirelength_to_one(ctx, objective):
    parts = objective.parts(ctx.warm_start)
    assert float(parts["wl_norm"]) == pytest.approx(1.0, abs=1e-5)


def test_rollout_costs_start_at_the_warm_start_and_are_finite(ctx, objective):
    def act(positions, _parts, _progress, _key):
        # A fixed nudge, so the transition is tested without a policy in the way.
        return jnp.full_like(positions, 0.25)

    final, costs = moves.rollout(
        ctx.warm_start, jax.random.PRNGKey(0), act, objective, ctx.lo, ctx.hi, steps=4, horizon=2
    )
    assert costs.shape == (5,)
    assert float(costs[0]) == pytest.approx(float(objective.total(ctx.warm_start)), rel=1e-5)
    assert jnp.all(jnp.isfinite(costs))
    assert not jnp.allclose(final, ctx.warm_start), "four nudges should have moved something"


def test_policy_gradient_is_finite_and_nonzero(ctx, objective):
    """One shared policy, differentiated through the episode: the update `train.py` performs."""
    view_fn, _ = view_mod.make(ctx, "m1")
    policy = SharedMacroPolicy(features=(16, 16))
    parts = objective.parts(ctx.warm_start)
    variables = policy.init(jax.random.PRNGKey(0), view_fn(ctx.warm_start, parts, jnp.float32(0.0)))

    def loss(variables):
        def act(positions, parts, progress, key):
            mean_raw, log_std = policy.apply(variables, view_fn(positions, parts, progress))
            return displacements(mean_raw, log_std, key, max_step=1.0, stochastic=True)

        _final, costs = moves.rollout(
            ctx.warm_start, jax.random.PRNGKey(1), act, objective, ctx.lo, ctx.hi,
            steps=4, horizon=2,
        )
        return costs.mean()

    value, grads = jax.value_and_grad(loss)(variables)
    flat = jnp.concatenate([leaf.ravel() for leaf in jax.tree_util.tree_leaves(grads)])
    assert jnp.isfinite(value)
    assert jnp.all(jnp.isfinite(flat))
    assert float(jnp.abs(flat).max()) > 0.0, "no gradient reached the shared policy"


def _rule(ctx, config, view="m1"):
    from multiagent.policy import make_act
    view_fn, n_features = view_mod.make(ctx, view)
    policy = SharedMacroPolicy(features=(8,))
    variables = policy.init(jax.random.PRNGKey(0), jnp.zeros((ctx.n_macros, n_features)))
    # A non-zero output layer, so the rule actually moves macros.
    variables = jax.tree_util.tree_map(lambda x: x + 0.5, variables)
    rule = make_act(policy, view_fn, config, ctx.connection_weights)
    return lambda objective, stochastic=True: rule(
        variables, ctx.warm_start, objective.parts(ctx.warm_start), jnp.float32(0.0),
        jax.random.PRNGKey(1), stochastic,
    )


def test_alignment_keeps_every_move_inside_the_step_bound(ctx, objective):
    for align in (0.0, 0.5, 1.0):
        deltas = _rule(ctx, {"max_step": 1.0, "align": align})(objective)
        assert jnp.all(jnp.abs(deltas) <= 1.0 + 1e-5)


def test_full_alignment_moves_a_macro_with_its_partners_average(ctx, objective):
    own = _rule(ctx, {"max_step": 1.0})(objective, stochastic=False)
    aligned = _rule(ctx, {"max_step": 1.0, "align": 1.0})(objective, stochastic=False)
    from multiagent.policy import alignment_matrix
    partners = alignment_matrix(ctx.connection_weights)
    connected = partners.sum(axis=1) > 0
    assert jnp.allclose(aligned[connected], (partners @ own)[connected], atol=1e-5)
    assert jnp.allclose(aligned[~connected], own[~connected], atol=1e-5)


def test_update_prob_zero_freezes_every_macro(ctx, objective):
    deltas = _rule(ctx, {"max_step": 1.0, "update_prob": 1e-9})(objective)
    assert jnp.allclose(deltas, 0.0)


def test_train_with_every_swarm_option_and_a_second_design(tmp_path, ctx):
    out = tmp_path / "swarm"
    train.main([
        f"--benchmark_dir={BENCHMARK}", f"--grid={GRID}", f"--macro_budget={BUDGET}",
        "--canvas=die", "--k_neighbors=3", "--view=m1all", "--hidden=16,16",
        "--steps=4", "--horizon=2", "--iterations=2", "--eval_every=1", f"--out={out}",
        "--align=0.7", "--update_prob=0.5", "--start_noise=2", "--overlap_weight=3",
        "--max_step_start=4", f"--extra_benchmarks={BENCHMARK}",
    ])
    lines = [json.loads(line) for line in (out / "log.jsonl").read_text().splitlines()]
    assert len(lines) == 2 and all(np.isfinite(line["loss"]) for line in lines)
    manifest = json.loads((out / "manifest.json").read_text())
    assert manifest["args"]["align"] == 0.7
    assert manifest["args"]["extra_benchmarks"] == [str(BENCHMARK)]


def test_train_entry_point_writes_a_run(tmp_path, ctx):
    out = tmp_path / "policy"
    train.main([
        f"--benchmark_dir={BENCHMARK}", f"--grid={GRID}", f"--macro_budget={BUDGET}",
        "--canvas=die", "--k_neighbors=3", "--view=m1", "--hidden=16,16",
        "--steps=4", "--horizon=2", "--iterations=2", "--eval_every=1", f"--out={out}",
    ])
    lines = [json.loads(line) for line in (out / "log.jsonl").read_text().splitlines()]
    assert len(lines) == 2
    assert lines[-1]["moves"] == 8 and lines[-1]["macro_moves"] == 8 * BUDGET
    assert "eval_real_hpwl_snapped" in lines[-1]
    manifest = json.loads((out / "manifest.json").read_text())
    assert manifest["method"] == "short_horizon"
    assert manifest["warm_start"]["is_legal"] is True
    assert json.loads((out / "summary.json").read_text())["warm_start"]


def test_a_run_directory_refuses_to_hold_a_second_run(tmp_path, ctx):
    out = tmp_path / "once"
    argv = [
        f"--benchmark_dir={BENCHMARK}", f"--grid={GRID}", f"--macro_budget={BUDGET}",
        "--canvas=die", "--steps=2", "--horizon=2", "--iterations=1", f"--out={out}",
    ]
    adam.main([f"--benchmark_dir={BENCHMARK}", f"--grid={GRID}", f"--macro_budget={BUDGET}",
               "--canvas=die", "--steps=2", "--eval_every=1", f"--out={out}"])
    with pytest.raises(SystemExit):
        train.main(argv)


def test_adam_lowers_the_objective_from_the_warm_start(tmp_path, ctx):
    out = tmp_path / "adam"
    adam.main([
        f"--benchmark_dir={BENCHMARK}", f"--grid={GRID}", f"--macro_budget={BUDGET}",
        "--canvas=die", "--steps=30", "--eval_every=10", f"--out={out}",
    ])
    lines = [json.loads(line) for line in (out / "log.jsonl").read_text().splitlines()]
    assert lines[-1]["cost"] < lines[0]["cost"], "30 Adam steps did not improve its own objective"


@pytest.mark.parametrize("mode", legalize.REPAIR_MODES)
def test_repair_leaves_an_already_legal_placement_alone(ctx, mode):
    """The greedy warm start is legal on integer cells, so the legalizer must be the identity on it.

    This is what keeps the comparison fair: the baseline's number cannot move just because every
    method is scored through the repair. It matters most for the wirelength-aware mode, which is
    capable of improving a placement on its own - it must not get the chance to on a legal one.
    """
    placed, stats = legalize.repair(ctx, ctx.warm_start, mode=mode)
    assert np.array_equal(placed, np.asarray(ctx.warm_start).astype(np.int32))
    assert stats["repair_moved_macros"] == 0
    assert stats["repair_unplaceable_macros"] == 0


@pytest.mark.parametrize("mode", legalize.REPAIR_MODES)
def test_repair_makes_a_deliberately_overlapping_placement_legal(ctx, mode):
    """Stack every macro on one cell - the worst case the continuous optimizer can hand over."""
    stacked = jnp.zeros_like(ctx.warm_start)
    assert not ctx.legality_of(stacked)["is_legal"], "the stacked placement should be illegal"

    placed, stats = legalize.repair(ctx, stacked, mode=mode)
    measured = ctx.legality_of(jnp.asarray(placed, dtype=jnp.float32))
    assert measured["is_legal"], measured
    assert stats["repair_unplaceable_macros"] == 0
    assert stats["repair_moved_macros"] > 0


def test_repaired_metrics_are_reported_under_their_own_prefix(ctx, objective):
    _placed, metrics = legalize.repair_and_report(ctx, objective, ctx.warm_start)
    assert metrics["repaired_is_legal"] is True
    assert metrics["repaired_real_hpwl_snapped"] == pytest.approx(
        objective_mod.report(ctx, objective, ctx.warm_start)["real_hpwl_snapped"], rel=1e-6
    )


def test_unknown_repair_mode_is_refused(ctx):
    with pytest.raises(ValueError, match="unknown repair mode"):
        legalize.repair(ctx, ctx.warm_start, mode="teleport")


def test_spread_leaves_a_legal_placement_exactly_where_it_is(ctx):
    spread = legalize.spread(ctx, ctx.warm_start)
    assert jnp.allclose(spread, jnp.clip(ctx.warm_start, 0.0, None), atol=1e-6)


def test_spread_reduces_overlap_on_a_jittered_placement(ctx):
    objective = objective_mod.make(ctx, overlap_weight=1.0)
    rng = np.random.default_rng(0)
    jittered = jnp.clip(ctx.warm_start + rng.standard_normal(ctx.warm_start.shape).astype(np.float32),
                        ctx.lo, ctx.hi)
    before = float(objective.parts(jittered)["overlap_norm"])
    after = float(objective.parts(legalize.spread(ctx, jittered))["overlap_norm"])
    assert before > 0 and after < before


def test_portfolio_repair_is_legal_and_no_worse_than_the_plain_one(ctx, objective):
    rng = np.random.default_rng(1)
    jittered = jnp.clip(ctx.warm_start + rng.standard_normal(ctx.warm_start.shape).astype(np.float32),
                        ctx.lo, ctx.hi)
    _, plain = legalize.repair_and_report(ctx, objective, jittered, spread_steps=0)
    _, both = legalize.repair_and_report(ctx, objective, jittered, spread_steps=100)
    assert both["repaired_is_legal"]
    assert both["repaired_real_hpwl_snapped"] <= plain["repaired_real_hpwl_snapped"] + 1e-6


def test_swap_descent_keeps_legality_and_never_lengthens_wires(ctx, objective):
    from multiagent.swap import swap_descent
    start = np.asarray(jnp.round(ctx.warm_start))
    final, _swaps = swap_descent(ctx, start)
    before = objective_mod.report(ctx, objective, jnp.asarray(start))
    after = objective_mod.report(ctx, objective, jnp.asarray(final))
    assert after["is_legal"]
    assert after["real_hpwl_snapped"] <= before["real_hpwl_snapped"] + 1e-6
