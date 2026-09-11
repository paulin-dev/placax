"""The comparison table itself - the one artifact this project exists to produce.

No test imported this module at all before, which is how it came to print a single budget line
under every row while the runner had already stopped one of those rows after a single iteration.
A table that misstates compute is worse than no table, because it looks like evidence.
"""
import dataclasses
import json
import pathlib

import pytest

from placax_agents.experiment.budget import Budget  # noqa: F401  must precede jax imports
from placax_agents.experiment.config import Spec
from placax_agents.experiment.presets import build_preset, training
from scripts.compare_agents import (
    _agent_spec, _format_table, _parse_args, _with_overrides, _write_results, build_comparison,
)


def _run(hpwl: float, env_steps: int, *, seed: int = 0, gradient_steps: int = 0,
         is_legal: bool = True, overlap: float = 0.0) -> dict:
    return {
        "real_hpwl": hpwl, "reward_return": -hpwl, "is_legal": is_legal,
        "overlap_ratio": overlap, "out_of_bounds_ratio": 0.0, "n_unplaced": 0,
        "seed": seed, "env_steps": env_steps, "eval_env_steps": 0,
        "gradient_steps": gradient_steps, "iterations": 1, "full_hash": f"hash{seed}",
    }


BUDGET = Budget(env_steps=500_000)


def test_the_table_reports_what_each_agent_actually_spent() -> None:
    """A converged agent stops early, and the table has to say so.

    `greedy_wiremask` reports `converged`, so the runner stops it after one iteration - roughly
    n_macros steps out of a budget that may run to millions. Printing the budget over that row
    claimed a compute match the run itself had already declined.
    """
    table = _format_table(
        {"greedy_wiremask": [_run(1000.0, env_steps=543)],
         "ppo": [_run(900.0, env_steps=500_000, gradient_steps=1_000)]},
        BUDGET, "environment",
    )
    assert "543" in table
    assert "500,000" in table
    # The budget is named as what was OFFERED, never as what every row spent.
    assert "budget OFFERED to every run" in table
    assert "all runs: 500,000 env steps" not in table


def test_the_table_separates_sample_matching_from_compute_matching() -> None:
    # env_steps deliberately does not price gradient work, so the table has to show the number
    # that lets a reader check rather than assume it.
    table = _format_table(
        {"ppo": [_run(900.0, env_steps=500_000, gradient_steps=10_000)],
         "random_search": [_run(1200.0, env_steps=500_000)]},
        BUDGET, "environment",
    )
    assert "grad steps" in table
    assert "10,000" in table
    assert "never compute-matched" in table


def test_an_illegal_row_is_visible_beside_its_score() -> None:
    # Overlapping macros have shorter wires, so an illegal placement tops the table on HPWL alone.
    table = _format_table(
        {"cheater": [_run(10.0, env_steps=543, is_legal=False, overlap=0.31)],
         "honest": [_run(900.0, env_steps=543)]},
        BUDGET, "environment",
    )
    cheating_row = next(line for line in table.splitlines() if line.startswith("cheater"))
    assert "0/1" in cheating_row
    assert "31.00%" in cheating_row


def test_the_table_is_ranked_by_mean_hpwl() -> None:
    table = _format_table(
        {"worse": [_run(2000.0, env_steps=1)], "better": [_run(100.0, env_steps=1)]},
        BUDGET, "environment",
    )
    rows = [line.split()[0] for line in table.splitlines() if line[:1].isalpha()]
    assert rows.index("better") < rows.index("worse")


def test_spread_across_seeds_is_reported() -> None:
    table = _format_table(
        {"ppo": [_run(100.0, env_steps=1, seed=0), _run(300.0, env_steps=1, seed=1)]},
        BUDGET, "environment",
    )
    row = next(line for line in table.splitlines() if line.startswith("ppo"))
    assert "200" in row  # mean
    assert "100" in row  # best


def test_results_json_carries_every_number_the_table_shows(tmp_path: pathlib.Path) -> None:
    """The docstring has always promised the table can be regenerated; a printout is not that.

    Reconstructing a row used to mean re-running the agent. Everything the table prints is now
    written beside the hash of the environment it ran in and the full hash of each run.
    """
    reference = training("benchmarks/adaptec1", budget=BUDGET)
    results = {"ppo": [_run(900.0, env_steps=500_000, seed=1, gradient_steps=7),
                       _run(950.0, env_steps=500_000, seed=0, gradient_steps=7)]}
    path = _write_results(tmp_path / "results.json", {"adaptec1": results},
                          {"adaptec1": reference}, BUDGET, "environment",
                          reference.protocol_hash())

    data = json.loads(path.read_text())
    assert data["level"] == "environment"
    assert data["protocol_hash"] == reference.protocol_hash()
    assert data["designs"]["adaptec1"]["shared_hash"] == reference.environment_hash()
    assert data["budget"]["env_steps"] == 500_000
    assert set(data["fingerprint"]) >= {"packages", "device", "git_revision"}
    assert "real_hpwl" in data["metrics"]
    # Seeds land in a stable order, so two runs of one comparison produce comparable files.
    runs = data["designs"]["adaptec1"]["runs"]["ppo"]
    assert [run["seed"] for run in runs] == [0, 1]
    assert runs[0]["env_steps"] == 500_000


def test_a_suite_ranks_within_each_design_rather_than_averaging_across_them() -> None:
    """HPWL is not comparable between designs, so the aggregate cannot be a mean of it.

    adaptec1 and bigblue1 differ by orders of magnitude; averaging their wirelengths would let the
    larger design decide the winner on its own. Ranking inside each design and averaging the ranks
    is what the comparison actually supports.
    """
    from scripts.compare_agents import _rank_summary

    summary = _rank_summary({
        # Consistent winner, despite tiny numbers on one design and huge ones on the other.
        "small": {"good": [_run(10.0, 1)], "bad": [_run(20.0, 1)]},
        "large": {"good": [_run(1_000_000.0, 1)], "bad": [_run(2_000_000.0, 1)]},
    })
    good_row = next(line for line in summary.splitlines() if line.startswith("good"))
    bad_row = next(line for line in summary.splitlines() if line.startswith("bad"))
    assert "1.00" in good_row and "2.00" in bad_row
    assert summary.index("good") < summary.index("bad")
    assert "ACROSS 2 DESIGNS" in summary


def test_the_suite_summary_flags_an_agent_that_was_illegal_anywhere() -> None:
    # Winning on rank while producing an unrealizable placement on one design is not winning.
    from scripts.compare_agents import _rank_summary

    summary = _rank_summary({
        "one": {"cheat": [_run(1.0, 1, is_legal=False, overlap=0.4)], "honest": [_run(9.0, 1)]},
        "two": {"cheat": [_run(1.0, 1)], "honest": [_run(9.0, 1)]},
    })
    cheat_row = next(line for line in summary.splitlines() if line.startswith("cheat"))
    honest_row = next(line for line in summary.splitlines() if line.startswith("honest"))
    assert cheat_row.rstrip().endswith("NO")
    assert honest_row.rstrip().endswith("yes")


def test_a_comparison_refuses_to_be_built_across_two_environments() -> None:
    # The guarantee the whole script rests on, exercised here because nothing else covers it.
    reference = training("benchmarks/adaptec1", budget=BUDGET)
    configs = build_comparison(reference, ["ppo", "random_search"], seeds=1, population=4)
    assert len({config.environment_hash() for config in configs}) == 1

    from placax_agents.experiment.config import assert_comparable

    tampered = dataclasses.replace(configs[0], environment=dataclasses.replace(
        configs[0].environment, budget=Budget(env_steps=1_000)
    ))
    with pytest.raises(ValueError, match="not comparable"):
        assert_comparable(tampered, configs[1])


# ------------------------------------- the environment axes an agent may REQUIRE, from the CLI


def test_the_action_space_and_warm_start_are_settable_from_the_command_line() -> None:
    """Two of five shipped agents could not be reached by this script at all.

    `local_search` requires a perturbation space and a full warm start, and neither was a flag -
    so it was refused at build() whatever the command line said, and the GA's orientation arm was
    equally unreachable. They are environment axes, so they are applied to the REFERENCE and every
    row inherits them.
    """
    args = _parse_args([
        "compare_agents", "--env_steps=400", "--agents=local_search",
        "--action_space=perturbation:n_moves=128",
        "--initial_placement=greedy_wiremask_prefix:n_macros=null",
    ])
    reference = _with_overrides(build_preset("training", "b", budget=Budget(env_steps=400)), args)
    assert reference.environment.action_space == Spec("perturbation", {"n_moves": 128})
    assert reference.environment.initial_placement == Spec(
        "greedy_wiremask_prefix", {"n_macros": None}
    )
    # ...and every agent in the comparison gets the same one, which is the point of the split.
    configs = build_comparison(reference, ["local_search"], seeds=1, population=2)
    assert all(c.environment.action_space.name == "perturbation" for c in configs)


def test_a_component_is_parsed_from_its_command_line_spelling() -> None:
    assert Spec.parse("oriented_grid") == Spec("oriented_grid")
    assert Spec.parse("perturbation:n_moves=64") == Spec("perturbation", {"n_moves": 64})
    # JSON values, so null is None and a bare word stays a string.
    assert Spec.parse("x:a=null,b=1.5,c=corner") == Spec("x", {"a": None, "b": 1.5, "c": "corner"})
    with pytest.raises(ValueError, match="key=value"):
        Spec.parse("perturbation:n_moves")


def test_per_agent_kwargs_reach_only_that_agent() -> None:
    # The agent is the thing under test, so its own hyperparameters differ per row by design -
    # while everything the environment owns stays shared.
    reference = build_preset("training", "b", budget=Budget(env_steps=400))
    kwargs = {"genetic": {"population": 64}}
    genetic = _agent_spec("genetic", reference, population=16, agent_kwargs=kwargs)
    random_search = _agent_spec("random_search", reference, population=16, agent_kwargs=kwargs)
    assert genetic.algorithm.kwargs == {"population": 64}
    assert random_search.algorithm.kwargs == {"population": 16}


def test_the_table_scores_a_placement_the_way_the_runner_does() -> None:
    """One scoring path, or the table and the per-run logs beside it disagree.

    `_run_design` used to call `score()` without the agent's orientations, its action space or its
    warm start - so an oriented GA row would have been ranked on an all-north reading of a
    placement whose legality was decided under rotation.
    """
    import inspect

    from scripts import compare_agents

    source = inspect.getsource(compare_agents._run_design)
    scoring = [line.strip() for line in source.splitlines() if "score(" in line]
    assert scoring, "the comparison must score the placements itself"
    assert "best_orientations" in source
    assert "built.action_space" in source and "built.initial_positions" in source
