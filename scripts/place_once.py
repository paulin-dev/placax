"""Loads a trained checkpoint and runs one greedy placement pass - no training, just inference."""
import argparse
import pathlib
import sys

from placax import _device  # noqa: F401  must precede jax imports
from placax.core import reset
from placax.log import Log
from placax_agents.ops.evaluate import evaluate
from placax_agents.policy.scale import to_real_centers
from placax_agents.training.loops.common import open_train_state
from scripts.run_maskplace import (
    MASKPLACE_MACRO_BUDGET,
    WIREMASK_MARGIN,
    _build_policy,
    _build_state_fn,
    _load_benchmark,
    maskplace_optimizer,
)

from jax import random


def _parse_args(argv: list[str]) -> tuple[pathlib.Path, pathlib.Path, int | None]:
    parser = argparse.ArgumentParser(description="Run one greedy placement pass from a trained checkpoint.")
    parser.add_argument("--benchmark_dir", type=pathlib.Path, default=pathlib.Path("benchmarks/adaptec1"))
    parser.add_argument("--checkpoint", type=pathlib.Path, default=None,
                         help="Defaults to <benchmark_dir>/output_maskplace/checkpoint.bin.")
    parser.add_argument("--macro_budget", type=str, default=str(MASKPLACE_MACRO_BUDGET),
                         help='Must match the budget the checkpoint was trained with; pass "all" for the whole netlist.')
    args = parser.parse_args(argv[1:])
    macro_budget = None if args.macro_budget.lower() == "all" else int(args.macro_budget)
    checkpoint_path = args.checkpoint or (args.benchmark_dir / "output_maskplace" / "checkpoint.bin")
    return args.benchmark_dir, checkpoint_path, macro_budget


def main() -> None:
    Log.configure()
    from placax_agents.policy.action import make_wiremask_quality_illegal

    benchmark_dir, checkpoint_path, macro_budget = _parse_args(sys.argv)
    if not checkpoint_path.exists():
        Log.error(f"'{checkpoint_path}' not found - train first with scripts/run_maskplace.py.")
        sys.exit(1)

    # 1. Load the same netlist/observation/policy setup the checkpoint was trained with.
    Log.info(f"loading {benchmark_dir} (macro_budget={macro_budget}) ...")
    benchmark = _load_benchmark(benchmark_dir, macro_budget)
    state_fn = _build_state_fn(benchmark)
    extra_illegal_fn = make_wiremask_quality_illegal(margin=WIREMASK_MARGIN, cell_size=benchmark.cell_size)
    policy = _build_policy(benchmark)

    # 2. Build a template pytree matching what was saved, then deserialize the checkpoint into it.
    key = random.PRNGKey(0)
    key, init_key = random.split(key)
    obs0 = state_fn(reset(benchmark.params), benchmark.params, benchmark.sizes_array)
    variables = policy.init(init_key, obs0)
    optimizer = maskplace_optimizer()
    variables, _opt_state, _running_stats, _key, iteration = open_train_state(
        variables, key, optimizer, checkpoint_path
    )
    Log.info(f"loaded checkpoint at iteration {iteration}")

    # 3. One greedy (no training, no sampling) rollout: place every macro, then report real HPWL.
    positions, hpwl_value = evaluate(
        variables, policy.apply, benchmark.params, benchmark.sizes_array, benchmark.cell_size,
        benchmark.padded_pin_idx, benchmark.padded_pin_offset, benchmark.valid_mask,
        state_fn, extra_illegal_fn,
    )
    real_centers = to_real_centers(positions, benchmark.sizes_array, benchmark.cell_size)

    print()
    print(f"real_hpwl = {float(hpwl_value):.2f}")
    print(f"placed {positions.shape[0]} macros - grid positions in `positions`, real-unit centers in `real_centers`.")


if __name__ == "__main__":
    main()
