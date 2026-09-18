"""Decentralized macro agents: every macro nudges itself, all of them at once.

This directory is deliberately OUTSIDE `placax/` and `placax_agents/`. It imports the core -
netlist loading, the smoothed wirelength, the differentiable density cost, the greedy warm start,
the legality measurement and the runner's own `score_placement` - and adds nothing to it, so a
week of experiments here cannot destabilize the environment every other result in this project
was produced under. If something here needs a change in the core, that is a bug or a real gap and
should be fixed there on purpose, not worked around here.

What it adds, in the order the pieces depend on each other:

  * `neighbors.py`  - which macros are wired to which, and each macro's top-k strongest partners.
  * `context.py`    - one loaded design, its greedy warm start, its bounds and its neighbour table.
  * `objective.py`  - the shared score every agent, every baseline and every plot is judged by:
                      smoothed wirelength plus a differentiable legality cost, both normalized.
  * `moves.py`      - the transition: every macro moves by its own small (dx, dy), simultaneously.
  * `view.py`       - what ONE macro sees (m0 / m1 / m2) - the ablation this experiment exists for.
  * `policy.py`     - one shared network, applied per macro. Same weights for every macro.
  * `train.py`      - short-horizon analytic-gradient training (SHAC's window, no critic).
  * `adam.py`       - the baseline that matters most: the same objective optimized directly.
  * `compare.py`    - the finished runs as one table.

The claim being tested, and the thing to check on day 7: a shared per-macro policy improves the
greedy placement, and does it better than Adam does - or, if it does not, why not.
"""
