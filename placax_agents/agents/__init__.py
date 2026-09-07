"""Agents: everything that can drive the shared placement kernel.

The kernel itself knows nothing about any of these - `step()` cannot tell whether the action it
received came from a policy network's argmax, a uniform draw over legal cells, or a greedy
wirelength scan. That was always the design claim; these three agents are what make it a tested
one rather than an asserted one.
"""
from placax_agents.agents.base import Agent, UpdateResult
from placax_agents.agents.baselines import GreedyWiremaskAgent, RandomSearchAgent
from placax_agents.agents.ppo import PPOAgent, is_ppo_state

__all__ = [
    "Agent", "GreedyWiremaskAgent", "PPOAgent", "RandomSearchAgent", "UpdateResult",
    "is_ppo_state",
]
