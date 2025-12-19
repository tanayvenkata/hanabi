"""
LLM-based Hanabi Agents

Two architectures available:

1. MonolithAgent - Single LLM makes all decisions
   - Simpler, fewer API calls
   - Good baseline

2. CouncilAgent - Multi-agent council of advisors
   - PlayAdvisor, HintAdvisor, DiscardAdvisor
   - MainDecisionAgent weighs recommendations
   - More interpretable, easier to debug/tune

Usage:
    # Monolith (single agent)
    from agents.llm import MonolithAgent
    agent = MonolithAgent(player_id=0, config={"verbose": True})

    # Council (multi-agent)
    from agents.llm import CouncilAgent
    agent = CouncilAgent(player_id=0, config={"verbose": True, "parallel": True})
"""

from agents.llm.monolith import MonolithAgent
from agents.llm.council_agent import CouncilAgent

__all__ = ["MonolithAgent", "CouncilAgent"]
