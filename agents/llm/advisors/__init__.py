"""
Council of Advisors for Hanabi

Three specialist advisors that each recommend an action:
- PlayAdvisor: Recommends which card to play (if any)
- HintAdvisor: Recommends which hint to give
- DiscardAdvisor: Recommends which card to discard

Each advisor gets context relevant to their specialty and returns
a recommendation with confidence level and reasoning.
"""

from agents.llm.advisors.base import BaseAdvisor, Recommendation
from agents.llm.advisors.play import PlayAdvisor
from agents.llm.advisors.hint import HintAdvisor
from agents.llm.advisors.discard import DiscardAdvisor

__all__ = [
    "BaseAdvisor",
    "Recommendation",
    "PlayAdvisor",
    "HintAdvisor",
    "DiscardAdvisor",
]
