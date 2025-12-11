"""
Example Hanabi Agent Implementation

This module provides:
- BaseAgent: Abstract base class for all Hanabi agents
- RandomAgent: A simple agent that plays random legal moves

To create your own agent:
1. Create a new folder in agents/ (e.g., agents/myagent/)
2. Subclass BaseAgent and implement the `act` method
3. Your `act` method receives an observation and returns a move
"""

import random
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

from hanabi_learning_environment import rl_env


class BaseAgent(ABC):
    """Abstract base class for Hanabi agents.

    All agents must implement the `act` method which takes an observation
    and returns a legal action.

    Attributes:
        player_id: The player's index (0 or 1 in 2-player games)
        config: Optional configuration dictionary
    """

    def __init__(self, player_id: int, config: Optional[Dict[str, Any]] = None):
        self.player_id = player_id
        self.config = config or {}

    @abstractmethod
    def act(self, observation: Dict[str, Any]) -> int:
        """Choose an action given the current observation.

        Args:
            observation: Dictionary containing:
                - 'current_player': int, whose turn it is
                - 'legal_moves': list of legal move dicts
                - 'legal_moves_as_int': list of legal action integers
                - 'observed_hands': list of visible hands (own hand is hidden)
                - 'discard_pile': list of discarded cards
                - 'fireworks': dict mapping color to highest played rank
                - 'information_tokens': int, remaining hint tokens
                - 'life_tokens': int, remaining life tokens
                - 'deck_size': int, cards remaining in deck

        Returns:
            An integer representing the chosen action (index into legal_moves_as_int)
        """
        pass

    def reset(self) -> None:
        """Called at the start of each new game. Override to reset any state."""
        pass

    def observe_move(self, move: Dict[str, Any], acting_player: int) -> None:
        """Called after each move in the game. Override to track game history.

        Args:
            move: The move that was played
            acting_player: The player who made the move
        """
        pass


class RandomAgent(BaseAgent):
    """An agent that plays uniformly random legal moves.

    This serves as a baseline and demonstrates the agent interface.
    Expected score: ~0-2 points on average.
    """

    def __init__(self, player_id: int, config: Optional[Dict[str, Any]] = None):
        super().__init__(player_id, config)
        self.seed = config.get("seed") if config else None
        if self.seed is not None:
            random.seed(self.seed)

    def act(self, observation: Dict[str, Any]) -> int:
        """Choose a random legal action."""
        legal_moves = observation["legal_moves_as_int"]
        return random.choice(legal_moves)


# Convenience function to create agents
def create_agent(agent_type: str, player_id: int, config: Optional[Dict[str, Any]] = None) -> BaseAgent:
    """Factory function to create agents by type name.

    Args:
        agent_type: One of "random" (add more as you implement them)
        player_id: Player index
        config: Optional configuration

    Returns:
        An instance of the requested agent type
    """
    agents = {
        "random": RandomAgent,
    }

    if agent_type not in agents:
        raise ValueError(f"Unknown agent type: {agent_type}. Available: {list(agents.keys())}")

    return agents[agent_type](player_id, config)
