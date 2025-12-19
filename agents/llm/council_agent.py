"""
Council Agent

The main agent class that implements the BaseAgent interface.
Uses a council of advisors (Play, Hint, Discard) coordinated by
an orchestrator and main decision agent.

This is the entry point for using the multi-agent architecture.

Usage:
    from agents.llm import CouncilAgent

    agent = CouncilAgent(player_id=0, config={"verbose": True})
    action = agent.act(observation)
"""

from typing import Any, Dict, Optional

from agents.example import BaseAgent
from agents.llm.orchestrator import Orchestrator


class CouncilAgent(BaseAgent):
    """
    Multi-agent Hanabi player using a council of advisors.

    Architecture:
        Game State
            │
            ├──► PlayAdvisor ──────┐
            ├──► HintAdvisor ──────┼──► MainDecisionAgent ──► Action
            └──► DiscardAdvisor ───┘

    Each advisor specializes in one action type and provides a
    recommendation with confidence and reasoning. The main agent
    weighs all recommendations and makes the final decision.

    Config options:
        - llm_url: LLM API endpoint (default: LM Studio local)
        - model: Model name (default: nvidia/nemotron-3-nano)
        - temperature: Sampling temperature (default: 0.3)
        - verbose: Print debug info (default: False)
        - parallel: Run advisors in parallel (default: False)
    """

    def __init__(self, player_id: int, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the council agent.

        Args:
            player_id: Player index (0 or 1 in 2-player games)
            config: Configuration options for all sub-agents
        """
        super().__init__(player_id, config)

        self.orchestrator = Orchestrator(config)
        self.verbose = (config or {}).get("verbose", False)

        # Tracking
        self.turn_count = 0

    def act(self, observation: Dict[str, Any]) -> int:
        """
        Choose an action using the council of advisors.

        Flow:
        1. Orchestrator runs all three advisors
        2. Main agent receives recommendations
        3. Main agent decides final action
        4. Decision mapped to game action integer

        Args:
            observation: Game state from environment

        Returns:
            Action integer for the game engine
        """
        self.turn_count += 1

        if self.verbose:
            print(f"\n{'#'*60}")
            print(f"# COUNCIL AGENT - Turn {self.turn_count} - Player {self.player_id}")
            print(f"{'#'*60}")

        # Get decision from orchestrator
        decision = self.orchestrator.get_decision(observation)

        if self.verbose:
            print(f"\n{'='*60}")
            print("FINAL DECISION")
            print(f"{'='*60}")
            print(f"Action: {decision.action_type}")
            print(f"Details: {decision.action_details}")
            print(f"Reasoning: {decision.reasoning}")

        # Map to game action
        action = self.orchestrator.map_decision_to_action(decision, observation)

        if self.verbose:
            print(f"Game action int: {action}")

        return action

    def reset(self) -> None:
        """Reset for a new game."""
        self.turn_count = 0
        self.orchestrator.reset()

    def observe_move(self, move: Dict[str, Any], acting_player: int) -> None:
        """
        Observe a move made in the game.

        TODO: Use this to build game history for context.

        Args:
            move: The move that was played
            acting_player: Who made the move
        """
        # TODO: Track moves for history context
        pass


# For testing
if __name__ == "__main__":
    from agents.llm.test_states import SCENARIO_1_HINT_PLAYABLE

    print("Testing CouncilAgent with Scenario 1...")

    agent = CouncilAgent(player_id=0, config={"verbose": True})
    obs = SCENARIO_1_HINT_PLAYABLE["observation"]

    action = agent.act(obs)
    print(f"\n\nFinal action: {action}")
