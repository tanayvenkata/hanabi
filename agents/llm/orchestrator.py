"""
Orchestrator

Coordinates the council of advisors and main decision agent.

Flow:
1. Receive game observation
2. Run all three advisors (can be parallelized)
3. Collect recommendations
4. Pass to main agent for final decision
5. Map decision to game action

TODO:
- Add parallel execution for advisors
- Add caching/memoization
- Add history tracking for stateful context
"""

from typing import Any, Dict, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

from agents.llm.advisors import PlayAdvisor, HintAdvisor, DiscardAdvisor, Recommendation
from agents.llm.main_agent import MainDecisionAgent, Decision
from shared.card_inference import calculate_card_probabilities


class Orchestrator:
    """
    Coordinates the multi-agent decision process.

    Manages:
    - Three specialist advisors
    - Main decision agent
    - Action mapping to game format
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize orchestrator and all agents.

        Args:
            config: Configuration dict passed to all agents
                - llm_url: LLM API endpoint
                - model: Model name
                - temperature: Sampling temperature
                - verbose: Print debug info
                - parallel: Run advisors in parallel (default: False)
        """
        self.config = config or {}
        self.verbose = self.config.get("verbose", False)
        self.parallel = self.config.get("parallel", False)

        # Initialize advisors
        self.play_advisor = PlayAdvisor(config)
        self.hint_advisor = HintAdvisor(config)
        self.discard_advisor = DiscardAdvisor(config)

        # Initialize main agent
        self.main_agent = MainDecisionAgent(config)

        # History tracking (for stateful context)
        # TODO: Implement history tracking
        self.history: list = []

    def _augment_observation(self, observation: Dict[str, Any]) -> Dict[str, Any]:
        """
        Augment observation with computed card probabilities.

        This is done once at the orchestrator level so all advisors
        can benefit from the same probabilistic reasoning without
        redundant computation.

        Args:
            observation: Raw game observation

        Returns:
            Observation dict with added 'card_probabilities' field
        """
        # Create a copy to avoid mutating the original
        augmented_obs = observation.copy()

        # Calculate card probabilities once
        card_probabilities = calculate_card_probabilities(observation)

        # Add to observation
        augmented_obs["card_probabilities"] = card_probabilities

        return augmented_obs

    def get_recommendations(
        self,
        observation: Dict[str, Any]
    ) -> tuple[Recommendation, Recommendation, Recommendation]:
        """
        Get recommendations from all three advisors.

        Args:
            observation: Game observation

        Returns:
            Tuple of (play_rec, hint_rec, discard_rec)
        """
        if self.parallel:
            return self._get_recommendations_parallel(observation)
        else:
            return self._get_recommendations_sequential(observation)

    def _get_recommendations_sequential(
        self,
        observation: Dict[str, Any]
    ) -> tuple[Recommendation, Recommendation, Recommendation]:
        """Run advisors sequentially."""
        if self.verbose:
            print("\n" + "="*60)
            print("RUNNING ADVISORS (sequential)")
            print("="*60)

        play_rec = self.play_advisor.recommend(observation)
        hint_rec = self.hint_advisor.recommend(observation)
        discard_rec = self.discard_advisor.recommend(observation)

        return play_rec, hint_rec, discard_rec

    def _get_recommendations_parallel(
        self,
        observation: Dict[str, Any]
    ) -> tuple[Recommendation, Recommendation, Recommendation]:
        """Run advisors in parallel using ThreadPoolExecutor."""
        if self.verbose:
            print("\n" + "="*60)
            print("RUNNING ADVISORS (parallel)")
            print("="*60)

        results = {}

        with ThreadPoolExecutor(max_workers=3) as executor:
            futures = {
                executor.submit(self.play_advisor.recommend, observation): "play",
                executor.submit(self.hint_advisor.recommend, observation): "hint",
                executor.submit(self.discard_advisor.recommend, observation): "discard",
            }

            for future in as_completed(futures):
                advisor_type = futures[future]
                try:
                    results[advisor_type] = future.result()
                except Exception as e:
                    # Create error recommendation
                    results[advisor_type] = Recommendation(
                        action_type=advisor_type.upper(),
                        action_details={},
                        confidence="LOW",
                        reasoning=f"Advisor error: {e}",
                        raw_response=str(e),
                    )

        return results["play"], results["hint"], results["discard"]

    def get_decision(self, observation: Dict[str, Any]) -> Decision:
        """
        Full decision pipeline: advisors -> main agent -> decision.

        Args:
            observation: Game observation

        Returns:
            Final Decision
        """
        # 0. Augment observation with card probabilities (computed once)
        augmented_obs = self._augment_observation(observation)

        # 1. Get recommendations from all advisors (using augmented observation)
        play_rec, hint_rec, discard_rec = self.get_recommendations(augmented_obs)

        # 2. Main agent makes final decision
        decision = self.main_agent.decide(
            observation,
            play_rec,
            hint_rec,
            discard_rec,
        )

        # 3. Record in history (for future stateful context)
        # TODO: Implement history tracking
        self.history.append({
            "play_rec": play_rec,
            "hint_rec": hint_rec,
            "discard_rec": discard_rec,
            "decision": decision,
        })

        return decision

    def map_decision_to_action(
        self,
        decision: Decision,
        observation: Dict[str, Any]
    ) -> int:
        """
        Map a Decision to a game action integer.

        Args:
            decision: Decision from main agent
            observation: Game observation (contains legal_moves)

        Returns:
            Action integer for the game engine
        """
        legal_moves = observation.get("legal_moves", [])
        legal_moves_as_int = observation.get("legal_moves_as_int", [])

        action_type = decision.action_type
        details = decision.action_details

        # Find matching action in legal moves
        for i, move in enumerate(legal_moves):
            move_type = move.get("action_type", "")

            if action_type == "PLAY" and move_type == "PLAY":
                if move.get("card_index") == details.get("slot"):
                    return legal_moves_as_int[i]

            elif action_type == "DISCARD" and move_type == "DISCARD":
                if move.get("card_index") == details.get("slot"):
                    return legal_moves_as_int[i]

            elif action_type == "HINT":
                hint_type = details.get("hint_type")
                value = details.get("value")

                if hint_type == "color" and move_type == "REVEAL_COLOR":
                    if move.get("color") == value:
                        return legal_moves_as_int[i]

                elif hint_type == "rank" and move_type == "REVEAL_RANK":
                    # Value is 1-5, move rank is 0-indexed
                    if move.get("rank") == value - 1:
                        return legal_moves_as_int[i]

        # Fallback: return first legal action if mapping fails
        if self.verbose:
            print(f"WARNING: Could not map decision {decision} to action, using fallback")
        return legal_moves_as_int[0] if legal_moves_as_int else 0

    def reset(self) -> None:
        """Reset state for new game."""
        self.history = []
