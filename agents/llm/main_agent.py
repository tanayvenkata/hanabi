"""
Main Decision Agent

The "judge" that receives recommendations from all three advisors
and makes the final decision on which action to take.

Responsibilities:
- Receive recommendations from Play, Hint, and Discard advisors
- Weigh confidence levels and reasoning
- Make final decision: PLAY, HINT, or DISCARD
- Return structured decision for execution

Key Context:
- Minimal game state (score, resources, turn)
- Three recommendations with confidence and reasoning
- Optionally: game history for context
"""

import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from agents.llm.advisors.base import Recommendation
from shared.llm_client import LLMClient


@dataclass
class Decision:
    """
    Final decision from the main agent.

    Attributes:
        action_type: "PLAY", "HINT", or "DISCARD"
        action_details: Specific action info from chosen recommendation
        reasoning: Why this action was chosen
        raw_response: Full LLM response (for debugging)
    """
    action_type: str
    action_details: Dict[str, Any]
    reasoning: str
    raw_response: str


class MainDecisionAgent:
    """
    The main decision maker that chooses between advisor recommendations.

    Receives pre-analyzed options from three specialists and decides
    which action to take based on confidence levels and reasoning.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize main agent.

        Args:
            config: Optional configuration dict with:
                - llm_provider: "local" or "openai" (default: from .env or "local")
                - llm_url: LLM API endpoint for local
                - model: Model name
                - temperature: Sampling temperature
                - openai_api_key: OpenAI API key (default: from .env)
                - verbose: Print debug info
        """
        config = config or {}

        # Create LLM client (handles local and OpenAI)
        self.llm_client = LLMClient(config)
        self.verbose = config.get("verbose", False)

        # TODO: Move these to prompts.py
        self.system_prompt = """You are the Hanabi Decision Agent.

Three specialist advisors have analyzed the game state:
- PLAY Advisor: Recommends which card to play
- HINT Advisor: Recommends which hint to give
- DISCARD Advisor: Recommends which card to discard

Your job is to choose the BEST action based on their recommendations.

Guidelines:
- HIGH confidence suggestions are usually correct
- Prefer PLAY if a safe play is available (advances score)
- Prefer HINT if partner has playable cards or critical cards need saving
- DISCARD is usually a fallback when nothing better is available
- Consider the game state (hint tokens, lives, score)

You MUST provide:
1. DECISION: PLAY, HINT, or DISCARD
2. REASONING: Brief explanation in 1-2 sentences"""

        self.user_prompt_template = """=== FINAL DECISION ===

GAME STATE:
{game_summary}

---

PLAY ADVISOR recommends:
{play_recommendation}

---

HINT ADVISOR recommends:
{hint_recommendation}

---

DISCARD ADVISOR recommends:
{discard_recommendation}

---

Based on the advisors' recommendations and confidence levels, what should we do?

Respond with:
DECISION: PLAY or HINT or DISCARD
REASONING: <brief explanation>"""

    def format_recommendation(self, rec: Recommendation) -> str:
        """Format a recommendation for display."""
        lines = [
            f"Action: {rec.action_type} {rec.action_details}",
            f"Confidence: {rec.confidence}",
            f"Reasoning: {rec.reasoning}",
        ]
        return "\n".join(lines)

    def format_game_summary(self, observation: Dict[str, Any]) -> str:
        """
        Format minimal game state summary.

        TODO: Customize what the main agent sees.
        Keep it brief - advisors already did deep analysis.
        """
        fireworks = observation.get("fireworks", {})
        score = sum(fireworks.values())

        lines = [
            f"Score: {score}/25",
            f"Fireworks: {' '.join(f'{c}:{r}' for c, r in fireworks.items())}",
            f"Hint tokens: {observation.get('information_tokens', 8)}/8",
            f"Life tokens: {observation.get('life_tokens', 3)}/3",
            f"Deck: {observation.get('deck_size', '?')} cards",
        ]
        return "\n".join(lines)

    def decide(
        self,
        observation: Dict[str, Any],
        play_rec: Recommendation,
        hint_rec: Recommendation,
        discard_rec: Recommendation,
    ) -> Decision:
        """
        Make final decision based on advisor recommendations.

        Args:
            observation: Game observation (for summary)
            play_rec: Recommendation from PlayAdvisor
            hint_rec: Recommendation from HintAdvisor
            discard_rec: Recommendation from DiscardAdvisor

        Returns:
            Decision with chosen action and reasoning
        """
        # Build prompt
        prompt = self.user_prompt_template.format(
            game_summary=self.format_game_summary(observation),
            play_recommendation=self.format_recommendation(play_rec),
            hint_recommendation=self.format_recommendation(hint_rec),
            discard_recommendation=self.format_recommendation(discard_rec),
        )

        if self.verbose:
            print(f"\n{'='*50}")
            print("MAIN DECISION AGENT")
            print(f"{'='*50}")
            print(f"\nPrompt:\n{prompt}")

        # Call LLM
        response = self.call_llm(prompt)

        if self.verbose:
            print(f"\nResponse:\n{response}")

        # Parse decision
        decision = self.parse_response(response, play_rec, hint_rec, discard_rec)

        if self.verbose:
            print(f"\nDecision: {decision}")

        return decision

    def call_llm(self, prompt: str) -> str:
        """
        Call the LLM API.

        Uses the shared LLM client which supports both local and OpenAI.
        """
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": prompt}
        ]
        return self.llm_client.chat_completion(messages)

    def parse_response(
        self,
        response: str,
        play_rec: Recommendation,
        hint_rec: Recommendation,
        discard_rec: Recommendation,
    ) -> Decision:
        """
        Parse LLM response into Decision.

        Args:
            response: Raw LLM response
            play_rec, hint_rec, discard_rec: Original recommendations

        Returns:
            Decision with chosen action
        """
        # Handle </think> tags
        if "</think>" in response:
            response = response.split("</think>")[-1]

        # Parse decision type
        decision_match = re.search(
            r'DECISION:\s*(PLAY|HINT|DISCARD)',
            response,
            re.IGNORECASE
        )

        if decision_match:
            action_type = decision_match.group(1).upper()
        else:
            # Default to DISCARD if parsing fails
            action_type = "DISCARD"

        # Get action details from corresponding recommendation
        if action_type == "PLAY":
            action_details = play_rec.action_details
        elif action_type == "HINT":
            action_details = hint_rec.action_details
        else:
            action_details = discard_rec.action_details

        # Parse reasoning
        reasoning_match = re.search(
            r'REASONING:\s*(.+?)(?=\n\n|\Z)',
            response,
            re.IGNORECASE | re.DOTALL
        )
        reasoning = reasoning_match.group(1).strip() if reasoning_match else ""

        return Decision(
            action_type=action_type,
            action_details=action_details,
            reasoning=reasoning,
            raw_response=response,
        )
