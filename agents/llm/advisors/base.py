"""
Base Advisor Class

Abstract base class for all advisors in the council.
Each advisor specializes in one action type (PLAY, HINT, or DISCARD).

To implement a new advisor:
1. Subclass BaseAdvisor
2. Override get_context() to filter observation for your specialty
3. Set system_prompt and user_prompt_template in __init__
4. Optionally override parse_response() for custom parsing
"""

import re
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, Optional

from shared.llm_client import LLMClient


@dataclass
class Recommendation:
    """
    A recommendation from an advisor.

    Attributes:
        action_type: "PLAY", "HINT", or "DISCARD"
        action_details: Specific action info, e.g.:
            - PLAY: {"slot": 0}
            - HINT: {"hint_type": "rank", "value": 1}
            - DISCARD: {"slot": 4}
        confidence: "HIGH", "MEDIUM", or "LOW"
        reasoning: Natural language explanation (shown to main agent)
        raw_response: Full LLM response (for debugging)
    """
    action_type: str
    action_details: Dict[str, Any]
    confidence: str
    reasoning: str
    raw_response: str


class BaseAdvisor(ABC):
    """
    Abstract base class for advisors.

    Each advisor:
    1. Receives game observation
    2. Filters to relevant context (get_context)
    3. Builds a prompt (get_prompt)
    4. Calls LLM
    5. Parses response into Recommendation
    """

    # Override these in subclasses
    advisor_type: str = "BASE"  # "PLAY", "HINT", or "DISCARD"

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize advisor.

        Args:
            config: Optional configuration dict with:
                - llm_provider: "local" or "openai" (default: from .env or "local")
                - llm_url: LLM API endpoint for local (default: LM Studio)
                - model: Model name to use
                - temperature: Sampling temperature
                - openai_api_key: OpenAI API key (default: from .env)
                - verbose: Print debug info
        """
        config = config or {}

        # Create LLM client (handles local and OpenAI)
        self.llm_client = LLMClient(config)
        self.verbose = config.get("verbose", False)

        # Prompt templates - override in subclasses or set via prompts.py
        self.system_prompt: str = ""
        self.user_prompt_template: str = ""

    @abstractmethod
    def get_context(self, observation: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract relevant context from game observation.

        Override this to filter the observation to only what this
        advisor needs. For example, PlayAdvisor might only need:
        - own hand knowledge
        - fireworks (what's playable)
        - life tokens

        Args:
            observation: Full game observation from environment

        Returns:
            Filtered context dict specific to this advisor's needs

        TODO: Define what context each advisor needs
        """
        pass

    def format_context(self, context: Dict[str, Any]) -> str:
        """
        Format context dict into a string for the prompt.

        Override this to customize how context is presented to LLM.
        Default implementation just converts dict to readable string.

        Args:
            context: Context dict from get_context()

        Returns:
            Formatted string representation

        TODO: Define format for each advisor type
        """
        # Placeholder - override per advisor
        lines = []
        for key, value in context.items():
            lines.append(f"{key}: {value}")
        return "\n".join(lines)

    def get_prompt(self, context: Dict[str, Any]) -> str:
        """
        Build the user prompt from context.

        Uses self.user_prompt_template and formats with context.

        Args:
            context: Context dict from get_context()

        Returns:
            Formatted user prompt string
        """
        context_str = self.format_context(context)

        # Simple template substitution
        # TODO: Use proper template system if needed
        return self.user_prompt_template.format(
            context=context_str,
            **context  # Allow direct field access in template
        )

    def call_llm(self, prompt: str) -> str:
        """
        Call the LLM API.

        Uses the shared LLM client which supports both local and OpenAI.

        Args:
            prompt: User prompt to send

        Returns:
            LLM response text
        """
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": prompt}
        ]
        return self.llm_client.chat_completion(messages)

    def parse_response(self, response: str) -> Recommendation:
        """
        Parse LLM response into a Recommendation.

        Default implementation looks for:
        - SUGGESTION: <action details>
        - CONFIDENCE: HIGH/MEDIUM/LOW
        - REASONING: <explanation>

        Override for custom parsing per advisor.

        Args:
            response: Raw LLM response text

        Returns:
            Recommendation dataclass
        """
        # Handle </think> tags from some models
        if "</think>" in response:
            response = response.split("</think>")[-1]

        # Parse confidence
        confidence_match = re.search(
            r'CONFIDENCE:\s*(HIGH|MEDIUM|LOW)',
            response,
            re.IGNORECASE
        )
        confidence = confidence_match.group(1).upper() if confidence_match else "MEDIUM"

        # Parse reasoning (everything after REASONING:)
        reasoning_match = re.search(
            r'REASONING:\s*(.+?)(?=\n\n|\Z)',
            response,
            re.IGNORECASE | re.DOTALL
        )
        reasoning = reasoning_match.group(1).strip() if reasoning_match else response[:200]

        # Parse action details - subclasses should override for specifics
        action_details = self._parse_action_details(response)

        return Recommendation(
            action_type=self.advisor_type,
            action_details=action_details,
            confidence=confidence,
            reasoning=reasoning,
            raw_response=response,
        )

    def _parse_action_details(self, response: str) -> Dict[str, Any]:
        """
        Parse action-specific details from response.

        Override in subclasses to extract slot numbers, hint types, etc.

        Args:
            response: Raw LLM response

        Returns:
            Dict with action details
        """
        # Default: empty dict, subclasses should override
        return {}

    def recommend(self, observation: Dict[str, Any]) -> Recommendation:
        """
        Generate a recommendation for the given game state.

        This is the main entry point. It:
        1. Extracts relevant context
        2. Builds prompt
        3. Calls LLM
        4. Parses response

        Args:
            observation: Full game observation

        Returns:
            Recommendation with action, confidence, and reasoning
        """
        # 1. Get filtered context
        context = self.get_context(observation)

        if self.verbose:
            print(f"\n{'='*50}")
            print(f"{self.advisor_type} ADVISOR")
            print(f"{'='*50}")
            print(f"Context: {context}")

        # 2. Build prompt
        prompt = self.get_prompt(context)

        if self.verbose:
            print(f"\nPrompt:\n{prompt}")

        # 3. Call LLM
        response = self.call_llm(prompt)

        if self.verbose:
            print(f"\nResponse:\n{response}")

        # 4. Parse and return
        recommendation = self.parse_response(response)

        if self.verbose:
            print(f"\nRecommendation: {recommendation}")

        return recommendation
