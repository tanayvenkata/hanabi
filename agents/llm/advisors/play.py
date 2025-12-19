"""
Play Advisor

Specializes in recommending which card to play (if any).

Responsibilities:
- Analyze what cards in our hand are safe to play
- Consider hints we've received about our cards
- Evaluate risk vs reward of playing uncertain cards
- Recommend the best play option with confidence level

Key Context Needed:
- TODO: Define exactly what context this advisor needs
- Our hand knowledge (hints received)
- Current fireworks (what's playable)
- Life tokens (risk tolerance)
- Maybe: cards visible in partner's hand (for deduction)
"""

import re
from typing import Any, Dict, Optional

from agents.llm.advisors.base import BaseAdvisor, Recommendation


class PlayAdvisor(BaseAdvisor):
    """
    Advisor that recommends which card to play.

    Focuses on:
    - Identifying known-safe plays (100% confidence)
    - Evaluating probable plays (card counting, deduction)
    - Assessing risk when information is incomplete
    """

    advisor_type = "PLAY"

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)

        # TODO: Move these to prompts.py
        self.system_prompt = """You are a Hanabi Play Advisor.

Your job is to recommend which card to play from your hand.
You CANNOT see your own cards - you only know what hints you've received.

A card is PLAYABLE if: rank = firework[color] + 1
Playing a wrong card loses a life token!

You MUST provide:
1. SUGGESTION: Which slot to play (0-4) or "NONE" if too risky
2. CONFIDENCE: HIGH (certain), MEDIUM (probable), or LOW (risky)
3. REASONING: Brief explanation in 1-2 sentences

Be conservative - only suggest HIGH confidence plays unless desperate."""

        self.user_prompt_template = """=== PLAY DECISION ===

{context}

Based on what you know about your hand, which card should you play?

Remember:
- You CANNOT see your cards, only hints received
- Playing wrong loses a life
- If nothing is safe, say CONFIDENCE: LOW

Respond with:
SUGGESTION: slot <number> or NONE
CONFIDENCE: HIGH/MEDIUM/LOW
REASONING: <brief explanation>"""

    def get_context(self, observation: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract context relevant to play decisions.

        TODO: Customize what the play advisor sees.
        Currently returns placeholder/full observation.

        Consider including:
        - card_knowledge for our hand (what hints we have)
        - fireworks (what ranks are playable)
        - life_tokens (how much risk we can take)
        - Maybe inferred info from visible cards
        """
        # TODO: Filter to only relevant info for play decisions
        #
        # Example of what we might want:
        # return {
        #     "my_hand_knowledge": observation.get("card_knowledge", [[]])[0],
        #     "fireworks": observation.get("fireworks", {}),
        #     "life_tokens": observation.get("life_tokens", 3),
        #     "playable_ranks": {c: r+1 for c, r in observation.get("fireworks", {}).items()},
        # }

        # For now, return key fields (customize later)
        return {
            "fireworks": observation.get("fireworks", {}),
            "life_tokens": observation.get("life_tokens", 3),
            "my_hand_knowledge": observation.get("card_knowledge", [[]])[0] if observation.get("card_knowledge") else [],
            # TODO: Add more context as needed
        }

    def format_context(self, context: Dict[str, Any]) -> str:
        """
        Format play context for the prompt.

        TODO: Make this more readable/useful for the LLM.
        """
        lines = []

        # Fireworks
        fireworks = context.get("fireworks", {})
        lines.append("FIREWORKS (current piles):")
        lines.append("  " + "  ".join(f"{c}:{r}" for c, r in fireworks.items()))
        lines.append("")

        # What's playable
        playable = {c: r+1 for c, r in fireworks.items() if r < 5}
        lines.append("PLAYABLE RANKS (what we need):")
        lines.append("  " + ", ".join(f"{c} needs {r}" for c, r in playable.items()))
        lines.append("")

        # Life tokens
        lines.append(f"LIFE TOKENS: {context.get('life_tokens', 3)}/3")
        lines.append("")

        # Our hand knowledge
        lines.append("YOUR HAND (only hints you've received):")
        hand_knowledge = context.get("my_hand_knowledge", [])
        for i in range(5):
            if i < len(hand_knowledge):
                k = hand_knowledge[i]
                color = k.get("color")
                rank = k.get("rank")
                known_parts = []
                if color:
                    known_parts.append(f"color={color}")
                if rank is not None and rank >= 0:
                    known_parts.append(f"rank={rank + 1}")
                known = ", ".join(known_parts) if known_parts else "nothing"
            else:
                known = "nothing"
            lines.append(f"  Slot {i}: [You know: {known}]")

        return "\n".join(lines)

    def _parse_action_details(self, response: str) -> Dict[str, Any]:
        """
        Extract slot number from play suggestion.
        """
        # Look for "slot X" or "SUGGESTION: slot X"
        slot_match = re.search(r'slot\s*(\d)', response, re.IGNORECASE)
        if slot_match:
            return {"slot": int(slot_match.group(1))}

        # Check for "NONE"
        if re.search(r'SUGGESTION:\s*NONE', response, re.IGNORECASE):
            return {"slot": None}

        return {"slot": None}
