"""
Discard Advisor

Specializes in recommending which card to safely discard.

Responsibilities:
- Identify known-useless cards (safe to discard)
- Find duplicates (cards also in partner's hand)
- Recommend safest discard when nothing is certain
- Default to chop (oldest unhinted card) when uncertain

Key Context Needed:
- TODO: Define exactly what context this advisor needs
- Our hand knowledge (what hints we have)
- Current fireworks (what's useless)
- Discard pile (what's been discarded)
- Partner's hand (check for duplicates)
- Critical cards info
"""

import re
from typing import Any, Dict, Optional

from agents.llm.advisors.base import BaseAdvisor, Recommendation


class DiscardAdvisor(BaseAdvisor):
    """
    Advisor that recommends which card to discard.

    Focuses on:
    - Safe discards (known useless cards)
    - Duplicate discards (card also in partner's hand)
    - Chop discards (standard fallback)
    - Avoiding critical card discards
    """

    advisor_type = "DISCARD"

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)

        # TODO: Move these to prompts.py
        self.system_prompt = """You are a Hanabi Discard Advisor.

Your job is to recommend which card to discard from your hand.
You CANNOT see your own cards - you only know what hints you've received.

Safe discards:
- KNOWN USELESS: You know color AND rank, and rank <= firework (already played)
- DUPLICATES: Same card exists in partner's hand
- CHOP: Oldest unhinted card (rightmost) - standard fallback

DANGER:
- Never discard critical cards (5s, or cards needed with only 1 copy left)
- Discarding blindly is risky

You MUST provide:
1. SUGGESTION: slot <number> (0-4)
2. CONFIDENCE: HIGH (proven safe), MEDIUM (probably safe), LOW (risky but necessary)
3. REASONING: Brief explanation in 1-2 sentences"""

        self.user_prompt_template = """=== DISCARD DECISION ===

{context}

Which card should you discard?

Remember:
- Discarding regains 1 hint token
- The card is lost forever
- Prefer known-useless cards
- Default to chop (oldest unhinted) if uncertain

Respond with:
SUGGESTION: slot <number>
CONFIDENCE: HIGH/MEDIUM/LOW
REASONING: <brief explanation>"""

    def get_context(self, observation: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract context relevant to discard decisions.

        TODO: Customize what the discard advisor sees.
        Currently returns placeholder/full observation.

        Consider including:
        - card_knowledge for our hand (what hints we have)
        - fireworks (what ranks are useless)
        - discard_pile (what's already discarded)
        - partner's hand (check for duplicates)
        """
        # TODO: Filter to only relevant info for discard decisions
        #
        # Example of what we might want:
        # return {
        #     "my_hand_knowledge": observation.get("card_knowledge", [[]])[0],
        #     "fireworks": observation.get("fireworks", {}),
        #     "discard_pile": observation.get("discard_pile", []),
        #     "partner_hand": observation.get("observed_hands", [[],[]])[1],
        #     "hint_tokens": observation.get("information_tokens", 8),
        # }

        card_knowledge = observation.get("card_knowledge", [[], []])
        my_knowledge = card_knowledge[0] if len(card_knowledge) > 0 else []

        observed_hands = observation.get("observed_hands", [[], []])
        partner_hand = observed_hands[1] if len(observed_hands) > 1 else []

        return {
            "my_hand_knowledge": my_knowledge,
            "fireworks": observation.get("fireworks", {}),
            "discard_pile": observation.get("discard_pile", []),
            "partner_hand": partner_hand,
            "hint_tokens": observation.get("information_tokens", 8),
            # TODO: Add more context as needed
        }

    def format_context(self, context: Dict[str, Any]) -> str:
        """
        Format discard context for the prompt.

        TODO: Make this more readable/useful for the LLM.
        """
        lines = []

        # Hint tokens (context for why we might discard)
        lines.append(f"HINT TOKENS: {context.get('hint_tokens', 8)}/8")
        if context.get('hint_tokens', 8) >= 8:
            lines.append("  (At max - cannot discard to gain tokens)")
        lines.append("")

        # Fireworks
        fireworks = context.get("fireworks", {})
        lines.append("FIREWORKS (current piles):")
        lines.append("  " + "  ".join(f"{c}:{r}" for c, r in fireworks.items()))
        lines.append("")

        # What's useless (rank <= firework)
        lines.append("USELESS CARDS (rank <= firework, already played):")
        for color, rank in fireworks.items():
            if rank > 0:
                useless = [str(i) for i in range(1, rank + 1)]
                lines.append(f"  {color}: ranks {', '.join(useless)} are useless")
        lines.append("")

        # Our hand knowledge
        lines.append("YOUR HAND (what you know from hints):")
        my_knowledge = context.get("my_hand_knowledge", [])
        for i in range(5):
            if i < len(my_knowledge):
                k = my_knowledge[i]
                color = k.get("color")
                rank = k.get("rank")
                known_parts = []
                if color:
                    known_parts.append(f"color={color}")
                if rank is not None and rank >= 0:
                    known_parts.append(f"rank={rank + 1}")
                known = ", ".join(known_parts) if known_parts else "nothing"

                # Check if known useless
                useless_marker = ""
                if color and rank is not None and rank >= 0:
                    fw_rank = fireworks.get(color, 0)
                    if rank + 1 <= fw_rank:  # rank is 0-indexed
                        useless_marker = " <- USELESS (already played)"
            else:
                known = "nothing"
                useless_marker = ""

            # Chop marker
            chop_marker = " <- CHOP" if i == 4 else ""

            lines.append(f"  Slot {i}: [You know: {known}]{useless_marker}{chop_marker}")

        lines.append("")

        # Partner's hand (for duplicate checking)
        lines.append("PARTNER'S HAND (for duplicate checking):")
        partner_hand = context.get("partner_hand", [])
        color_names = {"R": "Red", "G": "Green", "B": "Blue", "W": "White", "Y": "Yellow"}

        partner_cards = []
        for card in partner_hand:
            color = card.get("color")
            rank = card.get("rank")
            if color and rank is not None:
                color_display = color_names.get(color, color)
                partner_cards.append(f"{color_display} {rank + 1}")
            else:
                partner_cards.append("??")
        lines.append("  " + ", ".join(partner_cards))

        # Discard pile
        discard_pile = context.get("discard_pile", [])
        if discard_pile:
            lines.append("")
            lines.append("DISCARD PILE:")
            discards = []
            for card in discard_pile:
                color = card.get("color")
                rank = card.get("rank")
                if color and rank is not None:
                    color_display = color_names.get(color, color)
                    discards.append(f"{color_display} {rank + 1}")
            lines.append("  " + ", ".join(discards))

        return "\n".join(lines)

    def _parse_action_details(self, response: str) -> Dict[str, Any]:
        """
        Extract slot number from discard suggestion.
        """
        # Look for "slot X" or "SUGGESTION: slot X"
        slot_match = re.search(r'slot\s*(\d)', response, re.IGNORECASE)
        if slot_match:
            return {"slot": int(slot_match.group(1))}

        return {"slot": 4}  # Default to chop if parsing fails
