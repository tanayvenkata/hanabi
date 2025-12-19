"""
Hint Advisor

Specializes in recommending which hint to give to partner.

Responsibilities:
- Identify partner's playable cards that need hints
- Spot critical cards on partner's chop that need saving
- Choose between color hints and rank hints
- Prioritize hints that enable immediate plays

Key Context Needed:
- TODO: Define exactly what context this advisor needs
- Partner's actual hand (we can see it!)
- Partner's knowledge (what hints they've received)
- Current fireworks (what's playable)
- Information tokens available
"""

import re
from typing import Any, Dict, Optional

from agents.llm.advisors.base import BaseAdvisor, Recommendation


class HintAdvisor(BaseAdvisor):
    """
    Advisor that recommends which hint to give partner.

    Focuses on:
    - Enabling immediate plays (hint playable cards)
    - Saving critical cards (hint cards about to be discarded)
    - Efficient hints (touch multiple useful cards)
    """

    advisor_type = "HINT"

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)

        # TODO: Move these to prompts.py
        self.system_prompt = """You are a Hanabi Hint Advisor.

Your job is to recommend which hint to give your partner.
You CAN see your partner's cards (but they can't see their own).

Hint types:
- COLOR hint: "Your cards in slots [X,Y] are Red"
- RANK hint: "Your cards in slots [X,Y] are 1s"

Good hints:
- Tell partner about PLAYABLE cards (so they can play)
- SAVE critical cards (prevent bad discards)
- Efficient hints touch multiple useful cards

You MUST provide:
1. SUGGESTION: "color <C>" or "rank <N>" (where N is 1-5)
2. CONFIDENCE: HIGH (clearly best), MEDIUM (good option), LOW (no great hints)
3. REASONING: Brief explanation in 1-2 sentences"""

        self.user_prompt_template = """=== HINT DECISION ===

{context}

What hint should you give your partner?

Remember:
- You CAN see their cards
- They CANNOT see their own cards
- Prioritize hints that enable plays or save critical cards
- Costs 1 hint token

Respond with:
SUGGESTION: color <COLOR> or rank <NUMBER>
CONFIDENCE: HIGH/MEDIUM/LOW
REASONING: <brief explanation>"""

    def get_context(self, observation: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract context relevant to hint decisions.

        TODO: Customize what the hint advisor sees.
        Currently returns placeholder/full observation.

        Consider including:
        - Partner's full hand (we can see it!)
        - Partner's card_knowledge (what they know)
        - fireworks (what's playable)
        - information_tokens (can we afford hints)
        - Critical cards info
        """
        # TODO: Filter to only relevant info for hint decisions
        #
        # Example of what we might want:
        # partner_idx = 1  # In 2-player, partner is index 1
        # return {
        #     "partner_hand": observation.get("observed_hands", [[],[]])[partner_idx],
        #     "partner_knowledge": observation.get("card_knowledge", [[],[]])[partner_idx],
        #     "fireworks": observation.get("fireworks", {}),
        #     "hint_tokens": observation.get("information_tokens", 8),
        # }

        # Get partner's hand (index 1 in 2-player)
        observed_hands = observation.get("observed_hands", [[], []])
        partner_hand = observed_hands[1] if len(observed_hands) > 1 else []

        card_knowledge = observation.get("card_knowledge", [[], []])
        partner_knowledge = card_knowledge[1] if len(card_knowledge) > 1 else []

        return {
            "partner_hand": partner_hand,
            "partner_knowledge": partner_knowledge,
            "fireworks": observation.get("fireworks", {}),
            "hint_tokens": observation.get("information_tokens", 8),
            # TODO: Add more context as needed
        }

    def format_context(self, context: Dict[str, Any]) -> str:
        """
        Format hint context for the prompt.

        TODO: Make this more readable/useful for the LLM.
        """
        lines = []

        # Hint tokens
        lines.append(f"HINT TOKENS AVAILABLE: {context.get('hint_tokens', 8)}/8")
        lines.append("")

        # Fireworks
        fireworks = context.get("fireworks", {})
        lines.append("FIREWORKS (current piles):")
        lines.append("  " + "  ".join(f"{c}:{r}" for c, r in fireworks.items()))
        lines.append("")

        # What's playable
        playable = {c: r+1 for c, r in fireworks.items() if r < 5}
        lines.append("PLAYABLE RANKS (what cards can be played):")
        lines.append("  " + ", ".join(f"{c} needs {r}" for c, r in playable.items()))
        lines.append("")

        # Partner's hand (we can see it!)
        lines.append("PARTNER'S HAND (you can see these):")
        partner_hand = context.get("partner_hand", [])
        partner_knowledge = context.get("partner_knowledge", [])

        color_names = {"R": "Red", "G": "Green", "B": "Blue", "W": "White", "Y": "Yellow"}

        for i, card in enumerate(partner_hand):
            color = card.get("color")
            rank = card.get("rank")

            if color and rank is not None:
                color_display = color_names.get(color, color)
                rank_display = rank + 1  # 0-indexed to 1-5

                # What does partner know?
                if i < len(partner_knowledge):
                    pk = partner_knowledge[i]
                    known_parts = []
                    if pk.get("color"):
                        known_parts.append(f"color={pk['color']}")
                    if pk.get("rank") is not None and pk.get("rank") >= 0:
                        known_parts.append(f"rank={pk['rank'] + 1}")
                    known = ", ".join(known_parts) if known_parts else "nothing"
                else:
                    known = "nothing"

                # Is this playable?
                is_playable = playable.get(color) == rank_display
                playable_marker = " <- PLAYABLE!" if is_playable else ""

                # Is this critical (a 5)?
                critical_marker = " <- CRITICAL!" if rank == 4 else ""

                # Is this the chop (rightmost)?
                chop_marker = " <- CHOP" if i == len(partner_hand) - 1 else ""

                lines.append(
                    f"  Slot {i}: {color_display} {rank_display} "
                    f"[Partner knows: {known}]{playable_marker}{critical_marker}{chop_marker}"
                )
            else:
                lines.append(f"  Slot {i}: ??")

        return "\n".join(lines)

    def _parse_action_details(self, response: str) -> Dict[str, Any]:
        """
        Extract hint type and value from suggestion.
        """
        # Look for "color <X>" pattern
        color_match = re.search(
            r'(?:SUGGESTION:)?\s*color\s+(\w+)',
            response,
            re.IGNORECASE
        )
        if color_match:
            color = color_match.group(1).upper()
            # Normalize to single letter
            color_map = {
                "RED": "R", "R": "R",
                "GREEN": "G", "G": "G",
                "BLUE": "B", "B": "B",
                "WHITE": "W", "W": "W",
                "YELLOW": "Y", "Y": "Y",
            }
            return {
                "hint_type": "color",
                "value": color_map.get(color, color)
            }

        # Look for "rank <N>" pattern
        rank_match = re.search(
            r'(?:SUGGESTION:)?\s*rank\s+(\d)',
            response,
            re.IGNORECASE
        )
        if rank_match:
            rank = int(rank_match.group(1))
            return {
                "hint_type": "rank",
                "value": rank
            }

        return {"hint_type": None, "value": None}
