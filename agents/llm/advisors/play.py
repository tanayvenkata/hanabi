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
from collections import defaultdict

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

        Includes:
        - Game state (fireworks, tokens, deck size)
        - Teammates' visible cards
        - Probabilistic knowledge about our own cards (from orchestrator)
        - Hints we've received about our cards
        - Discard pile information
        """
        # Get teammates' hands (skip index 0 which is our hidden hand)
        observed_hands = observation.get("observed_hands", [])
        teammates_cards = []
        for i in range(1, len(observed_hands)):
            teammates_cards.append({
                "player_id": i,
                "cards": observed_hands[i]
            })

        return {
            # Game state
            "fireworks": observation.get("fireworks", {}),
            "life_tokens": observation.get("life_tokens", 3),
            "information_tokens": observation.get("information_tokens", 8),
            "deck_size": observation.get("deck_size", 0),

            # Our hand knowledge
            "my_hand_knowledge": observation.get("card_knowledge", [[]])[0] if observation.get("card_knowledge") else [],

            # Probabilistic card knowledge (computed once by orchestrator)
            "card_probabilities": observation.get("card_probabilities", []),

            # Teammates' visible cards
            "teammates_cards": teammates_cards,

            # Discard pile
            "discard_pile": observation.get("discard_pile", []),
        }

    def format_context(self, context: Dict[str, Any]) -> str:
        """
        Format play context for the prompt.

        Displays game state, teammates' cards, and probabilistic card knowledge.
        """
        lines = []

        # Game state
        lines.append("=== GAME STATE ===")
        fireworks = context.get("fireworks", {})
        lines.append("FIREWORKS (current piles):")
        lines.append("  " + "  ".join(f"{c}:{r}" for c, r in fireworks.items()))
        lines.append("")

        # What's playable
        playable = {c: r+1 for c, r in fireworks.items() if r < 5}
        lines.append("PLAYABLE RANKS (what we need next):")
        lines.append("  " + ", ".join(f"{c} needs {r}" for c, r in playable.items()))
        lines.append("")

        # Tokens and deck
        lines.append(f"LIFE TOKENS: {context.get('life_tokens', 3)}/3")
        lines.append(f"INFORMATION TOKENS: {context.get('information_tokens', 8)}/8")
        lines.append(f"DECK SIZE: {context.get('deck_size', 0)} cards remaining")
        lines.append("")

        # Teammates' cards (visible to us)
        lines.append("=== TEAMMATES' CARDS (visible to you) ===")
        teammates = context.get("teammates_cards", [])
        if teammates:
            for teammate in teammates:
                player_id = teammate["player_id"]
                cards = teammate["cards"]
                lines.append(f"Player {player_id}:")
                for i, card in enumerate(cards):
                    color = card.get("color", "?")
                    rank = card.get("rank", -1)
                    rank_display = rank + 1 if rank >= 0 else "?"
                    lines.append(f"  Slot {i}: {color}{rank_display}")
                lines.append("")
        else:
            lines.append("  (No teammates visible)")
            lines.append("")

        # Our hand - hints received
        lines.append("=== YOUR HAND (hints you've received) ===")
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
            lines.append(f"  Slot {i}: [Hints: {known}]")
        lines.append("")

        # Probabilistic card knowledge
        lines.append("=== PROBABILISTIC CARD KNOWLEDGE ===")
        lines.append("(What cards you likely have based on visible cards, discards, and hints)")
        lines.append("")
        card_probs = context.get("card_probabilities", [])
        for slot_data in card_probs:
            slot = slot_data.get("slot")
            possibilities = slot_data.get("possibilities", [])

            lines.append(f"Slot {slot}:")
            if possibilities:
                # Show top 3 most likely possibilities
                for i, poss in enumerate(possibilities[:3]):
                    color = poss["color"]
                    rank = poss["rank"]
                    prob = poss["probability"]
                    count = poss["count"]
                    lines.append(f"  {color}{rank}: {prob*100:.1f}% (count: {count})")
                if len(possibilities) > 3:
                    lines.append(f"  ... and {len(possibilities) - 3} other possibilities")
            else:
                lines.append("  (No valid possibilities)")
            lines.append("")

        # Discard pile summary
        discard_pile = context.get("discard_pile", [])
        if discard_pile:
            lines.append("=== DISCARD PILE ===")
            discard_counts = defaultdict(int)
            for card in discard_pile:
                color = card.get("color", "?")
                rank = card.get("rank", -1)
                rank_display = rank + 1 if rank >= 0 else "?"
                card_str = f"{color}{rank_display}"
                discard_counts[card_str] += 1

            for card_str, count in sorted(discard_counts.items()):
                lines.append(f"  {card_str}: {count}x")
            lines.append("")

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
