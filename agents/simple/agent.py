"""
Simple Hanabi Agent with Basic Awareness

A step up from random - uses simple heuristics:
1. If I have info about a card being playable, play it
2. If I have info tokens, give a useful hint
3. Otherwise, discard the oldest card
"""

from typing import Any, Dict, Optional
from agents.example import BaseAgent


class SimpleAgent(BaseAgent):
    """An agent with basic awareness of game state and card knowledge."""

    def act(self, observation: Dict[str, Any]) -> int:
        legal_moves = observation["legal_moves"]
        legal_moves_int = observation["legal_moves_as_int"]
        fireworks = observation["fireworks"]
        info_tokens = observation["information_tokens"]

        # Get my card knowledge (what hints have I received?)
        # card_knowledge is a list per player - index 0 is our hand
        all_knowledge = observation.get("card_knowledge", [])
        my_hand_knowledge = all_knowledge[0] if all_knowledge else []

        # Strategy 1: Play a card if I know it's playable
        for i, move in enumerate(legal_moves):
            if move["action_type"] == "PLAY":
                card_idx = move["card_index"]
                if self._is_probably_playable(card_idx, my_hand_knowledge, fireworks):
                    return legal_moves_int[i]

        # Strategy 2: Give a hint about a playable card (if we have tokens)
        if info_tokens > 0:
            hint_move = self._find_useful_hint(legal_moves, legal_moves_int, observation)
            if hint_move is not None:
                return hint_move

        # Strategy 3: If we got a hint about a card, try playing it (be optimistic!)
        for i, move in enumerate(legal_moves):
            if move["action_type"] == "PLAY":
                card_idx = move["card_index"]
                if card_idx < len(my_hand_knowledge):
                    card_info = my_hand_knowledge[card_idx]
                    # If we know ANYTHING about this card, partner hinted it for a reason
                    if card_info.get("color") is not None or card_info.get("rank") is not None:
                        return legal_moves_int[i]

        # Strategy 4: Discard oldest card (rightmost = oldest in hand)
        for i, move in enumerate(legal_moves):
            if move["action_type"] == "DISCARD":
                # Prefer discarding higher index (older cards)
                if move["card_index"] == 4:  # rightmost card
                    return legal_moves_int[i]

        # Fallback: discard any card, or play if no discard available
        for i, move in enumerate(legal_moves):
            if move["action_type"] == "DISCARD":
                return legal_moves_int[i]

        # Last resort: just pick first legal move
        return legal_moves_int[0]

    def _is_probably_playable(
        self,
        card_idx: int,
        knowledge: list,
        fireworks: Dict[str, int]
    ) -> bool:
        """Check if we have enough info to know a card is playable."""
        if not knowledge or card_idx >= len(knowledge):
            return False

        card_info = knowledge[card_idx]
        known_color = card_info.get("color")
        known_rank = card_info.get("rank")

        # If we know both color and rank, check if it's playable
        if known_color is not None and known_rank is not None:
            needed_rank = fireworks.get(known_color, 0)
            return known_rank == needed_rank

        # If we only know rank is 0 (a "1"), it might be playable
        if known_rank == 0:
            # Check if any color needs a 1
            for color, top in fireworks.items():
                if top == 0:
                    return True

        return False

    def _find_useful_hint(
        self,
        legal_moves: list,
        legal_moves_int: list,
        observation: Dict[str, Any]
    ) -> Optional[int]:
        """Find a hint that tells partner about a playable card."""
        fireworks = observation["fireworks"]
        observed_hands = observation["observed_hands"]

        # Partner's hand is at index 1 (we're index 0 from our perspective)
        if len(observed_hands) < 2:
            return None

        partner_hand = observed_hands[1]

        # Look for hints about playable cards
        for i, move in enumerate(legal_moves):
            if move["action_type"] == "REVEAL_COLOR":
                color = move["color"]
                needed_rank = fireworks.get(color, 0)
                # Check if partner has a playable card of this color
                for card in partner_hand:
                    if card.get("color") == color and card.get("rank") == needed_rank:
                        return legal_moves_int[i]

            elif move["action_type"] == "REVEAL_RANK":
                rank = move["rank"]
                # Hinting about 1s is often useful early game
                if rank == 0:  # rank 0 = "1" in the game
                    for card in partner_hand:
                        if card.get("rank") == 0:
                            # Check if any 1 is playable
                            for color, top in fireworks.items():
                                if top == 0 and card.get("color") == color:
                                    return legal_moves_int[i]

        return None
