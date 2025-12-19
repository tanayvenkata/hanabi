"""
Card Inference Utilities

Shared probabilistic reasoning about card identities based on visible information.

Used by all advisors (Play, Hint, Discard) to understand what cards are likely
in the agent's hand based on:
- Cards visible in other players' hands
- Cards in the discard pile
- Cards already played (on fireworks)
- Hints received about our cards
"""

from collections import defaultdict
from typing import Any, Dict, List


def calculate_card_probabilities(observation: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Calculate probabilistic knowledge about cards in our hand.

    For each card slot, calculate the probability distribution over possible
    card identities based on:
    - Cards visible in other players' hands
    - Cards in the discard pile
    - Cards already played (on fireworks)
    - Hints we've received about our cards

    Args:
        observation: Game observation dict from the environment

    Returns:
        List of dicts, one per card slot, containing:
        - slot: Slot index (0-4)
        - possibilities: List of possible cards with probabilities
          Each possibility contains:
          - color: Card color (R, G, B, W, Y)
          - rank: Card rank (1-5)
          - probability: Probability this is the card (0-1)
          - count: Number of this card type remaining unseen
    """
    # Standard Hanabi deck composition
    # Ranks: 1 (3 copies), 2 (2 copies), 3 (2 copies), 4 (2 copies), 5 (1 copy)
    colors = ["R", "G", "B", "W", "Y"]
    card_counts = {
        0: 3,  # Rank 1 (index 0)
        1: 2,  # Rank 2 (index 1)
        2: 2,  # Rank 3 (index 2)
        3: 2,  # Rank 4 (index 3)
        4: 1,  # Rank 5 (index 4)
    }

    # Count remaining cards of each type
    remaining = defaultdict(lambda: defaultdict(int))
    for color in colors:
        for rank in range(5):
            remaining[color][rank] = card_counts[rank]

    # Subtract cards we can see (other players' hands)
    observed_hands = observation.get("observed_hands", [])
    for i, hand in enumerate(observed_hands):
        if i == 0:  # Skip our own hand (we can't see it)
            continue
        for card in hand:
            color = card.get("color")
            rank = card.get("rank")
            if color and rank is not None and rank >= 0:
                remaining[color][rank] -= 1

    # Subtract discarded cards
    discard_pile = observation.get("discard_pile", [])
    for card in discard_pile:
        color = card.get("color")
        rank = card.get("rank")
        if color and rank is not None and rank >= 0:
            remaining[color][rank] -= 1

    # Subtract cards already played on fireworks
    fireworks = observation.get("fireworks", {})
    for color, max_rank in fireworks.items():
        for rank in range(max_rank):
            remaining[color][rank] -= 1

    # Now calculate probabilities for each card in our hand
    card_knowledge = observation.get("card_knowledge", [[]])[0] if observation.get("card_knowledge") else []
    probabilities = []

    for slot_idx in range(5):
        slot_prob = {"slot": slot_idx, "possibilities": []}

        if slot_idx < len(card_knowledge):
            knowledge = card_knowledge[slot_idx]
            known_color = knowledge.get("color")
            known_rank = knowledge.get("rank")

            # Filter possibilities based on what we know
            possible_cards = []
            total_remaining = 0

            for color in colors:
                # Skip if we know the color and this isn't it
                if known_color and color != known_color:
                    continue

                for rank in range(5):
                    # Skip if we know the rank and this isn't it
                    if known_rank is not None and known_rank >= 0 and rank != known_rank:
                        continue

                    count = remaining[color][rank]
                    if count > 0:
                        possible_cards.append({
                            "color": color,
                            "rank": rank + 1,  # Display rank (1-5)
                            "count": count
                        })
                        total_remaining += count

            # Calculate probabilities
            for card in possible_cards:
                prob = card["count"] / total_remaining if total_remaining > 0 else 0
                slot_prob["possibilities"].append({
                    "color": card["color"],
                    "rank": card["rank"],
                    "probability": prob,
                    "count": card["count"]
                })

            # Sort by probability (highest first)
            slot_prob["possibilities"].sort(key=lambda x: x["probability"], reverse=True)

        probabilities.append(slot_prob)

    return probabilities


def format_card_probabilities(card_probabilities: List[Dict[str, Any]], top_n: int = 3) -> str:
    """
    Format card probabilities for display in advisor prompts.

    Args:
        card_probabilities: Output from calculate_card_probabilities()
        top_n: Number of top possibilities to show per slot (default: 3)

    Returns:
        Formatted string showing probabilistic card knowledge
    """
    lines = []
    lines.append("=== PROBABILISTIC CARD KNOWLEDGE ===")
    lines.append("(What cards you likely have based on visible cards, discards, and hints)")
    lines.append("")

    for slot_data in card_probabilities:
        slot = slot_data.get("slot")
        possibilities = slot_data.get("possibilities", [])

        lines.append(f"Slot {slot}:")
        if possibilities:
            # Show top N most likely possibilities
            for i, poss in enumerate(possibilities[:top_n]):
                color = poss["color"]
                rank = poss["rank"]
                prob = poss["probability"]
                count = poss["count"]
                lines.append(f"  {color}{rank}: {prob*100:.1f}% (count: {count})")
            if len(possibilities) > top_n:
                lines.append(f"  ... and {len(possibilities) - top_n} other possibilities")
        else:
            lines.append("  (No valid possibilities)")
        lines.append("")

    return "\n".join(lines)
