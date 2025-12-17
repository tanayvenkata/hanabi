"""
Clue Efficiency Calculator for Hanabi

Calculates how efficient each possible hint is based on:
1. Number of cards revealed (more is better)
2. Whether it reveals playable cards (high priority)
3. Whether it reveals critical cards (cards needed for max score)
4. Information content (unique information is valuable)
"""

from typing import Any, Dict, List, Tuple


def calculate_clue_efficiency(
    observation: Dict[str, Any],
    player_id: int
) -> List[Dict[str, Any]]:
    """Calculate efficiency scores for all hint moves.

    Args:
        observation: The game observation from the environment
        player_id: The ID of the player giving the hint

    Returns:
        List of dicts with move info and efficiency scores, sorted by efficiency
    """
    # Get partner's hand
    partner_idx = 1 - player_id
    observed_hands = observation['observed_hands']

    if partner_idx >= len(observed_hands):
        return []

    partner_hand = observed_hands[partner_idx]

    # Get game state
    fireworks = observation['fireworks']
    discard_pile = observation.get('discard_pile', [])

    # Analyze all hint moves
    hint_analyses = []

    for move in observation['legal_moves']:
        action_type = move.get('action_type')

        if action_type == 'REVEAL_COLOR':
            color = move.get('color')
            analysis = _analyze_color_hint(color, partner_hand, fireworks, discard_pile)
            analysis['move'] = move
            hint_analyses.append(analysis)

        elif action_type == 'REVEAL_RANK':
            rank = move.get('rank')
            analysis = _analyze_rank_hint(rank, partner_hand, fireworks, discard_pile)
            analysis['move'] = move
            hint_analyses.append(analysis)

    # Sort by efficiency score (highest first)
    hint_analyses.sort(key=lambda x: x['efficiency_score'], reverse=True)

    return hint_analyses


def _analyze_color_hint(
    color: str,
    partner_hand: List[Dict[str, Any]],
    fireworks: Dict[str, int],
    discard_pile: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """Analyze efficiency of a color hint."""
    # Count matching cards
    matching_cards = [
        card for card in partner_hand
        if card.get('color') == color
    ]
    num_cards = len(matching_cards)

    # Check for playable cards (next rank needed for this color)
    next_rank_needed = fireworks.get(color, 0) + 1
    playable_cards = [
        card for card in matching_cards
        if card.get('rank') == next_rank_needed
    ]
    num_playable = len(playable_cards)

    # Check for critical cards (last copy of a needed rank)
    critical_cards = []
    for card in matching_cards:
        rank = card.get('rank', 0)
        if rank > fireworks.get(color, 0):  # Card is still needed
            if _is_last_copy(color, rank, discard_pile):
                critical_cards.append(card)
    num_critical = len(critical_cards)

    # Calculate efficiency score
    # Base: number of cards revealed
    # Bonus: +5 per playable card, +3 per critical card
    efficiency_score = (
        num_cards * 2.0 +           # Base value: cards revealed
        num_playable * 5.0 +         # Playable cards are very valuable
        num_critical * 3.0           # Critical cards need protection
    )

    return {
        'hint_type': 'color',
        'hint_value': color,
        'num_cards': num_cards,
        'num_playable': num_playable,
        'num_critical': num_critical,
        'efficiency_score': efficiency_score,
        'description': f"Hint {color}: reveals {num_cards} card(s), {num_playable} playable, {num_critical} critical"
    }


def _analyze_rank_hint(
    rank: int,
    partner_hand: List[Dict[str, Any]],
    fireworks: Dict[str, int],
    discard_pile: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """Analyze efficiency of a rank hint."""
    # Count matching cards
    matching_cards = [
        card for card in partner_hand
        if card.get('rank') == rank
    ]
    num_cards = len(matching_cards)

    # Check for playable cards (rank matches next needed rank for the color)
    playable_cards = []
    for card in matching_cards:
        color = card.get('color')
        next_rank_needed = fireworks.get(color, 0) + 1
        if rank == next_rank_needed:
            playable_cards.append(card)
    num_playable = len(playable_cards)

    # Check for critical cards
    critical_cards = []
    for card in matching_cards:
        color = card.get('color')
        if rank > fireworks.get(color, 0):  # Card is still needed
            if _is_last_copy(color, rank, discard_pile):
                critical_cards.append(card)
    num_critical = len(critical_cards)

    # Calculate efficiency score
    efficiency_score = (
        num_cards * 2.0 +           # Base value: cards revealed
        num_playable * 5.0 +         # Playable cards are very valuable
        num_critical * 3.0           # Critical cards need protection
    )

    return {
        'hint_type': 'rank',
        'hint_value': rank,
        'num_cards': num_cards,
        'num_playable': num_playable,
        'num_critical': num_critical,
        'efficiency_score': efficiency_score,
        'description': f"Hint rank {rank}: reveals {num_cards} card(s), {num_playable} playable, {num_critical} critical"
    }


def _is_last_copy(color: str, rank: int, discard_pile: List[Dict[str, Any]]) -> bool:
    """Check if a card is the last remaining copy (critical).

    In Hanabi, there are:
    - Three 1s of each color
    - Two 2s, 3s, 4s of each color
    - One 5 of each color
    """
    # Count how many of this card are in the discard pile
    discarded_count = sum(
        1 for card in discard_pile
        if card.get('color') == color and card.get('rank') == rank
    )

    # Determine total copies of this card in the deck
    if rank == 1:
        total_copies = 3
    elif rank == 5:
        total_copies = 1
    else:  # ranks 2, 3, 4
        total_copies = 2

    # If all but one are discarded, this is the last copy
    return discarded_count >= (total_copies - 1)


def format_hint_analysis(hint_analyses: List[Dict[str, Any]], top_n: int = 5) -> str:
    """Format hint analysis for display to LLM.

    Args:
        hint_analyses: List of hint analysis dicts from calculate_clue_efficiency
        top_n: Number of top hints to include

    Returns:
        Formatted string for inclusion in LLM prompt
    """
    if not hint_analyses:
        return "\nCLUE EFFICIENCY ANALYSIS: No hints available"

    lines = ["\nCLUE EFFICIENCY ANALYSIS (sorted by effectiveness):"]

    for i, analysis in enumerate(hint_analyses[:top_n], 1):
        lines.append(
            f"  {i}. {analysis['description']} "
            f"[Score: {analysis['efficiency_score']:.1f}]"
        )

    if len(hint_analyses) > top_n:
        lines.append(f"  ... and {len(hint_analyses) - top_n} more hints")

    lines.append("\nTIP: Higher scores = more information conveyed!")

    return "\n".join(lines)
