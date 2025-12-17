"""
Greedy Efficiency Agent for Hanabi

This agent uses a greedy strategy focused on maximizing clue efficiency:
- When giving hints, always choose the hint with the highest efficiency score
- For play/discard decisions, uses simple heuristics:
  - Discard when info tokens are low (< 3)
  - Otherwise play the oldest card (leftmost position)
"""

import random
from typing import Any, Dict, Optional

from agents.example import BaseAgent
from agents.llm.clue_efficiency import calculate_clue_efficiency


class GreedyEfficiencyAgent(BaseAgent):
    """An agent that maximizes clue efficiency when giving hints.

    Strategy:
    1. If hints are available and we have info tokens, pick the most efficient hint
    2. If info tokens are low (< 3), prefer discarding
    3. Otherwise, play the leftmost card (oldest card, likely to have been hinted)
    """

    def __init__(self, player_id: int, config: Optional[Dict[str, Any]] = None):
        super().__init__(player_id, config)
        self.prefer_hints = config.get("prefer_hints", True) if config else True
        self.hint_threshold = config.get("hint_threshold", 3) if config else 3

    def act(self, observation: Dict[str, Any]) -> int:
        """Choose action by maximizing clue efficiency or using simple heuristics."""
        legal_moves = observation['legal_moves']
        legal_moves_as_int = observation['legal_moves_as_int']

        # Separate moves by type
        hint_moves = []
        play_moves = []
        discard_moves = []

        for i, move in enumerate(legal_moves):
            action_type = move.get('action_type')
            if action_type in ['REVEAL_COLOR', 'REVEAL_RANK']:
                hint_moves.append(i)
            elif action_type == 'PLAY':
                play_moves.append(i)
            elif action_type == 'DISCARD':
                discard_moves.append(i)

        # Get game state
        info_tokens = observation['information_tokens']

        # Strategy: Prioritize hints if we have tokens and hints are available
        if hint_moves and info_tokens > 0 and self.prefer_hints:
            # Calculate efficiency for all hints
            hint_analyses = calculate_clue_efficiency(observation, self.player_id)

            if hint_analyses:
                # Find the most efficient hint
                best_hint = hint_analyses[0]  # Already sorted by efficiency
                best_move = best_hint['move']

                # Find this move in legal_moves
                for i, move in enumerate(legal_moves):
                    if self._moves_match(move, best_move):
                        return legal_moves_as_int[i]

        # If we're low on info tokens, prefer discarding
        if info_tokens < self.hint_threshold and discard_moves:
            # Discard the oldest card (leftmost, index 0)
            return legal_moves_as_int[discard_moves[0]]

        # Otherwise, try to play the oldest card
        if play_moves:
            return legal_moves_as_int[play_moves[0]]

        # Fallback: random legal move
        return random.choice(legal_moves_as_int)

    def _moves_match(self, move1: Dict[str, Any], move2: Dict[str, Any]) -> bool:
        """Check if two moves are the same."""
        if move1.get('action_type') != move2.get('action_type'):
            return False

        action_type = move1.get('action_type')

        if action_type == 'REVEAL_COLOR':
            return move1.get('color') == move2.get('color')
        elif action_type == 'REVEAL_RANK':
            return move1.get('rank') == move2.get('rank')
        elif action_type in ['PLAY', 'DISCARD']:
            return move1.get('card_index') == move2.get('card_index')

        return False


class BalancedGreedyAgent(GreedyEfficiencyAgent):
    """A variant that balances between hints and play/discard actions.

    Only gives hints when efficiency is high enough to justify spending a token.
    """

    def __init__(self, player_id: int, config: Optional[Dict[str, Any]] = None):
        super().__init__(player_id, config)
        # Only give hints if efficiency score is above this threshold
        self.min_efficiency = config.get("min_efficiency", 4.0) if config else 4.0

    def act(self, observation: Dict[str, Any]) -> int:
        """Choose action, but only give hints if efficiency is high enough."""
        legal_moves = observation['legal_moves']
        legal_moves_as_int = observation['legal_moves_as_int']

        # Separate moves by type
        hint_moves = []
        play_moves = []
        discard_moves = []

        for i, move in enumerate(legal_moves):
            action_type = move.get('action_type')
            if action_type in ['REVEAL_COLOR', 'REVEAL_RANK']:
                hint_moves.append(i)
            elif action_type == 'PLAY':
                play_moves.append(i)
            elif action_type == 'DISCARD':
                discard_moves.append(i)

        # Get game state
        info_tokens = observation['information_tokens']

        # Check if we should give a hint (only if efficiency is high)
        if hint_moves and info_tokens > 1:
            hint_analyses = calculate_clue_efficiency(observation, self.player_id)

            if hint_analyses:
                best_hint = hint_analyses[0]

                # Only give hint if it's efficient enough
                if best_hint['efficiency_score'] >= self.min_efficiency:
                    best_move = best_hint['move']

                    # Find this move in legal_moves
                    for i, move in enumerate(legal_moves):
                        if self._moves_match(move, best_move):
                            return legal_moves_as_int[i]

        # If we're low on info tokens, prefer discarding
        if info_tokens < self.hint_threshold and discard_moves:
            return legal_moves_as_int[discard_moves[0]]

        # Otherwise, try to play the oldest card
        if play_moves:
            return legal_moves_as_int[play_moves[0]]

        # Fallback: discard if possible, otherwise random
        if discard_moves:
            return legal_moves_as_int[discard_moves[0]]

        return random.choice(legal_moves_as_int)
