"""
BasicLLMAgent - An LLM-powered Hanabi agent

Uses a language model (via LM Studio or OpenAI-compatible API) to make decisions.
"""

import json
import random
import re
from typing import Any, Dict, List, Optional

from openai import OpenAI

from agents.example import BaseAgent
from agents.llm.clue_efficiency import calculate_clue_efficiency, format_hint_analysis


class BasicLLMAgent(BaseAgent):
    """An agent that uses a language model to play Hanabi.

    This agent sends game observations to an LLM and parses its responses
    to choose actions. It maintains conversation history for context.

    Config options:
        - api_base: API endpoint (default: "http://localhost:1234/v1" for LM Studio)
        - api_key: API key (default: "lm-studio" - LM Studio doesn't need a real key)
        - model: Model name (default: "local-model")
        - temperature: Sampling temperature (default: 0.7)
        - max_tokens: Max response tokens (default: 200)
        - memory_mode: "full" or "summary" (default: "summary")
            - "full": Maintains complete conversation history (more context, more tokens)
            - "summary": Compact summary of key events (cheaper, faster)
    """

    def __init__(self, player_id: int, config: Optional[Dict[str, Any]] = None):
        super().__init__(player_id, config)

        # Configuration
        self.api_base = self.config.get("api_base", "http://localhost:1234/v1")
        self.api_key = self.config.get("api_key", "lm-studio")
        self.model = self.config.get("model", "local-model")
        self.temperature = self.config.get("temperature", 0.7)
        self.max_tokens = self.config.get("max_tokens", 200)
        self.memory_mode = self.config.get("memory_mode", "summary")  # "full" or "summary"

        # Initialize OpenAI client
        self.client = OpenAI(base_url=self.api_base, api_key=self.api_key)

        # Game state tracking
        self.conversation_history: List[Dict[str, str]] = []
        self.turn_number = 0
        self.game_history: List[str] = []

        # Summarized state tracking (more compact than full history)
        self.hints_received: List[str] = []
        self.hints_given: List[str] = []
        self.my_plays: List[str] = []
        self.my_discards: List[str] = []
        self.partner_plays: List[str] = []
        self.partner_discards: List[str] = []

    def reset(self) -> None:
        """Reset agent state for a new game."""
        self.conversation_history = [
            {"role": "system", "content": self._get_system_prompt()}
        ]
        self.turn_number = 0
        self.game_history = []

        # Reset summary tracking
        self.hints_received = []
        self.hints_given = []
        self.my_plays = []
        self.my_discards = []
        self.partner_plays = []
        self.partner_discards = []

    def _get_system_prompt(self) -> str:
        """Create the system prompt explaining Hanabi and the agent's role."""
        return f"""You are an expert Hanabi player (Player {self.player_id}). Hanabi is a cooperative card game where players work together to play cards in the correct order.

RULES:
- Goal: Play cards 1-5 in each of 5 colors (R=Red, G=Green, B=Blue, Y=Yellow, W=White)
- You CANNOT see your own hand, but you can see your partner's cards
- Actions: PLAY a card, DISCARD a card, or HINT (tell partner about a color or rank)
- Information tokens: 8 total, spend 1 to give a hint, regain 1 when discarding
- Life tokens: 3 total, lose 1 for playing an invalid card (game over at 0)
- Maximum score: 25 points (all cards played correctly)

STRATEGY TIPS:
- Early game: Build information tokens by discarding useless cards
- Give hints about playable cards or critical cards to save
- When giving hints, prioritize clues with HIGH EFFICIENCY SCORES (reveal more cards)
- Play cards when you're certain they're playable
- Don't discard cards that might be the last copy of important cards
- Use the clue efficiency analysis to choose the most informative hints

YOUR TASK:
I will present you with the current game state and numbered action options.
Respond with ONLY the number of your chosen action and a brief reason.
Format: "Action: [number]. Reason: [brief explanation]"

Example response: "Action: 3. Reason: This card is likely playable based on previous hints."
"""

    def _format_observation(self, observation: Dict[str, Any]) -> str:
        """Convert observation dict into a clear text prompt for the LLM."""
        obs_lines = [
            f"\n=== TURN {self.turn_number + 1} ===",
            f"\nGAME STATE:",
            f"- Fireworks (cards played): {self._format_fireworks(observation['fireworks'])}",
            f"- Information tokens: {observation['information_tokens']}/8",
            f"- Life tokens: {observation['life_tokens']}/3",
            f"- Cards left in deck: {observation['deck_size']}",
        ]

        # Show discard pile
        if observation['discard_pile']:
            obs_lines.append(f"- Discard pile: {self._format_cards(observation['discard_pile'][-5:])} (last 5)")

        # Show partner's hand
        obs_lines.append(f"\nPARTNER'S HAND:")
        partner_idx = 1 - self.player_id
        if partner_idx < len(observation['observed_hands']):
            partner_hand = observation['observed_hands'][partner_idx]
            obs_lines.append(f"  {self._format_cards(partner_hand)}")
        else:
            obs_lines.append("  (Not visible)")

        # Show your hand (hidden, but with any hints you've received)
        obs_lines.append(f"\nYOUR HAND: [Hidden - you cannot see your own cards]")
        obs_lines.append("  (Use hints from your partner to reason about your cards)")

        # Format legal moves as numbered options
        obs_lines.append(f"\nAVAILABLE ACTIONS:")
        for i, move in enumerate(observation['legal_moves'], 1):
            obs_lines.append(f"  {i}. {self._format_move(move)}")

        # Add clue efficiency analysis if hints are available
        hint_analyses = calculate_clue_efficiency(observation, self.player_id)
        if hint_analyses:
            obs_lines.append(format_hint_analysis(hint_analyses, top_n=5))

        # Add recent game history if available
        if self.game_history:
            obs_lines.append(f"\nRECENT HISTORY:")
            for event in self.game_history[-3:]:  # Last 3 events
                obs_lines.append(f"  - {event}")

        obs_lines.append(f"\nChoose an action (1-{len(observation['legal_moves'])}):")

        return "\n".join(obs_lines)

    def _format_fireworks(self, fireworks: Dict[str, int]) -> str:
        """Format fireworks dict as readable string."""
        return ", ".join([f"{color}:{rank}" for color, rank in sorted(fireworks.items())])

    def _format_cards(self, cards: List[Dict[str, Any]]) -> str:
        """Format list of cards as readable string."""
        if not cards:
            return "[]"

        card_strs = []
        for card in cards:
            if 'color' in card and 'rank' in card:
                color = card['color']
                rank = card['rank']
                card_strs.append(f"{color}{rank}")
            else:
                card_strs.append("?")

        return "[" + ", ".join(card_strs) + "]"

    def _format_move(self, move: Dict[str, Any]) -> str:
        """Format a move dict as readable string."""
        action_type = move.get('action_type', 'UNKNOWN')

        if action_type == 'PLAY':
            return f"PLAY card at position {move.get('card_index', '?')}"
        elif action_type == 'DISCARD':
            return f"DISCARD card at position {move.get('card_index', '?')}"
        elif action_type == 'REVEAL_COLOR':
            target = move.get('target_offset', '?')
            color = move.get('color', '?')
            return f"HINT to partner: tell them about {color} cards"
        elif action_type == 'REVEAL_RANK':
            target = move.get('target_offset', '?')
            rank = move.get('rank', '?')
            return f"HINT to partner: tell them about rank {rank} cards"
        else:
            return f"UNKNOWN ACTION: {move}"

    def _get_summary_context(self) -> str:
        """Generate a compact summary of game events so far (for summary mode)."""
        summary_parts = []

        if self.hints_received:
            summary_parts.append("HINTS YOU RECEIVED:")
            for hint in self.hints_received[-5:]:  # Last 5 hints
                summary_parts.append(f"  - {hint}")

        if self.hints_given:
            summary_parts.append("\nHINTS YOU GAVE:")
            for hint in self.hints_given[-5:]:
                summary_parts.append(f"  - {hint}")

        if self.my_plays:
            summary_parts.append("\nYOUR PLAYS:")
            for play in self.my_plays[-3:]:
                summary_parts.append(f"  - {play}")

        if self.my_discards:
            summary_parts.append("\nYOUR DISCARDS:")
            for discard in self.my_discards[-3:]:
                summary_parts.append(f"  - {discard}")

        if self.partner_plays:
            summary_parts.append("\nPARTNER'S PLAYS:")
            for play in self.partner_plays[-3:]:
                summary_parts.append(f"  - {play}")

        if self.partner_discards:
            summary_parts.append("\nPARTNER'S DISCARDS:")
            for discard in self.partner_discards[-3:]:
                summary_parts.append(f"  - {discard}")

        if not summary_parts:
            return "\nGAME MEMORY: (No significant events yet)"

        return "\nGAME MEMORY:\n" + "\n".join(summary_parts)

    def act(self, observation: Dict[str, Any]) -> int:
        """Choose an action using the LLM."""
        try:
            # Format observation as prompt
            user_message = self._format_observation(observation)

            # Add summarized context if in summary mode
            if self.memory_mode == "summary":
                user_message = self._get_summary_context() + "\n" + user_message

            # Build messages list based on memory mode
            if self.memory_mode == "full":
                # Full conversation history
                messages = self.conversation_history + [{"role": "user", "content": user_message}]
            else:
                # Summary mode: only system prompt + current state with summary
                messages = [
                    self.conversation_history[0],  # System prompt
                    {"role": "user", "content": user_message}
                ]

            # Get LLM response
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=self.temperature,
                max_tokens=self.max_tokens
            )

            llm_response = response.choices[0].message.content

            # Parse action from response
            action_idx = self._parse_action_choice(llm_response, observation)

            # Update conversation history (only in full mode)
            if self.memory_mode == "full":
                self.conversation_history.append({"role": "user", "content": user_message})
                self.conversation_history.append({"role": "assistant", "content": llm_response})

            self.turn_number += 1

            return action_idx

        except Exception as e:
            print(f"[BasicLLMAgent] Error calling LLM: {e}")
            print(f"[BasicLLMAgent] Falling back to random action")
            return random.choice(observation['legal_moves_as_int'])

    def _parse_action_choice(self, llm_response: str, observation: Dict[str, Any]) -> int:
        """Parse the LLM's response to extract the chosen action number.

        Looks for patterns like:
        - "Action: 3"
        - "Choose option 3"
        - "3. Reason: ..."
        - Just "3"

        Falls back to random action if parsing fails.
        """
        legal_moves = observation['legal_moves_as_int']
        num_options = len(legal_moves)

        # Try multiple parsing strategies
        patterns = [
            r'Action:\s*(\d+)',           # "Action: 3"
            r'option\s*(\d+)',            # "option 3"
            r'^(\d+)\.?',                 # "3." or "3" at start
            r'\b(\d+)\b',                 # any number
        ]

        for pattern in patterns:
            match = re.search(pattern, llm_response, re.IGNORECASE)
            if match:
                try:
                    choice = int(match.group(1))
                    # Convert to 0-indexed and validate
                    if 1 <= choice <= num_options:
                        action_idx = legal_moves[choice - 1]
                        print(f"[BasicLLMAgent] Chose action {choice}: {self._format_move(observation['legal_moves'][choice - 1])}")
                        print(f"[BasicLLMAgent] Reasoning: {llm_response[:100]}...")
                        return action_idx
                except (ValueError, IndexError):
                    continue

        # Fallback: random legal move
        print(f"[BasicLLMAgent] Could not parse action from: {llm_response[:100]}")
        print(f"[BasicLLMAgent] Falling back to random action")
        action_idx = random.choice(legal_moves)
        return action_idx

    def observe_move(self, move: Dict[str, Any], acting_player: int) -> None:
        """Track moves for context (both full history and summary)."""
        player_name = "You" if acting_player == self.player_id else "Partner"
        move_desc = self._format_move(move)
        self.game_history.append(f"Turn {self.turn_number}: {player_name} - {move_desc}")

        # Update summary tracking based on move type
        action_type = move.get('action_type', '')
        is_me = (acting_player == self.player_id)

        if action_type == 'PLAY':
            card_idx = move.get('card_index', '?')
            event = f"Turn {self.turn_number}: Played card from position {card_idx}"
            if is_me:
                self.my_plays.append(event)
            else:
                self.partner_plays.append(event)

        elif action_type == 'DISCARD':
            card_idx = move.get('card_index', '?')
            event = f"Turn {self.turn_number}: Discarded card from position {card_idx}"
            if is_me:
                self.my_discards.append(event)
            else:
                self.partner_discards.append(event)

        elif action_type == 'REVEAL_COLOR':
            color = move.get('color', '?')
            if is_me:
                self.hints_given.append(f"Turn {self.turn_number}: Told partner about {color} cards")
            else:
                self.hints_received.append(f"Turn {self.turn_number}: Partner told you about {color} cards")

        elif action_type == 'REVEAL_RANK':
            rank = move.get('rank', '?')
            if is_me:
                self.hints_given.append(f"Turn {self.turn_number}: Told partner about rank {rank} cards")
            else:
                self.hints_received.append(f"Turn {self.turn_number}: Partner told you about rank {rank} cards")
