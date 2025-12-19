"""
Monolithic LLM Hanabi Agent

A pure LLM-driven agent that delegates all decision logic to a language model.
Python handles: state formatting, prompt building, response parsing.
The LLM handles: reasoning and decision making.
"""

import re
import requests
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from agents.example import BaseAgent
from agents.llm.prompts import GAME_RULES, TURN_PROMPT_TEMPLATE


@dataclass
class LLMDecision:
    """Parsed decision from LLM response."""
    action_index: int       # Index into legal_moves list
    reason: str             # One sentence explanation
    raw_reasoning: str      # Full LLM response (for debugging)


class MonolithAgent(BaseAgent):
    """
    Pure LLM-driven Hanabi agent.

    All decision logic is delegated to the LLM.
    Python handles: state formatting, prompt building, response parsing.
    """

    def __init__(self, player_id: int, config: Optional[Dict[str, Any]] = None):
        super().__init__(player_id, config)

        # LLM configuration
        self.llm_url = config.get("llm_url", "http://127.0.0.1:1234/v1/chat/completions") if config else "http://127.0.0.1:1234/v1/chat/completions"
        self.model = config.get("model", "nvidia/nemotron-3-nano") if config else "nvidia/nemotron-3-nano"
        self.temperature = config.get("temperature", 0.3) if config else 0.3
        self.verbose = config.get("verbose", False) if config else False

        # Tracking
        self.turn_count = 0
        self.discard_pile: List[Dict] = []  # Track discards for display

    def reset(self) -> None:
        """Reset at start of new game."""
        self.turn_count = 0
        self.discard_pile = []

    def observe_move(self, move: Dict[str, Any], acting_player: int) -> None:
        """Track discards for context."""
        if move.get("action_type") == "DISCARD":
            # We'll update discard pile from observation instead
            pass

    def act(self, observation: Dict[str, Any]) -> int:
        """Choose action by asking LLM."""
        self.turn_count += 1

        # 1. Format the game state
        state_prompt = self._format_game_state(observation)

        # 2. Format available actions
        actions_prompt = self._format_actions(observation)

        # 3. Build full prompt
        full_prompt = TURN_PROMPT_TEMPLATE.format(
            fireworks_display=state_prompt["fireworks"],
            needed_cards=state_prompt["needed"],
            hint_tokens=observation["information_tokens"],
            life_tokens=observation["life_tokens"],
            deck_size=observation["deck_size"],
            discard_pile=state_prompt["discards"],
            partner_hand_display=state_prompt["partner_hand"],
            my_hand_display=state_prompt["my_hand"],
            actions_display=actions_prompt,
        )

        if self.verbose:
            print(f"\n{'='*60}")
            print(f"TURN {self.turn_count} - Player {self.player_id}")
            print(f"{'='*60}")
            print(full_prompt)

        # 4. Call LLM
        response = self._call_llm(full_prompt)

        if self.verbose:
            print(f"\n--- LLM RESPONSE ---")
            print(response)

        # 5. Parse response
        decision = self._parse_response(response, observation)

        if self.verbose:
            print(f"\n--- DECISION ---")
            print(f"Action index: {decision.action_index}")
            print(f"Reason: {decision.reason}")

        # 6. Return the action integer
        return observation["legal_moves_as_int"][decision.action_index]

    def _format_game_state(self, obs: Dict[str, Any]) -> Dict[str, str]:
        """Convert observation to readable strings."""

        # Fireworks display
        fireworks = obs["fireworks"]
        fw_display = "  " + "  |  ".join(f"{c}: {r}" for c, r in fireworks.items())

        # What's needed next
        needed = ", ".join(f"{c}->{r+1}" for c, r in fireworks.items() if r < 5)
        if not needed:
            needed = "All complete!"

        # Discard pile
        discard_pile = obs.get("discard_pile", [])
        if discard_pile:
            discards = "[" + ", ".join(self._card_str(c) for c in discard_pile) + "]"
        else:
            discards = "[empty]"

        # Partner's hand (we CAN see this)
        # In 2-player: observed_hands[0] is our hidden hand, observed_hands[1] is partner
        partner_idx = 1 - self.player_id
        partner_cards = obs["observed_hands"][partner_idx]
        partner_knowledge = obs.get("card_knowledge", [[],[]])[partner_idx]

        partner_lines = []
        for i, card in enumerate(partner_cards):
            # What do we see
            card_str = self._card_str(card)
            # What does partner know about it
            if i < len(partner_knowledge):
                pk = partner_knowledge[i]
                known = self._knowledge_str(pk)
            else:
                known = "nothing"

            # Check if critical (5s are always critical since only 1 copy)
            critical_note = ""
            if card.get("rank") == 4:  # rank is 0-indexed, so 4 = "5"
                critical_note = "  <- CRITICAL (only one 5!)"

            # Check if chop (oldest unhinted)
            chop_note = ""
            if i == len(partner_cards) - 1:  # Simple: rightmost is chop
                chop_note = "  <- CHOP"

            partner_lines.append(f"  Slot {i}: {card_str:12} [Partner knows: {known}]{critical_note}{chop_note}")

        partner_hand = "\n".join(partner_lines)

        # My hand (I CANNOT see this - only hints)
        my_knowledge = obs.get("card_knowledge", [[],[]])[self.player_id]
        my_lines = []
        for i in range(5):  # Assume 5 cards
            if i < len(my_knowledge):
                mk = my_knowledge[i]
                known = self._knowledge_str(mk)
            else:
                known = "nothing"

            chop_note = ""
            if i == 4:  # Rightmost is chop
                chop_note = "  <- YOUR CHOP"

            my_lines.append(f"  Slot {i}: ??          [You know: {known}]{chop_note}")

        my_hand = "\n".join(my_lines)

        return {
            "fireworks": fw_display,
            "needed": needed,
            "discards": discards,
            "partner_hand": partner_hand,
            "my_hand": my_hand,
        }

    def _card_str(self, card: Dict) -> str:
        """Format a card as 'Color Rank' (e.g., 'Red 3')."""
        color = card.get("color")
        rank = card.get("rank")

        if color is None or rank is None or rank < 0:
            return "??"

        # Color might be a letter or full name
        color_names = {"R": "Red", "G": "Green", "B": "Blue", "W": "White", "Y": "Yellow"}
        color_display = color_names.get(color, color)

        # Rank is 0-indexed in the env, display as 1-5
        rank_display = rank + 1

        return f"{color_display} {rank_display}"

    def _knowledge_str(self, knowledge: Dict) -> str:
        """Format what's known about a card from hints."""
        color = knowledge.get("color")
        rank = knowledge.get("rank")

        parts = []
        if color:
            color_names = {"R": "Red", "G": "Green", "B": "Blue", "W": "White", "Y": "Yellow"}
            parts.append(f"color={color_names.get(color, color)}")
        if rank is not None and rank >= 0:
            parts.append(f"rank={rank + 1}")  # Display as 1-5

        return ", ".join(parts) if parts else "nothing"

    def _format_actions(self, obs: Dict[str, Any]) -> str:
        """Format legal actions as numbered list."""
        legal_moves = obs["legal_moves"]
        lines = []

        play_actions = []
        discard_actions = []
        hint_actions = []

        for i, move in enumerate(legal_moves):
            action_type = move.get("action_type", "")

            if action_type == "PLAY":
                slot = move.get("card_index", 0)
                play_actions.append(f"  [{i}] PLAY slot {slot}")

            elif action_type == "DISCARD":
                slot = move.get("card_index", 0)
                discard_actions.append(f"  [{i}] DISCARD slot {slot}")

            elif action_type == "REVEAL_COLOR":
                color = move.get("color", "?")
                target = move.get("target_offset", 1)
                color_names = {"R": "Red", "G": "Green", "B": "Blue", "W": "White", "Y": "Yellow"}
                color_name = color_names.get(color, color)
                hint_actions.append(f"  [{i}] HINT partner: color {color_name}")

            elif action_type == "REVEAL_RANK":
                rank = move.get("rank", 0)
                target = move.get("target_offset", 1)
                rank_display = rank + 1  # 0-indexed to 1-5
                hint_actions.append(f"  [{i}] HINT partner: rank {rank_display}")

        # Build organized output
        if play_actions:
            lines.append("PLAY actions:")
            lines.extend(play_actions)
            lines.append("")

        if discard_actions:
            lines.append("DISCARD actions:")
            lines.extend(discard_actions)
            lines.append("")

        if hint_actions:
            lines.append(f"HINT actions (you have {obs['information_tokens']} tokens):")
            lines.extend(hint_actions)

        return "\n".join(lines)

    def _call_llm(self, prompt: str) -> str:
        """Call LM Studio API."""
        try:
            response = requests.post(
                self.llm_url,
                json={
                    "model": self.model,
                    "messages": [
                        {"role": "system", "content": GAME_RULES},
                        {"role": "user", "content": prompt}
                    ],
                    "temperature": self.temperature,
                    # No max_tokens - let it think freely
                },
                timeout=120,
            )
            response.raise_for_status()
            return response.json()["choices"][0]["message"]["content"]
        except requests.exceptions.RequestException as e:
            # If LLM fails, return a fallback response
            return f"DECISION: 0\nREASON: LLM request failed ({e}), defaulting to first action."

    def _parse_response(self, response: str, obs: Dict[str, Any]) -> LLMDecision:
        """Extract structured decision from LLM response."""

        # Some models use </think> tags - get content after that if present
        if "</think>" in response:
            response_to_parse = response.split("</think>")[-1]
        else:
            response_to_parse = response

        # Find the LAST DECISION: N (in case model mentioned it in reasoning too)
        decision_matches = list(re.finditer(r'DECISION:\s*(\d+)', response_to_parse, re.IGNORECASE))
        reason_match = re.search(r'REASON:\s*(.+?)(?:\n|$)', response_to_parse, re.IGNORECASE)

        if decision_matches:
            action_index = int(decision_matches[-1].group(1))  # Take last match
            # Validate it's in range
            max_action = len(obs["legal_moves"]) - 1
            if action_index > max_action:
                action_index = 0  # Fallback to first action
        else:
            action_index = 0  # Fallback

        reason = reason_match.group(1).strip() if reason_match else "No reason provided"

        return LLMDecision(
            action_index=action_index,
            reason=reason,
            raw_reasoning=response,
        )


# For quick testing
if __name__ == "__main__":
    # Test with a mock observation
    mock_obs = {
        "current_player": 0,
        "legal_moves": [
            {"action_type": "PLAY", "card_index": 0},
            {"action_type": "PLAY", "card_index": 1},
            {"action_type": "DISCARD", "card_index": 0},
            {"action_type": "REVEAL_COLOR", "color": "R", "target_offset": 1},
            {"action_type": "REVEAL_RANK", "rank": 0, "target_offset": 1},
        ],
        "legal_moves_as_int": [0, 1, 5, 10, 15],
        "observed_hands": [
            # My hand (hidden)
            [{"color": None, "rank": -1}] * 5,
            # Partner's hand (visible)
            [
                {"color": "R", "rank": 0},  # Red 1
                {"color": "G", "rank": 2},  # Green 3
                {"color": "B", "rank": 1},  # Blue 2
                {"color": "W", "rank": 4},  # White 5
                {"color": "Y", "rank": 0},  # Yellow 1
            ],
        ],
        "fireworks": {"R": 0, "G": 0, "B": 0, "W": 0, "Y": 0},
        "information_tokens": 8,
        "life_tokens": 3,
        "deck_size": 40,
        "discard_pile": [],
        "card_knowledge": [
            # My knowledge
            [{"color": None, "rank": None}] * 5,
            # Partner's knowledge
            [{"color": None, "rank": None}] * 5,
        ],
    }

    agent = MonolithAgent(player_id=0, config={"verbose": True})
    action = agent.act(mock_obs)
    print(f"\nFinal action: {action}")
