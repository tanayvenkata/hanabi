"""
Static test scenarios for the Monolithic LLM Agent.

Each scenario is a mock observation with an expected "good" action category.
Use these to test LLM decision quality without running full games.
"""

from typing import Dict, Any, List


def make_observation(
    fireworks: Dict[str, int],
    partner_hand: List[Dict],
    my_knowledge: List[Dict],
    partner_knowledge: List[Dict],
    hint_tokens: int = 8,
    life_tokens: int = 3,
    deck_size: int = 40,
    discard_pile: List[Dict] = None,
) -> Dict[str, Any]:
    """Helper to build a mock observation."""

    if discard_pile is None:
        discard_pile = []

    # Build legal moves (simplified - always include standard options)
    legal_moves = []
    legal_moves_as_int = []
    action_id = 0

    # Play actions (5 cards)
    for i in range(5):
        legal_moves.append({"action_type": "PLAY", "card_index": i})
        legal_moves_as_int.append(action_id)
        action_id += 1

    # Discard actions (5 cards) - only if not at max hints
    if hint_tokens < 8:
        for i in range(5):
            legal_moves.append({"action_type": "DISCARD", "card_index": i})
            legal_moves_as_int.append(action_id)
            action_id += 1

    # Hint actions - only if tokens available
    if hint_tokens > 0:
        for color in ["R", "G", "B", "W", "Y"]:
            # Only add hint if partner has that color
            if any(c.get("color") == color for c in partner_hand):
                legal_moves.append({"action_type": "REVEAL_COLOR", "color": color, "target_offset": 1})
                legal_moves_as_int.append(action_id)
                action_id += 1

        for rank in range(5):
            # Only add hint if partner has that rank
            if any(c.get("rank") == rank for c in partner_hand):
                legal_moves.append({"action_type": "REVEAL_RANK", "rank": rank, "target_offset": 1})
                legal_moves_as_int.append(action_id)
                action_id += 1

    return {
        "current_player": 0,
        "legal_moves": legal_moves,
        "legal_moves_as_int": legal_moves_as_int,
        "observed_hands": [
            [{"color": None, "rank": -1}] * 5,  # My hand (hidden)
            partner_hand,  # Partner's hand (visible)
        ],
        "fireworks": fireworks,
        "information_tokens": hint_tokens,
        "life_tokens": life_tokens,
        "deck_size": deck_size,
        "discard_pile": discard_pile,
        "card_knowledge": [
            my_knowledge,
            partner_knowledge,
        ],
    }


# =============================================================================
# TEST SCENARIOS
# =============================================================================

SCENARIO_1_HINT_PLAYABLE = {
    "name": "Partner has playable 1s - should hint",
    "description": "All fireworks at 0, partner has Red 1 and Yellow 1. Should hint 1s.",
    "observation": make_observation(
        fireworks={"R": 0, "G": 0, "B": 0, "W": 0, "Y": 0},
        partner_hand=[
            {"color": "R", "rank": 0},  # Red 1 - playable!
            {"color": "G", "rank": 2},  # Green 3 - not playable
            {"color": "B", "rank": 1},  # Blue 2 - not playable
            {"color": "W", "rank": 3},  # White 4 - not playable
            {"color": "Y", "rank": 0},  # Yellow 1 - playable!
        ],
        my_knowledge=[{"color": None, "rank": None}] * 5,
        partner_knowledge=[{"color": None, "rank": None}] * 5,
        hint_tokens=8,
    ),
    "expected_action_type": "REVEAL_RANK",  # Hint about 1s
    "expected_reason": "Partner has two playable 1s",
}


SCENARIO_2_PLAY_KNOWN = {
    "name": "I know I have a playable card - should play",
    "description": "I've been hinted that slot 0 is Red 1, fireworks at 0. Should play it.",
    "observation": make_observation(
        fireworks={"R": 0, "G": 0, "B": 0, "W": 0, "Y": 0},
        partner_hand=[
            {"color": "G", "rank": 2},  # Green 3
            {"color": "B", "rank": 3},  # Blue 4
            {"color": "W", "rank": 2},  # White 3
            {"color": "Y", "rank": 1},  # Yellow 2
            {"color": "R", "rank": 4},  # Red 5
        ],
        my_knowledge=[
            {"color": "R", "rank": 0},  # I KNOW slot 0 is Red 1!
            {"color": None, "rank": None},
            {"color": None, "rank": None},
            {"color": None, "rank": None},
            {"color": None, "rank": None},
        ],
        partner_knowledge=[{"color": None, "rank": None}] * 5,
        hint_tokens=6,
    ),
    "expected_action_type": "PLAY",
    "expected_slot": 0,
    "expected_reason": "I know slot 0 is Red 1, which is playable",
}


SCENARIO_3_SAVE_CRITICAL = {
    "name": "Partner's chop is a 5 - must save!",
    "description": "Partner is about to discard a White 5 (critical). Must hint to save it.",
    "observation": make_observation(
        fireworks={"R": 2, "G": 1, "B": 0, "W": 3, "Y": 2},
        partner_hand=[
            {"color": "R", "rank": 3},  # Red 4
            {"color": "G", "rank": 2},  # Green 3
            {"color": "B", "rank": 1},  # Blue 2
            {"color": "Y", "rank": 3},  # Yellow 4
            {"color": "W", "rank": 4},  # White 5 - CRITICAL on chop!
        ],
        my_knowledge=[{"color": None, "rank": None}] * 5,
        partner_knowledge=[{"color": None, "rank": None}] * 5,  # Partner knows nothing!
        hint_tokens=4,
    ),
    "expected_action_type": "REVEAL_RANK",  # Hint 5 to save it
    "expected_reason": "Must save the White 5 before partner discards it",
}


SCENARIO_4_DISCARD_USELESS = {
    "name": "I have a known useless card - discard it",
    "description": "I know slot 2 is Red 1, but Red firework is at 3. Discard it.",
    "observation": make_observation(
        fireworks={"R": 3, "G": 2, "B": 1, "W": 0, "Y": 1},
        partner_hand=[
            {"color": "G", "rank": 3},  # Green 4
            {"color": "B", "rank": 2},  # Blue 3
            {"color": "W", "rank": 0},  # White 1
            {"color": "Y", "rank": 2},  # Yellow 3
            {"color": "R", "rank": 4},  # Red 5
        ],
        my_knowledge=[
            {"color": None, "rank": None},
            {"color": None, "rank": None},
            {"color": "R", "rank": 0},  # I know slot 2 is Red 1 - USELESS!
            {"color": None, "rank": None},
            {"color": None, "rank": None},
        ],
        partner_knowledge=[{"color": None, "rank": None}] * 5,
        hint_tokens=5,  # Not max, so can discard
    ),
    "expected_action_type": "DISCARD",
    "expected_slot": 2,
    "expected_reason": "Red 1 is useless since Red firework is already at 3",
}


SCENARIO_5_NO_INFO_DISCARD_CHOP = {
    "name": "No useful info, no hints - discard chop",
    "description": "Zero hint tokens, know nothing about my hand. Must discard chop (slot 4).",
    "observation": make_observation(
        fireworks={"R": 1, "G": 1, "B": 0, "W": 0, "Y": 0},
        partner_hand=[
            {"color": "G", "rank": 3},  # Green 4 - not playable
            {"color": "B", "rank": 2},  # Blue 3 - not playable
            {"color": "W", "rank": 3},  # White 4 - not playable
            {"color": "Y", "rank": 2},  # Yellow 3 - not playable
            {"color": "R", "rank": 3},  # Red 4 - not playable
        ],
        my_knowledge=[{"color": None, "rank": None}] * 5,
        partner_knowledge=[{"color": None, "rank": None}] * 5,
        hint_tokens=0,  # No hints available!
    ),
    "expected_action_type": "DISCARD",
    "expected_slot": 4,  # Chop
    "expected_reason": "No hint tokens and no info about my cards, must discard chop",
}


# All scenarios for iteration
ALL_SCENARIOS = [
    SCENARIO_1_HINT_PLAYABLE,
    SCENARIO_2_PLAY_KNOWN,
    SCENARIO_3_SAVE_CRITICAL,
    SCENARIO_4_DISCARD_USELESS,
    SCENARIO_5_NO_INFO_DISCARD_CHOP,
]


def run_scenario_test(agent, scenario: Dict) -> Dict:
    """Run a single scenario and check if decision matches expected."""
    print(f"\n{'='*60}")
    print(f"SCENARIO: {scenario['name']}")
    print(f"{'='*60}")
    print(f"Description: {scenario['description']}")
    print()

    obs = scenario["observation"]
    action_int = agent.act(obs)

    # Find which action was chosen
    action_idx = obs["legal_moves_as_int"].index(action_int)
    chosen_move = obs["legal_moves"][action_idx]

    # Check if it matches expected
    expected_type = scenario.get("expected_action_type")
    actual_type = chosen_move.get("action_type")

    type_match = (actual_type == expected_type)

    # Check slot if applicable
    expected_slot = scenario.get("expected_slot")
    actual_slot = chosen_move.get("card_index")
    slot_match = (expected_slot is None) or (actual_slot == expected_slot)

    success = type_match and slot_match

    print(f"\n--- RESULT ---")
    print(f"Expected: {expected_type}" + (f" slot {expected_slot}" if expected_slot is not None else ""))
    print(f"Actual:   {actual_type}" + (f" slot {actual_slot}" if actual_slot is not None else ""))
    print(f"Match: {'YES' if success else 'NO'}")

    return {
        "scenario": scenario["name"],
        "success": success,
        "expected": expected_type,
        "actual": actual_type,
        "chosen_move": chosen_move,
    }


def run_all_tests(agent) -> None:
    """Run all scenario tests and print summary."""
    results = []

    for scenario in ALL_SCENARIOS:
        result = run_scenario_test(agent, scenario)
        results.append(result)

    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")

    passed = sum(1 for r in results if r["success"])
    total = len(results)

    for r in results:
        status = "PASS" if r["success"] else "FAIL"
        print(f"  [{status}] {r['scenario']}")

    print(f"\nTotal: {passed}/{total} passed")


if __name__ == "__main__":
    from agents.llm.monolith import MonolithAgent

    agent = MonolithAgent(player_id=0, config={"verbose": True})
    run_all_tests(agent)
