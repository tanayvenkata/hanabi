"""
Test script to demonstrate clue efficiency feature

This script shows how the LLM agent now receives clue efficiency analysis
to help it make better decisions when giving hints.
"""

from agents.llm import BasicLLMAgent
from agents.example import RandomAgent
from shared import run_game
import sys

def test_clue_efficiency_display():
    """Run a game and show how clue efficiency is displayed."""
    print("="*60)
    print("Testing Clue Efficiency Feature")
    print("="*60)
    print()
    print("This enhanced agent now:")
    print("- Calculates efficiency scores for each possible hint")
    print("- Shows which hints reveal the most cards")
    print("- Highlights hints that reveal playable or critical cards")
    print("- Helps the LLM make better clue decisions")
    print()
    print("="*60)
    print()

    # Create enhanced LLM agent
    llm_agent = BasicLLMAgent(
        player_id=0,
        config={
            "api_base": "http://localhost:1234/v1",
            "api_key": "lm-studio",
            "model": "local-model",
            "memory_mode": "summary",
            "temperature": 0.7,
        }
    )

    # Create opponent
    random_agent = RandomAgent(player_id=1)

    # Run game with verbose output
    print("Running game with enhanced LLM agent...")
    print("Watch for 'CLUE EFFICIENCY ANALYSIS' in the output!")
    print()

    result = run_game(llm_agent, random_agent, seed=42, verbose=True)

    print()
    print("="*60)
    print(f"Final Score: {result['score']}/25")
    print(f"Turns played: {result['turns']}")
    print("="*60)
    print()
    print("The agent can now see which hints are most efficient!")
    print("Higher efficiency scores = more information conveyed")


def manual_clue_efficiency_demo():
    """Demonstrate clue efficiency calculation manually."""
    from hanabi_learning_environment import rl_env
    from agents.llm.clue_efficiency import calculate_clue_efficiency, format_hint_analysis

    print("="*60)
    print("Manual Clue Efficiency Demo")
    print("="*60)
    print()

    # Create environment
    env = rl_env.HanabiEnv({'players': 2})
    obs = env.reset()

    # Get first player's observation
    player_obs = obs['player_observations'][0]

    print("Game State:")
    print(f"  Fireworks: {player_obs['fireworks']}")
    print(f"  Info tokens: {player_obs['information_tokens']}")
    print()

    # Show partner's hand
    partner_hand = player_obs['observed_hands'][1]
    print("Partner's Hand:")
    for i, card in enumerate(partner_hand):
        print(f"  Card {i}: {card['color']}{card['rank']}")
    print()

    # Calculate clue efficiency
    hint_analyses = calculate_clue_efficiency(player_obs, player_id=0)

    if hint_analyses:
        print(format_hint_analysis(hint_analyses, top_n=10))
    else:
        print("No hints available")

    print()
    print("="*60)
    print()
    print("This analysis is now automatically provided to the LLM!")
    print("The agent can use these scores to choose the best hints.")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--demo":
        # Show manual demo of clue efficiency
        manual_clue_efficiency_demo()
    else:
        # Run full game with enhanced agent
        test_clue_efficiency_display()
