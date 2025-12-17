"""
Test script to compare Greedy Efficiency Agent performance

Compares different agent strategies:
1. RandomAgent (baseline)
2. GreedyEfficiencyAgent (always maximizes clue efficiency)
3. BalancedGreedyAgent (only gives hints when efficiency is high)

Runs multiple games and shows average scores.
"""

from agents.example import RandomAgent, GreedyEfficiencyAgent, BalancedGreedyAgent
from shared import run_games, print_game_summary


def test_greedy_vs_random(num_games: int = 50):
    """Test greedy efficiency agent against random agent."""
    print("="*60)
    print("TESTING GREEDY EFFICIENCY AGENT")
    print("="*60)
    print()

    # Test 1: Greedy vs Random
    print("\n" + "="*60)
    print("Test 1: GreedyEfficiencyAgent (P0) vs RandomAgent (P1)")
    print("="*60)
    greedy_agent = GreedyEfficiencyAgent(player_id=0)
    random_agent = RandomAgent(player_id=1)

    results_greedy_vs_random = run_games(
        greedy_agent, random_agent,
        num_games=num_games,
        verbose=False
    )
    print_game_summary(results_greedy_vs_random)

    # Test 2: Random vs Greedy (swap positions)
    print("\n" + "="*60)
    print("Test 2: RandomAgent (P0) vs GreedyEfficiencyAgent (P1)")
    print("="*60)
    random_agent2 = RandomAgent(player_id=0)
    greedy_agent2 = GreedyEfficiencyAgent(player_id=1)

    results_random_vs_greedy = run_games(
        random_agent2, greedy_agent2,
        num_games=num_games,
        verbose=False
    )
    print_game_summary(results_random_vs_greedy)

    # Test 3: Greedy vs Greedy
    print("\n" + "="*60)
    print("Test 3: GreedyEfficiencyAgent (P0) vs GreedyEfficiencyAgent (P1)")
    print("="*60)
    greedy_agent3 = GreedyEfficiencyAgent(player_id=0)
    greedy_agent4 = GreedyEfficiencyAgent(player_id=1)

    results_greedy_vs_greedy = run_games(
        greedy_agent3, greedy_agent4,
        num_games=num_games,
        verbose=False
    )
    print_game_summary(results_greedy_vs_greedy)

    # Test 4: Random vs Random (baseline)
    print("\n" + "="*60)
    print("Test 4: RandomAgent (P0) vs RandomAgent (P1) [Baseline]")
    print("="*60)
    random_agent3 = RandomAgent(player_id=0)
    random_agent4 = RandomAgent(player_id=1)

    results_random_vs_random = run_games(
        random_agent3, random_agent4,
        num_games=num_games,
        verbose=False
    )
    print_game_summary(results_random_vs_random)

    # Summary comparison
    print("\n" + "="*60)
    print("SUMMARY COMPARISON")
    print("="*60)

    def avg_score(results):
        return sum(r['score'] for r in results) / len(results)

    print(f"Greedy vs Random:  {avg_score(results_greedy_vs_random):.2f}")
    print(f"Random vs Greedy:  {avg_score(results_random_vs_greedy):.2f}")
    print(f"Greedy vs Greedy:  {avg_score(results_greedy_vs_greedy):.2f}")
    print(f"Random vs Random:  {avg_score(results_random_vs_random):.2f}")
    print("="*60)


def test_balanced_greedy(num_games: int = 50):
    """Test balanced greedy agent (only gives efficient hints)."""
    print("\n" + "="*60)
    print("TESTING BALANCED GREEDY AGENT")
    print("="*60)
    print()

    # Test different efficiency thresholds
    thresholds = [2.0, 4.0, 6.0, 8.0]

    for threshold in thresholds:
        print(f"\n--- Minimum Efficiency Threshold: {threshold} ---")
        balanced_agent = BalancedGreedyAgent(
            player_id=0,
            config={"min_efficiency": threshold}
        )
        random_agent = RandomAgent(player_id=1)

        results = run_games(balanced_agent, random_agent, num_games=num_games)
        avg_score = sum(r['score'] for r in results) / len(results)
        print(f"Average score: {avg_score:.2f}")

    print()


def test_single_game_verbose():
    """Run a single game with verbose output to see the strategy in action."""
    print("\n" + "="*60)
    print("SINGLE GAME - VERBOSE MODE")
    print("="*60)
    print("Watch how the Greedy agent chooses high-efficiency hints!")
    print()

    from shared import run_game

    greedy_agent = GreedyEfficiencyAgent(player_id=0)
    random_agent = RandomAgent(player_id=1)

    result = run_game(greedy_agent, random_agent, seed=42, verbose=True)

    print()
    print(f"Final Score: {result['score']}/25")
    print(f"Turns: {result['turns']}")


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        if sys.argv[1] == "--verbose":
            # Show a single game in detail
            test_single_game_verbose()
        elif sys.argv[1] == "--balanced":
            # Test balanced greedy agent
            test_balanced_greedy()
        elif sys.argv[1] == "--all":
            # Run all tests
            test_greedy_vs_random(num_games=100)
            test_balanced_greedy(num_games=100)
        else:
            print("Usage:")
            print("  python test_greedy_efficiency.py           # Standard test (50 games)")
            print("  python test_greedy_efficiency.py --verbose # Single game with details")
            print("  python test_greedy_efficiency.py --balanced # Test balanced variant")
            print("  python test_greedy_efficiency.py --all     # All tests (100 games each)")
    else:
        # Default: run standard test
        test_greedy_vs_random(num_games=50)
