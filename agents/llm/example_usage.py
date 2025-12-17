"""
Example usage of BasicLLMAgent with LM Studio

Before running:
1. Install LM Studio from https://lmstudio.ai/
2. Download a model (recommend: Llama 3.1 8B or Mistral 7B)
3. Start the local server in LM Studio (it runs on http://localhost:1234 by default)
4. Install dependencies: pip install openai

Then run this script to watch the LLM play Hanabi!
"""

from agents.llm import BasicLLMAgent
from agents.example import RandomAgent
from shared import run_game, run_games, print_game_summary


def test_single_game():
    """Run a single game with verbose output to see the LLM's reasoning."""
    print("Running single game with BasicLLMAgent...\n")

    # Create agents
    llm_agent = BasicLLMAgent(
        player_id=0,
        config={
            "api_base": "http://localhost:1234/v1",  # LM Studio default
            "api_key": "lm-studio",                  # Doesn't matter for LM Studio
            "model": "local-model",                  # LM Studio auto-detects
            "temperature": 0.7,
            "use_conversation_history": True,
        }
    )

    random_agent = RandomAgent(player_id=1)

    # Run game
    result = run_game(llm_agent, random_agent, seed=42, verbose=True)

    print(f"\n{'='*60}")
    print(f"Final Score: {result['score']}/25")
    print(f"Turns: {result['turns']}")
    print(f"{'='*60}")


def test_multiple_games():
    """Run multiple games and see average performance."""
    print("Running 5 games with BasicLLMAgent...\n")

    llm_agent = BasicLLMAgent(
        player_id=0,
        config={
            "api_base": "http://localhost:1234/v1",
            "model": "local-model",
            "temperature": 0.7,
        }
    )

    random_agent = RandomAgent(player_id=1)

    # Run multiple games
    results = run_games(llm_agent, random_agent, num_games=5, verbose=True)
    print_game_summary(results)

    # Show comparison
    print("\nPerformance Comparison:")
    print("  RandomAgent baseline: ~0-2 points")
    print(f"  BasicLLMAgent: {sum(r['score'] for r in results) / len(results):.1f} points")


def test_llm_vs_llm():
    """Two LLM agents playing together (cooperative mode)."""
    print("Running LLM vs LLM game...\n")

    llm_agent_0 = BasicLLMAgent(
        player_id=0,
        config={
            "api_base": "http://localhost:1234/v1",
            "model": "local-model",
        }
    )

    llm_agent_1 = BasicLLMAgent(
        player_id=1,
        config={
            "api_base": "http://localhost:1234/v1",
            "model": "local-model",
        }
    )

    result = run_game(llm_agent_0, llm_agent_1, seed=42, verbose=False)

    print(f"\n{'='*60}")
    print(f"LLM vs LLM Performance:")
    print(f"  Score: {result['score']}/25")
    print(f"  Turns: {result['turns']}")
    print(f"{'='*60}")


def test_with_different_apis():
    """Example configurations for different LLM providers."""

    # LM Studio (default)
    lm_studio_config = {
        "api_base": "http://localhost:1234/v1",
        "api_key": "lm-studio",
        "model": "local-model",
    }

    # OpenAI API
    openai_config = {
        "api_base": "https://api.openai.com/v1",
        "api_key": "your-api-key-here",  # Set this!
        "model": "gpt-4o-mini",  # or gpt-4
    }

    # Claude API (via OpenAI compatibility - requires additional setup)
    claude_config = {
        "api_base": "https://api.anthropic.com/v1",
        "api_key": "your-anthropic-key-here",  # Set this!
        "model": "claude-3-5-sonnet-20241022",
    }

    # Use whichever config you want
    agent = BasicLLMAgent(player_id=0, config=lm_studio_config)

    print("Agent configured successfully!")
    print(f"API Base: {agent.api_base}")
    print(f"Model: {agent.model}")


if __name__ == "__main__":
    import sys

    # Check if LM Studio is running
    try:
        import requests
        response = requests.get("http://localhost:1234/v1/models", timeout=2)
        if response.status_code != 200:
            print("WARNING: LM Studio doesn't appear to be running!")
            print("Please start LM Studio and load a model before continuing.\n")
            sys.exit(1)
    except Exception:
        print("WARNING: Cannot connect to LM Studio at http://localhost:1234")
        print("Please start LM Studio and load a model before continuing.\n")
        sys.exit(1)

    # Run tests
    print("="*60)
    print("BasicLLMAgent Test Suite")
    print("="*60)
    print()

    # Choose which test to run
    print("Choose a test:")
    print("1. Single game (verbose)")
    print("2. Multiple games (5 games)")
    print("3. LLM vs LLM")
    print()

    choice = input("Enter choice (1-3, default=1): ").strip() or "1"

    if choice == "1":
        test_single_game()
    elif choice == "2":
        test_multiple_games()
    elif choice == "3":
        test_llm_vs_llm()
    else:
        print("Invalid choice, running single game...")
        test_single_game()
