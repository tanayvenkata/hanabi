"""
Game Runner for Hanabi

Utilities for running games between agents and collecting results.

Usage:
    from shared import run_game, run_games
    from agents.example import RandomAgent

    # Run a single game
    agent0 = RandomAgent(player_id=0)
    agent1 = RandomAgent(player_id=1)
    result = run_game(agent0, agent1, seed=42)
    print(f"Score: {result['score']}")

    # Run multiple games
    results = run_games(agent0, agent1, num_games=10)
    print(f"Average score: {sum(r['score'] for r in results) / len(results)}")
"""

from typing import Any, Dict, List, Optional, Tuple

from hanabi_learning_environment import rl_env


def run_game(
    agent0,
    agent1,
    seed: Optional[int] = None,
    verbose: bool = False,
) -> Dict[str, Any]:
    """Run a single game between two agents.

    Args:
        agent0: Agent for player 0
        agent1: Agent for player 1
        seed: Random seed for reproducibility
        verbose: If True, print game state after each move

    Returns:
        Dictionary containing:
            - score: Final game score (0-25)
            - turns: Number of turns played
            - history: List of (player, action, observation) tuples
    """
    # Create environment
    env_config = {
        "colors": 5,
        "ranks": 5,
        "players": 2,
        "hand_size": 5,
        "max_information_tokens": 8,
        "max_life_tokens": 3,
    }
    if seed is not None:
        env_config["seed"] = seed

    env = rl_env.HanabiEnv(env_config)
    agents = [agent0, agent1]

    # Reset agents
    for agent in agents:
        agent.reset()

    # Get initial observations
    observations = env.reset()
    history = []
    turns = 0
    done = False

    while not done:
        current_player = observations["current_player"]

        # Get action from current player's agent
        obs = observations["player_observations"][current_player]
        action = agents[current_player].act(obs)

        if verbose:
            print(f"\n--- Turn {turns + 1} (Player {current_player}) ---")
            print(f"Fireworks: {obs['fireworks']}")
            print(f"Info tokens: {obs['information_tokens']}, Life tokens: {obs['life_tokens']}")
            print(f"Legal moves: {len(obs['legal_moves'])}")
            print(f"Chosen action: {action}")

        # Apply action
        observations, reward, done, info = env.step(action)

        # Let agents observe the move
        move = obs["legal_moves"][obs["legal_moves_as_int"].index(action)]
        for agent in agents:
            agent.observe_move(move, current_player)

        history.append({
            "turn": turns,
            "player": current_player,
            "action": action,
            "move": move,
        })
        turns += 1

    # Get final score
    score = sum(observations["player_observations"][0]["fireworks"].values())

    if verbose:
        print(f"\n=== Game Over ===")
        print(f"Final Score: {score}")
        print(f"Turns played: {turns}")

    return {
        "score": score,
        "turns": turns,
        "history": history,
        "seed": seed,
    }


def run_games(
    agent0,
    agent1,
    num_games: int = 10,
    seeds: Optional[List[int]] = None,
    verbose: bool = False,
) -> List[Dict[str, Any]]:
    """Run multiple games between two agents.

    Args:
        agent0: Agent for player 0
        agent1: Agent for player 1
        num_games: Number of games to play
        seeds: Optional list of seeds (one per game). If None, uses range(num_games)
        verbose: If True, print summary after each game

    Returns:
        List of result dictionaries from run_game()
    """
    if seeds is None:
        seeds = list(range(num_games))
    elif len(seeds) != num_games:
        raise ValueError(f"seeds list length ({len(seeds)}) must match num_games ({num_games})")

    results = []
    for i, seed in enumerate(seeds):
        result = run_game(agent0, agent1, seed=seed, verbose=False)
        results.append(result)

        if verbose:
            print(f"Game {i + 1}/{num_games}: Score = {result['score']}, Turns = {result['turns']}")

    return results


def print_game_summary(results: List[Dict[str, Any]]) -> None:
    """Print a summary of game results.

    Args:
        results: List of result dictionaries from run_games()
    """
    scores = [r["score"] for r in results]
    turns = [r["turns"] for r in results]

    print(f"\n{'=' * 40}")
    print(f"Games played: {len(results)}")
    print(f"Average score: {sum(scores) / len(scores):.2f}")
    print(f"Min score: {min(scores)}, Max score: {max(scores)}")
    print(f"Average turns: {sum(turns) / len(turns):.1f}")
    print(f"{'=' * 40}")


if __name__ == "__main__":
    # Quick test with random agents
    from agents.example import RandomAgent

    print("Running test games with RandomAgent...")
    agent0 = RandomAgent(player_id=0, config={"seed": 42})
    agent1 = RandomAgent(player_id=1, config={"seed": 43})

    results = run_games(agent0, agent1, num_games=5, verbose=True)
    print_game_summary(results)
