# Hanabi Agents

A collaborative project for building AI agents that play [Hanabi](https://en.wikipedia.org/wiki/Hanabi_(card_game)), the cooperative card game.

Uses [Google DeepMind's Hanabi Learning Environment](https://github.com/google-deepmind/hanabi-learning-environment) as the game engine.

# Agent Performance Metrics

**BalancedGreedyAgent**: Implemented with clue efficiency maximizer so the clue conveying the most information is selected -- 30% performance improvement with greedy strategy (selecting card with max efficiency) than random play with efficiency threshold of 6.0


## Setup

### 1. Clone the repository (with submodules)

```bash
git clone --recurse-submodules git@github.com:tanayvenkata/hanabi.git
cd hanabi
```

If you already cloned without submodules:
```bash
git submodule update --init --recursive
```

### 2. Create a virtual environment

```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install the Hanabi environment

```bash
pip install -e hanabi_env/
```

### 4. Install additional dependencies

```bash
pip install -r requirements.txt
```

### 5. Verify installation

```bash
python -c "from hanabi_learning_environment import rl_env; print('Success!')"
```

## Project Structure

```
hanabi/
├── hanabi_env/          # (submodule) DeepMind's Hanabi Learning Environment
├── agents/              # Agent implementations
│   └── example/         # Example agents (BaseAgent, RandomAgent)
├── shared/              # Common utilities
│   └── game_runner.py   # Run games between agents
├── README.md
└── requirements.txt
```

## Quick Start

Run a test game with random agents:

```bash
python -m shared.game_runner
```

Or in Python:

```python
from agents.example import RandomAgent
from shared import run_game, print_game_summary

# Create agents
agent0 = RandomAgent(player_id=0)
agent1 = RandomAgent(player_id=1)

# Run a game
result = run_game(agent0, agent1, seed=42, verbose=True)
print(f"Score: {result['score']}")
```

## Creating Your Own Agent

1. Create a new folder: `agents/youragent/`
2. Subclass `BaseAgent` from `agents.example.agent`
3. Implement the `act(observation)` method

```python
# agents/youragent/agent.py
from agents.example import BaseAgent

class MyAgent(BaseAgent):
    def act(self, observation):
        # observation contains:
        # - 'legal_moves': list of legal move dicts
        # - 'legal_moves_as_int': list of action integers
        # - 'observed_hands': other players' hands (yours is hidden)
        # - 'fireworks': current score per color
        # - 'information_tokens': remaining hints
        # - 'life_tokens': remaining lives
        # - 'discard_pile': discarded cards

        # Return an action from legal_moves_as_int
        return observation['legal_moves_as_int'][0]
```

## Game Rules Quick Reference

- **Objective**: Play cards 1-5 in each of 5 colors (max score: 25)
- **Actions**: Play a card, Discard a card, Give a hint
- **Hints**: Tell a teammate about all cards of one color OR one rank in their hand
- **Lives**: Lose one for playing an illegal card (game over at 0)
- **Info tokens**: Spend to give hints, regain by discarding

## Benchmarks

| Agent | Avg Score |
|-------|-----------|
| Random | ~0-2 |
| Your agent? | ??? |

## References

- [Hanabi Learning Environment](https://github.com/google-deepmind/hanabi-learning-environment)
- [Hanabi rules](https://en.wikipedia.org/wiki/Hanabi_(card_game))
