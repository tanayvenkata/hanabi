# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a Hanabi AI agent development project using Google DeepMind's Hanabi Learning Environment. The goal is to build cooperative AI agents that play the card game Hanabi.

## Common Commands

```bash
# Setup (first time)
python3 -m venv venv
source venv/bin/activate
pip install -e hanabi_env/
pip install -r requirements.txt

# Verify installation
python -c "from hanabi_learning_environment import rl_env; print('Success!')"

# Run test games with random agents
python -m shared.game_runner
```

## Architecture

### Core Components

- **`hanabi_env/`**: Git submodule containing DeepMind's Hanabi Learning Environment. Provides `rl_env.HanabiEnv` (OpenAI Gym-like interface) and `pyhanabi` (lower-level C++ bindings).

- **`agents/`**: Agent implementations. Each agent should be in its own subfolder (e.g., `agents/myagent/`).
  - `agents/example/agent.py`: Contains `BaseAgent` (abstract base class) and `RandomAgent` (baseline implementation).

- **`shared/game_runner.py`**: Game execution utilities. Use `run_game(agent0, agent1)` for single games or `run_games(agent0, agent1, num_games=N)` for batch runs.

### Creating a New Agent

1. Create a folder: `agents/youragent/`
2. Subclass `BaseAgent` from `agents.example`
3. Implement `act(observation) -> int` which returns an action from `observation['legal_moves_as_int']`
4. Optionally override `reset()` and `observe_move(move, acting_player)` for state tracking

### Key Observation Fields

When implementing `act()`, the observation dict contains:
- `legal_moves_as_int`: List of valid action integers (return one of these)
- `legal_moves`: List of move dicts with action details
- `observed_hands`: Other players' hands (your own hand at index 0 shows `color: None, rank: -1`)
- `fireworks`: Dict of current score per color (`{'R': 0, 'G': 0, 'B': 0, 'W': 0, 'Y': 0}`)
- `information_tokens`: Remaining hint tokens (start: 8)
- `life_tokens`: Remaining lives (start: 3, game over at 0)
- `card_knowledge`: Hints received about your own cards
