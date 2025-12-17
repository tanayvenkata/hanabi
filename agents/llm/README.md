# BasicLLMAgent - LLM-powered Hanabi Agent

An intelligent Hanabi agent that uses Large Language Models (LLMs) to make decisions.

## Features

- **Multiple Memory Modes:**
  - `summary`: Compact memory tracking (cheaper, faster) - **DEFAULT**
  - `full`: Complete conversation history (more context)

- **Flexible LLM Support:**
  - LM Studio (local, free)
  - OpenAI API (GPT-4, GPT-3.5)
  - Any OpenAI-compatible API

- **Smart Parsing:**
  - Robust action parsing from LLM responses
  - Automatic fallback to random moves if parsing fails

- **Context Tracking:**
  - Tracks hints received and given
  - Remembers plays and discards
  - Maintains game state awareness

## Quick Start

### 1. Install Dependencies

```bash
pip install openai requests
```

### 2. Start LM Studio

1. Download and open [LM Studio](https://lmstudio.ai/)
2. Load a model (recommended: Llama 3.1 8B, Mistral 7B, Qwen 8B)
3. Go to "Local Server" tab → Click "Start Server"
4. Server runs at `http://localhost:1234`

### 3. Run Your First Game

```python
from agents.llm import BasicLLMAgent
from agents.example import RandomAgent
from shared import run_game

# Create LLM agent (uses summary mode by default)
llm_agent = BasicLLMAgent(
    player_id=0,
    config={
        "api_base": "http://localhost:1234/v1",
        "model": "local-model",
        "memory_mode": "summary",  # or "full"
    }
)

random_agent = RandomAgent(player_id=1)

# Run game
result = run_game(llm_agent, random_agent, seed=42, verbose=True)
print(f"Score: {result['score']}/25")
```

## Configuration Options

```python
config = {
    # API Configuration
    "api_base": "http://localhost:1234/v1",  # LM Studio default
    "api_key": "lm-studio",                  # Not needed for LM Studio
    "model": "local-model",                  # Auto-detected by LM Studio

    # LLM Parameters
    "temperature": 0.7,                      # 0.0 = deterministic, 1.0 = creative
    "max_tokens": 200,                       # Max response length

    # Memory Mode
    "memory_mode": "summary",                # "summary" or "full"
}
```

## Memory Modes Explained

### Summary Mode (Default)
- Maintains compact summaries of key events
- Tracks: hints received/given, plays, discards (last 3-5 each)
- **Pros:** Cheaper, faster, scales to long games
- **Cons:** May miss subtle details
- **Best for:** Most games, cost-conscious usage

### Full Mode
- Keeps complete conversation history
- Every observation and response preserved
- **Pros:** Maximum context, better reasoning
- **Cons:** More expensive, slower, context limit issues
- **Best for:** Short games, debugging, maximum performance

## Using Different LLM Providers

### LM Studio (Local)
```python
config = {
    "api_base": "http://localhost:1234/v1",
    "api_key": "lm-studio",
    "model": "local-model",
}
```

### OpenAI API
```python
config = {
    "api_base": "https://api.openai.com/v1",
    "api_key": "sk-your-key-here",
    "model": "gpt-4o-mini",  # or "gpt-4"
}
```

### Other OpenAI-Compatible APIs
Most modern LLM APIs support OpenAI's format (Anthropic, Together, etc.)

## Example Scripts

See `example_usage.py` for:
- Single game with verbose output
- Multiple games for benchmarking
- LLM vs LLM cooperative play

Run examples:
```bash
python -m agents.llm.example_usage
```

## Performance Expectations

| Agent Type | Avg Score | Notes |
|-----------|-----------|-------|
| RandomAgent | 0-2 | Baseline |
| BasicLLMAgent (small model) | 2-5 | Depends on model quality |
| BasicLLMAgent (large model) | 5-15 | GPT-4, Claude, etc. |
| Two BasicLLMAgents | 10-20+ | Cooperative play |

## Troubleshooting

### "Cannot connect to LM Studio"
- Make sure LM Studio is open
- Click "Start Server" in Local Server tab
- Check server is running on port 1234

### "Error calling LLM"
- Agent automatically falls back to random moves
- Check your API key and endpoint
- Verify model is loaded in LM Studio

### Poor Performance
- Try a larger model (8B → 13B → 70B)
- Use `temperature: 0.3` for more consistent play
- Switch to `memory_mode: "full"` for better context

## Claude API Pricing (as requested)

**Claude 3.5 Sonnet (Latest):**
- Input: $3 per million tokens
- Output: $15 per million tokens

**Typical Hanabi Game Costs:**
- Summary mode: ~$0.01-0.05 per game
- Full mode: ~$0.05-0.20 per game

**Comparison:**
- GPT-4o: $2.50 / $10 per million tokens
- GPT-4o-mini: $0.15 / $0.60 per million tokens
- LM Studio: **FREE** (runs locally)

## Next Steps

1. Try both memory modes and compare performance
2. Experiment with different models
3. Create custom prompts in `_get_system_prompt()`
4. Add more sophisticated reasoning strategies
5. Implement chain-of-thought prompting

Happy coding! 🎴
