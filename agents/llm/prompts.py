"""
Prompt Templates for Hanabi LLM Agents

This file contains all prompt templates for:
1. MonolithAgent - Single agent that makes all decisions
2. CouncilAgent - Multi-agent architecture with specialists

=== STRUCTURE ===

MONOLITH PROMPTS:
- GAME_RULES: Full game explanation (system prompt)
- TURN_PROMPT_TEMPLATE: Per-turn state + decision request

COUNCIL PROMPTS (TODO - customize these):
- PLAY_ADVISOR_*: Prompts for play specialist
- HINT_ADVISOR_*: Prompts for hint specialist
- DISCARD_ADVISOR_*: Prompts for discard specialist
- MAIN_AGENT_*: Prompts for final decision maker

=== CUSTOMIZATION GUIDE ===

To tune agent behavior:
1. Modify the relevant prompt template
2. Test with test_states.py scenarios
3. Iterate based on results

Key principles:
- Be specific about output format
- Include examples where helpful
- Keep context focused (don't overload with info)
"""

GAME_RULES = """You are playing Hanabi, a cooperative card game.

=== GAME RULES ===

CARDS:
- 5 colors: Red (R), Green (G), Blue (B), White (W), Yellow (Y)
- Ranks 1-5 per color
- Distribution per color: 3x 1s, 2x 2s, 2x 3s, 2x 4s, 1x 5s (only one 5!)

GOAL: Build 5 firework piles (one per color) from 1 to 5. Max score: 25.

CRITICAL RULE: You CANNOT see your own cards. You can only see your partner's cards.

RESOURCES:
- 8 hint tokens (spend to give hints, regain 1 when discarding)
- 3 life tokens (lose one on wrong play, game over at 0)

ACTIONS (pick one per turn):
1. PLAY a card from your hand (hope it's correct!)
2. DISCARD a card (regain 1 hint token, but lose the card forever)
3. HINT your partner about their cards (costs 1 hint token)
   - Can hint a COLOR: "Your cards in slots [X, Y] are Red"
   - Can hint a RANK: "Your cards in slots [X, Y] are 1s"
   - Hints must touch at least one card

WHAT MAKES A CARD PLAYABLE:
- A card is playable if rank = firework[color] + 1
- Example: If Red pile is at 2, then Red 3 is playable
- Playing the wrong card loses a life token!

TERMINOLOGY:
- "Chop" = oldest unhinted card (usually rightmost) - the card a player would discard
- "Critical" = last remaining copy of a card needed to complete a color (losing it = can't get 25)
- "Useless/Dead" = a card that can never be played (rank <= current firework)"""


TURN_PROMPT_TEMPLATE = """=== CURRENT GAME STATE ===

FIREWORKS (current piles):
{fireworks_display}

  Cards needed next: {needed_cards}

RESOURCES:
  Hint tokens: {hint_tokens}/8
  Life tokens: {life_tokens}/3
  Deck remaining: {deck_size} cards

DISCARD PILE: {discard_pile}

---

YOUR PARTNER'S HAND (you can see these):
{partner_hand_display}

YOUR HAND (you CANNOT see these - only hints you've received):
{my_hand_display}

---

=== AVAILABLE ACTIONS ===

{actions_display}

---

=== YOUR TASK ===

Think through this step by step in 2-3 short paragraphs:
1. What cards are immediately playable? (need rank = firework + 1)
2. Does your partner have playable cards you should hint?
3. Is your partner's chop card critical (must save it)?
4. Do you know enough about YOUR cards to play safely?
5. If you must discard, which is safest?

Priorities:
- Play known-safe cards (advances score, no risk)
- Hint playable cards to partner (enables future plays)
- Save critical cards (prevents losing the game)
- Discard known-useless cards (safe, regains hint token)
- Discard chop as fallback (risky but standard)

After reasoning, give your final answer.

=== OUTPUT FORMAT ===

End your response with EXACTLY:

DECISION: [number]
REASON: [one sentence explaining why]

Example:
DECISION: 10
REASON: Hinting Red tells partner their slot 0 is playable now."""
