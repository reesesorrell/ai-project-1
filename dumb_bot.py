# threes_basic_bot.py
"""
Basic Threes! bot that uses the provided engine.

Strategy:
- Alternate between Right (1) and Down (3) on each turn.
- If the intended move doesn't change the board, try the *other* primary move.
- If both are invalid, try Left (0) **only if necessary**.
- Stop when no moves remain.

Engine directions: 0=L, 1=R, 2=U, 3=D
"""

from __future__ import annotations
import argparse
from typing import Optional
import random
import time
import statistics

# Import the engine you shared (must be on PYTHONPATH or in the same folder)
from threes_engine import Threes

def play(seed: Optional[int] = None, max_steps: int = 10_000, sleep: float = 0.0, verbose: bool = True) -> list[int]:
    """
    Run one game with the alternating R/D, fallback L strategy.
    Returns the max tile value reached.
    """
    game = Threes.new(seed=seed)

    # Start alternating between all directions
    current_index = 0  # toggles each turn
    move_order = [0, 2, 1, 3]

    step = 0
    while game.has_moves():

        if current_index >= 4:
            current_index = 0

        game.try_move(move_order[current_index])
        current_index += 1
        step += 1
        

    if verbose:
        print("\nGame over.")
        print("Final board:")
        game.print_board()
        print(f"Max tile: {game.max_value()}")

    return [game.max_value(), game.score_board()]

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Basic Threes bot: alternate Right/Down; fallback Left only if needed.")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility.")
    parser.add_argument("--steps", type=int, default=10000, help="Safety cap on number of steps.")
    parser.add_argument("--sleep", type=float, default=0.0, help="Seconds to sleep between moves (for viewing).")
    parser.add_argument("--quiet", action="store_true", help="Suppress per-move prints.")
    args = parser.parse_args()

    game_num = 10000
    max_tile_list = []
    high_score_list = []
    for i in range(game_num):
        game_play = play(seed=args.seed, max_steps=args.steps, sleep=args.sleep, verbose=args.quiet)
        max_tile_list.append(game_play[0])
        high_score_list.append(game_play[1])
        if (i%5 == 0):
            print(f"{i} / {game_num} Games Played")

    print(f"\nMax tile reached average: {statistics.mean(max_tile_list)}\nHigh Score: {statistics.mean(high_score_list)}")
