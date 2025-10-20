
# threes_expectimax_bot.py
"""
Expectimax Threes! bot using the provided compact engine.

- Uses one-ply or multi-ply expectimax with explicit chance over spawn POSITION.
- The next-card value is known from the engine's preview; we average equally over
  all spawn candidate positions for a move.
- For deeper plies (>1), the "next preview" randomness (deck/bonus) is sampled.
  You can control that sampling count with --chance-samples.

Engine directions: 0=L, 1=R, 2=U, 3=D
"""

from __future__ import annotations
import argparse
from dataclasses import dataclass
from typing import Optional, Tuple, List, Callable
import random
import math
import statistics
import time

from threes_engine import Threes, get_cell, set_cell, value_to_code

Move = int  # 0..3

# --------------------
# Utility: clone game
# --------------------

def clone_game(g: Threes) -> Threes:
    # Copy RNG state
    rng = random.Random()
    rng.setstate(g.rng.getstate())
    # Copy deck (shares same RNG instance for reshuffles)
    deck = type(g.deck)(rng)
    deck.cards = list(g.deck.cards)
    deck.idx = g.deck.idx
    # New game object
    ng = Threes(rng=rng, board=g.board, deck=deck, next_card=g.next_card, next_is_bonus=g.next_is_bonus)
    return ng

# --------------------
# Heuristic
# --------------------

def count_empty(board: int) -> int:
    empties = 0
    for i in range(16):
        if ((board >> (i*4)) & 0xF) == 0:
            empties += 1
    return empties

def corner_max_bonus(g: Threes) -> int:
    # Reward having the max tile in a corner
    m = g.max_value()
    # corners: (0,0),(0,3),(3,0),(3,3)
    corners = [(0,0),(0,3),(3,0),(3,3)]
    for r,c in corners:
        code = (g.board >> (((r<<2)+c) * 4)) & 0xF
        if code != 0 and m == ( (1 << (code-1)) if code>=3 else code ):
            return 1
    return 0

def evaluate(g: Threes) -> float:
    # Weighted sum of: score, empties, corner bonus, slight penalty for preview being "bad"
    score_term = g.score_board()
    empty_term = count_empty(g.board) * 150.0
    corner_term = 500.0 * corner_max_bonus(g)
    # prefer when next preview is "3+" rather than 1/2 (slightly)
    preview = g.next_preview()
    preview_bonus = 60.0 if preview == "3+" else (20.0 if preview == "2" else 0.0)
    # Scale combined
    return score_term + empty_term + corner_term + preview_bonus

# --------------------
# Move simulation helpers (without sampling spawn position)
# --------------------

def _apply_direction_no_spawn(g: Threes, direction: Move) -> Tuple[int, List[int], bool]:
    """Return (new_board, spawn_candidates, moved) without placing the new card."""
    if direction == 0:
        return g._move_left_rows(g.board)
    elif direction == 1:
        return g._move_right_rows(g.board)
    elif direction == 2:
        return g._move_up_cols(g.board)
    elif direction == 3:
        return g._move_down_cols(g.board)
    else:
        raise ValueError("direction must be 0..3")

def _place_and_draw(ng: Threes, board_after_slide: int, pos: int) -> None:
    """Place the CURRENT preview card at pos on the given board, then update preview."""
    r, c = divmod(pos, 4)
    ng.board = set_cell(board_after_slide, r, c, ng.next_card)
    # draw new preview based on updated board / deck
    ng._draw_next_card()

# --------------------
# Expectimax (averaging over spawn position; sampling for preview randomness)
# --------------------

@dataclass
class EMParams:
    depth: int = 2
    chance_samples: int = 1  # times to sample preview per chance node (>=1)
    move_order: Tuple[Move,Move,Move,Move] = (1,3,0,2)  # R, D, L, U

def expectimax_value(g: Threes, params: EMParams, depth: int) -> float:
    if depth == 0 or not g.has_moves():
        return evaluate(g)

    best = -math.inf
    # Consider moves in preferred order
    for move in params.move_order:
        # Get slide result + candidate positions
        nb, candidates, moved = _apply_direction_no_spawn(g, move)
        if not moved:
            continue

        # Expected value over spawn positions (uniform across candidates)
        if not candidates:
            continue
        pos_expect = 0.0
        for pos in candidates:
            # Average over preview randomness for NEXT turn (sampled)
            sample_sum = 0.0
            for _ in range(max(1, params.chance_samples)):
                child = clone_game(g)
                _place_and_draw(child, nb, pos)
                sample_sum += expectimax_value(child, params, depth-1)
            pos_expect += (sample_sum / max(1, params.chance_samples))
        pos_expect /= len(candidates)

        if pos_expect > best:
            best = pos_expect

    if best == -math.inf:
        # No valid moves; terminal
        return evaluate(g)
    return best

def choose_move(g: Threes, params: EMParams) -> Optional[Move]:
    """Return best move by expectimax; None if no moves apply."""
    best_move = None
    best_val = -math.inf
    for move in params.move_order:
        nb, candidates, moved = _apply_direction_no_spawn(g, move)
        if not moved or not candidates:
            continue
        # Compute expected value of this move
        val = 0.0
        for pos in candidates:
            sample_sum = 0.0
            for _ in range(max(1, params.chance_samples)):
                child = clone_game(g)
                _place_and_draw(child, nb, pos)
                sample_sum += expectimax_value(child, params, params.depth-1)
            val += (sample_sum / max(1, params.chance_samples))
        val /= len(candidates)
        if val > best_val:
            best_val = val
            best_move = move
    return best_move

# --------------------
# Play loop
# --------------------

def play(seed: Optional[int] = None, max_steps: int = 10_000, sleep: float = 0.0, verbose: bool = True,
         depth: int = 2, chance_samples: int = 1) -> list[int]:
    """
    Run one game using Expectimax (depth plies), averaging over spawn positions.
    Returns [max_tile, total_score].
    """
    params = EMParams(depth=depth, chance_samples=chance_samples)
    game = Threes.new(seed=seed)

    step = 0

    start_time = time.perf_counter()

    while game.has_moves() and step < max_steps:
        mv = choose_move(game, params)
        if mv is None:
            break
        _ = game.try_move(mv)

        if verbose:
            print(f"\nMove: {mv} (0=L,1=R,2=U,3=D)")
            game.print_board()
            end_time = time.perf_counter()
            elapsed_time = end_time - start_time
            print(f"Elapsed time: {elapsed_time:.4f} seconds")

        step += 1
        if sleep > 0:
            time.sleep(sleep)

    if verbose:
        print("\nGame over.")
        print("Final board:")
        game.print_board()
        print(f"Max tile: {game.max_value()}")
        print(f"Score: {game.score_board()}")

    return [game.max_value(), game.score_board()]

# --------------------
# CLI for batch runs
# --------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Expectimax Threes bot")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility")
    parser.add_argument("--steps", type=int, default=10_000, help="Max steps per game")
    parser.add_argument("--sleep", type=float, default=0.0, help="Pause between moves (seconds)")
    parser.add_argument("--quiet", action="store_true", help="Suppress per-move prints")
    parser.add_argument("--depth", type=int, default=3, help="Search depth (plies)")
    parser.add_argument("--chance-samples", type=int, default=1, help="Samples per chance node for preview randomness")
    parser.add_argument("--games", type=int, default=10, help="Number of games to run")
    args = parser.parse_args()

    max_tile_list = []
    high_score_list = []
    start_time = time.perf_counter()

    for i in range(args.games):
        # To vary seeds if not provided, offset by i
        seed = args.seed if args.seed is not None else None
        result = play(seed=seed, max_steps=args.steps, sleep=args.sleep, verbose=args.quiet,
                      depth=args.depth, chance_samples=args.chance_samples)
        max_tile_list.append(result[0])
        high_score_list.append(result[1])
        print(f"{i + 1} / {args.games} Games Played")

    end_time = time.perf_counter()
    elapsed_time = end_time - start_time
    print(f"Elapsed time: {elapsed_time:.4f} seconds")

    print(f"Max tile reached average: {statistics.mean(max_tile_list)}\nHigh Score average: {statistics.mean(high_score_list)}")
