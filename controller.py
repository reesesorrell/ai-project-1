"""
threes_solver_cli_auto_fixed.py
-------------------------------
Interactive Threes! solver that applies real moves and shows the correct post-move board.
"""

from __future__ import annotations
from typing import List
import numpy as np
import sys, time

from threes_engine import (
    set_cell, get_cell, value_to_code, code_to_value,
    Threes
)
from monte_carlo_bot import (
    choose_move_monte_carlo, MCParams, counts_from_deck,
)

MOVE_NAMES = ["Left", "Right", "Up", "Down"]

# ---------- Helpers ----------

def encode_board(grid: List[List[int]]) -> int:
    board = 0
    for r in range(4):
        for c in range(4):
            board = set_cell(board, r, c, value_to_code(grid[r][c]))
    return board

def print_board(board: int):
    for r in range(4):
        row = [f"{code_to_value(get_cell(board, r, c)):>4}" for c in range(4)]
        print(" ".join(row))

# ---------- Main ----------

def main():
    print("=== Threes Solver (accurate board update) ===")
    print("Enter your initial 4×4 board and preview.\n")

    # initial setup
    grid = []
    for r in range(4):
        vals = [int(x) for x in input(f"Row {r+1}: ").split()]
        if len(vals) != 4:
            print("Each row must have 4 numbers.")
            return
        grid.append(vals)
    nxt_val = int(input("Next card (1,2,or >=3): "))

    g = Threes.new(seed=1337)
    g.board = encode_board(grid)
    g.next_card = value_to_code(nxt_val)
    g.next_is_bonus = nxt_val >= 4

    rng = np.random.default_rng(12345)
    params = MCParams(depth=8, rollouts=256)

    while True:
        print("\nCurrent board:")
        print_board(g.board)
        print(f"Next preview: {code_to_value(g.next_card)}")
        print("=" * 40)

        # choose best move
        n1, n2, n3 = counts_from_deck(g)
        t0 = time.perf_counter()
        mv = choose_move_monte_carlo(g.board, g.next_card, n1, n2, n3, params, rng)
        dt = time.perf_counter() - t0

        if mv is None:
            print("No valid moves left — game over.")
            break

        print(f"Suggested move: {MOVE_NAMES[mv]} (computed in {dt:.2f}s)")

        # ask spawn location (row/col only)
        if mv in (0, 1):
            prompt = "Enter ROW (1–4) where new card spawned: "
        else:
            prompt = "Enter COLUMN (1–4) where new card spawned: "

        s_idx = input(prompt).strip()
        if s_idx.lower() == "q":
            print("Exiting.")
            break
        try:
            s_idx = int(s_idx)
            if not (1 <= s_idx <= 4):
                raise ValueError
        except ValueError:
            print("Invalid index (1–4 expected).")
            continue

        # new preview
        new_prev = input("Next preview card: ").strip()
        if new_prev.lower() == "q":
            print("Goodbye!")
            break
        next_val = int(new_prev)

        # perform the actual move on engine (so merges happen)
        if mv == 0:
            nb, _, _ = g._move_left_rows(g.board)
            sr, sc = s_idx - 1, 3
        elif mv == 1:
            nb, _, _ = g._move_right_rows(g.board)
            sr, sc = s_idx - 1, 0
        elif mv == 2:
            nb, _, _ = g._move_up_cols(g.board)
            sr, sc = 3, s_idx - 1
        else:
            nb, _, _ = g._move_down_cols(g.board)
            sr, sc = 0, s_idx - 1

        # apply spawn (the old preview)
        new_board = set_cell(nb, sr, sc, g.next_card)
        g.board = new_board
        g.next_card = value_to_code(next_val)
        g.next_is_bonus = next_val >= 4

        print("\nUpdated board after move:")
        print_board(g.board)
        print(f"Next preview: {next_val}")

if __name__ == "__main__":
    main()
