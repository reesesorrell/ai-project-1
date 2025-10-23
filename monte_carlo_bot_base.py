# threes_expectimax_board.py
"""
Board-centric, deck-aware Monte Carlo for Threes! (uses threes_engine_numba.py)

- Immediate spawn: uses the CURRENT preview deterministically (no branching).
- Deeper plies: simulate the NEXT preview over {1,2,3} using remaining shoe counts.
  * If counts are (n1, n2, n3), use probabilities n1/T, n2/T, n3/T (T=n1+n2+n3).
  * If T == 0 (shoe exhausted), reshuffle to (4,4,4) and use uniform 1/3 each.
- Ignores bonus (big) cards for preview probabilities (per request). If the current
  preview is a bonus (>=4), we still place it deterministically; shoe counts unchanged.

- Search state is (board:int, next_card:int, n1:int, n2:int, n3:int).
- Provides Monte Carlo playout policy (fixed depth), serial-only.
- Supports multi-game runs and prints a summary (low/median/high + tile thresholds).
"""

from __future__ import annotations
import argparse
import math
import time
from dataclasses import dataclass
from typing import Callable, List, Optional, Tuple
import statistics
import numpy as np

# CHANGE WHICH IMPORT IS ACTIVE TO TEST YOUR HEURISTIC
from heuristic_reese import evaluate_board
# from heuristic_landon import evaluate_board
# from heuristic_mirage import evaluate_board

# ---- Engine glue ----
from threes_engine import (
    Threes,            # only used to start/print games
    set_cell,          # nibble setter
    move_left_nb,
    move_right_nb,
    move_up_nb,
    move_down_nb,
)

U64_MASK = (1 << 64) - 1

# ---- Numba warmup ----
def _numba_warmup():
    """Touch kernels once so they JIT before timing the real game."""
    import numpy as np
    from threes_engine import move_left_nb, move_right_nb, move_up_nb, move_down_nb
    b = np.uint64(0x0000_0000_0000_0000)
    _ = move_left_nb(b)
    _ = move_right_nb(b)
    _ = move_up_nb(b)
    _ = move_down_nb(b)

# ---- Small helpers ----

def _bits4(mask: int):
    """Iterate set-bit indices (0..3) of a 4-bit mask."""
    m = mask & 0xF
    while m:
        lsb = m & -m
        yield (lsb.bit_length() - 1)
        m ^= lsb

def _format_total_time(seconds: float) -> str:
    """Format as HH:MM:SS.fffffff (7 fractional digits)."""
    t = seconds
    h = int(t // 3600); t -= 3600 * h
    m = int(t // 60);   t -= 60 * m
    s = int(t);         t -= s
    return f"{h:02d}:{m:02d}:{s:02d}.{int(round(t * 10_000_000)):07d}"

# Wrap the Numba kernels with Python-friendly return types + candidate mapping
def _move_left(board: int) -> Tuple[int, List[int], bool]:
    nb, mask, moved = move_left_nb(np.uint64(board & U64_MASK))
    if not moved:
        return int(nb), [], False
    cands = [r * 4 + 3 for r in _bits4(int(mask))]  # right edge of changed rows
    return int(nb), cands, True

def _move_right(board: int) -> Tuple[int, List[int], bool]:
    nb, mask, moved = move_right_nb(np.uint64(board & U64_MASK))
    if not moved:
        return int(nb), [], False
    cands = [r * 4 + 0 for r in _bits4(int(mask))]  # left edge
    return int(nb), cands, True

def _move_up(board: int) -> Tuple[int, List[int], bool]:
    nb, mask, moved = move_up_nb(np.uint64(board & U64_MASK))
    if not moved:
        return int(nb), [], False
    cands = [3 * 4 + c for c in _bits4(int(mask))]  # bottom row
    return int(nb), cands, True

def _move_down(board: int) -> Tuple[int, List[int], bool]:
    nb, mask, moved = move_down_nb(np.uint64(board & U64_MASK))
    if not moved:
        return int(nb), [], False
    cands = [0 * 4 + c for c in _bits4(int(mask))]  # top row
    return int(nb), cands, True

MOVE_FUNS: Tuple[Callable[[int], Tuple[int, List[int], bool]], ...] = (
    _move_left, _move_right, _move_up, _move_down
)

# ---- Deck helpers ----

def counts_from_deck(g: Threes) -> Tuple[int, int, int]:
    """
    Read remaining basic-card counts (1,2,3) from the engine's current shoe:
    counts are for g.deck.cards[g.deck.idx:] only.
    Note: the engine preview has already been drawn; these are *post-preview* counts.
    """
    n1 = n2 = n3 = 0
    for code in g.deck.cards[g.deck.idx:]:
        if code == 1: n1 += 1
        elif code == 2: n2 += 1
        elif code == 3: n3 += 1
    return n1, n2, n3

def next_preview_outcomes(n1: int, n2: int, n3: int) -> List[Tuple[int, float, Tuple[int,int,int]]]:
    """
    Given remaining counts, return list of (card_code, probability, new_counts_after_drawing_it).
    If shoe exhausted (n1+n2+n3==0), reshuffle to (4,4,4) and return uniform over {1,2,3}.
    """
    T = n1 + n2 + n3
    outcomes: List[Tuple[int, float, Tuple[int,int,int]]] = []
    if T == 0:
        outcomes.append((1, 1.0/3.0, (3, 4, 4)))
        outcomes.append((2, 1.0/3.0, (4, 3, 4)))
        outcomes.append((3, 1.0/3.0, (4, 4, 3)))
        return outcomes

    if n1:
        outcomes.append((1, n1 / T, (n1 - 1, n2, n3)))
    if n2:
        outcomes.append((2, n2 / T, (n1, n2 - 1, n3)))
    if n3:
        outcomes.append((3, n3 / T, (n1, n2, n3 - 1)))
    return outcomes

# ---- Monte Carlo (deck-aware) ----

@dataclass
class MCParams:
    depth: int = 8           # playout agent-move depth
    rollouts: int = 512      # rollouts per root move
    move_order: Tuple[int, int, int, int] = (1, 3, 0, 2)  # R, D, L, U (for root iteration)

def _sample_next_preview(n1: int, n2: int, n3: int, rng: np.random.Generator) -> Tuple[int, int, int, int]:
    """
    Sample the NEXT preview and update counts accordingly.
    Returns (nxt_card, m1, m2, m3).
    """
    outcomes = next_preview_outcomes(n1, n2, n3)
    probs = [p for _, p, _ in outcomes]
    idx = rng.choice(len(outcomes), p=np.array(probs, dtype=np.float64))
    nxt, _, (m1, m2, m3) = outcomes[idx]
    return nxt, m1, m2, m3

def _random_legal_move(board: int, rng: np.random.Generator) -> Optional[Tuple[int, int, List[int]]]:
    """Return a random legal move: (mv, nb, cands)."""
    legal = []
    for mv in (0, 1, 2, 3):
        nb, cands, moved = MOVE_FUNS[mv](board)
        if moved and cands:
            legal.append((mv, nb, cands))
    if not legal:
        return None
    return legal[rng.integers(len(legal))]

def mc_rollout(board: int, next_card: int, n1: int, n2: int, n3: int,
               depth: int, rng: np.random.Generator) -> float:
    """
    Random playout for 'depth' agent moves.
    Uses uniform choice among legal moves and uniform spawn position; next preview sampled by deck.
    """
    if depth <= 0:
        return evaluate_board(board, next_card)

    move_pack = _random_legal_move(board, rng)
    if move_pack is None:
        return evaluate_board(board, next_card)

    _, nb, cands = move_pack
    pos = int(rng.choice(cands))
    r, c = divmod(pos, 4)
    child_board = set_cell(nb, r, c, next_card) & U64_MASK

    nxt, m1, m2, m3 = _sample_next_preview(n1, n2, n3, rng)
    return mc_rollout(child_board, nxt, m1, m2, m3, depth - 1, rng)

def choose_move_monte_carlo(board: int, next_card: int, n1: int, n2: int, n3: int,
                            params: MCParams, rng: np.random.Generator) -> Optional[int]:
    """
    For each legal ROOT move, run 'rollouts' random playouts starting from that move and
    pick the move with the highest average terminal heuristic.
    """
    best_mv: Optional[int] = None
    best_avg = -math.inf

    for mv in params.move_order:
        nb, cands, moved = MOVE_FUNS[mv](board)
        if not moved or not cands:
            continue

        total = 0.0
        R = params.rollouts
        for _ in range(R):
            pos = int(rng.choice(cands))
            r, c = divmod(pos, 4)
            child_board = set_cell(nb, r, c, next_card) & U64_MASK

            nxt, m1, m2, m3 = _sample_next_preview(n1, n2, n3, rng)
            total += mc_rollout(child_board, nxt, m1, m2, m3, params.depth - 1, rng)

        avg = total / R
        if avg > best_avg:
            best_avg = avg
            best_mv = mv

    return best_mv

# ---- Play a single game (deck-aware, MC-only) ----

def run_game(seed: Optional[int], max_steps: int, mc_depth: int, rollouts: int, verbose: bool):
    g = Threes.new(seed=seed)
    step = 0
    start = time.perf_counter()
    rng = np.random.default_rng(seed if seed is not None else None)
    mc_params = MCParams(depth=mc_depth, rollouts=rollouts)

    while step < max_steps:
        board = g.board
        next_card = g.next_card  # guaranteed preview for this turn
        # Remaining counts after this preview was drawn:
        n1, n2, n3 = counts_from_deck(g)

        mv = choose_move_monte_carlo(board, next_card, n1, n2, n3, mc_params, rng)
        if mv is None:
            break

        g.try_move(mv)

        if verbose:
            print(f"\nStep {step}  Move: {mv}  MC: depth={mc_params.depth}, rollouts={mc_params.rollouts}")
            g.print_board()
            print(f"Score: {g.score_board()}  Next preview: {g.next_preview()}  Elapsed: {time.perf_counter()-start:.3f}s")

        step += 1

    if verbose:
        print("\nGame over.")
        g.print_board()
        print(f"Max tile: {g.max_value()}  Score: {g.score_board()}")

    return g.max_value(), g.score_board()

# ---- Summary helpers ----

_TILE_THRESHOLDS = [384, 768, 1536, 3072, 6144]

def print_summary(scores: List[int], max_tiles: List[int], total_seconds: float):
    n = len(scores)
    print(f"{n} games completed!")
    print(f"Total time: {_format_total_time(total_seconds)}")

    low = min(scores) if n else 0
    med = int(round(statistics.median(scores))) if n else 0
    high = max(scores) if n else 0

    print(f"Low Score: {low}")
    print(f"Median Score: {med}")
    print(f"High Score: {high}")

    if n == 0:
        for t in _TILE_THRESHOLDS:
            print(f"% of games with at least a {t}: 0%")
        return

    for t in _TILE_THRESHOLDS:
        count = sum(1 for v in max_tiles if v >= t)
        pct = int(round(100.0 * count / n))
        print(f"% of games with at least a {t}: {pct}%")

# ---- CLI ----

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Deck-aware Threes! — Monte Carlo (serial only)")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--steps", type=int, default=100_000)
    ap.add_argument("--mc-depth", type=int, default=7, help="Monte Carlo playout depth")
    ap.add_argument("--rollouts", type=int, default=128, help="Monte Carlo rollouts per root move")
    ap.add_argument("--quiet", action="store_true")
    ap.add_argument("--games", type=int, default=5, help="Number of games to run (serial)")
    args = ap.parse_args()

    # -------- Main execution (serial only) --------
    scores: List[int] = []
    max_tiles: List[int] = []
    t0 = time.perf_counter()

    # Warm up Numba once to avoid first-move stalls
    _numba_warmup()
    for i in range(args.games):
        seed = args.seed if args.seed is not None else None
        max_val, score = run_game(seed, args.steps, args.mc_depth, args.rollouts, args.quiet)
        scores.append(score)
        max_tiles.append(max_val)
        if not args.quiet:
            print(f"{i + 1} / {args.games} games played")

    total_time = time.perf_counter() - t0
    print_summary(scores, max_tiles, total_time)
