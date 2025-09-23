# threes_engine.py
# Compact Threes engine with 64-bit board, one-step slide, deck + bonus pool.
# Python 3.10+

from __future__ import annotations
from dataclasses import dataclass
from typing import List, Tuple, Optional, Iterable
import random

# ---------------------------
# Encoding helpers (nibbles)
# ---------------------------

def pos_shift(r: int, c: int) -> int:
    return ((r << 2) + c) << 2  # (r*4 + c) * 4

def get_cell(board: int, r: int, c: int) -> int:
    return (board >> pos_shift(r, c)) & 0xF

def set_cell(board: int, r: int, c: int, val: int) -> int:
    s = pos_shift(r, c)
    return (board & ~(0xF << s)) | ((val & 0xF) << s)

def get_row16(board: int, r: int) -> int:
    return (board >> (r * 16)) & 0xFFFF

def set_row16(board: int, r: int, row16: int) -> int:
    m = 0xFFFF << (r * 16)
    return (board & ~m) | ((row16 & 0xFFFF) << (r * 16))

def reverse_row16(row16: int) -> int:
    # Swap nibbles: [a b c d] -> [d c b a]
    a = (row16 >> 0) & 0xF
    b = (row16 >> 4) & 0xF
    c = (row16 >> 8) & 0xF
    d = (row16 >> 12) & 0xF
    return (a << 12) | (b << 8) | (c << 4) | (d << 0)

def transpose_4x4(board: int) -> int:
    # Generic (tiny) transpose; simple loops are plenty fast for Python here.
    out = 0
    for r in range(4):
        for c in range(4):
            v = (board >> pos_shift(r, c)) & 0xF
            out |= (v << pos_shift(c, r))
    return out

# ---------------------------
# Value/code conversions
# ---------------------------

def code_to_value(code: int) -> int:
    if code == 0: return 0
    if code == 1: return 1
    if code == 2: return 2
    # 3,6,12,... doubling sequence
    return 3 << (code - 3)

def value_to_code(value: int) -> int:
    if value == 0: return 0
    if value == 1: return 1
    if value == 2: return 2
    # value must be 3 * 2^k
    k = (value // 3).bit_length() - 1  # since value is a power of two times 3
    return 3 + k

def can_merge(left_code: int, right_code: int) -> bool:
    if left_code == 0 or right_code == 0: return False
    if (left_code, right_code) in ((1,2),(2,1)): return True
    if left_code >= 3 and left_code == right_code: return True
    return False

def merge_code(left_code: int, right_code: int) -> int:
    # Precondition: can_merge(left_code, right_code) is True
    if (left_code, right_code) in ((1,2),(2,1)): return 3
    return left_code + 1  # equal >=3 -> next code (double value)

# ---------------------------
# Precompute row-left transitions (one-step slide)
# Table entry: 0..65535 -> (new_row16 | (changed<<16))
# ---------------------------

ROW_LEFT_TABLE: List[int] = [0] * 65536

def _simulate_row_left(row16: int) -> Tuple[int, bool]:
    # Extract 4 nibbles into a mutable list
    cells = [
        (row16 >> 0) & 0xF,
        (row16 >> 4) & 0xF,
        (row16 >> 8) & 0xF,
        (row16 >> 12) & 0xF,
    ]
    changed = False
    merged = [False, False, False, False]  # track merges at target slots

    # Process leftward one step, left-to-right
    for i in range(1, 4):
        v = cells[i]
        if v == 0:
            continue
        left = cells[i-1]
        if left == 0:
            cells[i-1] = v
            cells[i] = 0
            changed = True
        elif can_merge(left, v) and not merged[i-1]:
            cells[i-1] = merge_code(left, v)
            cells[i] = 0
            merged[i-1] = True
            changed = True
        else:
            # stays
            pass

    new_row16 = (cells[0] << 0) | (cells[1] << 4) | (cells[2] << 8) | (cells[3] << 12)
    return new_row16, changed

def _build_row_table() -> None:
    for row16 in range(65536):
        new_row, ch = _simulate_row_left(row16)
        ROW_LEFT_TABLE[row16] = new_row | (int(ch) << 16)

_build_row_table()

# ---------------------------
# Deck + bonus pool
# ---------------------------

@dataclass
class BasicDeck:
    rng: random.Random
    cards: List[int] = None
    idx: int = 0

    def __post_init__(self):
        self._fresh()

    def _fresh(self):
        self.cards = [1]*4 + [2]*4 + [3]*4  # codes
        self.rng.shuffle(self.cards)
        self.idx = 0

    def peek(self) -> int:
        return self.cards[self.idx]

    def draw(self) -> int:
        v = self.cards[self.idx]
        self.idx += 1
        if self.idx >= len(self.cards):
            # Immediately create a fresh shuffled deck for future peeks/draws.
            self._fresh()
        return v

def bonus_pool_codes(max_board_value: int) -> List[int]:
    # Pool: {6,12,24,...} up to floor(high/8)
    cap = max_board_value // 8
    if cap < 6:
        return []
    # Find highest 6*2^k <= cap
    out = []
    v = 6
    while v <= cap:
        out.append(value_to_code(v))
        v <<= 1
    return out

# ---------------------------
# Engine
# ---------------------------

MoveDir = int  # 0=L,1=R,2=U,3=D

@dataclass
class Threes:
    rng: random.Random
    board: int = 0
    deck: BasicDeck = None
    next_card: int = 0  # code
    next_is_bonus: bool = False

    @staticmethod
    def new(seed: Optional[int] = None) -> "Threes":
        rng = random.Random(seed)
        g = Threes(rng=rng, board=0, deck=BasicDeck(rng))
        g._init_board()
        g._draw_next_card()  # initial preview
        return g

    # ---------- Board init ----------

    def _init_board(self):
        # Fresh deck already created in __post_init__
        # Draw 9 basic cards and place randomly
        cells = [(r, c) for r in range(4) for c in range(4)]
        self.rng.shuffle(cells)
        for (r, c) in cells[:8]:
            code = self.deck.draw()  # at start: always from deck (no bonus yet)
            self.board = set_cell(self.board, r, c, code)

    # ---------- Utilities ----------

    def max_value(self) -> int:
        m = 0
        b = self.board
        for i in range(16):
            code = (b >> (i*4)) & 0xF
            if code:
                v = code_to_value(code)
                if v > m:
                    m = v
        return m

    def _draw_next_card(self):
        # Choose next card based on current board (after last placement)
        pool = bonus_pool_codes(self.max_value())
        use_bonus = bool(pool) and (self.rng.randrange(21) == 0)
        if use_bonus:
            self.next_card = self.rng.choice(pool)
            self.next_is_bonus = True
        else:
            self.next_card = self.deck.draw()
            self.next_is_bonus = False

    def next_preview(self) -> str:
        """
        Returns (kind, display):
          kind in {"exact","ambiguous"}
          display: "1", "2", or "3+"
        """
        if self.next_card in (1, 2):
            return ("1" if self.next_card == 1 else "2")
        return ("3+")

    # ---------- Moves ----------

    def _move_left_rows(self, board: int) -> Tuple[int, List[int], bool]:
        """
        Returns (new_board, spawn_candidates, moved)
        spawn_candidates are absolute cell indices (0..15) at the right edge for rows that changed.
        """
        moved_any = False
        out = board
        candidates = []
        for r in range(4):
            row = get_row16(out, r)
            entry = ROW_LEFT_TABLE[row]
            new_row = entry & 0xFFFF
            changed = (entry >> 16) & 1
            if changed:
                moved_any = True
                candidates.append(r*4 + 3)  # right-edge cell of this row
            if new_row != row:
                out = set_row16(out, r, new_row)
        return out, candidates, moved_any

    def _move_right_rows(self, board: int) -> Tuple[int, List[int], bool]:
        moved_any = False
        out = board
        candidates = []
        for r in range(4):
            row = get_row16(out, r)
            rev_row = reverse_row16(row)
            entry = ROW_LEFT_TABLE[rev_row]
            new_rev = entry & 0xFFFF
            changed = (entry >> 16) & 1
            new_row = reverse_row16(new_rev)
            if changed:
                moved_any = True
                candidates.append(r*4 + 0)  # left-edge cell of this row
            if new_row != row:
                out = set_row16(out, r, new_row)
        return out, candidates, moved_any

    def _move_up_cols(self, board: int) -> Tuple[int, List[int], bool]:
        # Transpose, do left, transpose back.
        tb = transpose_4x4(board)
        tb2, row_candidates, moved = self._move_left_rows(tb)
        # Map candidates (in transposed space) back to absolute positions: right-edge in transposed -> bottom row in original.
        candidates = []
        for idx in row_candidates:
            tr = idx // 4
            # right-edge col=3 in transposed -> row=3 in original
            candidates.append(3*4 + tr)  # (row=3, col=tr)
        return transpose_4x4(tb2), candidates, moved

    def _move_down_cols(self, board: int) -> Tuple[int, List[int], bool]:
        tb = transpose_4x4(board)
        tb2, row_candidates, moved = self._move_right_rows(tb)
        # Left-edge in transposed -> top row in original.
        candidates = []
        for idx in row_candidates:
            tr = idx // 4
            candidates.append(0*4 + tr)  # (row=0, col=tr)
        return transpose_4x4(tb2), candidates, moved

    def try_move(self, direction: MoveDir) -> bool:
        """
        Apply one move if valid; return True if anything changed (and a card was spawned).
        Direction: 0=L,1=R,2=U,3=D
        """
        if direction == 0:
            nb, candidates, moved = self._move_left_rows(self.board)
        elif direction == 1:
            nb, candidates, moved = self._move_right_rows(self.board)
        elif direction == 2:
            nb, candidates, moved = self._move_up_cols(self.board)
        elif direction == 3:
            nb, candidates, moved = self._move_down_cols(self.board)
        else:
            raise ValueError("direction must be 0..3")

        if not moved or not candidates:
            return False  # invalid/no-op

        # Place the already-previewed next_card onto a random candidate edge slot
        spawn_pos = self.rng.choice(candidates)
        sr, sc = divmod(spawn_pos, 4)
        # Should be empty; assert in debug
        # assert get_cell(nb, sr, sc) == 0
        nb = set_cell(nb, sr, sc, self.next_card)

        self.board = nb

        # Draw new preview for next turn (uses UPDATED board for bonus pool limits)
        self._draw_next_card()
        return True

    # ---------- Convenience ----------

    def has_moves(self) -> bool:
        # If any direction yields a change, we have moves.
        for d in range(4):
            if self._peek_move(d):
                return True
        return False

    def _peek_move(self, direction: MoveDir) -> bool:
        if direction == 0:
            _, _, moved = self._move_left_rows(self.board)
        elif direction == 1:
            _, _, moved = self._move_right_rows(self.board)
        elif direction == 2:
            _, _, moved = self._move_up_cols(self.board)
        else:
            _, _, moved = self._move_down_cols(self.board)
        return moved

    # Classic Threes! score for a 64-bit nibble-encoded board.
    # Each tile is stored as a 4-bit *code*:
    #   0 -> empty, 1 -> tile '1', 2 -> tile '2', 3 -> '3', 4 -> '6', 5 -> '12', ...
    # Score rule: only codes >=3 contribute; code c contributes 3^(c-2).

    def score_board(self) -> int:
        """
        Compute the classic Threes! score from a 64-bit nibble board.
        Args:
            board: int, 16 tiles packed as 4-bit codes (row-major), i*4 shift per tile.
        Returns:
            int: total score.
        """
        SCORE_LUT = tuple(0 if c < 3 else 3 ** (c - 2) for c in range(16))
        total = 0
        # 16 tiles, each 4 bits
        for i in range(16):
            code = (self.board >> (i * 4)) & 0xF
            total += SCORE_LUT[code]
        return total


    def print_board(self):
        # For debugging: print raw values (not codes)
        rows = []
        for r in range(4):
            row = []
            for c in range(4):
                v = code_to_value(get_cell(self.board, r, c))
                row.append(f"{v:>4}")
            rows.append(" ".join(row))
        print("\n".join(rows))

# ---------------------------
# Quick demo
# ---------------------------

if __name__ == "__main__":

    rng = random.Random(1337)  # set a seed for reproducibility in this demo
    game = Threes.new(seed=rng)

    print("Threes! – demo")
    print("Controls: w/a/s/d for up/left/down/right; q to quit\n")

    while True:
        disp = game.next_preview()
        print("Initial board:")
        game.print_board()
        print("Next preview:", disp)

        ch = input("Move (w/a/s/d/q): ").strip().lower()
        if ch == "q":
            break
        key_to_dir = {"w": 2, "s": 3, "a": 0, "d": 1}
        if ch in key_to_dir:
            moved = game.try_move(key_to_dir[ch])
            if not moved:
                print("No change – invalid move.")
        else:
            print("Invalid input.")
