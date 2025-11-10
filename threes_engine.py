# threes_engine_numba.py
# Threes engine optimized for optimax: precomputed LUTs + Numba-compiled hot paths.
# Python 3.10+

from __future__ import annotations
from dataclasses import dataclass
from typing import List, Tuple, Optional
import random
import numpy as _np  # type: ignore
from numba import njit  # type: ignore
import numpy as _np
U64_MASK = (1 << 64) - 1
_NUMBA_OK = True

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
    # Fast nibble-swap: [a b c d] -> [d c b a]
    return ((row16 & 0x000F) << 12) | ((row16 & 0x00F0) << 4) | \
           ((row16 & 0x0F00) >> 4)  | ((row16 & 0xF000) >> 12)

def transpose_4x4(board: int) -> int:
    """Unrolled transpose of 4x4 nibbles packed into 64-bit int."""
    result = 0
    # Row 0 -> Column 0
    result |= ((board >>  0) & 0xF) <<  0
    result |= ((board >>  4) & 0xF) << 16
    result |= ((board >>  8) & 0xF) << 32
    result |= ((board >> 12) & 0xF) << 48
    # Row 1 -> Column 1
    result |= ((board >> 16) & 0xF) <<  4
    result |= ((board >> 20) & 0xF) << 20
    result |= ((board >> 24) & 0xF) << 36
    result |= ((board >> 28) & 0xF) << 52
    # Row 2 -> Column 2
    result |= ((board >> 32) & 0xF) <<  8
    result |= ((board >> 36) & 0xF) << 24
    result |= ((board >> 40) & 0xF) << 40
    result |= ((board >> 44) & 0xF) << 56
    # Row 3 -> Column 3
    result |= ((board >> 48) & 0xF) << 12
    result |= ((board >> 52) & 0xF) << 28
    result |= ((board >> 56) & 0xF) << 44
    result |= ((board >> 60) & 0xF) << 60
    return result

# ---------------------------
# Value/code conversions
# ---------------------------

def code_to_value(code: int) -> int:
    if code == 0: return 0
    if code == 1: return 1
    if code == 2: return 2
    return 3 << (code - 3)

def value_to_code(value: int) -> int:
    if value == 0: return 0
    if value == 1: return 1
    if value == 2: return 2
    k = (value // 3).bit_length() - 1
    return 3 + k

def can_merge(left_code: int, right_code: int) -> bool:
    if left_code == 0 or right_code == 0: return False
    if (left_code, right_code) in ((1,2),(2,1)): return True
    if left_code >= 3 and left_code == right_code: return True
    return False

def merge_code(left_code: int, right_code: int) -> int:
    if (left_code, right_code) in ((1,2),(2,1)): return 3
    return left_code + 1

# ---------------------------
# Row transition tables (Python build)
# ---------------------------

def _simulate_row_left_py(row16: int) -> Tuple[int, bool]:
    cells = [
        (row16 >> 0) & 0xF,
        (row16 >> 4) & 0xF,
        (row16 >> 8) & 0xF,
        (row16 >> 12) & 0xF,
    ]
    changed = False
    merged = [False, False, False, False]
    for i in range(1, 4):
        v = cells[i]
        if v == 0:
            continue
        left = cells[i-1]
        if left == 0:
            cells[i-1] = v
            cells[i] = 0
            changed = True
        elif ((left, v) in ((1,2),(2,1)) or (left >= 3 and left == v)) and not merged[i-1]:
            cells[i-1] = 3 if (left, v) in ((1,2),(2,1)) else left + 1
            cells[i] = 0
            merged[i-1] = True
            changed = True
    new_row16 = (cells[0] << 0) | (cells[1] << 4) | (cells[2] << 8) | (cells[3] << 12)
    return new_row16, changed

# Build tables (Python, once)
ROW_LEFT_TABLE_PY: List[int] = [0] * 65536
REVERSE_ROW16_PY:  List[int] = [0] * 65536
ROW_RIGHT_TABLE_PY: List[int] = [0] * 65536

for row16 in range(65536):
    new_row, ch = _simulate_row_left_py(row16)
    ROW_LEFT_TABLE_PY[row16] = new_row | (int(ch) << 16)

for r in range(65536):
    REVERSE_ROW16_PY[r] = reverse_row16(r)

for r in range(65536):
    entry = ROW_LEFT_TABLE_PY[REVERSE_ROW16_PY[r]]
    new_rev = entry & 0xFFFF
    changed = (entry >> 16) & 1
    new_row = REVERSE_ROW16_PY[new_rev]
    ROW_RIGHT_TABLE_PY[r] = new_row | (changed << 16)

# Numba-friendly versions of the tables
if _NUMBA_OK:
    ROW_LEFT_TABLE = _np.asarray(ROW_LEFT_TABLE_PY, dtype=_np.uint32)
    ROW_RIGHT_TABLE = _np.asarray(ROW_RIGHT_TABLE_PY, dtype=_np.uint32)
else:
    ROW_LEFT_TABLE = ROW_LEFT_TABLE_PY
    ROW_RIGHT_TABLE = ROW_RIGHT_TABLE_PY

# Score table
SCORE_LUT: Tuple[int, ...] = tuple(0 if c < 3 else 3 ** (c - 2) for c in range(16))

# ---------------------------
# Numba hot functions
# ---------------------------

if _NUMBA_OK:

    @njit(cache=True, fastmath=True)
    def _transpose_4x4_nb(board: _np.uint64) -> _np.uint64:
        b = board
        result = _np.uint64(0)
        result |= ((b >>  0) & 0xF) <<  0
        result |= ((b >>  4) & 0xF) << 16
        result |= ((b >>  8) & 0xF) << 32
        result |= ((b >> 12) & 0xF) << 48
        result |= ((b >> 16) & 0xF) <<  4
        result |= ((b >> 20) & 0xF) << 20
        result |= ((b >> 24) & 0xF) << 36
        result |= ((b >> 28) & 0xF) << 52
        result |= ((b >> 32) & 0xF) <<  8
        result |= ((b >> 36) & 0xF) << 24
        result |= ((b >> 40) & 0xF) << 40
        result |= ((b >> 44) & 0xF) << 56
        result |= ((b >> 48) & 0xF) << 12
        result |= ((b >> 52) & 0xF) << 28
        result |= ((b >> 56) & 0xF) << 44
        result |= ((b >> 60) & 0xF) << 60
        return result

    @njit(cache=True, fastmath=True)
    def _get_row16_nb(board: _np.uint64, r: _np.uint8) -> _np.uint16:
        return _np.uint16((board >> (r * 16)) & _np.uint64(0xFFFF))

    @njit(cache=True, fastmath=True)
    def _set_row16_nb(board: _np.uint64, r: _np.uint8, row16: _np.uint16) -> _np.uint64:
        m = _np.uint64(0xFFFF) << (r * 16)
        return (board & ~m) | (_np.uint64(row16) << (r * 16))

    @njit(cache=True, fastmath=True)
    def _move_rows_nb(board: _np.uint64, table: _np.ndarray) -> Tuple[_np.uint64, _np.uint8, _np.uint8]:
        """
        Apply row-wise move using a precomputed table.
        Returns (new_board, candidates_mask, moved_flag)
        - candidates_mask: 4-bit mask, bit r=1 if row r changed (spawn at edge determined by caller)
        - moved_flag: 1 if any row changed
        """
        out = board
        candidates_mask = _np.uint8(0)
        moved_any = _np.uint8(0)
        for r in range(4):
            row = _get_row16_nb(out, r)
            entry = _np.uint32(table[row])
            new_row = _np.uint16(entry & 0xFFFF)
            changed = _np.uint8((entry >> 16) & 1)
            if changed:
                moved_any = _np.uint8(1)
                candidates_mask |= _np.uint8(1 << r)
            if new_row != row:
                out = _set_row16_nb(out, r, new_row)
        return out, candidates_mask, moved_any

    @njit(cache=True, fastmath=True)
    def move_left_nb(board: _np.uint64) -> Tuple[_np.uint64, _np.uint8, _np.uint8]:
        return _move_rows_nb(board, ROW_LEFT_TABLE)

    @njit(cache=True, fastmath=True)
    def move_right_nb(board: _np.uint64) -> Tuple[_np.uint64, _np.uint8, _np.uint8]:
        return _move_rows_nb(board, ROW_RIGHT_TABLE)

    @njit(cache=True, fastmath=True)
    def move_up_nb(board: _np.uint64) -> Tuple[_np.uint64, _np.uint8, _np.uint8]:
        tb = _transpose_4x4_nb(board)
        nb, mask, moved = _move_rows_nb(tb, ROW_LEFT_TABLE)
        return _transpose_4x4_nb(nb), mask, moved  # mask bits correspond to columns

    @njit(cache=True, fastmath=True)
    def move_down_nb(board: _np.uint64) -> Tuple[_np.uint64, _np.uint8, _np.uint8]:
        tb = _transpose_4x4_nb(board)
        nb, mask, moved = _move_rows_nb(tb, ROW_RIGHT_TABLE)
        return _transpose_4x4_nb(nb), mask, moved

# ---------------------------
# Deck + bonus pool
# ---------------------------

@dataclass
class BasicDeck:
    rng: random.Random
    cards: Optional[List[int]] = None
    idx: int = 0

    def __post_init__(self):
        self._fresh()

    def _fresh(self):
        self.cards = [1]*4 + [2]*4 + [3]*4  # codes
        self.rng.shuffle(self.cards)
        self.idx = 0

    def peek(self) -> int:
        if self.idx >= len(self.cards):
            self._fresh()
        return self.cards[self.idx]

    def draw(self) -> int:
        v = self.cards[self.idx]
        self.idx += 1
        if self.idx >= len(self.cards):
            self._fresh()
        return v

def bonus_pool_codes(max_board_value: int) -> List[int]:
    cap = max_board_value // 8
    if cap < 6:
        return []
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
        g = Threes(rng=rng, deck=BasicDeck(rng))
        g._init_board()
        g._draw_next_card()
        return g

    # ---------- Board init ----------

    def _init_board(self):
        # Draw 8 basic cards and place randomly
        cells = [(r, c) for r in range(4) for c in range(4)]
        self.rng.shuffle(cells)
        for (r, c) in cells[:8]:
            code = self.deck.draw()
            self.board = set_cell(self.board, r, c, code)

    # ---------- Utilities ----------

    def max_value(self) -> int:
        b = self.board
        max_code = 0
        for i in range(16):
            c = (b >> (i * 4)) & 0xF
            if c > max_code:
                max_code = c
        if max_code <= 2:
            return max_code
        return 3 << (max_code - 3)

    def _draw_next_card(self):
        pool = bonus_pool_codes(self.max_value())
        use_bonus = bool(pool) and (self.rng.randrange(21) == 0)
        if use_bonus:
            self.next_card = self.rng.choice(pool)
            self.next_is_bonus = True
        else:
            self.next_card = self.deck.draw()
            self.next_is_bonus = False

    def next_preview(self) -> str:
        return "1" if self.next_card == 1 else ("2" if self.next_card == 2 else "3+")

    # ---------- Candidates helpers ----------

    @staticmethod
    def _mask_to_indices(mask: int) -> List[int]:
        # Convert 4-bit mask to indices [0..3]
        out = []
        if mask & 1: out.append(0)
        if mask & 2: out.append(1)
        if mask & 4: out.append(2)
        if mask & 8: out.append(3)
        return out

    # ---------- Moves ----------

    def _move_left_rows(self, board: int) -> Tuple[int, List[int], bool]:
        if _NUMBA_OK:
            nb, mask, moved = move_left_nb(_np.uint64(board & U64_MASK))
            # mask bits denote rows that changed; spawn at right edge col=3
            candidates = [r*4 + 3 for r in self._mask_to_indices(int(mask))]
            return int(nb), candidates, bool(moved)
        # Fallback Python path
        moved_any = False
        out = board
        candidates = []
        table = ROW_LEFT_TABLE_PY
        for r in range(4):
            row = get_row16(out, r)
            entry = table[row]
            new_row = entry & 0xFFFF
            changed = (entry >> 16) & 1
            if changed:
                moved_any = True
                candidates.append(r*4 + 3)
            if new_row != row:
                out = set_row16(out, r, new_row)
        return out, candidates, moved_any

    def _move_right_rows(self, board: int) -> Tuple[int, List[int], bool]:
        if _NUMBA_OK:
            nb, mask, moved = move_right_nb(_np.uint64(board & U64_MASK))
            candidates = [r*4 + 0 for r in self._mask_to_indices(int(mask))]
            return int(nb), candidates, bool(moved)
        moved_any = False
        out = board
        candidates = []
        table = ROW_RIGHT_TABLE_PY
        for r in range(4):
            row = get_row16(out, r)
            entry = table[row]
            new_row = entry & 0xFFFF
            changed = (entry >> 16) & 1
            if changed:
                moved_any = True
                candidates.append(r*4 + 0)
            if new_row != row:
                out = set_row16(out, r, new_row)
        return out, candidates, moved_any

    def _move_up_cols(self, board: int) -> Tuple[int, List[int], bool]:
        if _NUMBA_OK:
            nb, mask, moved = move_up_nb(_np.uint64(board & U64_MASK))
            # mask bits denote columns (because we operated in transposed space)
            candidates = [3*4 + c for c in self._mask_to_indices(int(mask))]  # bottom row
            return int(nb), candidates, bool(moved)
        tb = transpose_4x4(board)
        tb2, row_candidates, moved = self._move_left_rows(tb)
        candidates = []
        for idx in row_candidates:
            tr = idx // 4
            candidates.append(3*4 + tr)
        return transpose_4x4(tb2), candidates, moved

    def _move_down_cols(self, board: int) -> Tuple[int, List[int], bool]:
        if _NUMBA_OK:
            nb, mask, moved = move_down_nb(_np.uint64(board & U64_MASK))
            candidates = [0*4 + c for c in self._mask_to_indices(int(mask))]  # top row
            return int(nb), candidates, bool(moved)
        tb = transpose_4x4(board)
        tb2, row_candidates, moved = self._move_right_rows(tb)
        candidates = []
        for idx in row_candidates:
            tr = idx // 4
            candidates.append(0*4 + tr)
        return transpose_4x4(tb2), candidates, moved

    def try_move(self, direction: MoveDir) -> bool:
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
            return False

        # Place next_card on a random candidate edge slot
        spawn_pos = self.rng.choice(candidates)
        sr, sc = divmod(spawn_pos, 4)
        self.board = set_cell(nb, sr, sc, self.next_card)

        self._draw_next_card()
        return True

    # ---------- Convenience ----------

    def has_moves(self) -> bool:
        for d in (0, 1, 2, 3):
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

    def score_board(self) -> int:
        total = 0
        b = self.board
        for i in range(16):
            total += SCORE_LUT[(b >> (i * 4)) & 0xF]
        return total

    def print_board(self):
        rows = []
        for r in range(4):
            row = []
            for c in range(4):
                v = code_to_value(get_cell(self.board, r, c))
                row.append(f"{v:>4}")
            rows.append(" ".join(row))
        print("\n".join(rows))

# ---------------------------
# Quick demo (interactive)
# ---------------------------

if __name__ == "__main__":
    rng = random.Random(1337)
    game = Threes.new(seed=1337)

    print("Threes! – demo")
    print("Controls: w/a/s/d for up/left/down/right; q to quit\n")

    while True:
        disp = game.next_preview()
        print("Current board:")
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
