# ---- Lightweight eval (replace with your heuristic) ----

# Same scoring rule as engine, local for speed
_SCORE_LUT = tuple(0 if c < 3 else 3 ** (c - 2) for c in range(16))

def _score_board(board: int) -> int:
    b = board
    s = 0
    for i in range(16):
        s += _SCORE_LUT[(b >> (i * 4)) & 0xF]
    return s

def _count_empty(board: int) -> int:
    b = board
    e = 0
    for i in range(16):
        e += 1 if ((b >> (i * 4)) & 0xF) == 0 else 0
    return e

def evaluate_board(board: int, next_card: int) -> float:
    
    return _score_board(board) + 150.0 * _count_empty(board) + (5.0 if next_card >= 3 else next_card * 2.0)
