# ---- Lightweight eval (replace with your heuristic) ----

# Same scoring rule as engine, local for speed
_SCORE_LUT = tuple(0 if c < 3 else 3 ** (c - 2) for c in range(16))

from math import sqrt

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

    # These medians are listed with depth 4, rollouts 32 across 25 games at seed 1000:

    #return sqrt(_score_board(board)) + 50 * _count_empty(board) (Median: 10,611)
    #return sqrt(_score_board(board)) + 200 * _count_empty(board)  (Median: 20,600)
    #return sqrt(_score_board(board)) + 400 * _count_empty(board) (Median: 20,600)
    #return sqrt(_score_board(board)) + 10 * (_count_empty(board) ** 3)   (Median: 10,600)
    #return _score_board(board) + 20 * (_count_empty(board) ** 2) (Median: 10,824)
    #return (_score_board(board) ** (1/3)) + 20 * (_count_empty(board) ** 2) (Median: 22,500)
    #return sqrt(_score_board(board)) + 40 * (_count_empty(board) ** 2) (Median: 21,600)
    #return sqrt(_score_board(board)) + 20 * (_count_empty(board) ** 2)    (Median: 23,080)
    #return sqrt(_score_board(board)) + 10 * (_count_empty(board) ** 2) (Median: 23,300)
    #return sqrt(_score_board(board)) + 5 * (_count_empty(board) ** 2) (Median: 23,577)
    #return sqrt(_score_board(board)) + 2 * (_count_empty(board) ** 2) (Median: 22,767)

    # These medians are listed with depth 9, rollouts 128 across 25 games at seed 2000:
    #return sqrt(_score_board(board)) + 5 * (_count_empty(board) ** 2) (Median: 69,000)
    #return sqrt(_score_board(board)) + 10 * (_count_empty(board) ** 2) (Median: 31,000)

    # These medians are listed with depth 10, rollouts 256 across 100 games at seed 3000:
    return sqrt(_score_board(board)) + 5 * (_count_empty(board) ** 2)

    # These medians are listed with depth 4 expectimax across 10 games at seed 3000:
    # return sqrt(_score_board(board)) + 5 * (_count_empty(board) ** 2) 
