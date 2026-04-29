from typing import List, Tuple


ROWS = 6
COLS = 7
Board = List[int]  # ROWS*COLS ints: 0 empty, 1 Yellow/Human, -1 Red/AI


def _idx(row: int, col: int) -> int:
    return row * COLS + col


def get_legal_moves(board: Board) -> List[int]:
    """Return column indices where a piece can be dropped (top row is empty)."""
    return [c for c in range(COLS) if board[c] == 0]


def make_move(board: Board, col: int, player: int) -> Board:
    """Drop player's piece in col via gravity (lowest empty row). Raises if col is full."""
    for row in range(ROWS - 1, -1, -1):
        if board[_idx(row, col)] == 0:
            new_board = board.copy()
            new_board[_idx(row, col)] = player
            return new_board
    raise ValueError(f"Column {col} is full")


def _build_win_lines() -> Tuple[Tuple[int, int, int, int], ...]:
    lines: List[Tuple[int, int, int, int]] = []
    for r in range(ROWS):
        for c in range(COLS - 3):
            lines.append((_idx(r, c), _idx(r, c + 1), _idx(r, c + 2), _idx(r, c + 3)))
    for c in range(COLS):
        for r in range(ROWS - 3):
            lines.append((_idx(r, c), _idx(r + 1, c), _idx(r + 2, c), _idx(r + 3, c)))
    for r in range(ROWS - 3):
        for c in range(COLS - 3):
            lines.append((_idx(r, c), _idx(r + 1, c + 1), _idx(r + 2, c + 2), _idx(r + 3, c + 3)))
    for r in range(ROWS - 3):
        for c in range(3, COLS):
            lines.append((_idx(r, c), _idx(r + 1, c - 1), _idx(r + 2, c - 2), _idx(r + 3, c - 3)))
    return tuple(lines)


WIN_LINES: Tuple[Tuple[int, int, int, int], ...] = _build_win_lines()


def check_winner(board: Board) -> int:
    for a, b, c, d in WIN_LINES:
        s = board[a] + board[b] + board[c] + board[d]
        if s == 4:
            return 1
        if s == -4:
            return -1
    return 0


def is_draw(board: Board) -> bool:
    return check_winner(board) == 0 and all(v != 0 for v in board)
