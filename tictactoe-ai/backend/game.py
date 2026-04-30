from typing import List, Tuple


Board = List[int]  # 9 ints: 0 empty, 1 X, -1 O


WIN_LINES: Tuple[Tuple[int, int, int], ...] = (
    (0, 1, 2),
    (3, 4, 5),
    (6, 7, 8),
    (0, 3, 6),
    (1, 4, 7),
    (2, 5, 8),
    (0, 4, 8),
    (2, 4, 6),
)


def get_legal_moves(board: Board) -> List[int]:
    return [i for i, v in enumerate(board) if v == 0]


def make_move(board: Board, position: int, player: int) -> Board:
    if board[position] != 0:
        raise ValueError("Invalid move: position already taken")
    new_board = board.copy()
    new_board[position] = player
    return new_board


def check_winner(board: Board) -> int:
    for a, b, c in WIN_LINES:
        s = board[a] + board[b] + board[c]
        if s == 3:
            return 1
        if s == -3:
            return -1
    return 0


def is_draw(board: Board) -> bool:
    return check_winner(board) == 0 and all(v != 0 for v in board)


def print_board(board: Board) -> None:
    symbols = {1: "X", -1: "O", 0: " "}
    rows = [
        f" {symbols[board[0]]} | {symbols[board[1]]} | {symbols[board[2]]} ",
        f" {symbols[board[3]]} | {symbols[board[4]]} | {symbols[board[5]]} ",
        f" {symbols[board[6]]} | {symbols[board[7]]} | {symbols[board[8]]} ",
    ]
    sep = "---+---+---"
    print("\n".join([rows[0], sep, rows[1], sep, rows[2]]))
