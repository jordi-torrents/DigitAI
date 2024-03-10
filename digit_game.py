from itertools import pairwise
from typing import Iterator

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes

CONNECTIONS = (
    0,
    0,
    1,
    3,
    6,
    8,
    11,
    14,
    17,
    20,
    23,
    26,
    29,
    32,
    35,
    38,
    41,
    44,
    47,
)


def matches_in_board(board: np.ndarray) -> Iterator[tuple[int, int, int, int, int]]:
    for col_i, col_j in pairwise(range(5)):
        if board[0, col_i] == board[0, col_j]:
            yield col_i, col_j, 0, 0, board[0, col_i]
        if board[col_i, 0] == board[col_j, 0]:
            yield 0, 0, col_i, col_j, board[col_i, 0]

    for row in range(1, 5):
        for col_i, col_j in pairwise(range(5)):
            if board[row, col_i] == board[row, col_j]:
                yield col_i, col_j, row, row, board[row, col_i]

            if board[col_i, row] == board[col_j, row]:
                yield row, row, col_i, col_j, board[col_i, row]

            if board[row, col_i] == board[row - 1, col_j]:
                yield col_i, col_j, row, row - 1, board[row, col_i]

            if board[row - 1, col_i] == board[row, col_j]:
                yield col_i, col_j, row - 1, row, board[row, col_j]


def compute_absolute_score(board: np.ndarray):
    return sum(match[4] for match in matches_in_board(board))


class Game:
    size: int = 5

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.numbers = np.random.randint(1, 10, self.size * self.size, np.uint8)
        self.board = np.zeros((self.size, self.size), np.int64)
        self.step: int = 0

    def print_board(self):
        fig, ax = plt.subplots(figsize=(3, 3))
        assert isinstance(ax, Axes)
        ax.set(
            xlim=(-0.0, 4.5),
            ylim=(4.5, -0.5),
            xticks=np.arange(5) - 0.5,
            yticks=np.arange(5) - 0.5,
            xticklabels=(),
            yticklabels=(),
            aspect=1,
        )

        for i, j in np.ndindex(self.board.shape):
            ax.text(
                i,
                j,
                self.board[j, i] or "",
                fontsize=20,
                horizontalalignment="center",
                verticalalignment="center",
            )
        ax.grid()

        for match in matches_in_board(self.board):
            col_i, col_j, row_i, row_j, value = match

            if value == 0:
                continue
            col_pad = 0.15 * (col_j - col_i)
            row_pad = 0.25 * (row_j - row_i)

            ax.plot(
                (col_i + col_pad, col_j - col_pad),
                (row_i + row_pad, row_j - row_pad),
                "r",
            )

        fig.tight_layout(pad=0)
        return fig

    def board_absolute_score(self) -> int:
        return compute_absolute_score(self.board)

    def board_score(self) -> float:
        return self.board_absolute_score() / self.max_score()

    def max_score(self) -> int:
        return sum(
            val * CONNECTIONS[count]
            for val, count in enumerate(np.bincount(self.numbers))
        )

    def empty_cells(self) -> np.ndarray:
        return np.asarray(np.where(self.board == 0)).T

    @property
    def next_number(self) -> int:
        try:
            return self.numbers[self.step]
        except IndexError:
            return 0

    def pop_next_number(self) -> int:
        next_number = self.next_number
        if next_number != 0:
            self.step += 1
        return next_number
