from itertools import pairwise, permutations
from typing import Iterator

import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np
import torch
from scipy.ndimage import gaussian_filter1d
from torch import Tensor, nn, optim, tensor
from tqdm import trange

DEVICE = "cuda"


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

    def reset(self):
        self.numbers = np.random.randint(1, 10, self.size * self.size, np.uint8)
        self.board = np.zeros((self.size, self.size), np.uint8)
        self.step: int = 0

    def print_board(self, filename: str = ""):
        fig, ax = plt.subplots(figsize=(3, 3))

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

        plt.tight_layout(pad=0)
        plt.tick_params("both", size=0)
        if filename:
            fig.savefig(filename + f"_{self.step}.png")
        else:
            plt.show()
        plt.close()

    def board_absolute_score(self) -> int:
        return compute_absolute_score(self.board)

    def board_score(self) -> float:
        return self.board_absolute_score() / self.max_score()

    def max_score(self) -> int:
        connections = (
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

        return sum(
            val * connections[count]
            for val, count in enumerate(np.bincount(self.numbers))
        )

    def empty_cells(self):
        return np.asarray(np.where(self.board == 0)).T

    @property
    def next_number(self) -> int | None:
        try:
            return self.numbers[self.step]
        except IndexError:
            return None

    def pop_next_number(self) -> int | None:
        next_number = self.next_number
        if next_number is not None:
            self.step += 1
        return next_number


class ScorePredictor(nn.Module):
    def __init__(self, board_size: int = 5, n_numbers: int = 9):
        super().__init__()

        # self.layers = nn.Sequential(
        #     nn.Conv2d(n_numbers, 16, kernel_size=3, padding=1),
        #     nn.ReLU(),
        #     nn.Flatten(),
        #     nn.Linear(16 * (board_size ) ** 2, 128),
        #     nn.ReLU(),
        #     nn.Linear(128, 1),
        #     nn.Sigmoid(),
        # )

        self.layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(n_numbers * board_size * board_size, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid(),
        )

    def forward(self, game: Tensor) -> Tensor:
        # score = torch.rand(1)

        # score = torch.tensor(game.board_absolute_score())

        score = self.layers(game)
        return score


class Agent:
    def __init__(self, game: Game, score_predictor: ScorePredictor):
        self.game = game
        self.score_predictor = score_predictor
        self.optimizer = optim.Adam(self.score_predictor.parameters(), lr=0.001)
        self.predicted_scores: list[Tensor] = []

    @staticmethod
    def generate_batch(
        board: np.ndarray,
        possible_moves: np.ndarray,
        next_number: int,
        next_next_number: int | None = None,
    ) -> tuple[np.ndarray, Tensor]:
        one_hot_encoded_board: Tensor = (
            nn.functional.one_hot(
                tensor(board, dtype=torch.int64),
                num_classes=10,
            )[..., 1:]
            .permute(2, 0, 1)
            .float()
        )

        n_moves = len(possible_moves)

        if next_next_number is None:
            batch = one_hot_encoded_board.repeat(n_moves, 1, 1, 1)
            for batch_element, (row, col) in zip(batch, possible_moves):
                batch_element[next_number - 1, row, col] = 1
            return possible_moves, batch

        batch = one_hot_encoded_board.repeat((n_moves * (n_moves - 1)), 1, 1, 1)
        for batch_element, ((row1, col1), (row2, col2)) in zip(
            batch, permutations(possible_moves, 2)
        ):
            batch_element[next_number - 1, row1, col1] = 1.0
            batch_element[next_next_number - 1, row2, col2] = 1.0
        return possible_moves.repeat(n_moves - 1, 0), batch

    def generate_one_size_batch(self, board: np.ndarray) -> Tensor:

        return (
            nn.functional.one_hot(
                tensor(board, dtype=torch.int64),
                num_classes=10,
            )[..., 1:]
            .permute(2, 0, 1)
            .float()
            .unsqueeze(0)
        )

    def perform_step(self) -> bool:
        next_number = self.game.pop_next_number()
        next_next_number = self.game.next_number
        if next_number is None:
            return True
        possible_moves = self.game.empty_cells()

        possible_moves, board_batch = self.generate_batch(
            self.game.board, possible_moves, next_number, next_next_number
        )

        predicted_scores = self.score_predictor.forward(board_batch.to(DEVICE))
        choosen_index = predicted_scores.argmax()

        if self.game.step == 24:
            assert len(possible_moves) == 2
            assert next_next_number is not None
            first_option = self.game.board.copy()
            first_option[possible_moves[0]] = next_number
            first_option[possible_moves[1]] = next_next_number
            second_option = self.game.board.copy()
            second_option[possible_moves[1]] = next_number
            second_option[possible_moves[0]] = next_next_number
            if compute_absolute_score(first_option) > compute_absolute_score(
                second_option
            ):
                choosen_index = 0
            else:
                choosen_index = 1

        choosen_move = possible_moves[choosen_index]
        # choosen_score = predicted_scores[choosen_index]
        # self.predicted_scores.append(choosen_score)
        self.game.board[choosen_move[0], choosen_move[1]] = next_number

        current_encoded_board = self.generate_one_size_batch(self.game.board)
        prediction = self.score_predictor(current_encoded_board.to(DEVICE))
        self.predicted_scores.append(prediction)

        return False

    def play(self, n_games: int = 1, learn: bool = False, output_file: str = ""):
        if output_file:
            with open(output_file, "w") as f:
                pass
        scores = []
        losses = []
        for game_number in trange(n_games):
            self.game.reset()
            finished = False
            while not finished:
                finished = self.perform_step()

            final_score = self.game.board_score()
            scores.append(final_score)

            loss = sum(
                (final_score - score).square() for score in self.predicted_scores
            )
            self.predicted_scores.clear()
            assert isinstance(loss, Tensor)
            losses.append(loss.item())
            if output_file:
                with open(output_file, "a") as f:
                    f.write(f"{final_score:.4f}, {loss.item():.4f}\n")

            if learn:
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            if game_number % 50 == 0:
                plot_results(scores, losses)

        plot_results(scores, losses)


def plot_results(scores, losses):
    fig, ax = plt.subplots(2, sharex=True, figsize=(6, 4))

    ax[0].plot(scores, lw=1)
    ax[1].plot(losses, lw=1)
    ax[0].set(
        ylim=(0, 1),
        ylabel="Score",
        title=f"{np.mean(scores[-500:]):.1%} mean 500 scores",
    )
    ax[0].yaxis.set_major_formatter(mtick.PercentFormatter())
    ax[0].axhline(0.272, color="gray", linestyle=":", label="Random")
    ax[0].axhline(0.465, color="gray", linestyle=":", label="Maximize")
    ax[1].set(ylim=(0, None), ylabel="Loss", xlabel="Games")

    if len(scores) > 200:
        N = 100
        ax[0].plot(gaussian_filter1d(scores, 30, mode="nearest"))
        ax[1].plot(gaussian_filter1d(losses, 30, mode="nearest"))
    plt.tight_layout(pad=0.2)
    fig.savefig("scores")
    plt.close()


game = Game()
score_predictor = ScorePredictor().to(DEVICE)
agent = Agent(game, score_predictor)

agent.play(n_games=5000, learn=True, output_file="scores.dat")
