from collections import deque
from random import randint

import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np
import torch
from scipy.ndimage import gaussian_filter1d
from torch import Tensor, nn, optim
from torch.backends import cudnn
from tqdm import trange

from digit_game import Game, compute_absolute_score

cudnn.benchmark = True
DEVICE = torch.device("cuda")


class ScorePredictor(nn.Module):
    def __init__(self, board_size: int = 5, n_numbers: int = 10) -> None:
        super().__init__()
        self.n_numbers = n_numbers

        self.layers = nn.Sequential(
            nn.Linear(9 * (board_size * board_size + 1), 1024),
            nn.ReLU(),
            nn.Linear(1024, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid(),
        )

    def forward(self, game: Tensor) -> Tensor:
        return self.layers(
            nn.functional.one_hot(game.long(), num_classes=self.n_numbers)[:, :, 1:]
            .float()
            .flatten(1)
        )


class LossManager:
    def __init__(self, memory: int = 1000) -> None:
        self.loss_history: deque[np.ndarray] = deque(maxlen=memory)

    def losses_trend(self) -> Tensor:
        trend: np.ndarray = np.asarray(self.loss_history).mean(0)
        normalization = trend.mean() / trend
        return torch.tensor(normalization, device=DEVICE)

    def add_losses(self, losses: np.ndarray) -> None:
        self.loss_history.append(losses)


class Agent:
    breaking_step: int = -1
    "Step at which choose a random move"

    def __init__(self, game: Game, score_predictor: ScorePredictor) -> None:
        self.loss_manager = LossManager()
        self.game = game
        self.score_predictor = score_predictor
        self.optimizer = optim.Adam(
            self.score_predictor.parameters(),
            lr=0.0005,
        )
        self.schedule = optim.lr_scheduler.MultiStepLR(
            self.optimizer, [3000, 4000, 5000, 6000, 7000, 8000, 9000], 0.3
        )
        self.predicted_scores: list[Tensor] = []

    @staticmethod
    def generate_batch(
        board: np.ndarray,
        possible_moves: np.ndarray,
        current_number: int,
        next_number: int,
    ) -> Tensor:
        # prepare a board for each possible move -> shape (N,5,5)
        batch = board[None, ...].repeat(len(possible_moves), 0)

        # populate boards with possible moves -> shape (N,5,5)
        batch[
            range(len(possible_moves)), possible_moves[:, 0], possible_moves[:, 1]
        ] = current_number

        # add the next number information to each board -> shape (N,26)
        batch = np.concatenate(
            (
                batch.reshape(len(batch), -1),
                np.full((len(batch), 1), next_number, batch.dtype),
            ),
            1,
        )

        return torch.from_numpy(batch)

    def perform_step(self) -> bool:
        """Return finished"""
        current_number = self.game.pop_next_number()
        next_number = self.game.next_number
        if current_number == 0:
            return True
        possible_moves = self.game.empty_cells()

        if self.game.step == 25:
            assert len(possible_moves) == 1
            assert next_number == 0
            self.game.board[possible_moves[0, 0], possible_moves[0, 1]] = current_number
            return True

        board_batch = self.generate_batch(
            self.game.board, possible_moves, current_number, next_number
        )
        predicted_scores: Tensor = self.score_predictor.forward(board_batch.to(DEVICE))

        if self.game.step == self.breaking_step:
            choosen_index = randint(0, len(predicted_scores) - 1)
        else:
            choosen_index = predicted_scores.argmax()

        if self.breaking_step == -1 and self.game.step == 24:
            assert len(possible_moves) == 2
            assert next_number != 0
            first_option = self.game.board.copy()
            first_option[possible_moves[0, 0], possible_moves[0, 1]] = current_number
            first_option[possible_moves[1, 0], possible_moves[1, 1]] = next_number
            second_option = self.game.board.copy()
            second_option[possible_moves[1, 0], possible_moves[1, 1]] = current_number
            second_option[possible_moves[0, 0], possible_moves[0, 1]] = next_number
            if compute_absolute_score(first_option) > compute_absolute_score(
                second_option
            ):
                choosen_index = 0
            else:
                choosen_index = 1

        choosen_move = possible_moves[choosen_index]
        choosen_score = predicted_scores[choosen_index]
        self.predicted_scores.append(choosen_score)
        self.game.board[choosen_move[0], choosen_move[1]] = current_number

        return False

    def play(self, n_games: int = 1, learn: bool = False) -> None:
        scores = []
        all_losses = []
        batch_losses = []
        for game_number in trange(n_games):
            self.game.reset()
            # if learn:
            #     self.breaking_step = randint(0, 25)
            # else:
            #     self.breaking_step = -1
            finished = False
            self.predicted_scores.clear()
            while not finished:
                finished = self.perform_step()
            assert np.count_nonzero(self.game.board) == 25

            final_score = self.game.board_score()
            # absolute_final_score = self.game.board_absolute_score()
            scores.append(final_score)

            batch_losses = (torch.stack(self.predicted_scores) - final_score).square()

            if learn:
                self.optimizer.zero_grad()
                batch_losses.mean().backward()
                self.optimizer.step()
                # loss_normalization = losses.mean(0).detach()
                # loss_normalization = loss_normalization.mean() / loss_normalization

            if game_number % 300 == 0:
                plot_results(scores, all_losses)

        plot_results(scores, all_losses)


def plot_results(scores: list, losses: list) -> None:
    fig = plt.figure(figsize=(6, 4))
    ax0 = fig.add_subplot(2, 1, 1)
    ax1 = fig.add_subplot(2, 1, 2)

    ax0.plot(scores, lw=1)
    ax1.plot(losses, lw=1)
    ax0.set(
        ylim=(0, 1),
        ylabel="Score",
        title=f"{np.mean(scores[-500:]):.1%} mean 500 scores",
    )
    ax0.yaxis.set_major_formatter(mtick.PercentFormatter())
    ax0.axhline(0.272, color="gray", linestyle=":", label="Random")
    ax0.axhline(0.465, color="gray", linestyle=":", label="Maximize")
    ax1.set(ylim=(0, None), ylabel="Loss", xlabel="Games")

    if len(scores) > 200:
        ax0.plot(gaussian_filter1d(scores, 30, mode="nearest"))
        ax1.plot(gaussian_filter1d(losses, 30, mode="nearest"))
    plt.tight_layout(pad=0.2)
    fig.savefig("scores")
    plt.close()


game = Game()
score_predictor = ScorePredictor().to(DEVICE)
agent = Agent(game, score_predictor)

agent.play(n_games=100000, learn=True)
# agent.play(n_games=1000, learn=False, output_file="scores_test.dat")

torch.save(score_predictor.state_dict(), "model.pth")
