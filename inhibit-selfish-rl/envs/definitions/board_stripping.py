"""
Within this file, multiple board stripping rules are defined.
In addition, the board stripping function and a configuration class are defined.

Board stripping rules are the input to board_stripping_function.
They are supposed to act on a board, and return a new board where some tiles have been removed.
"""

from typing import Tuple, Union

import numpy as np

from envs.board import BOARD_VALUES

from typing import Tuple

from util.random_util import ProbabilityRange


class BoardStrippingConfig:
    """This class functions a configuration for the board stripping function.

    It takes a list of tuples as input, where each tuple contains a probability and a board value."""
    def __init__(
            self,
            *config: Tuple[Union[float, Tuple[float, float], ProbabilityRange], int]
    ):
        """
        Example usage:
            stripping_config = _BoardStrippingConfig(
                (1, BOARD_VALUES["door"]),
                (0.5, BOARD_VALUES["player_a_resources"]),
                ((0, 0.7), BOARD_VALUES["shared_resources"]),
            )



        :param config:
            A variable number of tuples containing a probability (or a tuple of lower and upper bounds
                or a ProbabilityRange object) and a board value.
        """
        processed_config = []
        for prob, value in config:
            if isinstance(prob, (int, float)):
                prob = ProbabilityRange(prob, prob)
            elif isinstance(prob, (int, tuple)):
                prob = ProbabilityRange(*prob)
            elif not isinstance(prob, ProbabilityRange):
                raise ValueError("Invalid probability type. Must be float, tuple, or ProbabilityRange.")

            processed_config.append((prob, value))

        self._config = processed_config

    @property
    def config(self):
        return self._config

    def __repr__(self):
        return f"BoardStrippingConfig({self._config})"

    def __iter__(self):
        return iter(self._config)


"""This is to be used to open the door and remove half of the resources collectable by player A
To be used when training policy for player B. Foreign resources are removes, to make the agent
adapt to a variable amount of positions of resources it will observe, rather than overfitting
to a specific resource configuration."""
DOOR_AND_FOREIGN_RESOURCES = BoardStrippingConfig(
    (1, BOARD_VALUES["door"]),
    (0.5, BOARD_VALUES["player_a_resources"]),
    (0.5, BOARD_VALUES["shared_resources"]),
)

DOOR_AND_RAND_RESOURCES = BoardStrippingConfig(
    (1, BOARD_VALUES["door"]),
    (0.5, BOARD_VALUES["player_a_resources"]),
    (0.5, BOARD_VALUES["shared_resources"]),
    (0.5, BOARD_VALUES["player_b_resources"]),
)

RAND_DOOR_AND_RAND_RESOURCES = BoardStrippingConfig(
    (0.8, BOARD_VALUES["door"]),
    (0.5, BOARD_VALUES["player_a_resources"]),
    ((0, 1), BOARD_VALUES["shared_resources"]),
    ((0, 1), BOARD_VALUES["player_b_resources"]),
)

RAND_RESOURCES = BoardStrippingConfig(
    (0.5, BOARD_VALUES["player_a_resources"]),
    ((0, 1), BOARD_VALUES["shared_resources"]),
    ((0, 1), BOARD_VALUES["player_b_resources"]),
)

NO_STRIPPING = BoardStrippingConfig()

DOOR = BoardStrippingConfig(
    (1, BOARD_VALUES["door"]),
)

ALL = BoardStrippingConfig(
    (1, BOARD_VALUES["door"]),
    (1, BOARD_VALUES["player_a_resources"]),
    (1, BOARD_VALUES["shared_resources"]),
    (1, BOARD_VALUES["player_b_resources"]),
)


def strip(
        board: np.ndarray,
        strip_config: BoardStrippingConfig,
) -> np.ndarray:
    """Takes a board state and strips it of select values, as specified probabilities.

    Config to be supplied as a BoardStrippingConfig object.

    Args:
        :param board: The board state to strip.
        :param strip_config: The configuration of the stripping function.
    """
    assert strip_config, 'The strip config must not be empty'

    board = np.copy(board)

    for probability_range, strip_value in strip_config:
        strip_prob = probability_range.sample()
        strippable = np.where(board == strip_value)
        strippable = list(zip(strippable[0], strippable[1]))

        strip_threshold = np.random.rand(len(strippable))
        strip_threshold = np.where(strip_threshold <= strip_prob, 1, 0)
        strip_threshold = np.where(strip_threshold == 1)[0]

        for i in strip_threshold:
            board[strippable[i]] = 0

    return board


# if __name__ == '__main__':
#     a = np.array([[0, 0, 1, 1, 2, 2, 3, 3],
#                   [0, 0, 1, 1, 2, 2, 3, 3],
#                   [4, 4, 5, 5, 6, 6, 7, 7],
#                   [4, 4, 5, 5, 6, 6, 7, 7],
#                   [8, 8, 9, 9, 10, 10, 11, 11],
#                   [8, 8, 9, 9, 10, 10, 11, 11],
#                   [12, 12, 13, 13, 14, 14, 15, 15],
#                   [12, 12, 13, 13, 14, 14, 15, 15]])
#     stripping_config = BoardStrippingConfig(
#         (1, 10),
#         (1, 15),
#         (0.5, 1)
#     )
#
#     print(board_stripping_function(a, stripping_config))
