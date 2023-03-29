import random
from collections.abc import Callable

import numpy as np
from numpy import ndarray

from envs.board import BOARD_VALUES
from envs.definitions.board_stripping import BoardStrippingConfig


def random_board_generator(
        board_dimension: int,
        reward_frac_a: float = 0.05,
        reward_frac_b: float = 0.05,
        reward_frac_shared: float = 0.05,
        deterministic: bool = False
) -> tuple[Callable[[], ndarray], tuple[int, int], tuple[int, int]]:
    """Generates a random board state.

    Absolute values take precedence over fractions in all cases.

    Can specify a seed to ensure reproducibility.

    Args:
        :param board_dimension: The dimension of the board.
        :param reward_frac_a: The fraction of the board that will be filled with rewards for player A.
        :param reward_frac_b: The fraction of the board that will be filled with rewards for player B.
        :param reward_frac_shared: The fraction of the board that will be filled with rewards for both players.
        :param deterministic: Whether to use a deterministic random number generator.
    """
    _random = random if deterministic is None else random.Random(deterministic)
    arr = np.zeros((board_dimension, board_dimension), dtype=np.int8)
    a_pos = (_random.randint(0, board_dimension - 1), _random.randint(0, board_dimension - 1))
    b_pos = a_pos
    while a_pos == b_pos:  # make sure a's and b's starting points are distinct
        b_pos = (_random.randint(0, board_dimension - 1), _random.randint(0, board_dimension - 1))
    # arr = random_wall_generator(arr, wall_frac, wall_abs, exclude_pos_a, exclude_pos_b)
    arr = random_reward_generator(
        arr,
        reward_frac_a=reward_frac_a,
        reward_frac_b=reward_frac_b,
        reward_frac_shared=reward_frac_shared,
        exclude_pos_a=a_pos,
        exclude_pos_b=b_pos,
        deterministic=deterministic
    )

    return lambda: np.copy(arr), a_pos, b_pos


def random_reward_generator(
        arr: np.ndarray,
        reward_frac_a: float = 0.05,
        reward_frac_b: float = 0.05,
        reward_frac_shared: float = 0.05,
        reward_abs_a: int = None,
        reward_abs_b: int = None,
        reward_abs_shared: int = None,
        exclude_pos_a: tuple[int, int] = None,
        exclude_pos_b: tuple[int, int] = None,
        deterministic: bool = False
) -> np.ndarray:
    """Takes a board state and populates it with randomly placed resources.

    Absolute values take precedence over fractions in all cases.

    Args:
        :param arr: The board state to populate.
        :param reward_frac_a: The fraction of resources to place for player A.
        :param reward_frac_b: The fraction of resources to place for player B.
        :param reward_frac_shared: The fraction of resources to place for both players.
        :param reward_abs_a: The absolute number of resources to place for player A.
        :param reward_abs_b: The absolute number of resources to place for player B.
        :param reward_abs_shared: The absolute number of resources to place for both players.S
        :param exclude_pos_a: The position of player A to exclude from the resource placement.
        :param exclude_pos_b: The position of player B to exclude from the resource placement.
        :param deterministic: Whether to use a deterministic random number generator.
    """
    assert reward_frac_a + reward_frac_b + reward_frac_shared <= 1.0, \
        'The sum of the fractions must be less than or equal to 1.0'

    _random = random if not deterministic else random.Random(0)

    arr = np.copy(arr)

    populatable = np.where(arr == 0)
    populatable = set(zip(populatable[0], populatable[1]))

    populatable.discard(exclude_pos_a)
    populatable.discard(exclude_pos_b)

    a_resources, b_resources, shared_resources = reward_abs_a, reward_abs_b, reward_abs_shared
    if reward_abs_a is None:
        a_resources = int(len(populatable) * reward_frac_a)
    if reward_abs_b is None:
        b_resources = int(len(populatable) * reward_frac_b)
    if reward_abs_shared is None:
        shared_resources = int(len(populatable) * reward_frac_shared)

    a_resources = _random.sample(tuple(populatable), a_resources)
    populatable.difference_update(a_resources)

    b_resources = _random.sample(tuple(populatable), b_resources)
    populatable.difference_update(b_resources)

    shared_resources = _random.sample(tuple(populatable), shared_resources)
    populatable.difference_update(shared_resources)

    a_resources = {pos: BOARD_VALUES['player_a_resources'] for pos in a_resources}
    b_resources = {pos: BOARD_VALUES['player_b_resources'] for pos in b_resources}
    shared_resources = {pos: BOARD_VALUES['shared_resources'] for pos in shared_resources}

    resources = {**a_resources, **b_resources, **shared_resources}

    for pos, value in resources.items():
        arr[pos] = value

    return arr


# if __name__ == '__main__':
#     # a = np.array([[0, 0, 1, 1, 2, 2, 3, 3],
#     #               [0, 0, 1, 1, 2, 2, 3, 3],
#     #               [4, 4, 5, 5, 6, 6, 7, 7],
#     #               [4, 4, 5, 5, 6, 6, 7, 7],
#     #               [8, 8, 9, 9, 10, 10, 11, 11],
#     #               [8, 8, 9, 9, 10, 10, 11, 11],
#     #               [12, 12, 13, 13, 14, 14, 15, 15],
#     #               [12, 12, 13, 13, 14, 14, 15, 15]])
#     # stripping_config = BoardStrippingConfig(
#     #     (1, 10),
#     #     (1, 15),
#     #     (0.5, 1)
#     # )
#     #
#     # print(board_stripping_function(a, stripping_config))
#     # arr = np.zeros((10, 10), dtype=np.int8)
#     # arr = random_reward_generator(arr, exclude_pos_a=(0, 0), exclude_pos_b=(9, 9))
#     # print(arr)
