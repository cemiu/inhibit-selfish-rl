from typing import Callable, Tuple

import numpy as np

from envs.board import BOARD_VALUES
from util.board_util import random_reward_generator

_E = BOARD_VALUES["empty"]
_W = BOARD_VALUES["wall"]
_D = BOARD_VALUES["door"]

DEFAULT_BOARD_FUNCTION = Callable[[], Tuple[Callable[[], np.ndarray], Tuple[int, int], Tuple[int, int]]]


def board_with_obstacles(random_rewards=True, rule='mutex'):
    """7x7 board with obstacles and walls"""
    _board_with_obstacles = np.array([
        [_E, _E, _E, _E, _E, _E, _E],
        [_E, _E, _E, _W, _W, _W, _E],
        [_E, _W, _E, _E, _E, _W, _E],
        [_E, _W, _W, _E, _E, _E, _E],
        [_E, _W, _E, _E, _E, _E, _E],
        [_E, _E, _E, _W, _E, _W, _W],
        [_W, _E, _E, _E, _E, _D, _E],
    ])

    _a_pos = (0, 0)
    _b_pos = (6, 6)

    return _define_rewards_environment(_board_with_obstacles, _a_pos, _b_pos, not random_rewards, rule)


def board_with_obstacles_big_mutex(random_rewards=True, rule='mutex', a_pos=(0, 0), b_pos=(9, 9)):
    """10x10 board with obstacles, walls, and red locked behind a door.

    This board is created to be used in a mutex reward setting."""
    _board_with_obstacles_big = np.array([
        [_E, _E, _E, _E, _E, _W, _W, _E, _E, _W],
        [_E, _E, _E, _E, _E, _E, _E, _E, _E, _W],
        [_E, _E, _W, _W, _W, _E, _E, _W, _E, _E],
        [_E, _E, _E, _W, _W, _E, _E, _W, _E, _E],
        [_E, _W, _E, _E, _E, _E, _E, _W, _W, _E],
        [_E, _W, _E, _E, _E, _W, _E, _E, _E, _E],
        [_E, _E, _E, _W, _W, _W, _E, _W, _W, _W],
        [_E, _W, _E, _E, _E, _W, _E, _E, _E, _E],
        [_E, _W, _W, _E, _E, _E, _E, _E, _W, _W],
        [_E, _E, _E, _E, _W, _E, _E, _E, _D, _E],
    ])

    _a_pos = a_pos
    _b_pos = b_pos

    return _define_rewards_environment(_board_with_obstacles_big, _a_pos, _b_pos, not random_rewards, rule)


def board_with_obstacles_shared(random_rewards=True, rule='shared', a_pos=(0, 5), b_pos=(5, 0)):
    """10x10 board with obstacles, walls, and red locked behind a door.

    This board is created to be used in a shared reward setting."""
    _board_with_obstacles_big = np.array([
        [_E, _W, _E, _E, _E, _E, _E, _E, _E, _E],
        [_E, _E, _E, _W, _E, _W, _E, _E, _E, _W],
        [_E, _W, _W, _W, _E, _W, _W, _W, _E, _W],
        [_E, _E, _E, _E, _E, _E, _E, _E, _E, _E],
        [_W, _W, _E, _E, _W, _W, _E, _E, _W, _E],
        [_E, _D, _E, _E, _W, _W, _E, _W, _W, _W],
        [_W, _W, _W, _E, _E, _E, _E, _E, _E, _W],
        [_E, _E, _W, _E, _E, _W, _E, _E, _E, _W],
        [_E, _E, _E, _E, _W, _W, _W, _E, _E, _E],
        [_E, _E, _W, _E, _E, _E, _E, _E, _E, _E],
    ])

    _a_pos = a_pos
    _b_pos = b_pos

    return _define_rewards_environment(_board_with_obstacles_big, _a_pos, _b_pos, not random_rewards, rule)


def board_with_obstacles_narrow_pass(random_rewards=True, rule='mutex', a_pos=(4, 3), b_pos=(4, 5)):
    """10x10 board with obstacles

    This board is created to work in a slow-down setting (touching red reduces red's speed).

    For this, it consists of many narrow passages."""
    _board_with_obstacles_big = np.array([

        [_E, _W, _E, _E, _E, _W, _E, _E, _E, _E],  # clone
        [_E, _W, _E, _W, _W, _W, _W, _W, _W, _E],
        [_E, _E, _E, _E, _E, _E, _E, _E, _E, _E],
        [_E, _E, _W, _W, _W, _W, _W, _W, _W, _E],
        [_E, _E, _E, _E, _E, _E, _E, _E, _E, _E],
        [_E, _W, _E, _E, _W, _W, _W, _W, _W, _E],
        [_E, _W, _E, _W, _E, _E, _E, _E, _E, _E],
        [_E, _W, _E, _E, _W, _E, _E, _W, _E, _E],
        [_E, _W, _E, _E, _E, _E, _W, _W, _E, _E],
        [_E, _E, _E, _W, _W, _W, _W, _E, _E, _E],

        # [_E, _W, _E, _E, _E, _W, _E, _E, _E, _E],  # original
        # [_E, _W, _E, _W, _W, _W, _W, _W, _W, _E],
        # [_E, _E, _E, _E, _E, _E, _E, _E, _E, _E],
        # [_E, _E, _W, _W, _W, _W, _W, _W, _W, _E],
        # [_E, _W, _E, _E, _E, _E, _E, _E, _E, _W],
        # [_E, _W, _E, _E, _W, _W, _W, _W, _W, _E],
        # [_E, _E, _E, _W, _E, _E, _E, _E, _E, _E],
        # [_E, _W, _E, _E, _W, _E, _E, _W, _E, _E],
        # [_E, _W, _E, _E, _E, _E, _W, _W, _E, _E],
        # [_E, _E, _E, _W, _W, _W, _W, _E, _E, _E],
    ])

    _a_pos = a_pos
    _b_pos = b_pos

    return _define_rewards_environment(_board_with_obstacles_big, _a_pos, _b_pos, not random_rewards, rule)


def board_with_obstacles_wide_pass(random_rewards=True, rule='mutex', a_pos=(7, 3), b_pos=(0, 9)):
    """10x10 board with walls

    This board is created to work in a speed-up setting (touching red increases red's speed).

    For this, it consists of many wide passages."""
    _board_with_obstacles_big = np.array([
        [_E, _E, _E, _E, _W, _W, _E, _E, _W, _E],
        [_E, _E, _E, _E, _E, _W, _E, _E, _E, _E],
        [_E, _E, _W, _E, _E, _E, _E, _E, _E, _E],
        [_E, _E, _W, _E, _E, _E, _E, _W, _W, _E],
        [_E, _E, _E, _E, _E, _E, _E, _E, _W, _E],
        [_W, _E, _E, _W, _W, _W, _E, _E, _W, _E],
        [_W, _E, _E, _E, _E, _E, _E, _E, _E, _E],
        [_E, _E, _W, _E, _E, _E, _E, _W, _E, _E],
        [_E, _E, _W, _W, _W, _E, _E, _W, _E, _E],
        [_E, _E, _E, _E, _E, _E, _E, _E, _E, _E],
    ])

    _a_pos = a_pos
    _b_pos = b_pos

    return _define_rewards_environment(_board_with_obstacles_big, _a_pos, _b_pos, not random_rewards, rule)


def _define_rewards_environment(board, a_pos, b_pos, deterministic=False, rule='mutex'):
    assert rule in ['mutex', 'shared']
    if rule == 'mutex':
        frac_a = 0.3
        frac_b = 0.3
        frac_shared = 0
    else:
        frac_a = 0
        frac_b = 0
        frac_shared = 0.5

    new_board = random_reward_generator(
        arr=board,
        reward_frac_a=frac_a,
        reward_frac_b=frac_b,
        reward_frac_shared=frac_shared,
        exclude_pos_a=a_pos,
        exclude_pos_b=b_pos,
        deterministic=deterministic
    )

    return (
        lambda: np.array(new_board, copy=True),
        a_pos,
        b_pos
    )
