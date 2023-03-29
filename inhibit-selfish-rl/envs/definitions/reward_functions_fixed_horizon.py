from envs.board import TwoPlayerBoard

"""Rewards function to be used in fixed horizon environments.

Fixed horizon environments do not enter the terminal state until the maximum number of time steps is reached.

Thus, the time penalty should not be applied beyond the point that would have been the terminal state
in an equivalent variable horizon environment.
"""


def _is_collecting_state(
        board: TwoPlayerBoard,
        is_a_player: bool = True
) -> float:
    """Returns True if the agent has more resources to collect.

    Once there aren't any left, applying the time penalty becomes unnecessary."""
    if is_a_player:
        return board.are_a_resources_left
    else:
        return board.are_b_resources_left


def reward_default(
        board: TwoPlayerBoard,
        is_a_player: bool = True,
        **_
) -> float:
    """Default reward function. Simply the sum of resources collected by each player,
    each weighted with 20 reward.
    """
    if is_a_player:
        return board.has_a_collected_resource * 20
    else:
        return board.has_b_collected_resource * 20


def reward_time_penalty(
        board: TwoPlayerBoard,
        is_a_player: bool = True,
        **_
) -> float:
    """Reward function that penalizes time.

    Subtracts 1 reward on each time step."""
    return reward_default(board, is_a_player=is_a_player, **_) \
        - 1 * _is_collecting_state(board, is_a_player=is_a_player)


def reward_time_penalty_finish_reward(
        board: TwoPlayerBoard,
        is_a_player: bool = True,
        **_
) -> float:
    """Reward function that penalizes time (-1) and rewards finishing the game.

    Subtracts 1 reward for each time step over the course of the game.
    Adds 100 reward if the game is finished."""
    if is_a_player:
        return reward_time_penalty(board, is_a_player=is_a_player, **_) + 40 * (not board.are_a_resources_left)
    else:
        return reward_time_penalty(board, is_a_player=is_a_player, **_) + 40 * (not board.are_b_resources_left)


def selfless_reward_time_penalty_door_block_reward(
        board: TwoPlayerBoard,
        is_a_player: bool = True,
        **_
) -> float:
    """Reward function that penalizes time (-1) and blocks positive rewards while the door is locked.

    Subtracts 1 reward for each time step over the course of the game.
    """
    return reward_default(board, is_a_player=is_a_player, **_) * (not board.door_exists) \
        - 1 * _is_collecting_state(board, is_a_player=is_a_player)


def selfless_reward_time_penalty_door_reward(
        board: TwoPlayerBoard,
        is_a_player: bool = True,
        **_
) -> float:
    """Reward function that penalizes time (-1) and rewards opening the door.

    Subtracts 1 reward for each time step over the course of the game.
    Adds 50 reward if the door is opened.
    """
    if is_a_player:
        return reward_time_penalty(board, is_a_player=is_a_player, **_) + 10 * board.has_a_opened_door
    else:
        raise NotImplementedError('Player A specific reward function')


def selfless_reward_time_penalty_reward_touch(
        board: TwoPlayerBoard,
        is_a_player: bool = True,
        touch_reward: float = 40,
        **_
):
    """Reward function that penalizes time (-1) and rewards touching the other player."""
    assert is_a_player, 'Player A specific reward function'
    return reward_time_penalty(board, is_a_player=is_a_player, **_) + touch_reward * board.a_touches_b
    # return reward_time_penalty(board, is_a_player=is_a_player, **_) + 100000 * board.a_touches_b


def selfless_reward_time_penalty_punish_touch(
        board: TwoPlayerBoard,
        is_a_player: bool = True,
        **_
):
    """Reward function that penalizes time (-1) and punishes touching the other player."""
    assert is_a_player, 'Player A specific reward function'
    return reward_time_penalty(board, is_a_player=is_a_player, **_) - 40 * board.a_touches_b


def selfless_reward_time_penalty_b_actions(
        board: TwoPlayerBoard,
        is_a_player: bool = True,
        multiplier: float = 0.25,
        **_
) -> float:
    """Reward function that penalizes time (-1) and rewards Player B's actions with a multiplier."""
    if is_a_player:
        return reward_time_penalty(board, is_a_player=is_a_player, **_) \
                  + reward_time_penalty(board, is_a_player=not is_a_player, **_) * multiplier
    else:
        raise NotImplementedError('Player A specific reward function')
