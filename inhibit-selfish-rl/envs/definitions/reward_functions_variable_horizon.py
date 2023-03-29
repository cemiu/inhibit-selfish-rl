from envs.board import TwoPlayerBoard

"""Rewards function to be used for agents in variable horizon environments (non-fixed episode lengths).

For fixed horizon environments, see reward_functions_fixed_horizon.py.
"""


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
    return reward_default(board, is_a_player=is_a_player, **_) - 1


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
    return reward_default(board, is_a_player=is_a_player, **_) * (not board.door_exists) - 1


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
