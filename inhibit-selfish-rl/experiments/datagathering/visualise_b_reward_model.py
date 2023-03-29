"""
This experiment is for analyzing learned reward functions for Agent B.

Boards are generated, placing B in every possible position on the board.
On these boards, all possible actions are taken.

If B's position has changed, the model is used to predict a reward, and adds it to a list for the resulting position.

Boards with all rewards can be compared to boards with no rewards, to see if the model has a preference
for collecting rewards.
"""
import random
from collections.abc import Callable
from functools import partial

import gym
import numpy as np
from gym.vector.utils import spaces
from matplotlib import pyplot as plt

from envs import board
from envs.board import BOARD_VALUES
from envs.definitions import predefined_boards, board_stripping
from envs.definitions.board_stripping import BoardStrippingConfig
from envs.definitions.observations import PartialBoardObservation
from envs.envlist.door_shared import DoorSharedEnv
from ml.reward_networks import LoadableRewardNet
from util import board_util
from util.matrix_utils import calculate_statistics

BOARD_OBSERVATION = PartialBoardObservation
# DEFAULT_GENERATOR = partial(predefined_boards.board_with_obstacles_big, random_rewards=False)

# values which make a position invalid
INVALID_TILES = {board.BOARD_VALUES['wall'], board.BOARD_VALUES['door']}

# values which should be removed from the board, if the agent occupies them
REMOVE_TILES = {board.BOARD_VALUES['player_b_resources'], board.BOARD_VALUES['shared_resources']}

_to_stacked_obs = lambda obs: np.stack([obs])

act_map = {'left': 0, 'right': 1, 'up': 2, 'down': 3}
act_lookup = {v: k for k, v in act_map.items()}


def get_valid_grid(
        b_pos,
        stripping_rule: BoardStrippingConfig,
        board_generator,
) -> tuple[tuple[int, int], tuple[int, int], Callable] | None:
    """Generate a board with B in the given position, and A in the default position."""

    '''The following approach does not work, because changing initial B position
    changes the reward generator's deterministic output.'''
    # pos_board_generator = partial(random_board_generator, b_pos=b_pos)

    # generate board using reward function
    start_grid, a_pos, _ = board_generator()
    gen_grid = start_grid()

    gen_grid = board_stripping.strip(gen_grid, strip_config=stripping_rule)

    b_x, b_y = b_pos

    if gen_grid[b_x][b_y] in INVALID_TILES:
        return None

    if gen_grid[b_x][b_y] in REMOVE_TILES:
        gen_grid[b_x][b_y] = 0

    return a_pos, b_pos, lambda: np.array(gen_grid, copy=True)

    # new_board = board.TwoPlayerBoard(
    #     board_dimension=10,
    #     board_state=gen_grid,
    #     player_a_pos=a_pos,
    #     player_b_pos=b_pos,
    # )


def get_all_valid_grids(stripping_rule, board_generator):
    for x in range(10):
        for y in range(10):
            result = get_valid_grid((x, y), stripping_rule, board_generator)
            if result is not None:
                yield result


def predictions_for_grid(actions, a_pos, b_pos, board_generator, observation, reward_network, predicted_rewards):
    for action in actions:
        if type(action) is str:
            action = act_map[action]

        new_board = board.TwoPlayerBoard(
            board_dimension=10,
            board_state=board_generator(),
            player_a_pos=a_pos,
            player_b_pos=b_pos,
        )
        new_board.move_player_a(random.randint(0, 3))  # make A perform a random action

        state_stack = _to_stacked_obs(observation.get_observation(new_board))

        new_board.move_player_b(action)  # make B perform the given action

        if new_board.player_b_pos == b_pos:
            continue  # B did not move, so discard this board, action pair

        action_stack = _to_stacked_obs(action)
        state_stack_new = _to_stacked_obs(observation.get_observation(new_board))
        done_stack = _to_stacked_obs(False)

        reward = reward_network.predict(state_stack, action_stack, state_stack_new, done_stack)[0]

        b_x, b_y = new_board.player_b_pos
        predicted_rewards[b_x][b_y].append(reward)


def get_averaged_reward_grid(
        env: type[gym.Env],
        observation: BOARD_OBSERVATION,
        board_generator,
        stripping_config: BoardStrippingConfig,
        samples: int,
        reward_network: LoadableRewardNet,
):
    for actions in [
        # ['left'],
        # ['right'],
        # ['up'],
        # ['down'],
        list(range(4))
    ]:
        # list of lists of lists. Each stores predicted reward values for a field (10x10x0)
        predicted_rewards = [[[] for _ in range(10)] for _ in range(10)]

        for _ in range(samples):
            for grid in get_all_valid_grids(stripping_config, board_generator):
                predictions_for_grid(actions, *grid, observation=observation, reward_network=reward_network,
                                     predicted_rewards=predicted_rewards)

        avg_grid = [[np.mean(rewards) for rewards in row] for row in predicted_rewards]
        avg_grid = np.array(avg_grid).reshape(10, 10)

        actions = [act_lookup[action] if type(action) is int else action for action in actions]

        return avg_grid, actions


def main():
    environment = DoorSharedEnv
    board_generator = environment.DEFAULT_BOARD
    stripping_config = board_stripping.ALL

    samples = 100

    observation = BOARD_OBSERVATION(10)

    reward_network = LoadableRewardNet.loadnet(
        path='..//training/models/reward_net_shared_100k.pt',
        observation_space=observation.get_observation_space(),
        action_space=spaces.Discrete(4),
    )

    grid, *_ = get_averaged_reward_grid(
        env=environment,
        observation=observation,
        board_generator=board_generator,
        stripping_config=stripping_config,
        samples=samples,
        reward_network=reward_network,
    )

    plt.imshow(grid)
    # action_list = ', '.join(actions)
    action_list = ''
    stripping_info = ''
    # stripping_info = ', without rewards' if STRIPPING else ', with rewards'

    g_mean, g_min, g_max = calculate_statistics(grid)

    plt.title(
        f'Average reward for {action_list}{stripping_info}\nmean={g_mean:.2f}, min={g_min:.2f}, max={g_max:.2f}')

    plt.show()


if __name__ == '__main__':
    main()
