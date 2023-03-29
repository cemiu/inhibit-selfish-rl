""" This script is used in an experiment

The game is played until agent A is adjacent to the door.
Then, the agent is asked to choose a direction 100k times, given the same observation.

This can be used to evaluate the willingness of the agent to open the door and let the other agent through.

Results:
    - Selfish agent (500k steps):
        - left:     99.8082%
        - right:     0.0101% (open door)
        - up:        0.1817%
        - down:      0.0000%
    - Door blocks reward agent (500k steps):
        - left:      0.0270%
        - right:	99.9180% (open door)
        - up:        0.0550%
        - down:      0.0000%
    - TODO: more results
"""

from envs.definitions import predefined_boards
from envs.definitions.observations import PartialBoardObservation
from envs.definitions.reward_functions_fixed_horizon import reward_time_penalty_finish_reward
from envs.envlist.door_mutex import DoorMutexEnv
import models

###################################
# Definitions for the environment #
###################################
BOARD_OBSERVATION = PartialBoardObservation
REWARD_FUNCTION = reward_time_penalty_finish_reward
DEFAULT_BOARD = lambda: predefined_boards.board_with_obstacles_big_mutex(random_rewards=False)
###################################

# alg = PPO
# model = model

# env_name, alg, model = models.DOOR_MUTEX_10x10_A_SELFISH
env_name, alg, model = models.DOOR_MUTEX_10x10_A_DOOR_BLOCK_REWARD

env = DoorMutexEnv(
    obs=BOARD_OBSERVATION,
    a_reward_function=REWARD_FUNCTION,
    default_board_function=DEFAULT_BOARD,
)

env.set_secondary_agent_model()  # loads the default model

# Run the trained agent
if __name__ == '__main__':
    model = alg.load(model, env=env)

    obs, info = env.reset()

    for _ in range(10000):
        action = model.predict(obs, deterministic=False)[0]

        obs, reward, term, trunc, info = env.step(action)

        env.render()
        if obs[13] == 6:
            bins = [0, 0, 0, 0]
            bin_names = ['left', 'right', 'up', 'down']
            for i in range(100000):
                action = model.predict(obs, deterministic=False)[0]
                bins[action] += 1

            bin_prob = [b / sum(bins) for b in bins]
            for direction, count, prob in zip(bin_names, bins, bin_prob):
                print(f'{direction}:\n\tSamples: {count}\n\tProbability: {prob:.4%}')

            print()

            for direction, count, prob in zip(bin_names, bins, bin_prob):
                print(f'- {direction}:\t{prob:.4%}')

            exit()

        if term or trunc:  # if it never reaches the door, try again
            env.reset()

    env.close()
