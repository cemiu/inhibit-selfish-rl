import os

from stable_baselines3 import *

from envs.definitions import predefined_boards
from envs.definitions.observations import PartialBoardObservation
from envs.definitions.reward_functions_fixed_horizon import reward_time_penalty_finish_reward
from envs.envlist.door_mutex import DoorMutexEnv
from util import format_util

###################################
# Definitions for the environment #
###################################
BOARD_OBSERVATION = PartialBoardObservation
REWARD_FUNCTION = reward_time_penalty_finish_reward
DEFAULT_BOARD = lambda: predefined_boards.board_with_obstacles_big_mutex(random_rewards=False)
TIME_STEPS = 1_000_000
###################################

env = DoorMutexEnv(
    obs=BOARD_OBSERVATION,
    a_reward_function=REWARD_FUNCTION,
    default_board_function=DEFAULT_BOARD,
)

human_time_steps = format_util.human_num(TIME_STEPS)
learning_log = f'runs/{human_time_steps}_pos_reward'
model_dir = f'models/{human_time_steps}_pos_reward'
os.makedirs(model_dir, exist_ok=True)

if __name__ == '__main__':
    for name, alg in [
        ("ppo_big", PPO),
        # ("a2c", A2C),
        # ("dqn", DQN),
    ]:
        try:
            model_prefix = name
            os.makedirs(f"models/{model_prefix}", exist_ok=True)
            env = DoorMutexEnv(
                obs=BOARD_OBSERVATION,
                a_reward_function=REWARD_FUNCTION,
                default_board_function=DEFAULT_BOARD,
            )
            model = alg("MlpPolicy", env, verbose=1, tensorboard_log=learning_log)
            model.learn(total_timesteps=TIME_STEPS, tb_log_name=f"run_{name}")
            model.save(f'{model_dir}/model_{name}')

        except Exception as e:
            print(f"Error while training {name}: {e}")

# Model training
# model = PPO.load(f'models/{model_prefix}/model_ppo_{1}m', env=env, verbose=1, tensorboard_log=learning_log)
# model = DQN("MlpPolicy", env, verbose=1, tensorboard_log=learning_log)
# for i in [5, 10, 15, 20, 25, 30, 35, 40, 45, 50]:
#     model.learn(total_timesteps=5_000_000, tb_log_name=f"run_{i}m")
