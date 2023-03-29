from stable_baselines3 import *

import models
from envs.definitions import reward_functions_fixed_horizon
from envs.definitions.observations import PartialBoardObservation
from envs.envlist import *  # noqa

###################################
# Definitions for the environment #
###################################
BOARD_OBSERVATION = PartialBoardObservation
REWARD_FUNCTION = reward_functions_fixed_horizon.reward_time_penalty
# DEFAULT_BOARD = lambda: predefined_boards.board_with_obstacles_big_mutex(random_rewards=False)
###################################

# env_name, alg, model = models.TEST_DOOR_MUTEX_10x10_A_SELFISH
# env_name, alg, model = models.DOOR_MUTEX_10x10_A_DOOR_BLOCK_REWARD
# env_name, alg, model = models.DOOR_MUTEX_10x10_A_B_REWARD_0_25
# env_name, alg, model = models.DOOR_MUTEX_10x10_A_B_REWARD_1
# env_name, alg, model = models.DOOR_MUTEX_10x10_A_B_REWARD_5
# env_name, alg, model = models.TEST_DOOR_MUTEX_10x10_B_REWARD_LEARNED

# uncomment models above or replace path & env_name (DoorMutexEnv / DoorSharedEnv / SlowdownMutexEnv / SpeedupMutexEnv)
env_name, alg, model = "DoorMutexEnv", PPO, "../../#output/#models_output/a_ppo_door_mutex_env-inhib_300K_ZGGXWK7A.zip"

env = eval(env_name)(
    obs=BOARD_OBSERVATION,
    a_reward_function=REWARD_FUNCTION,
)

# Run the trained agent
if __name__ == '__main__':
    model = PPO.load(model, env=env)

    obs, info = env.reset()
    survived_steps, reward_sum = 0, 0

    for _ in range(100000):
        env.render()

        action = model.predict(obs, deterministic=False)[0]

        obs, reward, term, trunc, info = env.step(action)
        # print(np.array(obs[:25]).reshape(5, 5), '\n')

        reward_sum += reward

        if term or trunc:
            env.render()
            env.reset()
            print(f"Survived {survived_steps} steps; reward: {reward_sum}")
            survived_steps, reward_sum = -1, 0

        survived_steps += 1

    env.close()