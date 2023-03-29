import multiprocessing as mp
import os
import time
from functools import partial
from typing import Optional, Union, Callable, Type

import gym
from stable_baselines3 import PPO
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor

from envs.definitions import reward_functions_fixed_horizon
from envs.envlist import *  # noqa
from ml.reward_wrapper import RewardWrapperConfig
from mods.on_policy_logging_mod import patch_on_policy_logging
from util import format_util

patch_on_policy_logging()  # insert custom logging into stable-baselines3


def train_a(
        log_queue: Optional[mp.Queue],
        a_reward_function: tuple[str, Union[Callable, RewardWrapperConfig, str]],
        time_steps: int,
        env_type: Union[str, gym.Env] = "DoorMutexEnv",
        algorithm: Type[BaseAlgorithm] = PPO,
) -> None:
    """Main function to run the training and testing of the agent.

    Args:
        log_queue: The queue to log to.
        a_reward_function: A tuple containing the textual description and
            the reward function to use for A, either as a reward function,
            a RewardWrapperConfig or a string='learned_reward' which self-configures.
        time_steps: Number of time steps for training.
        env_type: The environment to use.
        algorithm: The algorithm class (PPO, DQN, A2C) to use for the agent.
    """
    env_name = env_type if isinstance(env_type, str) else env_type.__name__
    env_name = format_util.convert_case(env_name, drop_count=1)
    env_class = eval(env_type) if isinstance(env_type, str) else env_type

    inhibition_desc, a_reward_function = a_reward_function

    alg_name = algorithm.__name__.lower()
    step_name = format_util.human_num(time_steps)

    agent_code = format_util.generate_random_string(8)

    model_name = f'a_{alg_name}_{env_name}_{inhibition_desc}_{step_name}_{agent_code}'
    os.makedirs(f"training/models", exist_ok=True)
    learning_log = 'training/runs/a_training_logs_new'

    # Select the reward function / reward wrapper
    if isinstance(a_reward_function, RewardWrapperConfig):
        reward_config = a_reward_function
    elif isinstance(a_reward_function, str):  # self-configure reward wrapper
        # assert a_reward_function in ['known_b_reward', 'env_inhib', 'learned_reward'], \
        #     "known_b_reward, env_inhib, learned_reward are supported"

        if a_reward_function == 'env_inhib':
            reward_config = RewardWrapperConfig(
                a_reward_function=env_class.PRE_SELFLESS_ENVIRONMENT_REWARD,
            )
        elif a_reward_function.startswith('env_inhib'):
            rew = int(a_reward_function.split('_')[-1])
            reward_config = RewardWrapperConfig(
                a_reward_function=partial(reward_functions_fixed_horizon.selfless_reward_time_penalty_reward_touch, touch_reward=rew),
            )
        elif a_reward_function == 'learned_reward':
            reward_config = RewardWrapperConfig(
                a_reward_function=env_class.DEFAULT_A_REWARD_FUNCTION,
                b_reward_model_path=env_class.PRE_LEARNED_REWARD_NET,
                b_reward_model_weight=1.0,
            )
        elif a_reward_function == 'learned_reward3':
            reward_config = RewardWrapperConfig(
                a_reward_function=env_class.DEFAULT_A_REWARD_FUNCTION,
                b_reward_model_path=env_class.PRE_LEARNED_REWARD_NET,
                b_reward_model_weight=3.0,
            )
        elif a_reward_function == 'known_b_reward':
            reward_config = RewardWrapperConfig(
                a_reward_function=reward_functions_fixed_horizon.selfless_reward_time_penalty_b_actions,
            )
        else:
            raise ValueError(f"Unsupported reward function: {a_reward_function}")
    elif isinstance(a_reward_function, Callable):  # use only reward function
        reward_config = RewardWrapperConfig(
            a_reward_function=a_reward_function,
        )
    else:
        raise ValueError(f"Unsupported reward function type: {type(a_reward_function)}")

    env = env_class(
        a_reward_function=reward_config,
        fixed_horizon=60,  # usually its 120
        title=model_name,
    )

    env = Monitor(env, info_keywords=env.info_keywords())

    time_start = time.time()

    model = train_agent(env, algorithm, time_steps, model_name, learning_log)

    total_time = time.time() - time_start

    mean_reward, std_reward = evaluate_policy(model, model.get_env(), n_eval_episodes=100)

    if log_queue is not None:
        log_queue.put(f'{model_name}: mean_reward:{mean_reward:.2f} +/- {std_reward:.2f} in {total_time:.2f} seconds')
    # run_agent(model, env)


def train_agent(
        env: Monitor,
        algorithm: Type[BaseAlgorithm],
        time_steps: int,
        model_name: str,
        learning_log: str,
) -> BaseAlgorithm:
    """Train the agent and save the model.

    Args:
        env: The environment to train the agent in.
        algorithm: The algorithm class (PPO, DQN, A2C) to use for the agent.
        time_steps: Number of time steps for training.
        model_name: Model name for saving the model.
        learning_log: Path for the learning log.

    Returns:
        The trained model.
    """
    model = algorithm(
        policy="MlpPolicy",
        env=env,
        learning_rate=0.0003,
        n_steps=2048,  # noqa
        batch_size=64,  # noqa
        n_epochs=10,  # noqa
        gamma=0.98,  # noqa
        verbose=0,
        # policy_kwargs=dict(net_arch=[64, 64]),
        tensorboard_log=learning_log
    )  # noqa

    # print(learning_log, model_name)

    if learning_log is not None:
        model.learn(total_timesteps=time_steps, tb_log_name=model_name)
    else:
        model.learn(total_timesteps=time_steps)

    if model_name is not None:
        model.save(f'training/models/{model_name}')

    return model


def run_agent(model: BaseAlgorithm, env: Monitor) -> None:
    """Test the trained agent in the environment.
    Args:
        model: The trained model.
        env: The environment to test the agent in.
    """
    obs, info = env.reset()
    survived_steps, reward_sum = 0, 0

    test_time_start = time.time()
    env.render()
    while time.time() - test_time_start < 60 * 10:
        action = model.predict(obs, deterministic=False)[0]

        obs, reward, term, trunc, info = env.step(action)
        env.render()
        reward_sum += reward

        grid_obs = obs[:25]
        grid_obs = grid_obs.reshape((5, 5))
        print(grid_obs, 2 in grid_obs)

        if term or trunc:
            env.reset()
            print(f"Survived {survived_steps} steps; reward: {reward_sum}")
            survived_steps, reward_sum = -1, 0

        survived_steps += 1

    env.close()


# def init():
#     env_class = SlowdownMutexEnv
#
#     reward_function = reward_functions_fixed_horizon.reward_time_penalty
#     time_steps = 100_000
#     train = True
#     eval_after_training = False
#     run_after_training = True
#     algorithm = PPO
#     train_a(reward_function, time_steps, train, eval_after_training, run_after_training, env_class, algorithm)
#
#
# if __name__ == '__main__':
#     init()

# def exp():
#     env = SpeedupMutexEnv(
#         a_reward_function=reward_functions_fixed_horizon.reward_time_penalty,
#         fixed_horizon=60,
#         title='fixed_board',
#     )
#
#     env = Monitor(env, info_keywords=env.info_keywords())
#
#     model = train_agent(env, PPO, 200000, None, None)
#     run_agent(model, env)
#
#
# if __name__ == '__main__':
#     exp()
