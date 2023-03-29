import logging
import os
import time
from typing import Union, Optional

import gym
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
from envs.definitions import reward_functions_fixed_horizon, board_stripping
from envs.definitions.observations import PartialBoardObservation
from envs.definitions.secondary_agent_config import SecondaryAgentConfig
from envs.envlist import *  # noqa
from mods.on_policy_logging_mod import patch_on_policy_logging
from util import format_util

patch_on_policy_logging()


def train_b(
        log_queue: Optional[logging.Logger] = None,
        env_type: Union[str, gym.Env] = "DoorMutexEnv",
        time_steps: int = 1_000_000,
        horizon_length: Optional[int] = None,
):
    horizon = "variable" if horizon_length is None else "fixed"
    env_name = env_type if isinstance(env_type, str) else env_type.__name__
    env_name = format_util.convert_case(env_name, drop_count=1)

    # Automatically generate the model prefix
    model_prefix = f"b_ppo" \
                   f"_{horizon}_horizon" \
                   f"{f'_{horizon_length}' if horizon_length is not None else ''}" \
                   f"_{env_name}"

    # Create directories if not exists
    os.makedirs("models", exist_ok=True)
    learning_log = 'logs/b_training_logs'
    os.makedirs(learning_log, exist_ok=True)

    # Set the secondary agent config
    secondary_agent_config = SecondaryAgentConfig(
        secondary_agent_mode=True,
        reward_function=reward_functions_fixed_horizon.reward_time_penalty,
        board_stripping_config=board_stripping.RAND_DOOR_AND_RAND_RESOURCES,
        epsilon_greedy=0.25,
    )

    # Create the environment
    env_class = eval(env_type) if isinstance(env_type, str) else env_type
    env = env_class(
        # obs=PartialBoardObservation,
        secondary_agent_config=secondary_agent_config,
        fixed_horizon=horizon_length,
        title=model_prefix,
    )

    # Add a monitor to the environment
    env = Monitor(env, info_keywords=env.info_keywords())

    # Train and evaluate the agent
    _train_and_evaluate_agent(log_queue, model_prefix, time_steps, env, learning_log)


def _train_and_evaluate_agent(log_queue, model_prefix, time_steps, env, learning_log):
    # Train the agent
    human_time_steps = format_util.human_num(time_steps)
    model = PPO(
        policy="MlpPolicy",
        env=env,
        learning_rate=0.0003,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.98,
        verbose=0,
        tensorboard_log=learning_log,
        policy_kwargs=dict(net_arch=[64, 64]),
    )

    """
    model = DQN(
        "MlpPolicy",
        env,
        verbose=1,
        tensorboard_log=learning_log,
        learning_rate=0.0001,
        batch_size=512,
        gamma=0.9,  # discount factor
        exploration_final_eps=0.01,
        policy_kwargs=dict(net_arch=[64, 64, 64],),
    )
    """

    # Little concern for duplicates as sample space is 36^8 ~= 2.8e12
    agent_code = format_util.generate_random_string(8)
    agent_name = f"{model_prefix}_{human_time_steps}_{agent_code}"

    start_time = time.time()
    model.learn(total_timesteps=time_steps, tb_log_name=agent_name)

    total_time = time.time() - start_time

    # Evaluate the agent
    mean_reward, std_reward = evaluate_policy(model, model.get_env(), n_eval_episodes=100)

    model.save(f'models/{agent_name}')
    if log_queue is not None:
        log_queue.put(f'{agent_name}: mean_reward:{mean_reward:.2f} +/- {std_reward:.2f} in {total_time:.2f} seconds')


# if __name__ == '__main__':
#     logger = init_logger('logs/b_training_logs_%s.log')
#     main(env_type="SlowdownMutexEnv", time_steps=10_000_000)
