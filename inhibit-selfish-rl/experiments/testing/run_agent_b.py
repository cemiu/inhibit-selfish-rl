import logging
from typing import Union, Tuple, Type, Optional

import gym
from stable_baselines3 import PPO
from stable_baselines3.common.base_class import BaseAlgorithm

import models
from envs.definitions import reward_functions_fixed_horizon, board_stripping
from envs.definitions.observations import PartialBoardObservation
from envs.definitions.secondary_agent_config import SecondaryAgentConfig
from envs.envlist import *  # noqa
from util.format_util import ActionTable
from util.state_util import State


def run_agent(
        env_name: Union[str, Type[gym.Env]],
        alg_model: Tuple[BaseAlgorithm, str],
        fixed_horizon: Optional[int] = None,
        logger: Optional[logging.Logger] = None,
):
    env_class = eval(env_name) if isinstance(env_name, str) else env_name
    print(alg_model)
    alg, model_path = alg_model

    # Create the environment
    secondary_agent_config = SecondaryAgentConfig(
        secondary_agent_mode=True,
        reward_function=reward_functions_fixed_horizon.reward_time_penalty,
        board_stripping_config=board_stripping.RAND_DOOR_AND_RAND_RESOURCES,
        epsilon_greedy=0.0,
    )

    env = env_class(
        obs=PartialBoardObservation,
        a_reward_function=reward_functions_fixed_horizon.reward_time_penalty,
        fixed_horizon=fixed_horizon,
        secondary_agent_config=secondary_agent_config,
    )

    # Run the trained agent
    model = alg.load(model_path, env=env)
    _run_trained_agent(env, model, logger)


def _run_trained_agent(env, model, logger):
    state = State(env.reset())
    survived_steps, reward_sum = 0, 0
    actions = [0, 0, 0, 0]
    action_printer = ActionTable(immediate=True)

    env.render()
    for _ in range(100000):
        action = model.predict(state.obs, deterministic=False)[0]
        actions[action] += 1
        state.update(env.step(action))
        reward_sum += state.reward

        env.render()

        if state.done:
            action_printer.add_line(actions, 'closed' if env.board.door_exists else 'open')
            actions = [0, 0, 0, 0]
            env.reset()
            if logger is not None:
                logger.info(f"Horizon: {survived_steps} steps; reward: {reward_sum}")
            survived_steps, reward_sum = -1, 0

        survived_steps += 1

    env.close()


def main():
    # Preconfigured models to select from
    # envlist: DoorMutexEnv / DoorSharedEnv / SlowdownMutexEnv / SpeedupMutexEnv
    model_configs = [
        models.B_DOOR_MUTEX,
        models.B_DOOR_SHARED,
        models.B_SLOWDOWN_MUTEX,
        models.B_SPEEDUP_MUTEX,
        ('DoorMutexEnv', PPO, '$PATH_TO_B_MODEL'),
    ]

    # Select the desired model config
    env_name, *alg_model = model_configs[3]

    run_agent(
        env_name=env_name,
        alg_model=alg_model,
    )


if __name__ == '__main__':
    main()