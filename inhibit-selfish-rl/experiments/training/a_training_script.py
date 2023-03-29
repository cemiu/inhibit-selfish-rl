"""
This script trains the B agents on all environments and exports their models, and stores performance
evaluations to log files.

Trains up B agents on all environments, 3 times per environment, with 1M, 5M, and 10M time steps.

This is for training up expert policies which will be used as the secondary agents when training A
These agents will also be used to apply the IRL algorithm to, to derive the expert reward function.
"""
import itertools
import os
import multiprocessing as mp
from functools import partial

from envs.definitions import reward_functions_fixed_horizon
from experiments.training import train_agent_a
from util.logging_util import MultiProcessLogger


def train_a_agent(log_queue, env, time_steps, a_reward_function):
    train_agent_a.train_a(
        log_queue=log_queue,
        a_reward_function=a_reward_function,
        time_steps=time_steps,
        env_type=env,
    )


def main():
    logger = MultiProcessLogger('logs/a_training_logs_rerun_%s.log')

    queue = [logger.queue]

    env = [
        # 'DoorMutexEnv',
        # 'DoorSharedEnv',
        # 'SlowdownMutexEnv',
        'SpeedupMutexEnv',
    ]

    time_steps = [300_000] * 10

    """ Using the following inhibition methods:
        - 'none': No inhibition, use the standard reward function, expect selfish behaviour
        - 'learned-b-reward': Use the learned reward function from the B agents (IRL based approach)
        - 'env-inhib': Use the environment inhibition reward function (closed door blocks rewards, reward on touching)
        - 'known-b-reward-0.25': Use the known reward function with a multiplier of 0.25
    """
    inhibition_methods = [
        ('none', reward_functions_fixed_horizon.reward_time_penalty),
        ('learned-b-reward', 'learned_reward'),
        ('learned-b-reward-x3', 'learned_reward3'),
        ('env-inhib', 'env_inhib'),
        # ('env-inhib-35', 'env_inhib_35'),
        # ('env-inhib-40', 'env_inhib_40'),
        (
            'known-b-reward-0-25',
            partial(reward_functions_fixed_horizon.selfless_reward_time_penalty_b_actions, multiplier=0.25)
        ),
        (
            'known-b-reward-0-75',
            partial(reward_functions_fixed_horizon.selfless_reward_time_penalty_b_actions, multiplier=0.75)
        ),
    ]

    """ In addition to the above, we explore adding known reward with different multipliers to the
        shared reward environment (zero-sum game).
        - 'known-b-reward-1.00': Use the known reward function with a multiplier of 1
        - 'known-b-reward-2.00': Use the known reward function with a multiplier of 2
    """
    # training_runs_shared_extra = itertools.product(
    #     queue,
    #     ['DoorSharedEnv'],
    #     time_steps,
    #     [(
    #         'known-b-reward-1-00',
    #         partial(reward_functions_fixed_horizon.selfless_reward_time_penalty_b_actions, multiplier=1)
    #     ), (
    #         'known-b-reward-2-00',
    #         partial(reward_functions_fixed_horizon.selfless_reward_time_penalty_b_actions, multiplier=2)
    #     )]
    # )

    training_runs_shared_extra = []

    training_runs_main = itertools.product(queue, env, time_steps, inhibition_methods)
    training_runs = itertools.chain(training_runs_main, training_runs_shared_extra)

    pool_count = os.cpu_count() // 2  # use half of the available cores
    pool_count = 6  # override
    pool_count = max(1, pool_count)   # but at least 1

    with mp.Pool(processes=pool_count) as pool:
        pool.starmap(train_a_agent, training_runs)

    logger.close()


if __name__ == '__main__':
    main()
