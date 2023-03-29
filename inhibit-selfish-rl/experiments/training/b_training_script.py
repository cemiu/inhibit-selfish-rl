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

from experiments.training import train_agent_b
from util.logging_util import MultiProcessLogger


def train_b_agent(log_queue, env, time_steps):
    train_agent_b.train_b(log_queue, env, time_steps)


def main():
    logger = MultiProcessLogger('logs/b_training_logs_%s.log')

    queue = [logger.queue]

    env = [
        'DoorMutexEnv',
        'DoorSharedEnv',
        'SlowdownMutexEnv',
        'SpeedupMutexEnv',
    ]
    time_steps = [1_000_000, 1_000_000, 5_000_000, 5_000_000, 10_000_000, 10_000_000]

    training_runs = itertools.product(queue, env, time_steps)
    # training_runs = map(wait_for_1_seconds, *zip(*training_runs))

    pool_count = os.cpu_count() // 2  # use half of the available cores
    pool_count = max(1, pool_count)   # but at least 1

    with mp.Pool(processes=pool_count) as pool:
        pool.starmap(train_b_agent, training_runs)

    logger.close()


if __name__ == '__main__':
    main()
