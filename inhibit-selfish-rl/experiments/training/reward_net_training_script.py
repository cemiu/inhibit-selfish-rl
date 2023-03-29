"""
This script learns B's reward model using IRL on environments and exports the networks
and stores performance evaluations to log files.

Train up reward models for all environments, 6 times per environment,
with 10k, 50k, and 100k preference comparisons.
"""
import itertools
import os
import multiprocessing as mp

import models
from experiments.training.irl_learn_b_reward_model import train_reward_network
from util.logging_util import MultiProcessLogger


def main():
    logger = MultiProcessLogger('logs/reward_net_learning_%s.log')

    queue = [logger.queue]

    runs_models = [
        models.B_DOOR_MUTEX,
        models.B_DOOR_SHARED,
        models.B_SLOWDOWN_MUTEX,
        models.B_SPEEDUP_MUTEX,
    ]

    # runs_comparisons = [
    #     10_000, 10_000,
    #     50_000, 50_000,
    #     100_000, 100_000,
    # ]

    runs_comparisons = [
        5_000, 5_000, 10_000, 10_000,
    ]

    training_runs = itertools.product(runs_models, runs_comparisons, queue)

    pool_count = os.cpu_count() // 2  # use half of the available cores
    pool_count = max(1, pool_count)   # but at least 1

    with mp.Pool(processes=pool_count) as pool:
        pool.starmap(train_reward_network, training_runs)

    logger.close()


if __name__ == '__main__':
    main()
