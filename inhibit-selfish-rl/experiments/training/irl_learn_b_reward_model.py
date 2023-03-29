import logging
import logging
import os
import time
from typing import Optional

import numpy as np
from imitation.algorithms import preference_comparisons
from imitation.policies.base import NormalizeFeaturesExtractor
from imitation.util import logger as imit_logger
from imitation.util.networks import RunningNorm
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.on_policy_algorithm import OnPolicyAlgorithm

import models
from envs.definitions import reward_functions_fixed_horizon, board_stripping
from envs.definitions.secondary_agent_config import SecondaryAgentConfig
from envs.envlist import *  # noqa
from envs.irl_trajectory_yielder import TrajectoryYielder
from ml.reward_networks import LoadableRewardNet
from util import format_util


def train_reward_network(
        model: tuple[str, OnPolicyAlgorithm, str],  # (env_name, algorithm, model_path)
        comparisons: int,
        log_queue: Optional[logging.Logger] = None,
        normalise: bool = False,
):
    env_name, expert_alg, expert_model_path = model
    env_class = eval(env_name)

    model_prefix = f"reward_net_{format_util.convert_case(env_name, drop_count=1)}"

    os.makedirs("models", exist_ok=True)

    expert_agent_config = SecondaryAgentConfig(
        secondary_agent_mode=True,
        reward_function=reward_functions_fixed_horizon.reward_time_penalty,
        board_stripping_config=board_stripping.RAND_DOOR_AND_RAND_RESOURCES,
        epsilon_greedy=0.25,
    )

    # inits env and maps it to a vectorised env
    env = env_class(
        secondary_agent_config=expert_agent_config,
        fixed_horizon=80,
    )

    venv = make_vec_env(lambda: env)

    # sets normalisation of features, if needed
    expert_kwargs = dict()
    if normalise:
        expert_kwargs['policy_kwargs'] = dict(
            features_extractor_class=NormalizeFeaturesExtractor,
            features_extractor_kwargs=dict(normalize_class=RunningNorm),
        )

    expert = expert_alg.load(
        path=expert_model_path,
        env=venv,
        n_steps=2048 // venv.num_envs,
        batch_size=64,
        ent_coef=0.0,
        learning_rate=0.0003,
        n_epochs=10,
        verbose=1,
        **expert_kwargs,
    )

    _train_reward_model(
        log_queue=log_queue,
        model_prefix=model_prefix,
        comparison_steps=comparisons,
        venv=venv,
        expert=expert,
    )


def _train_reward_model(log_queue, model_prefix, comparison_steps, venv, expert):
    agent_code = format_util.generate_random_string(8)
    human_comparisons = format_util.human_num(comparison_steps)
    agent_name = f"{model_prefix}_{human_comparisons}_{agent_code}"

    log_to_tensorboard = True
    if log_to_tensorboard:
        imitation_logger = imit_logger.configure(
            folder=f"logs/reward_net_logs/{agent_name}",
            format_strs=['tensorboard']
        )  # tensorboard logging
    else:
        imitation_logger = imit_logger.configure(format_strs=[])  # no logging

    # ANN of reward network: mapping from (state, action, next_state) to reward
    reward_net = LoadableRewardNet(
        venv.observation_space,
        venv.action_space,
        use_state=True,
        use_action=True,
        use_next_state=True,
    )

    # inits fragmenter, gatherer, preference model and reward trainer
    fragmenter = preference_comparisons.RandomFragmenter(
        warning_threshold=0,
        rng=np.random.default_rng(),
    )
    gatherer = preference_comparisons.SyntheticGatherer(
        rng=np.random.default_rng(),
    )
    preference_model = preference_comparisons.PreferenceModel(reward_net)
    reward_trainer = preference_comparisons.BasicRewardTrainer(
        preference_model=preference_model,
        loss=preference_comparisons.CrossEntropyRewardLoss(),
        epochs=3,
        rng=np.random.default_rng(),
    )

    start_time = time.time()

    trajectory_generator = TrajectoryYielder(
        policy=expert,
        venv=venv,
    )

    pref_comparisons = preference_comparisons.PreferenceComparisons(
        trajectory_generator,
        reward_net,
        num_iterations=5,
        fragmenter=fragmenter,
        preference_gatherer=gatherer,
        reward_trainer=reward_trainer,
        fragment_length=10,
        transition_oversampling=1,
        initial_comparison_frac=0.1,
        allow_variable_horizon=False,
        initial_epoch_multiplier=1,
        custom_logger=imitation_logger,
    )

    _comp_train = pref_comparisons.train(
        total_timesteps=0,  # no need to train the agent, as we use a trajectory yielder, from an expert agent
        total_comparisons=comparison_steps,  # Number of preference comparisons to perform
    )

    reward_loss = _comp_train['reward_loss']
    reward_accuracy = _comp_train['reward_accuracy']

    total_time = time.time() - start_time

    reward_net.save(f'models/{agent_name}.pt')
    if log_queue is not None:
        log_queue.put(f"{agent_name}: loss={reward_loss:.4f}, accuracy={reward_accuracy:.4f} in {total_time:.2f}s")


if __name__ == '__main__':
    test_model = models.B_DOOR_MUTEX
    comps = 10_000

    train_reward_network(test_model, comps)
