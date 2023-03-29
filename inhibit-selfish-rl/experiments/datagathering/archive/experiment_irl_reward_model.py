import itertools
import logging
import time
from functools import partial
from typing import Union, Type

import numpy as np
from imitation.algorithms import preference_comparisons
from imitation.policies.base import NormalizeFeaturesExtractor
from imitation.rewards.reward_wrapper import RewardVecEnvWrapper
from imitation.util.networks import RunningNorm
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.policies import ActorCriticPolicy

from envs.definitions import reward_functions_fixed_horizon, predefined_boards, board_stripping
from envs.definitions.observations import PartialBoardObservation
from envs.definitions.secondary_agent_config import SecondaryAgentConfig
from envs.envlist.door_mutex import DoorMutexEnv
from envs.envlist.door_shared import DoorSharedEnv
from envs.irl_trajectory_yielder import TrajectoryYielder
from ml.reward_networks import LoadableRewardNet
import models
from util import board_util
from util.format_util import round_elements

BOARD_OBSERVATION = PartialBoardObservation
REWARD_FUNCTION = reward_functions_fixed_horizon.reward_time_penalty
# DEFAULT_BOARD = lambda: predefined_boards.board_with_obstacles_big(random_rewards=False)
DEFAULT_BOARD = lambda: predefined_boards.board_with_obstacles_shared(random_rewards=False)

SECONDARY_AGENT_CONFIG = SecondaryAgentConfig(
    secondary_agent_mode=True,
    reward_function=reward_functions_fixed_horizon.reward_time_penalty,
    board_stripping_config=board_stripping.RAND_DOOR_AND_RAND_RESOURCES,
    epsilon_greedy=0.25,
)


def run_experiment(
        steps: int,
        policy: Union[str, Type[ActorCriticPolicy]] = "MlpPolicy",
        normalise: bool = True
):
    env = DoorSharedEnv(
        obs=BOARD_OBSERVATION,
        a_reward_function=REWARD_FUNCTION,
        default_board_function=DEFAULT_BOARD,
        secondary_agent_config=SECONDARY_AGENT_CONFIG,
        fixed_horizon=80,
    )

    venv = make_vec_env(lambda: env)

    reward_net = LoadableRewardNet(
        venv.observation_space,
        venv.action_space,
        # normalize_input_layer=RunningNorm,
        use_state=True,
        use_action=True,
        use_next_state=True,
    )

    fragmenter = preference_comparisons.RandomFragmenter(
        warning_threshold=0,
        rng=np.random.default_rng(),
    )
    gatherer = preference_comparisons.SyntheticGatherer(rng=np.random.default_rng())
    preference_model = preference_comparisons.PreferenceModel(reward_net)
    reward_trainer = preference_comparisons.BasicRewardTrainer(
        preference_model=preference_model,
        loss=preference_comparisons.CrossEntropyRewardLoss(),
        epochs=3,
        rng=np.random.default_rng(),
    )

    kwargs = dict()
    if normalise:
        kwargs['policy_kwargs'] = dict(
            features_extractor_class=NormalizeFeaturesExtractor,
            features_extractor_kwargs=dict(normalize_class=RunningNorm),
        )

    # agent_alg, agent_model = models.DOOR_MUTEX_10x10_B_1M
    agent_alg, agent_model = models.DOOR_SHARED_10x10_B_10M

    agent = agent_alg.load(
        path=agent_model,
        env=venv,
        n_steps=2048 // venv.num_envs,
        batch_size=64,
        ent_coef=0.0,
        learning_rate=0.0003,
        n_epochs=10,
        verbose=1,
        **kwargs,
    )

    expert_reward, expert_rew_std = evaluate_policy(agent, venv, n_eval_episodes=100)
    venv.reset()

    print(f'Expert agent performance: {expert_reward=} +/- {expert_rew_std:.2f}')

    _str_time = time.time()

    trajectory_generator = TrajectoryYielder(
        policy=agent,
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
    )

    _comp_train = pref_comparisons.train(
        total_timesteps=0,  # For good performance this should be 1_000_000
        total_comparisons=100_000,  # For good performance this should be 5_000
    )

    print(_comp_train)  # {"reward_loss": reward_loss, "reward_accuracy": reward_accuracy}

    reward_net.save('models/reward_net_shared_100k.pt')

    learned_reward_venv = RewardVecEnvWrapper(venv, reward_net.predict)

    learner = PPO(
        policy=policy,
        env=learned_reward_venv,
        batch_size=64,
        ent_coef=0.0,
        learning_rate=0.0003,
        n_epochs=10,
        n_steps=64,
        verbose=1,
    )
    learner.learn(steps, tb_log_name=str(steps))  # Note: set to 100000 to train a proficient expert

    _end_time = time.time()
    _total_time = _end_time - _str_time

    learned_reward, learned_std = evaluate_policy(learner.policy, venv, 100)

    if isinstance(policy, type):
        policy = policy.__name__

    result = steps, expert_reward, expert_rew_std, learned_reward, learned_std, _total_time, _comp_train['reward_loss'], _comp_train['reward_accuracy'], policy

    r_str = ','.join(map(str, round_elements(result)))

    logging.info(r_str)
    print(r_str)

    return result


if __name__ == '__main__':
    start_time = time.time()

    # runs_steps = [0, 300000]
    # runs_policies = [FeedForward32Policy, "MlpPolicy"]
    # runs_normalisation = [True, False]

    runs_steps = [0]
    runs_policies = ["MlpPolicy"]
    runs_normalisation = [False]

    runs_inputs = (runs_steps, runs_policies, runs_normalisation)

    # produce a map of all experiment combinations
    experiments = itertools.product(*runs_inputs)
    experiments = map(run_experiment, *zip(*experiments))

    result_list = []

    for r in experiments:
        r = ','.join(map(str, r))
        result_list.append(r)

    print("finished all")

    end_time = time.time()
    print(f'Total time: {end_time - start_time:.2f} seconds')

    for r in result_list:
        print(r)
