import os

from stable_baselines3 import *

_BASE_PATH = os.path.dirname(os.path.abspath(__file__))


def _t(env, algo, env_dir, name):
    """For testing models"""
    return env, algo, os.path.join(_BASE_PATH, f'test/{env_dir}/{name}')


def _m(env, algo, name):
    """For models"""
    return env, algo, os.path.join(_BASE_PATH, f'm/{name}')


"""Below are the models and reward models that are used for the experiments in the paper.

The B agents were trained on each environment for 1M, 5M, and 10M time steps (twice), all with a variable horizon,
and the best performing model was selected for each environment as the expert policy.

Note: the best performing model is not the one to collect the rewards the quickest, but the one that collects
ALL the rewards first. Collecting all rewards in 200 steps is better than collecting n-1 rewards in 50 steps,
and taking 300 for the last reward."""
B_DOOR_MUTEX = _m("DoorMutexEnv", PPO, 'b_ppo_variable_horizon_door_mutex_10M_J68T7D47')
B_DOOR_SHARED = _m("DoorSharedEnv", PPO, 'b_ppo_variable_horizon_door_shared_10M_RB7F1DMV')
B_SLOWDOWN_MUTEX = _m("SlowdownMutexEnv", PPO, 'b_ppo_variable_horizon_slowdown_mutex_5M_4346LPOW')
B_SPEEDUP_MUTEX = _m("SpeedupMutexEnv", PPO, 'b_ppo_variable_horizon_speedup_mutex_5M_6516K24H')

REWARD_NET_DOOR_MUTEX = _m("DoorMutexEnv", None, 'reward_net_door_mutex_100K_LICKEY5T.pt')
REWARD_NET_DOOR_SHARED = _m("DoorSharedEnv", None, 'reward_net_door_shared_100K_77AGICUJ.pt')
REWARD_NET_SLOWDOWN_MUTEX = _m("SlowdownMutexEnv", None, 'reward_net_slowdown_mutex_100K_CK8VYCZZ.pt')
REWARD_NET_SPEEDUP_MUTEX = _m("SpeedupMutexEnv", None, 'reward_net_speedup_mutex_100K_NI7XKOYB.pt')


"""Below are the models that are used for testing throughout the development of the project."""
# door mutex models
TEST_DOOR_MUTEX_10x10_A_SELFISH = _t("DoorMutexEnv", PPO, 'door-mutex_10x10', 'a_ppo_selfish_500K')
TEST_DOOR_MUTEX_10x10_A_DOOR_BLOCK_REWARD = _t("DoorMutexEnv", PPO, 'door-mutex_10x10', 'a_ppo_door-block-reward_500K')
TEST_DOOR_MUTEX_10x10_A_B_REWARD_0_25 = _t("DoorMutexEnv", PPO, 'door-mutex_10x10', 'a_ppo_b-reward-0.25_500K')
TEST_DOOR_MUTEX_10x10_A_B_REWARD_1 = _t("DoorMutexEnv", PPO, 'door-mutex_10x10', 'a_ppo_b-reward-1_500K')
TEST_DOOR_MUTEX_10x10_A_B_REWARD_5 = _t("DoorMutexEnv", PPO, 'door-mutex_10x10', 'a_ppo_b-reward-5_500K')

# door mutex B models
TEST_DOOR_MUTEX_10x10_B_150K = _t("DoorMutexEnv", PPO, 'door-mutex_10x10', 'b_ppo_e-0.25_150K')
TEST_DOOR_MUTEX_10x10_B_1M = _t("DoorMutexEnv", PPO, 'door-mutex_10x10', 'b_ppo_e-0.25_1M')
TEST_DOOR_MUTEX_10x10_B_5M = _t("DoorMutexEnv", PPO, 'door-mutex_10x10', 'b_ppo_e-0.25_5M')
TEST_DOOR_MUTEX_10x10_B_REWARD_LEARNED = _t("DoorMutexEnv", PPO, 'door-mutex_10x10', 'b_ppo_learned_500K')

# door mutex B reward model
TEST_DOOR_MUTEX_10x10_REWARD_MODEL = _t(None, None, 'door-mutex_10x10', 'b_reward_model_1.pt')

# door shared B models
TEST_DOOR_SHARED_10x10_B_1M = _t("DoorSharedEnv", PPO, 'door-shared_10x10', 'b_ppo_e-0.25_1M')
TEST_DOOR_SHARED_10x10_B_10M = _t("DoorSharedEnv", PPO, 'door-shared_10x10', 'b_ppo_e-0.25_10M')

# door shared B reward model
TEST_DOOR_SHARED_10x10_REWARD_MODEL = _t(None, None, 'door-shared_10x10', 'b_reward_net_100K.pt')

# slowdown mutex B models
TEST_SLOWDOWN_MUTEX_10x10_B_1M = _t("SlowdownMutexEnv", PPO, 'slowdown-mutex_10x10', 'b_ppo_e-0.25_1M')
TEST_SLOWDOWN_MUTEX_10x10_B_10M = _t("SlowdownMutexEnv", PPO, 'slowdown-mutex_10x10', 'b_ppo_e-0.25_10M')
TEST_SLOWDOWN_MUTEX_10x10_B_FIX_10M = _t("SlowdownMutexEnv", PPO, 'slowdown-mutex_10x10', 'b_ppo_e-0.25_10M')
