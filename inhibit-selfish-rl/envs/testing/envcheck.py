from stable_baselines3.common.env_checker import check_env

from envs.envlist.door_mutex import DoorMutexEnv
from envs.testing.trivial_env import TrivialEnv

trivial_env = TrivialEnv()
check_env(trivial_env, warn=True, skip_render_check=True)

door_mutex_env = DoorMutexEnv()
check_env(door_mutex_env, warn=True, skip_render_check=True)

# door_mutex_env = DoorMutexEnv()
# door_mutex_env.reset()
# for i in range(100):
#     door_mutex_env.render()
#     door_mutex_env.step(random.randint(0, 3))
