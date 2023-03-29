import gym
import numpy as np
from gym import spaces


class TrivialEnv(gym.Env):
    """Custom Environment that follows gym interface."""

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 50}

    def __init__(self):
        super(TrivialEnv, self).__init__()
        self.terminated, self.truncated, self.reward, self.observation, self.info = (None,) * 5
        self.action_space = spaces.Discrete(4)

        self.observation_space = spaces.Box(low=-500, high=500,
                                            shape=(4,), dtype=np.int64)

    def step(self, action):
        self.reward = 0
        self.observation = np.random.randint(0, 1, size=(4,))
        self.info = {}
        return self.observation, self.reward, self.terminated, self.truncated, self.info

    def reset(self, **kwargs):
        self.terminated = False
        self.truncated = False

        # self.observation = np.random.randint(0, 1, size=(4,))
        self.observation = np.array([0, 0, 0, 0])
        self.info = {}

        print(self.observation)
        print(self.observation.dtype)
        print(self.observation.shape)

        return_tuple = (self.observation, self.info)

        return return_tuple

    def render(self, mode="human"):
        pass

    def close(self):
        pass
