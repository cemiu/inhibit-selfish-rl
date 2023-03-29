from typing import Dict, Any

import torch as th

import gym
from imitation.rewards.reward_nets import RewardNet
from imitation.util import networks
from stable_baselines3.common import preprocessing


class LoadableRewardNet(RewardNet):
    """A reward networks that can be loaded from a file.

    This network is an MLP that takes a state, action, next state and done flag
    as input and outputs a scalar reward.

    Inputs are flattened and concatenated to one another.

    Inputs can be selectively disabled when initializing the network."""
    def __init__(
        self,
        observation_space: gym.Space,
        action_space: gym.Space,
        use_state: bool = True,
        use_action: bool = True,
        use_next_state: bool = True,
        use_done: bool = False,
        **kwargs,
    ):
        """Builds reward MLP network.

        Args:
            observation_space: The observation space.
            action_space: The action space.
            use_state: should the current state be included as an input to the MLP?
            use_action: should the current action be included as an input to the MLP?
            use_next_state: should the next state be included as an input to the MLP?
            use_done: should the "done" flag be included as an input to the MLP?
            kwargs: passed straight through to `build_mlp`.
        """
        super().__init__(observation_space, action_space)
        combined_size = 0

        self.use_state = use_state
        if self.use_state:
            combined_size += preprocessing.get_flattened_obs_dim(observation_space)

        self.use_action = use_action
        if self.use_action:
            combined_size += preprocessing.get_flattened_obs_dim(action_space)

        self.use_next_state = use_next_state
        if self.use_next_state:
            combined_size += preprocessing.get_flattened_obs_dim(observation_space)

        self.use_done = use_done
        if self.use_done:
            combined_size += 1

        full_build_mlp_kwargs: Dict[str, Any] = {
            "hid_sizes": (32, 32),
            **kwargs,
            # we do not want the values below to be overridden
            "in_size": combined_size,
            "out_size": 1,
            "squeeze_output": True,
        }

        self.mlp = networks.build_mlp(**full_build_mlp_kwargs)

    def forward(self, state, action, next_state, done):
        inputs = []
        if self.use_state:
            inputs.append(th.flatten(state, 1))
        if self.use_action:
            inputs.append(th.flatten(action, 1))
        if self.use_next_state:
            inputs.append(th.flatten(next_state, 1))
        if self.use_done:
            inputs.append(th.reshape(done, [-1, 1]))

        inputs_concat = th.cat(inputs, dim=1)

        outputs = self.mlp(inputs_concat)
        assert outputs.shape == state.shape[:1]

        return outputs

    @classmethod
    def loadnet(
            cls,
            path: str,
            observation_space: gym.Space,
            action_space: gym.Space,
            use_state: bool = True,
            use_action: bool = True,
            use_next_state: bool = True,
            use_done: bool = False,
            **kwargs,
    ):
        """Loads a reward network from a file."""
        reward_net = cls(
            observation_space,
            action_space,
            use_state=use_state,
            use_action=use_action,
            use_next_state=use_next_state,
            use_done=use_done,
            **kwargs,
        )
        reward_net.load_state_dict(th.load(path))
        return reward_net

    def setnet(self, path):
        """Sets the network state to a preconfigured state."""
        self.load_state_dict(th.load(path))
        return self

    def save(self, path: str):
        """Saves a reward network to a file."""
        th.save(self.state_dict(), path)


if __name__ == '__main__':
    LoadableRewardNet.loadnet('reward_net.pt')
