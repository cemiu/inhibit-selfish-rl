from imitation.rewards.reward_nets import BasicRewardNet
from imitation.util.networks import RunningNorm
from stable_baselines3.common.env_util import make_vec_env

import torch as th

venv = make_vec_env("Pendulum-v1", n_envs=1)

reward_net = BasicRewardNet(venv.observation_space, venv.action_space, normalize_input_layer=RunningNorm)
reward_net2 = BasicRewardNet(venv.observation_space, venv.action_space)  # , normalize_input_layer=RunningNorm)

# reward_net.mlp.load_state_dict(th.load("mlp.pt"))
# reward_net2.mlp.load_state_dict(th.load("mlp2.pt"))
# th.save(reward_net.mlp.state_dict(), "mlp.pt")
# th.save(reward_net.state_dict(), "mlp2.pt")


state_dict1 = reward_net.state_dict()
th.save(state_dict1, "mlp.pt")

reward_net2.load_state_dict(th.load("mlp.pt"))

mlp_state_dict1 = reward_net.mlp.state_dict()
mlp_state_dict2 = reward_net2.mlp.state_dict()

for var_name in mlp_state_dict1:
    allclose = th.allclose(mlp_state_dict1[var_name], mlp_state_dict2[var_name])
    print(f"{var_name}: {allclose}")

# for var_name in state_dict1:
#     var_name2 = var_name.replace("mlp.", "")
#     allclose = th.allclose(state_dict1[var_name], state_dict2[var_name])
#     print(f"{var_name}: {allclose}")

# print(reward_net2.mlp.state_dict())


# export
# th.save(mlp.state_dict(), "mlp.pt")
# th.save(mlp.state_dict(), "mlp2.pt")

# import
#
# reward_net2 = BasicRewardNet(
#     venv.observation_space, venv.action_space, normalize_input_layer=RunningNorm
# )
#
# state_dict2 = th.load("mlp.pt")
# reward_net.mlp.load_state_dict(state_dict2)
#
# print(reward_net.mlp.state_dict())
#
# # print(reward_net2.mlp.state_dict())
# #
# # print(state_dict2)
#
# # th.save(reward_net2.mlp.state_dict(), "mlp2.pt")
#
# state_dict3 = th.load("mlp2.pt")
#
# print(state_dict2)
# print(state_dict3)
