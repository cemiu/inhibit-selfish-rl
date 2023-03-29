from gym.vector.utils import spaces
from matplotlib import pyplot as plt
from matplotlib import colors
import numpy as np

from envs.definitions import board_stripping
from envs.envlist import *
from experiments.datagathering import visualise_b_reward_model
from models import *
from experiments.datagathering.visualise_b_reward_model import get_averaged_reward_grid
from ml.reward_networks import LoadableRewardNet
from util.matrix_utils import calculate_statistics, normalise_arrays


def get_reward_grid(stripping_config):
    grid, *_ = get_averaged_reward_grid(
        env=env,
        observation=observation,
        board_generator=board_generator,
        stripping_config=stripping_config,
        samples=samples,
        reward_network=reward_network,
    )
    return grid


def plot_grid(ax, grid, title):
    mean, min_val, max_val = calculate_statistics(grid)
    im = ax.imshow(grid, cmap=cmap, norm=norm)
    ax.set_title(title)
    ax.text(
        0.5,
        -0.1,
        f"Mean: {mean:.2f}\nMin: {min_val:.2f}\nMax: {max_val:.2f}",
        transform=ax.transAxes,
        ha="center",
        va="top",
        bbox=dict(facecolor="white", alpha=0.8),
    )
    ax.axis("off")
    return im


if __name__ == "__main__":
    # config
    reward_models = [
        REWARD_NET_DOOR_MUTEX,
        REWARD_NET_DOOR_SHARED,
        REWARD_NET_SPEEDUP_MUTEX,
        REWARD_NET_SLOWDOWN_MUTEX,
    ]

    env, _, model_path = reward_models[1]
    env = eval(env)
    board_generator = env.DEFAULT_BOARD
    samples = 100
    # config end

    observation = visualise_b_reward_model.BOARD_OBSERVATION(10)
    reward_network = LoadableRewardNet.loadnet(
        path=model_path,
        observation_space=observation.get_observation_space(),
        action_space=spaces.Discrete(4),
    )

    # Get reward grids
    grid_all = get_reward_grid(board_stripping.ALL)
    grid_random = get_reward_grid(board_stripping.DOOR_AND_RAND_RESOURCES)
    grid_door = get_reward_grid(board_stripping.DOOR)

    grid_all, grid_random, grid_door = normalise_arrays(grid_all, grid_random, grid_door)
    print(grid_door)

    # Define the colormap
    cmap = "viridis"

    # Normalize colours
    max_val = max(calculate_statistics(grid_all)[2], calculate_statistics(grid_door)[2])
    norm = colors.Normalize(vmin=0, vmax=max_val)

    # Plot the arrays
    # fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5), gridspec_kw={'wspace': 0.1})

    im1 = plot_grid(ax1, grid_all, "No rewards")
    im3 = plot_grid(ax2, grid_random, "Reduced rewards (50%)")
    im2 = plot_grid(ax3, grid_door, "All rewards")

    # fig.colorbar(im1, ax=[ax1, ax2], shrink=0.6, aspect=20)
    fig.colorbar(im1, ax=[ax1, ax2, ax3], shrink=0.6, aspect=20)

    fig.suptitle('Reward Grids for DoorShared environment', fontsize=22, y=0.9)

    plt.show()
