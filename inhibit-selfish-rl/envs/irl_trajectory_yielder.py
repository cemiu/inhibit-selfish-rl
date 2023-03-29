import math as maths
from functools import partial
from typing import Optional, Sequence

import numpy as np
from imitation.algorithms import preference_comparisons
from imitation.data import rollout
from imitation.data.types import TrajectoryWithRew
from imitation.util import logger as imit_logger
from stable_baselines3.common.vec_env import VecEnv


class TrajectoryYielder(preference_comparisons.TrajectoryGenerator):
    """Class for generating trajectories on demand from a policy
    and a vectorized environment.

    As opposed to AgentTrainer, the input policy can be a pre-trained expert policy
    used for reward learning.

    As opposed to TrajectoryDataset, an unlimited number of trajectories can be
    generated on demand, instead of being limited to a fixed dataset stored in memory.
    """
    def __init__(
            self,
            policy: rollout.AnyPolicy,
            venv: VecEnv,
            fixed_horizon_length: Optional[int] = None,
            rng: Optional[np.random.Generator] = None,
            custom_logger: Optional[imit_logger.HierarchicalLogger] = None,
    ):
        """Builds TrajectoryYielder.

        Args:
            policy: The policy to use for generating trajectories. (sb3 policy/algorithm)
            venv: The vectorized environment to generate trajectories in.
            fixed_horizon_length: Can be set if episodes have a fixed length.
            rng: Random number generator to use for sampling.
            custom_logger: Logger to use for logging. If None, a default logger
                is used.
        """
        super().__init__(custom_logger=custom_logger)
        self.policy = policy
        self.venv = venv
        self.fixed_horizon_length = fixed_horizon_length

        if rng is None:
            rng = np.random.default_rng()
        self.rng = rng

    def sample(self, steps: int) -> Sequence[TrajectoryWithRew]:
        """Generates trajectories until the minimum number of steps has been reached."""
        print(f"Generating {steps} steps of trajectories...")
        return rollout.generate_trajectories(
            policy=self.policy,
            venv=self.venv,
            sample_until=self.make_minimum_step_count(steps),
            rng=self.rng,
        )

    def make_minimum_step_count(
            self,
            step_count: int,
    ) -> rollout.GenTrajTerminationFn:
        """Returns a function that terminates rollout generation when the
        minimum number of steps has been reached.

        Selects between two terminations function depending on whether
        a fixed horizon is set, and thus the step count can be inferred
        from the episode count."""
        if self.fixed_horizon_length is not None:
            return partial(
                _reached_minimum_episode_length,
                self.fixed_horizon_length,
                step_count,
            )
        else:
            return partial(_reached_minimum_step_count, step_count)


def _reached_minimum_step_count(
        step_count: int,
        trajectories: Sequence[TrajectoryWithRew],
) -> bool:
    return sum(len(t.obs) - 1 for t in trajectories) >= step_count


def _reached_minimum_episode_length(
        episode_length: int,
        step_count: int,
        trajectories: Sequence[TrajectoryWithRew],
) -> bool:
    return len(trajectories) >= maths.ceil(step_count / episode_length)
