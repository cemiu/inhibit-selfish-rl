import sys
import time

from stable_baselines3.common.on_policy_algorithm import SelfOnPolicyAlgorithm, OnPolicyAlgorithm
from stable_baselines3.common.type_aliases import MaybeCallback
from stable_baselines3.common.utils import safe_mean

"""This modifies the learn method of the on_policy_algorithm class.

The goal is to log additional information to tensorboard.

The learn method is copied from the original source code and modified where stated."""

_original_on_policy_learn = None


def patch_on_policy_logging(modified: bool = True):
    """Patches the on_policy_algorithm class to log the b_reward."""
    if modified:
        _patch_on_policy_algorithm()
    else:
        _unpatch_on_policy_algorithm()


def _learn(
        self: SelfOnPolicyAlgorithm,
        total_timesteps: int,
        callback: MaybeCallback = None,
        log_interval: int = 1,
        tb_log_name: str = "OnPolicyAlgorithm",
        reset_num_timesteps: bool = True,
        progress_bar: bool = False,
) -> SelfOnPolicyAlgorithm:
    iteration = 0

    total_timesteps, callback = self._setup_learn(
        total_timesteps,
        callback,
        reset_num_timesteps,
        tb_log_name,
        progress_bar,
    )

    callback.on_training_start(locals(), globals())

    while self.num_timesteps < total_timesteps:
        continue_training = self.collect_rollouts(self.env, callback, self.rollout_buffer, n_rollout_steps=self.n_steps)

        if continue_training is False:
            break

        iteration += 1
        self._update_current_progress_remaining(self.num_timesteps, total_timesteps)

        if log_interval is not None and iteration % log_interval == 0:
            time_elapsed = max((time.time_ns() - self.start_time) / 1e9, sys.float_info.epsilon)
            fps = int((self.num_timesteps - self._num_timesteps_at_start) / time_elapsed)
            self.logger.record("time/iterations", iteration, exclude="tensorboard")
            if len(self.ep_info_buffer) > 0 and len(self.ep_info_buffer[0]) > 0:
                self.logger.record("rollout/ep_rew_mean", safe_mean([ep_info["r"] for ep_info in self.ep_info_buffer]))
                self.logger.record("rollout/ep_len_mean", safe_mean([ep_info["l"] for ep_info in self.ep_info_buffer]))

                # Start of the modification
                # Define the keys to be logged
                keys_to_log = {
                    "rollout": ('ep_rew_mean_a', 'ep_rew_mean_b'),
                    "performance": ('ep_b_speed_mean', 'ep_b_speed_active_mean', 'ep_touches', 'door_open_step',
                                    'a_rew_collect_step', 'b_rew_collect_step')
                }

                # Log the values for the keys present in the self.ep_info_buffer
                for group_key, keys in keys_to_log.items():
                    for key in keys:
                        if key in self.ep_info_buffer[0]:
                            values = [ep_info[key] for ep_info in self.ep_info_buffer]
                            if len(values) > 0:
                                self.logger.record(f"{group_key}/{key}", safe_mean(values))
                # End of the modification

            self.logger.record("time/fps", fps)
            self.logger.record("time/time_elapsed", int(time_elapsed), exclude="tensorboard")
            self.logger.record("time/total_timesteps", self.num_timesteps, exclude="tensorboard")
            self.logger.dump(step=self.num_timesteps)

        self.train()

    callback.on_training_end()

    return self


def _patch_on_policy_algorithm():
    global _original_on_policy_learn
    if _original_on_policy_learn is None:
        _original_on_policy_learn = OnPolicyAlgorithm.learn
    OnPolicyAlgorithm.learn = _learn


def _unpatch_on_policy_algorithm():
    if _original_on_policy_learn is not None:
        OnPolicyAlgorithm.learn = _original_on_policy_learn
