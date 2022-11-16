import sys

import numpy as np
import torch as th

from stable_baselines3.common.buffers import DictReplayBuffer
from stable_baselines3.common.type_aliases import DictReplayBufferSamples

class DissRelabeler():

    def __init__(self, replay_buffer: DictReplayBuffer):
        self.replay_buffer = replay_buffer

    def _get_batch_inds(self, batch_size):
        if self.replay_buffer.optimize_memory_usage:
            if self.replay_buffer.full:
                return (np.random.randint(1, self.replay_buffer.buffer_size, size=batch_size) + self.replay_buffer.pos) % self.replay_buffer.buffer_size
            else:
                return np.random.randint(0, self.replay_buffer.pos, size=batch_size)
        else:
            upper_bound = self.replay_buffer.buffer_size if self.replay_buffer.full else self.replay_buffer.pos
            return np.random.randint(0, upper_bound, size=batch_size)

    def _get_env_indices(self, batch_inds):
        return np.random.randint(0, high=self.replay_buffer.n_envs, size=(len(batch_inds),))

    def _get_samples(self, env, batch_inds, env_indices):
        # Normalize if needed and remove extra dimension (we are using only one env for now)
        obs_ = self.replay_buffer._normalize_obs({key: obs[batch_inds, env_indices, :] for key, obs in self.replay_buffer.observations.items()}, env)
        next_obs_ = self.replay_buffer._normalize_obs(
            {key: obs[batch_inds, env_indices, :] for key, obs in self.replay_buffer.next_observations.items()}, env
        )

        # Convert to torch tensor
        observations = {key: obs for key, obs in obs_.items()}
        next_observations = {key: obs for key, obs in next_obs_.items()}

        return DictReplayBufferSamples(
            observations=observations,
            actions=self.replay_buffer.actions[batch_inds, env_indices],
            next_observations=next_observations,
            # Only use dones that are not due to timeouts
            # deactivated by default (timeouts is initialized as an array of False)
            dones=self.replay_buffer.dones[batch_inds, env_indices] * (1 - self.replay_buffer.timeouts[batch_inds, env_indices]).reshape(
                -1, 1
            ),
            rewards=self.replay_buffer._normalize_reward(self.replay_buffer.rewards[batch_inds, env_indices].reshape(-1, 1), env),
        )

    def _relabel_samples(self, batch_inds, env_indices, new_observations, new_actions, new_next_observations, new_rewards, new_dones):
        # This function should rewrite the existing observations in the replay buffer. Questions: When we do things asynch, do we need a mutex for the relabeled observations?
        assert (self.replay_buffer.observations["dfa"][batch_inds, env_indices] == new_observations["dfa"]).all()
        self.replay_buffer.observations["dfa"][batch_inds, env_indices] = new_observations["dfa"]
        assert (self.replay_buffer.actions[batch_inds, env_indices] == new_actions).all()
        self.replay_buffer.actions[batch_inds, env_indices] = new_actions
        assert (self.replay_buffer.next_observations["dfa"][batch_inds, env_indices] == new_next_observations["dfa"]).all()
        self.replay_buffer.next_observations["dfa"][batch_inds, env_indices] = new_next_observations["dfa"]
        assert (self.replay_buffer.rewards[batch_inds, env_indices] == new_rewards.reshape(self.replay_buffer.rewards[batch_inds, env_indices].shape)).all()
        self.replay_buffer.rewards[batch_inds, env_indices] = new_rewards.reshape(self.replay_buffer.rewards[batch_inds, env_indices].shape)

    def relabel(self, env, batch_size):

        batch_inds = self._get_batch_inds(batch_size)
        env_indices = self._get_env_indices(batch_inds)
        replay_data = self._get_samples(env, batch_inds, env_indices)

        # print("batch_inds", batch_inds)
        # print("env_indices", env_indices)

        observations = replay_data.observations
        actions = replay_data.actions
        next_observations = replay_data.next_observations
        rewards = replay_data.rewards
        dones = replay_data.dones

        #### THE DISS MAGIC STARTS HERE ####

        new_observations = observations
        new_actions = actions
        new_next_observations = next_observations
        new_rewards = rewards
        new_dones = dones

        #### THE DISS MAGIC ENDS HERE ####

        self._relabel_samples(batch_inds, env_indices, new_observations, new_actions, new_next_observations, new_rewards, new_dones)
    