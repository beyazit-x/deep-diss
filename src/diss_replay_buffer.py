import warnings
from abc import ABC, abstractmethod
from typing import Any, Dict, Generator, List, Optional, Union

import random
import numpy as np
import torch as th
from gym import spaces

from stable_baselines3.common.vec_env import VecNormalize
from stable_baselines3.common.buffers import DictReplayBuffer
from stable_baselines3.common.type_aliases import (
    DictReplayBufferSamples,
    DictRolloutBufferSamples,
    ReplayBufferSamples,
    RolloutBufferSamples,
)

# TODO: How to delete stuff from her replay buffer if it is full?
# Once the deletion strategy for traces is decided, clear (fill with zeros) trace slots in the her replay buffer.
# TODO: How to compute reward wrt to the achieved goal? And where to do it?

class DissReplayBuffer(DictReplayBuffer):
    def __init__(
        self,
        buffer_size: int,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        max_episode_length: int,
        her_replay_buffer_size: int,
        device: Union[th.device, str] = "auto",
        n_envs: int = 1,
        optimize_memory_usage: bool = False,
        handle_timeout_termination: bool = True
    ):
        super().__init__(buffer_size=buffer_size,
            observation_space=observation_space,
            action_space=action_space,
            device=device,
            n_envs=n_envs,
            optimize_memory_usage=optimize_memory_usage,
            handle_timeout_termination=handle_timeout_termination
            )
        self.her_replay_buffer_size = her_replay_buffer_size
        self.max_episode_length = max_episode_length
        self.n_envs = n_envs
        self.input_shape = {
            "features": (self.n_envs, self.her_replay_buffer_size, self.max_episode_length + 1,) + self.obs_shape["features"],
            "dfa": (self.n_envs, self.her_replay_buffer_size, self.max_episode_length + 1,) + self.obs_shape["dfa"],
            "action": (self.n_envs, self.her_replay_buffer_size, self.max_episode_length + 1,) + (self.action_dim,),
            "reward": (self.n_envs, self.her_replay_buffer_size, self.max_episode_length + 1,) + (1,),
            "next_features": (self.n_envs, self.her_replay_buffer_size, self.max_episode_length + 1,) + self.obs_shape["features"],
            "next_dfa": (self.n_envs, self.her_replay_buffer_size, self.max_episode_length + 1,) + self.obs_shape["dfa"],
            "done": (self.n_envs, self.her_replay_buffer_size, self.max_episode_length + 1,) + (1,)
        }
        self.her_replay_buffer_not_relabeled = {
            key: np.zeros(dim, dtype=np.float32) if key != "action" else np.zeros(dim, dtype=np.int64)
            for key, dim in self.input_shape.items()
        }
        self.her_replay_buffer_relabeled = {
            key: np.zeros((dim[0] * dim[1],) + dim[2:], dtype=np.float32) if key != "action" else np.zeros((dim[0] * dim[1],) + dim[2:], dtype=np.int64)
            for key, dim in self.input_shape.items()
        }
        self.not_relabeled_traces = []
        self.episode_lengths = np.zeros((self.n_envs * self.her_replay_buffer_size), dtype=np.int64) # This should keep track of relabeled ones
        self.current_episode_idx_not_relabeled = np.zeros(self.n_envs, dtype=np.int64)
        self.current_episode_step_idx_not_relabeled = np.zeros(self.n_envs, dtype=np.int64)
        self.current_episode_idx_relabeled = 0
        self.is_her_replay_buffer_relabeled_full = False
        self.her_ratio = 0.1
        self.env_indices = np.arange(self.n_envs)

    def add(
        self,
        obs: Dict[str, np.ndarray],
        next_obs: Dict[str, np.ndarray],
        action: np.ndarray,
        reward: np.ndarray,
        done: np.ndarray,
        infos: List[Dict[str, Any]],
    ) -> None:
        super().add(obs, next_obs, action, reward, done, infos)
        for i in range(self.n_envs):
            self.her_replay_buffer_not_relabeled["features"][i][self.current_episode_idx_not_relabeled[i]][self.current_episode_step_idx_not_relabeled[i]] = np.array(obs["features"][i]).copy()
            self.her_replay_buffer_not_relabeled["dfa"][i][self.current_episode_idx_not_relabeled[i]][self.current_episode_step_idx_not_relabeled[i]] = np.array(obs["dfa"][i]).copy()
            self.her_replay_buffer_not_relabeled["action"][i][self.current_episode_idx_not_relabeled[i]][self.current_episode_step_idx_not_relabeled[i]] = np.array(action[i]).copy()
            self.her_replay_buffer_not_relabeled["reward"][i][self.current_episode_idx_not_relabeled[i]][self.current_episode_step_idx_not_relabeled] = np.array(reward[i]).copy()
            self.her_replay_buffer_not_relabeled["next_features"][i][self.current_episode_idx_not_relabeled[i]][self.current_episode_step_idx_not_relabeled[i]] = np.array(next_obs["features"][i]).copy()
            self.her_replay_buffer_not_relabeled["next_dfa"][i][self.current_episode_idx_not_relabeled[i]][self.current_episode_step_idx_not_relabeled[i]] = np.array(next_obs["dfa"][i]).copy()
            self.her_replay_buffer_not_relabeled["done"][i][self.current_episode_idx_not_relabeled[i]][self.current_episode_step_idx_not_relabeled[i]] = np.array(done[i]).copy()
            self.current_episode_step_idx_not_relabeled[i] += 1
            if done[i] or self.current_episode_step_idx_not_relabeled[i] > self.max_episode_length:
                self.not_relabeled_traces.append((i, self.current_episode_idx_not_relabeled[i]))
                self.current_episode_idx_not_relabeled[i] = (self.current_episode_idx_not_relabeled[i] + 1) % self.her_replay_buffer_size
                self.current_episode_step_idx_not_relabeled[i] = 0
                if (i, self.current_episode_idx_not_relabeled[i]) in self.not_relabeled_traces:
                    self.not_relabeled_traces.remove((i, self.current_episode_idx_not_relabeled[i])) # We will start writing to that env-episode pair so remove
                self.her_replay_buffer_not_relabeled["features"][i][self.current_episode_idx_not_relabeled[i]] = np.zeros(self.her_replay_buffer_not_relabeled["features"][i][self.current_episode_idx_not_relabeled[i]].shape)
                self.her_replay_buffer_not_relabeled["dfa"][i][self.current_episode_idx_not_relabeled[i]] = np.zeros(self.her_replay_buffer_not_relabeled["dfa"][i][self.current_episode_idx_not_relabeled[i]].shape)
                self.her_replay_buffer_not_relabeled["action"][i][self.current_episode_idx_not_relabeled[i]] = np.zeros(self.her_replay_buffer_not_relabeled["action"][i][self.current_episode_idx_not_relabeled[i]].shape)
                self.her_replay_buffer_not_relabeled["reward"][i][self.current_episode_idx_not_relabeled[i]] = np.zeros(self.her_replay_buffer_not_relabeled["reward"][i][self.current_episode_idx_not_relabeled[i]].shape)
                self.her_replay_buffer_not_relabeled["next_features"][i][self.current_episode_idx_not_relabeled[i]] = np.zeros(self.her_replay_buffer_not_relabeled["next_features"][i][self.current_episode_idx_not_relabeled[i]].shape)
                self.her_replay_buffer_not_relabeled["next_dfa"][i][self.current_episode_idx_not_relabeled[i]] = np.zeros(self.her_replay_buffer_not_relabeled["next_dfa"][i][self.current_episode_idx_not_relabeled[i]].shape)
                self.her_replay_buffer_not_relabeled["done"][i][self.current_episode_idx_not_relabeled[i]] = np.zeros(self.her_replay_buffer_not_relabeled["done"][i][self.current_episode_idx_not_relabeled[i]].shape)


    def sample_traces(self, batch_size, env):
        n_not_relabeled_traces = len(self.not_relabeled_traces)
        if batch_size > n_not_relabeled_traces or batch_size <= 0:
            return None
        sample_inds = np.array(self.not_relabeled_traces[:batch_size])
        self.not_relabeled_traces = self.not_relabeled_traces[batch_size:]
        sample_env_inds = sample_inds[:, 0]
        sample_eps_inds = sample_inds[:, 1]

        obs_ = self._normalize_obs({"features": self.her_replay_buffer_not_relabeled["features"][sample_env_inds, sample_eps_inds].copy(), "dfa": self.her_replay_buffer_not_relabeled["dfa"][sample_env_inds, sample_eps_inds].copy()}, env)
        next_obs_ = self._normalize_obs({"features": self.her_replay_buffer_not_relabeled["next_features"][sample_env_inds, sample_eps_inds].copy(), "dfa": self.her_replay_buffer_not_relabeled["next_dfa"][sample_env_inds, sample_eps_inds].copy()}, env)

        observations = {key: obs for key, obs in obs_.items()}
        actions = self.her_replay_buffer_not_relabeled["action"][sample_env_inds, sample_eps_inds].copy()
        next_observations = {key: next_obs for key, next_obs in next_obs_.items()}
        dones = self.her_replay_buffer_not_relabeled["done"][sample_env_inds, sample_eps_inds].copy()
        rewards = self._normalize_reward(self.her_replay_buffer_not_relabeled["reward"][sample_env_inds, sample_eps_inds].copy(), env)

        return DictReplayBufferSamples(
            observations=observations,
            actions=actions,
            next_observations=next_observations,
            dones=dones,
            rewards=rewards
        )

    def relabel_traces(self, batch_size, dict_replay_buffer_samples):
        self.her_replay_buffer_relabeled["features"][self.current_episode_idx_relabeled : self.current_episode_idx_relabeled + batch_size] = dict_replay_buffer_samples.observations["features"].copy()
        self.her_replay_buffer_relabeled["dfa"][self.current_episode_idx_relabeled : self.current_episode_idx_relabeled + batch_size] = dict_replay_buffer_samples.observations["dfa"].copy()
        self.her_replay_buffer_relabeled["action"][self.current_episode_idx_relabeled : self.current_episode_idx_relabeled + batch_size] = dict_replay_buffer_samples.actions.copy()
        self.her_replay_buffer_relabeled["reward"][self.current_episode_idx_relabeled : self.current_episode_idx_relabeled + batch_size] = dict_replay_buffer_samples.rewards.copy()
        self.her_replay_buffer_relabeled["next_features"][self.current_episode_idx_relabeled : self.current_episode_idx_relabeled + batch_size] = dict_replay_buffer_samples.next_observations["features"].copy()
        self.her_replay_buffer_relabeled["next_dfa"][self.current_episode_idx_relabeled : self.current_episode_idx_relabeled + batch_size] = dict_replay_buffer_samples.next_observations["dfa"].copy()
        self.her_replay_buffer_relabeled["done"][self.current_episode_idx_relabeled : self.current_episode_idx_relabeled + batch_size] = dict_replay_buffer_samples.dones.copy()
        self.episode_lengths[self.current_episode_idx_relabeled] = dict_replay_buffer_samples.dones.squeeze().nonzero()[0].item()
        self.current_episode_idx_relabeled = (self.current_episode_idx_relabeled + batch_size) % (self.n_envs * self.her_replay_buffer_size)
        if self.current_episode_idx_relabeled == 0:
            self.is_her_replay_buffer_relabeled_full = True

    def get_her_transitions_from_inds(self, her_batch_size, env):
        sample_space = None
        if self.is_her_replay_buffer_relabeled_full:
            sample_space = [(i, j) for i in range(self.n_envs * self.her_replay_buffer_size) for j in range(self.episode_lengths[i])]
        else:
            sample_space = [(i, j) for i in range(self.current_episode_idx_relabeled) for j in range(self.episode_lengths[i])]
        if her_batch_size > len(sample_space):
            return
        sample_inds = np.array(random.sample(sample_space, her_batch_size))
        sample_trc_inds = sample_inds[:, 0]
        sample_stp_inds = sample_inds[:, 1]

        obs_ = self._normalize_obs({"features": self.her_replay_buffer_relabeled["features"][sample_trc_inds, sample_stp_inds].copy(), "dfa": self.her_replay_buffer_relabeled["dfa"][sample_trc_inds, sample_stp_inds].copy()}, env)
        next_obs_ = self._normalize_obs({"features": self.her_replay_buffer_relabeled["next_features"][sample_trc_inds, sample_stp_inds].copy(), "dfa": self.her_replay_buffer_relabeled["next_dfa"][sample_trc_inds, sample_stp_inds].copy()}, env)

        observations = {key: self.to_torch(obs) for key, obs in obs_.items()}
        actions = self.to_torch(self.her_replay_buffer_relabeled["action"][sample_trc_inds, sample_stp_inds].copy())
        next_observations = {key: self.to_torch(next_obs) for key, next_obs in next_obs_.items()}
        dones = self.to_torch(self.her_replay_buffer_relabeled["done"][sample_trc_inds, sample_stp_inds].copy())
        rewards = self.to_torch(self._normalize_reward(self.her_replay_buffer_relabeled["reward"][sample_trc_inds, sample_stp_inds].copy(), env))

        return DictReplayBufferSamples(
            observations=observations,
            actions=actions,
            next_observations=next_observations,
            dones=dones,
            rewards=rewards
        )

    def sample(self, batch_size: int, env: Optional[VecNormalize] = None) -> DictReplayBufferSamples:
        her_batch_size = int(self.her_ratio * batch_size)
        regular_batch_size = batch_size - her_batch_size
        regular_samples = super().sample(regular_batch_size, env)
        her_samples = self.get_her_transitions_from_inds(her_batch_size, env)
        if her_samples is None:
            her_samples = super().sample(her_batch_size, env)
        return DictReplayBufferSamples(
            observations={"features": th.concat((regular_samples.observations["features"], her_samples.observations["features"])), "dfa": th.concat((regular_samples.observations["dfa"], her_samples.observations["dfa"]))},
            actions=th.concat((regular_samples.actions, her_samples.actions)),
            next_observations={"features": th.concat((regular_samples.next_observations["features"], her_samples.next_observations["features"])), "dfa": th.concat((regular_samples.next_observations["dfa"], her_samples.next_observations["dfa"]))},
            dones=th.concat((regular_samples.dones, her_samples.dones)),
            rewards=th.concat((regular_samples.rewards, her_samples.rewards)),
        )

