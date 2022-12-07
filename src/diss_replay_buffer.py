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
            "observation": (self.n_envs, self.her_replay_buffer_size, self.max_episode_length,) + self.obs_shape["features"],
            "achieved_goal": (self.n_envs, self.her_replay_buffer_size, self.max_episode_length,) + self.obs_shape["dfa"],
            "desired_goal": (self.n_envs, self.her_replay_buffer_size, self.max_episode_length,) + self.obs_shape["dfa"],
            "action": (self.n_envs, self.her_replay_buffer_size, self.max_episode_length,) + (self.action_dim,),
            "reward": (self.n_envs, self.her_replay_buffer_size, self.max_episode_length,) + (1,),
            "next_observation": (self.n_envs, self.her_replay_buffer_size, self.max_episode_length,) + self.obs_shape["features"],
            "next_achieved_goal": (self.n_envs, self.her_replay_buffer_size, self.max_episode_length,) + self.obs_shape["dfa"],
            "next_desired_goal": (self.n_envs, self.her_replay_buffer_size, self.max_episode_length,) + self.obs_shape["dfa"],
            "done": (self.n_envs, self.her_replay_buffer_size, self.max_episode_length,) + (1,)
        }
        self.her_replay_buffer = {
            key: np.zeros(dim, dtype=np.float32) if key != "action" else np.zeros(dim, dtype=np.int64)
            for key, dim in self.input_shape.items()
        }
        self.is_relabeled = np.zeros((self.n_envs, self.her_replay_buffer_size), dtype=np.int64)
        self.is_in_progress = np.zeros((self.n_envs, self.her_replay_buffer_size), dtype=np.int64)
        self.episode_lengths = np.zeros((self.n_envs, self.her_replay_buffer_size), dtype=np.int64)
        self.current_episode_idx = np.zeros(self.n_envs, dtype=np.int64)
        self.current_episode_step_idx = np.zeros(self.n_envs, dtype=np.int64)
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
        self.her_replay_buffer["observation"][self.env_indices, self.current_episode_idx, self.current_episode_step_idx] = np.array(obs["features"]).copy()
        self.her_replay_buffer["achieved_goal"][self.env_indices, self.current_episode_idx, self.current_episode_step_idx] = np.array(obs["dfa"]).copy()
        self.her_replay_buffer["desired_goal"][self.env_indices, self.current_episode_idx, self.current_episode_step_idx] = np.array(obs["dfa"]).copy()
        self.her_replay_buffer["action"][self.env_indices, self.current_episode_idx, self.current_episode_step_idx] = np.array(action).copy()
        self.her_replay_buffer["reward"][self.env_indices, self.current_episode_idx, self.current_episode_step_idx] = np.array(reward).copy()
        self.her_replay_buffer["next_observation"][self.env_indices, self.current_episode_idx, self.current_episode_step_idx] = np.array(next_obs["features"]).copy()
        self.her_replay_buffer["next_achieved_goal"][self.env_indices, self.current_episode_idx, self.current_episode_step_idx] = np.array(next_obs["dfa"]).copy()
        self.her_replay_buffer["next_desired_goal"][self.env_indices, self.current_episode_idx, self.current_episode_step_idx] = np.array(next_obs["dfa"]).copy()
        self.her_replay_buffer["done"][self.env_indices, self.current_episode_idx, self.current_episode_step_idx] = np.array(done).copy()
        self.is_relabeled[self.env_indices, self.current_episode_idx] = np.zeros(self.n_envs, dtype=np.int64)
        self.is_in_progress[self.env_indices, self.current_episode_idx] = np.zeros(self.n_envs, dtype=np.int64)
        self.current_episode_step_idx += 1
        inds = (done == 1) | (self.current_episode_step_idx >= self.max_episode_length)
        self.episode_lengths[inds, self.current_episode_idx] = self.current_episode_step_idx[inds]
        self.current_episode_idx[inds] = (self.current_episode_idx[inds] + 1) % self.her_replay_buffer_size
        self.current_episode_step_idx[inds] = 0
        # Below is the for loop version of the above batched code
        # for i in range(self.n_envs):
        #     self.her_replay_buffer["observation"][i][self.current_episode_idx[i]][self.current_episode_step_idx[i]] = np.array(obs["features"][i]).copy()
        #     self.her_replay_buffer["achieved_goal"][i][self.current_episode_idx[i]][self.current_episode_step_idx[i]] = np.array(obs["dfa"][i]).copy()
        #     self.her_replay_buffer["desired_goal"][i][self.current_episode_idx[i]][self.current_episode_step_idx[i]] = np.array(obs["dfa"][i]).copy()
        #     self.her_replay_buffer["action"][i][self.current_episode_idx[i]][self.current_episode_step_idx[i]] = np.array(action[i]).copy()
        #     self.her_replay_buffer["reward"][i][self.current_episode_idx[i]][self.current_episode_step_idx] = np.array(reward[i]).copy()
        #     self.her_replay_buffer["next_observation"][i][self.current_episode_idx[i]][self.current_episode_step_idx[i]] = np.array(next_obs["features"][i]).copy()
        #     self.her_replay_buffer["next_achieved_goal"][i][self.current_episode_idx[i]][self.current_episode_step_idx[i]] = np.array(next_obs["dfa"][i]).copy()
        #     self.her_replay_buffer["next_desired_goal"][i][self.current_episode_idx[i]][self.current_episode_step_idx[i]] = np.array(next_obs["dfa"][i]).copy()
        #     self.her_replay_buffer["done"][i][self.current_episode_idx[i]][self.current_episode_step_idx[i]] = np.array(done[i]).copy()
        #     self.is_relabeled[i][self.current_episode_idx[i]] = 0
        #     self.is_in_progress[i][self.current_episode_idx[i]] = 0
        #     self.current_episode_step_idx[i] += 1
        #     if done[i] or self.current_episode_step_idx[i] >= self.max_episode_length:
        #         self.episode_lengths[i][self.current_episode_idx[i]] = self.current_episode_step_idx[i]
        #         self.current_episode_idx[i] = (self.current_episode_idx[i] + 1) % self.her_replay_buffer_size
        #         self.current_episode_step_idx[i] = 0

    def sample_traces(self, batch_size, env):
        inds = np.argwhere(self.is_relabeled <= 0)
        env_inds = inds[:, 0]
        eps_inds = inds[:, 1]
        n_not_relabeled_traces = env_inds.size
        if batch_size > n_not_relabeled_traces or batch_size <= 0:
            return None
        return self.get_traces_from_inds(batch_size, env_inds, eps_inds, env)

    def relabel_traces(self, dic_replay_buffer_samples):
        # relabeled_indices = []
        # for sampled_episode_ind, sampled_env_ind, achieved_goal in achieved_goals:
        #     self.her_replay_buffer["achieved_goal"][sampled_episode_ind, :, sampled_env_ind] = achieved_goal
        #     self.her_replay_buffer["relabeled"][sampled_episode_ind, sampled_env_ind] = 1
        #     # Compute the new reward here and write it in ["reward"] in her replay buffer (do this in DissRelabeler and call this function with the computed reward)
        #     relabeled_indices.append([sampled_episode_ind, sampled_env_ind])
        # self.indices_in_progress = list(filter(lambda ind: ind not in relabeled_indices, self.indices_in_progress))
        pass

    def get_traces_from_inds(self, batch_size, env_inds, eps_inds, env):
        sample_space = [(i, j) for i, j in zip(env_inds, eps_inds)]
        sample_inds = np.array(random.sample(sample_space, batch_size))
        sample_env_inds = sample_inds[:, 0]
        sample_eps_inds = sample_inds[:, 1]

        obs_ = self._normalize_obs({"features": self.her_replay_buffer["observation"][sample_env_inds, sample_eps_inds], "dfa": self.her_replay_buffer["achieved_goal"][sample_env_inds, sample_eps_inds]}, env)
        next_obs_ = self._normalize_obs({"features": self.her_replay_buffer["next_observation"][sample_env_inds, sample_eps_inds], "dfa": self.her_replay_buffer["next_achieved_goal"][sample_env_inds, sample_eps_inds]}, env)

        observations = {key: self.to_torch(obs) for key, obs in obs_.items()}
        actions = self.to_torch(self.her_replay_buffer["action"][sample_env_inds, sample_eps_inds])
        next_observations = {key: self.to_torch(next_obs) for key, next_obs in next_obs_.items()}
        dones = self.to_torch(self.her_replay_buffer["done"][sample_env_inds, sample_eps_inds])
        rewards = self.to_torch(self._normalize_reward(self.her_replay_buffer["reward"][sample_env_inds, sample_eps_inds], env))

        return DictReplayBufferSamples(
            observations=observations,
            actions=actions,
            next_observations=next_observations,
            dones=dones,
            rewards=rewards
        )

    def get_her_transitions_from_inds(self, her_batch_size, env_inds, eps_inds, env):
        sample_space = [(i, j, k) for i, j in zip(env_inds, eps_inds) for k in range(self.episode_lengths[i, j])]
        sample_inds = np.array(random.sample(sample_space, her_batch_size))
        sample_env_inds = sample_inds[:, 0]
        sample_eps_inds = sample_inds[:, 1]
        sample_stp_inds = sample_inds[:, 2]

        obs_ = self._normalize_obs({"features": self.her_replay_buffer["observation"][sample_env_inds, sample_eps_inds, sample_stp_inds], "dfa": self.her_replay_buffer["achieved_goal"][sample_env_inds, sample_eps_inds, sample_stp_inds]}, env)
        next_obs_ = self._normalize_obs({"features": self.her_replay_buffer["next_observation"][sample_env_inds, sample_eps_inds, sample_stp_inds], "dfa": self.her_replay_buffer["next_achieved_goal"][sample_env_inds, sample_eps_inds, sample_stp_inds]}, env)

        observations = {key: self.to_torch(obs) for key, obs in obs_.items()}
        actions = self.to_torch(self.her_replay_buffer["action"][sample_env_inds, sample_eps_inds, sample_stp_inds])
        next_observations = {key: self.to_torch(next_obs) for key, next_obs in next_obs_.items()}
        dones = self.to_torch(self.her_replay_buffer["done"][sample_env_inds, sample_eps_inds, sample_stp_inds])
        rewards = self.to_torch(self._normalize_reward(self.her_replay_buffer["reward"][sample_env_inds, sample_eps_inds, sample_stp_inds], env))

        return DictReplayBufferSamples(
            observations=observations,
            actions=actions,
            next_observations=next_observations,
            dones=dones,
            rewards=rewards
        )

    def sample(self, batch_size: int, env: Optional[VecNormalize] = None) -> DictReplayBufferSamples:
        her_batch_size = int(self.her_ratio * batch_size)
        inds = np.argwhere(self.is_relabeled > 0)
        env_inds = inds[:, 0]
        eps_inds = inds[:, 1]
        n_relabeled_transitions = np.sum(self.episode_lengths[env_inds, eps_inds])
        if her_batch_size > n_relabeled_transitions or her_batch_size <= 0:
            return super().sample(batch_size, env)
        regular_batch_size = batch_size - her_batch_size
        regular_samples = super().sample(regular_batch_size, env)
        her_samples = self.get_her_transitions_from_inds(her_batch_size, env_inds, eps_inds, env)
        return DictReplayBufferSamples(
            observations={"features": th.concat((regular_samples.observations["features"], her_samples.observations["features"])), "dfa": th.concat((regular_samples.observations["dfa"], her_samples.observations["dfa"]))},
            actions=th.concat((regular_samples.actions, her_samples.actions)),
            next_observations={"features": th.concat((regular_samples.next_observations["features"], her_samples.next_observations["features"])), "dfa": th.concat((regular_samples.next_observations["dfa"], her_samples.next_observations["dfa"]))},
            dones=th.concat((regular_samples.dones, her_samples.dones)),
            rewards=th.concat((regular_samples.rewards, her_samples.rewards)),
        )
        return regular_samples

