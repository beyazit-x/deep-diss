import warnings
from abc import ABC, abstractmethod
from typing import Any, Dict, Generator, List, Optional, Union

import random
import numpy as np
import torch as th
from gym import spaces

from stable_baselines3.common.vec_env import VecNormalize
from stable_baselines3.common.buffers import DictReplayBuffer
from stable_baselines3.common.preprocessing import get_obs_shape
from stable_baselines3.common.type_aliases import (
    DictReplayBufferSamples,
    DictRolloutBufferSamples,
    ReplayBufferSamples,
    RolloutBufferSamples,
)

# TODO: How to delete stuff from her replay buffer if it is full?

class DissReplayBuffer(DictReplayBuffer):
    def __init__(
        self,
        buffer_size: int,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        device: Union[th.device, str] = "auto",
        n_envs: int = 1,
        optimize_memory_usage: bool = False,
        handle_timeout_termination: bool = True,
        max_episode_length: Optional[int] = None,
        her_replay_buffer_size: Optional[int] = None
    ):
        super().__init__(buffer_size=buffer_size,
            observation_space=observation_space,
            action_space=action_space,
            device=device,
            n_envs=n_envs,
            optimize_memory_usage=optimize_memory_usage,
            handle_timeout_termination=handle_timeout_termination
            )
        assert max_episode_length is not None
        assert her_replay_buffer_size is not None
        self.episode_positions = []
        self.is_first_obs = True
        self.episode_start_position = None
        self.her_replay_buffer_size = her_replay_buffer_size
        self.max_episode_length = max_episode_length
        self.n_envs = n_envs
        self.obs_shape = get_obs_shape(self.observation_space.spaces["features"])
        self.goal_shape = get_obs_shape(self.observation_space.spaces["dfa"])
        self.input_shape = {
            "observation": (self.her_replay_buffer_size, self.max_episode_length, self.n_envs,) + self.obs_shape,
            "achieved_goal": (self.her_replay_buffer_size, self.max_episode_length, self.n_envs,) + self.goal_shape,
            "desired_goal": (self.her_replay_buffer_size, self.max_episode_length, self.n_envs,) + self.goal_shape,
            "action": (self.her_replay_buffer_size, self.max_episode_length, self.n_envs,) + (self.action_dim,),
            "reward": (self.her_replay_buffer_size, self.max_episode_length, self.n_envs,) + (1,),
            "next_observation": (self.her_replay_buffer_size, self.max_episode_length, self.n_envs,) + self.obs_shape,
            "next_achieved_goal": (self.her_replay_buffer_size, self.max_episode_length, self.n_envs,) + self.goal_shape,
            "next_desired_goal": (self.her_replay_buffer_size, self.max_episode_length, self.n_envs,) + self.goal_shape,
            "done": (self.her_replay_buffer_size, self.max_episode_length, self.n_envs,) + (1,),
            "relabeled": (self.her_replay_buffer_size, self.n_envs)
        }
        self.her_replay_buffer = {
            key: np.zeros(dim, dtype=np.float32) if key != "action" else np.zeros(dim, dtype=np.int64)
            for key, dim in self.input_shape.items()
        }
        self.current_episode_idx = 0
        self.current_episode_step_idx = 0
        self.indices_in_progress = []
        self.her_ratio = 0.1
        self.my_full = False

    def add(
        self,
        obs: Dict[str, np.ndarray],
        next_obs: Dict[str, np.ndarray],
        action: np.ndarray,
        reward: np.ndarray,
        done: np.ndarray,
        infos: List[Dict[str, Any]],
    ) -> None:
        if not self.my_full:
            episode_indices_in_progress = [ind for ind, _ in self.indices_in_progress]
            if len(episode_indices_in_progress) < self.her_replay_buffer_size * self.n_envs:
                while self.current_episode_idx in episode_indices_in_progress:
                    self.current_episode_idx = (self.current_episode_idx + 1) % self.her_replay_buffer_size
                self.her_replay_buffer["observation"][self.current_episode_idx][self.current_episode_step_idx] = np.array(obs["features"]).copy()
                self.her_replay_buffer["achieved_goal"][self.current_episode_idx][self.current_episode_step_idx] = np.array(obs["dfa"]).copy()
                self.her_replay_buffer["desired_goal"][self.current_episode_idx][self.current_episode_step_idx] = np.array(obs["dfa"]).copy()
                self.her_replay_buffer["action"][self.current_episode_idx][self.current_episode_step_idx] = np.array(action).copy()
                self.her_replay_buffer["reward"][self.current_episode_idx][self.current_episode_step_idx] = np.array(reward).copy()
                self.her_replay_buffer["next_observation"][self.current_episode_idx][self.current_episode_step_idx] = np.array(next_obs["features"]).copy()
                self.her_replay_buffer["next_achieved_goal"][self.current_episode_idx][self.current_episode_step_idx] = np.array(next_obs["dfa"]).copy()
                self.her_replay_buffer["next_desired_goal"][self.current_episode_idx][self.current_episode_step_idx] = np.array(next_obs["dfa"]).copy()
                self.her_replay_buffer["done"][self.current_episode_idx][self.current_episode_step_idx] = np.array(done).copy()
                self.her_replay_buffer["relabeled"][self.current_episode_idx] = np.ones((self.n_envs,))
                # if self.current_episode_idx == 5 or self.current_episode_idx == 7: # This is just for testing
                #     self.her_replay_buffer["relabeled"][self.current_episode_idx] = np.ones((self.n_envs,))
                self.current_episode_step_idx += 1
                if done or self.current_episode_step_idx >= self.max_episode_length:
                    self.current_episode_idx = (self.current_episode_idx + 1) % self.her_replay_buffer_size
                    self.current_episode_step_idx = 0
        if self.current_episode_idx == self.her_replay_buffer_size - 1:
            self.my_full = True
        super().add(obs, next_obs, action, reward, done, infos)

    def sample_traces(self, n):
        try:
            indices = np.argwhere(self.her_replay_buffer["relabeled"] <= 0).tolist()
            self.indices_in_progress = random.sample(indices, n)
            return [(sampled_episode_ind, sampled_env_ind, self.her_replay_buffer["observation"][sampled_episode_ind, :, sampled_env_ind]) for sampled_episode_ind, sampled_env_ind in self.indices_in_progress], [(sampled_episode_ind, sampled_env_ind, self.her_replay_buffer["desired_goal"][sampled_episode_ind, :, sampled_env_ind]) for sampled_episode_ind, sampled_env_ind in self.indices_in_progress]
        except:
            return None, None

    def relabel_traces(self, achieved_goals):
        relabeled_indices = []
        for sampled_episode_ind, sampled_env_ind, achieved_goal in achieved_goals:
            self.her_replay_buffer["achieved_goal"][sampled_episode_ind, :, sampled_env_ind] = achieved_goal
            self.her_replay_buffer["relabeled"][sampled_episode_ind, sampled_env_ind] = 1
            relabeled_indices.append([sampled_episode_ind, sampled_env_ind])
        self.indices_in_progress = list(filter(lambda ind: ind not in relabeled_indices, self.indices_in_progress))

    def get_her_samples(self, batch_size, env):
        temp = np.argwhere(self.her_replay_buffer["relabeled"] > 0).tolist()
        indices = random.sample(temp, batch_size)

        batch_inds = [i for i, _ in indices]
        env_indices = [j for _, j in indices]
        obs_ = self._normalize_obs({"features": self.her_replay_buffer["observation"][batch_inds, env_indices, :].squeeze(axis=1), "dfa": self.her_replay_buffer["achieved_goal"][batch_inds, env_indices, :].squeeze(axis=1)}, env)
        next_obs_ = self._normalize_obs({"features": self.her_replay_buffer["next_observation"][batch_inds, env_indices, :].squeeze(axis=1), "dfa": self.her_replay_buffer["next_achieved_goal"][batch_inds, env_indices, :].squeeze(axis=1)}, env)

        observations = {key: self.to_torch(obs) for key, obs in obs_.items()}
        next_observations = {key: self.to_torch(obs) for key, obs in next_obs_.items()}

        return DictReplayBufferSamples(
            observations=observations,
            actions=self.to_torch(self.actions[batch_inds, env_indices]),
            next_observations=next_observations,
            # Only use dones that are not due to timeouts
            # deactivated by default (timeouts is initialized as an array of False)
            dones=self.to_torch(self.dones[batch_inds, env_indices] * (1 - self.timeouts[batch_inds, env_indices])).reshape(
                -1, 1
            ),
            rewards=self.to_torch(self._normalize_reward(self.rewards[batch_inds, env_indices].reshape(-1, 1), env)),
        )

    def sample(self, batch_size: int, env: Optional[VecNormalize] = None) -> DictReplayBufferSamples:
        # print(self.current_episode_idx)
        her_batch_size = int(self.her_ratio * batch_size) if self.my_full else 0
        regular_batch_size = batch_size - her_batch_size
        regular_samples = super().sample(regular_batch_size, env)
        # print(regular_samples)
        if her_batch_size > 0:
            her_samples = self.get_her_samples(her_batch_size, env)
            # print(regular_samples.observations["features"].shape, her_samples.observations["features"].shape)
            # print(regular_samples.observations["dfa"].shape, her_samples.observations["dfa"].shape)
            # print(regular_samples.next_observations["features"].shape, her_samples.next_observations["features"].shape)
            # print(regular_samples.next_observations["dfa"].shape, her_samples.next_observations["dfa"].shape)
            # print(regular_samples.dones.shape, her_samples.dones.shape)
            # print(regular_samples.rewards.shape, her_samples.rewards.shape)
            # input(">>>>")
            # return {key: th.cat(regular_samples[key], her_samples[key]) for key in self.input_shape.keys()}
            return DictReplayBufferSamples(
                observations={"features": th.concat((regular_samples.observations["features"], her_samples.observations["features"])), "dfa": th.concat((regular_samples.observations["dfa"], her_samples.observations["dfa"]))},
                actions=th.concat((regular_samples.actions, her_samples.actions)),
                next_observations={"features": th.concat((regular_samples.next_observations["features"], her_samples.next_observations["features"])), "dfa": th.concat((regular_samples.next_observations["dfa"], her_samples.next_observations["dfa"]))},
                # Only use dones that are not due to timeouts
                # deactivated by default (timeouts is initialized as an array of False)
                dones=th.concat((regular_samples.dones, her_samples.dones)),
                rewards=th.concat((regular_samples.rewards, her_samples.rewards)),
            )
        return regular_samples



