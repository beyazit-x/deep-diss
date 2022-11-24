import warnings
from abc import ABC, abstractmethod
from typing import Any, Dict, Generator, List, Optional, Union

import random
import numpy as np
import torch as th
from gym import spaces

from stable_baselines3.common.buffers import DictReplayBuffer

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
    ):
        super().__init__(buffer_size=buffer_size,
            observation_space=observation_space,
            action_space=action_space,
            device=device,
            n_envs=n_envs,
            optimize_memory_usage=optimize_memory_usage,
            handle_timeout_termination=handle_timeout_termination
            )
        self.episode_positions = []
        self.is_first_obs = True
        self.episode_start_position = None

    def _position_filter(self, positions):
        if positions[0] < positions[1]:
            return not (positions[0] <= self.pos and self.pos <= positions[1])
        else:
            return not ((positions[0] <= self.pos and self.pos <= self.buffer_size) or (0 <= self.pos and self.pos <= positions[1]))

    def add(
        self,
        obs: Dict[str, np.ndarray],
        next_obs: Dict[str, np.ndarray],
        action: np.ndarray,
        reward: np.ndarray,
        done: np.ndarray,
        infos: List[Dict[str, Any]],
    ) -> None:
        self.episode_positions = list(filter(self._position_filter, self.episode_positions))
        if self.episode_start_position is None:
            self.episode_start_position = self.pos
        else:
            if done:
                self.episode_positions.append((self.episode_start_position, self.pos))
                self.episode_start_position = None
            else:
                pass
        super().add(obs, next_obs, action, reward, done, infos)

    def sample_traces(self, n):
        sample_positions = random.sample(self.episode_positions, n)
        # TODO: format traces so that we can easily call DISS on them.
        # sample_positions has the indices for traces. Below is a simple
        # check for the consistency of the replay buffer.
        # TODO: Answer the following question: "when we are running
        # multiple environment in parallel, can we still assume that
        # traces are sequentially positioned in the buffer?"
        # The reason why we are keeping indices instead of a matrix
        # for traces is that storing the same data twice might be too costly.
        # Note that super().add(...) copies observations etc.; therefore,
        # we cannot have a trace buffer pointing to the same tensors
        # as the replay buffer. This implies that we have to do the position
        # bookkeeping if we want to change the goals (i.e., DFAs) in
        # the original replay buffer. As an alternative to in-place relabelling,
        # we can also consider adding relabeled traces to the buffer as new traces.
        # start_pos, end_pos = sample_positions[0]
        # print(start_pos-1, self.observations["features"][start_pos-1][0].transpose(2, 0, 1)[0], self.dones[start_pos-1])
        # for i in range(start_pos, end_pos + 2):
        #     print(i, self.observations["features"][i][0].transpose(2, 0, 1)[0], self.dones[i])
        #     input("->")






