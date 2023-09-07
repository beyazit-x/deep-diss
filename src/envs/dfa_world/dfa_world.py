from __future__ import annotations

import gym
import numpy as np

from itertools import product
from typing import Any, Literal, Optional, Union

import attr
import funcy as fn

class DummyEnv(gym.Env):

    def __init__(self, propositions):
        """
            argument:
                -description
        """

        self.props = list(propositions)
        self.props.sort()
        self.init_state = len(self.props)
        self.state_prop = self.init_state

        self.observation_space = gym.spaces.Discrete(len(self.props)+1)
        self.action_space = gym.spaces.Discrete(len(self.props))
        self.actions = list(self.props)
        self.timeout = 10
        self.time = 0
        self.num_episodes = 0

    def step(self, action_idx):
        """
        This function executes an action in the environment
        """

        action = self.actions[action_idx]
        self.state_prop = action_idx

        obs = action_idx # no observations in this dummy environment
        self.time += 1
        done = self.time > self.timeout
        reward = 0.0

        return obs, reward, done, {}

    def seed(self, seed=None):
        np.random.seed(seed)

    def reset(self):
        """
        This function resets the world and collects the first observation.
        """

        self.state_prop = self.init_state
        self.time = 0
        self.num_episodes += 1

        return self.state_prop

    def get_events(self):
        return self.props[self.state_prop]

    def get_events_given_obss(self, obss):
        return [self.get_events_given_obs(obs) for obs in obss]

    def get_events_given_obs(self, obs):
        obs = obs.squeeze()
        if obs == len(self.props):
            return ""
        else:
            return self.props[int(obs)]

    def get_propositions(self):
        return self.props



class DFADummyEnv(DummyEnv):
    def __init__(self):
        super().__init__(propositions=['a','b','c','d','e','f','g','h','i','j','k','l'])


