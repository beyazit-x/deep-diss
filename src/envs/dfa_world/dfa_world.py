from __future__ import annotations

import gym
import numpy as np

from itertools import product
from typing import Any, Literal, Optional, Union

import attr
import funcy as fn

class GridworldEnv(gym.Env):

    def __init__(self, propositions):
        """
            argument:
                -description
        """

        self.state_prop = None
        self.props = list(propositions)
        self.props.sort()

        self.action_space = gym.spaces.Discrete(len(self.props))
        self.actions = list(self.props)

    def step(self, action_idx):
        """
        This function executes an action in the environment
        """

        action = self.actions[action_idx]
        self.state_prop = action

        obs = None # no observations in this dummy environment
        reward = 0.0
        done = False

        return obs, reward, done, {}

    def seed(self, seed=None):
        np.random.seed(seed)

    def reset(self):
        """
        This function resets the world and collects the first observation.
        """

        self.state_prop = None

        return None

    def get_events(self):
        return self.state_prop

    def get_propositions(self):
        return self.props



class DFADummyEnv(GridworldEnv):
    def __init__(self):
        super().__init__(propositions=['a','b','c','d','e','f','g','h','i','j','k','l'])


