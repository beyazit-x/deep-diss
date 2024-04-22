import numpy as np
import gym
from gym import spaces
from copy import deepcopy
import random
import dfa_samplers
# from envs.safety.zones_env import zone
import networkx as nx
import pickle
from dfa import DFA
import math
from utils.parameters import GNN_EMBEDDING_SIZE

class PretrainEnv(gym.Wrapper):
    def __init__(self, env, mean=2, reject_reward=-1):
        super().__init__(env)
        self.propositions = self.env.get_propositions()

        self.observation_space = spaces.Dict({"features": env.observation_space,
                                              "reach_avoid" : spaces.Box(low=0, high=2, shape=(2*len(self.propositions),), dtype=np.bool8)})

        self.mean = mean
        self.reject_reward = reject_reward

    def sample(self):
        # sample from exponential distribution with mean 2 and truncate so the problem isn't impossible
        num_reach = np.round(np.clip(np.random.exponential(self.mean), 1, 3/4*len(self.propositions))).astype(int)
        num_avoid = np.round(np.clip(np.random.exponential(self.mean), 0, len(self.propositions) - num_reach)).astype(int)

        reach = np.random.choice(np.arange(len(self.propositions)), num_reach, replace=False)
        avoid = np.random.choice(list(set(np.arange(len(self.propositions))) - set(reach)), num_avoid, replace=False)

        return reach, avoid
    
    def encode(self, reach, avoid):
        reach_avoid = np.zeros(2*len(self.propositions), dtype=np.bool8)
        reach_avoid[reach] = 1
        reach_avoid[len(self.propositions)+avoid] = 1
        return reach_avoid
    
    def reset(self):
        obs = self.env.reset()
        self.reach_idxs, self.avoid_idxs = self.sample()
        self.reach_avoid_encoding = self.encode(self.reach_idxs, self.avoid_idxs)
        combined_obs = {"features": obs, "reach_avoid": self.reach_avoid_encoding}
        return combined_obs

    def step(self, action):
        next_obs, original_reward, env_done, env_info = self.env.step(action)

        event = self.get_events()
        event_index = self.propositions.index(event) if event in self.propositions else None

        if event_index in self.reach_idxs:
            prop_reward = 1.0
            prop_done = True
        elif event_index in self.avoid_idxs:
            prop_reward = self.reject_reward
            prop_done = True
        else:
            prop_reward = 0 
            prop_done = False

        dfa_obs = {"features": next_obs, "reach_avoid": self.reach_avoid_encoding}
        reward  = original_reward + prop_reward
        done    = env_done or prop_done

        return dfa_obs, reward, done, env_info


    def get_events(self):
        # This function must return the events that currently hold on the environment
        # NOTE: The events are represented by a string containing the propositions with positive values only (e.g., "ac" means that only propositions "a" and "b" hold)
        return self.env.get_events()

