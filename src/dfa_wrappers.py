import numpy as np
import gym
from gym import spaces
from copy import deepcopy
import random
import dfa_samplers
from envs.safety.zones_env import zone
import networkx as nx
import pickle

class DFAEnv(gym.Wrapper):
    def __init__(self, env, dfa_sampler=None):
        super().__init__(env)
        self.propositions = self.env.get_propositions()
        self.sampler = dfa_samplers.getDFASampler(dfa_sampler, self.propositions)

        self.N = 1000 # TODO compute this

        self.observation_space = spaces.Dict({"features": env.observation_space,
                                              "dfa"     : spaces.MultiBinary(self.N)})

        self.dfa_goal = None
        self.dfa_goal_binary_seq = None

    def sample_dfa_goal(self):
        # This function must return a DFA for a task.
        # Format: networkx graph
        # NOTE: The propositions must be represented by a char
        # dfa = self.sampler.sample()
        from dfa import DFA
        dfa = self.sampler.sample_dfa_formula()
        # print("DFA", type(dfa), dfa)
        return dfa

    def get_events(self, obs, act, next_obs):
        # This function must return the events that currently hold on the environment
        # NOTE: The events are represented by a string containing the propositions with positive values only (e.g., "ac" means that only propositions 'a' and 'b' hold)
        return self.env.get_events()

    def get_binary_seq(self, dfa):
        binary_string = bin(dfa.to_int())[2:]
        binary_seq = np.array([int(i) for i in binary_string])
        return np.pad(binary_seq, (self.N - binary_seq.shape[0], 0), 'constant', constant_values=(0, 0))

    def reset(self):
        self.known_progressions = {}
        self.obs = self.env.reset()
        self.dfa_goal = self.sample_dfa_goal()
        self.dfa_goal_binary_seq = self.get_binary_seq(self.dfa_goal)
        dfa_obs = {"features": self.obs, "dfa": self.dfa_goal_binary_seq}
        return dfa_obs

    def get_reward_and_done(self):
        dfa_reward = 0.0
        dfa_done = False

        start_state = self.dfa_goal.start
        start_state_label = self.dfa_goal._label(start_state)
        states = self.dfa_goal.states()
        if start_state_label == True: # If starting state of self.dfa_goal is accepting, then dfa_reward is 1.0.
            dfa_reward = 1.0
            dfa_done = True
        elif len(states) == 1: # If starting state of self.dfa_goal is rejecting and self.dfa_goal has a single state, then dfa_reward is -1.0.
            dfa_reward = -1.0
            dfa_done = True
        else:
            dfa_reward = 0.0 # If starting state of self.dfa_goal is rejecting and self.dfa_goal has a multiple states, then dfa_reward is 0.0.
            dfa_done = False
        return dfa_reward, dfa_done


    def step(self, action):
        # executing the action in the environment

        next_obs, original_reward, env_done, info = self.env.step(action)

        truth_assignment = self.get_events(self.obs, action, next_obs)
        next_dfa_goal = self.dfa_goal.advance(truth_assignment)
        if next_dfa_goal != self.dfa_goal:
            next_dfa_goal = next_dfa_goal.minimize()
            self.dfa_goal = next_dfa_goal
            self.dfa_goal_binary_seq = self.get_binary_seq(self.dfa_goal)

        dfa_reward, dfa_done = self.get_reward_and_done()

        self.obs = next_obs

        dfa_obs = {'features': self.obs, 'dfa': self.dfa_goal_binary_seq}

        reward  = original_reward + dfa_reward
        done    = env_done or dfa_done

        return dfa_obs, reward, done, info
