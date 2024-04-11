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

class DFAEnv(gym.Wrapper):
    def __init__(self, env, dfa_sampler=None, reject_reward=-1):
        super().__init__(env)
        self.propositions = self.env.get_propositions()
        self.sampler = dfa_samplers.getDFASampler(dfa_sampler, self.propositions)

        self.N = self.sampler.get_size_bound()
        self.dfa_n_conjunctions = self.sampler.get_n_conjunctions()
        self.dfa_n_disjunctions = self.sampler.get_n_disjunctions()
        self.per_dfa_bin_size = self.N // (self.dfa_n_conjunctions * self.dfa_n_disjunctions)

        self.observation_space = spaces.Dict({"features": env.observation_space,
                                              "dfa"     : spaces.MultiBinary(self.N)})

        self.dfa_goal = None # In CNF format, i.e., tuple of tuples
        self.dfa_goal_binary_seq = None

        self.reject_reward = reject_reward

    def _to_bin(self, dfa_goal):
        seqs = []
        for dfa_clause in dfa_goal:
            for dfa in dfa_clause:
                seqs.append(self.get_binary_seq(dfa.to_int()))
            for _ in range(self.dfa_n_disjunctions - len(dfa_clause)):
                seqs.append(np.zeros(self.per_dfa_bin_size))
        for _ in range(self.dfa_n_conjunctions - len(dfa_goal)):
            for _ in range(self.dfa_n_disjunctions):
                seqs.append(np.zeros(self.per_dfa_bin_size))
        return np.concatenate(seqs)

    def _advance(self, dfa_goal, truth_assignment):
        return tuple(tuple(dfa.advance(truth_assignment) for dfa in dfa_clause) for dfa_clause in dfa_goal)

    def _minimize(self, dfa_goal):
        return tuple(tuple(dfa.minimize() for dfa in dfa_clause) for dfa_clause in dfa_goal)

    def reset(self):
        self.obs = self.env.reset()
        self.dfa_goal = self.sampler.sample()
        self.dfa_goal_binary_seq = self._to_bin(self.dfa_goal)
        dfa_obs = {"features": self.obs, "dfa": self.dfa_goal_binary_seq}
        return dfa_obs

    def step(self, action):
        next_obs, original_reward, env_done, info = self.env.step(action)

        truth_assignment = self.get_events(self.obs, action, next_obs)
        next_dfa_goal = self._advance(self.dfa_goal, truth_assignment)

        if next_dfa_goal != self.dfa_goal:
            self.dfa_goal = self._minimize(next_dfa_goal)
            dfa_reward, dfa_done, self.dfa_goal = self.get_dfa_goal_reward_and_done(self.dfa_goal)
            self.dfa_goal_binary_seq = self._to_bin(self.dfa_goal)
        else:
            dfa_reward, dfa_done = 0.0, False

        self.obs = next_obs

        dfa_obs = {'features': self.obs, 'dfa': self.dfa_goal_binary_seq}

        reward  = original_reward + dfa_reward
        done    = env_done or dfa_done

        return dfa_obs, reward, done, info


    def step_given_obs(self, obs, action, time):

        # advance the environment
        next_feature, original_reward, env_done, info = self.env.step_from_obs(obs["features"], action, time)

        # advance the dfa
        dfa_bin_seq = obs["dfa"]
        dfa_goal = self._from_bin(dfa_bin_seq)
        truth_assignment = self.env.get_events_given_obs(next_feature)
        next_dfa_goal = self._advance(dfa_goal, truth_assignment)

        if next_dfa_goal != dfa_goal:
            next_dfa_goal = self._minimize(next_dfa_goal)
            dfa_reward, dfa_done, next_dfa_goal = self.get_dfa_goal_reward_and_done(next_dfa_goal)
            next_dfa_bin_seq = np.expand_dims(self._to_bin(next_dfa_goal), axis=0)
        else:
            next_dfa_bin_seq = dfa_bin_seq
            dfa_reward, dfa_done = 0.0, False

        next_obs = {'features': next_feature, 'dfa': next_dfa_bin_seq}

        reward  = original_reward + dfa_reward
        done    = env_done or dfa_done

        return next_obs, reward, done, info

    def get_dfa_goal_reward_and_done(self, dfa_goal):
        dfa_clause_rewards = []
        dfa_clause_dones = []
        dfa_clause_actives = []
        for dfa_clause in dfa_goal:
            dfa_clause_reward, dfa_clause_done, dfa_clause = self.get_dfa_clause_reward_and_done(dfa_clause)
            dfa_clause_rewards.append(dfa_clause_reward)
            dfa_clause_dones.append(dfa_clause_done)
            if not dfa_clause_done:
                dfa_clause_actives.append(dfa_clause)
        return min(dfa_clause_rewards), all(dfa_clause_dones), tuple(dfa_clause_actives)

    def get_dfa_clause_reward_and_done(self, dfa_clause):
        dfa_rewards = []
        dfa_dones = []
        dfa_actives = []
        for dfa in dfa_clause:
            dfa_reward, dfa_done = self.get_dfa_reward_and_done(dfa)
            dfa_rewards.append(dfa_reward)
            dfa_dones.append(dfa_done)
            if not dfa_done:
                dfa_actives.append(dfa)
        return max(dfa_rewards), any(dfa_dones), tuple(dfa_actives)

    def get_dfa_reward_and_done(self, dfa):
        start_state = dfa.start
        start_state_label = dfa._label(start_state)
        states = dfa.states()

        if start_state_label == True: # If starting state of dfa is accepting, then dfa_reward is 1.0.
            dfa_reward = 1.0
            dfa_done = True
        elif len(states) == 1: # If starting state of dfa is rejecting and self.dfa_goal has a single state, then dfa_reward is reject_reward.
            dfa_reward = self.reject_reward # Or maybe 0.0
            dfa_done = True
        else:
            dfa_reward = 0.0 # If starting state of dfa is rejecting and self.dfa_goal has a multiple states, then dfa_reward is 0.0.
            dfa_done = False

        return dfa_reward, dfa_done

    def get_events(self, obs, act, next_obs):
        # This function must return the events that currently hold on the environment
        # NOTE: The events are represented by a string containing the propositions with positive values only (e.g., "ac" means that only propositions 'a' and 'b' hold)
        return self.env.get_events()

    def get_binary_seq(self, dfa_int):
        binary_string = bin(dfa_int)[2:]
        binary_seq = np.array([int(i) for i in binary_string])
        return np.pad(binary_seq, (self.per_dfa_bin_size - binary_seq.shape[0], 0), 'constant', constant_values=(0, 0))

    def get_dfa_from_bin(self, dfa_binary_seq):
        dfa_binary_str = "".join(str(int(i)) for i in dfa_binary_seq.squeeze().tolist())
        dfa_int = int(dfa_binary_str, 2)
        dfa = DFA.from_int(dfa_int, self.propositions)
        return dfa

    def _from_bin(self, dfa_binary_seq):
        dfa_binary_seq.reshape(self.dfa_n_conjunctions, self.dfa_n_disjunctions, self.per_dfa_bin_size)
        return tuple(tuple(get_dfa_from_bin(dfa_bin) for dfa_bin in dfa_clause_bin) for dfa_clause_bin in dfa_binary_seq)
