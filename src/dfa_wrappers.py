import numpy as np
import gym
from gym import spaces
import random
from dfa_samplers import getDFASampler
from dfa.utils import min_distance_to_accept_by_state
from functools import reduce
import operator as OP

class DFAEnv(gym.Wrapper):
    def __init__(self, env, dfa_sampler=None, reject_reward=-1):
        super().__init__(env)
        self.propositions = self.env.get_propositions()
        self.sampler = getDFASampler(dfa_sampler, self.propositions)

        self.max_depth = 1_000_000

        self.N = self.sampler.get_size_bound()
        self.dfa_n_conjunctions = self.sampler.get_n_conjunctions()
        self.dfa_n_disjunctions = self.sampler.get_n_disjunctions()
        self.per_dfa_int_seq_size = self.N // (self.dfa_n_conjunctions * self.dfa_n_disjunctions)

        self.observation_space = spaces.Dict({"features": env.observation_space,
                                              "dfa"     : spaces.Box(low=0, high=9, shape=(self.N,), dtype=np.int64)})

        # self.observation_space = spaces.Dict({"features": env.observation_space,
        #                                       "dfa"     : spaces.MultiBinary(n=self.N)})

        self.dfa_goal = None # In CNF format, i.e., tuple of tuples
        self.dfa_goal_int_seq = None

        self.reject_reward = reject_reward

    def reset(self):
        self.obs = self.env.reset()
        self.dfa_goal = self.sampler.sample()
        self.dfa_goal_int_seq = self._to_int_seq(self.dfa_goal)
        dfa_obs = {"features": self.obs, "dfa": self.dfa_goal_int_seq}
        return dfa_obs

    def step(self, action):
        # executing the action in the environment
        next_obs, original_reward, env_done, info = self.env.step(action)

        # progressing the DFA formula
        truth_assignment = self.get_events()

        old_dfa_goal = self.dfa_goal
        self.dfa_goal = self._advance(self.dfa_goal, truth_assignment)
        self.obs      = next_obs

        if old_dfa_goal != self.dfa_goal:
            dfa_reward, dfa_done = self.get_dfa_reward(old_dfa_goal, self.dfa_goal)
            # dfa_reward, dfa_done = self.get_depth_reward(old_dfa_goal, self.dfa_goal)
            self.dfa_goal_int_seq = self._to_int_seq(self.dfa_goal)
        else:
            dfa_reward, dfa_done = 0.0, False

        dfa_obs = {"features": self.obs, "dfa": self.dfa_goal_int_seq}

        reward  = original_reward + dfa_reward
        done    = env_done or dfa_done

        assert dfa_reward >= -1 and dfa_reward <= 1
        assert dfa_reward !=  1 or dfa_done
        assert dfa_reward != -1 or dfa_done
        assert (dfa_reward <=  -1 or dfa_reward >= 1) or not dfa_done

        return dfa_obs, reward, done, info

    def step_given_obs(self, obs, action, time):
        raise NotImplemented

    def _to_monolithic_dfa(self, dfa_goal):
        return reduce(OP.and_, map(lambda dfa_clause: reduce(OP.or_, dfa_clause), dfa_goal))

    def get_dfa_reward(self, old_dfa_goal, dfa_goal):
        mono_dfa = self._to_monolithic_dfa(dfa_goal)
        if mono_dfa._label(mono_dfa.start):
            return 1.0, True
        if mono_dfa.find_word() is None:
            return -1.0, True
        return 0.0, False

    def min_distance_to_accept_by_state(self, dfa, state):
        depths = min_distance_to_accept_by_state(dfa)
        if state in depths:
            return depths[state]
        return self.max_depth

    def get_depth_reward(self, old_dfa_goal, dfa_goal):
        old_dfa = self._to_monolithic_dfa(old_dfa_goal).minimize()
        dfa = self._to_monolithic_dfa(dfa_goal).minimize()

        if dfa._label(dfa.start):
            return 1.0, True
        old_depth = self.min_distance_to_accept_by_state(old_dfa, old_dfa.start)
        depth = self.min_distance_to_accept_by_state(dfa, dfa.start)
        if depth == self.max_depth:
            return -1.0, True
        depth_reward = (old_depth - depth)/self.max_depth
        if depth_reward < 0:
            depth_reward *= 1_000
        if depth_reward > 1:
            return 1, True
        elif depth_reward < -1:
            return -1, True
        return depth_reward, False

    def _advance(self, dfa_goal, truth_assignment):
        return tuple(tuple(dfa.advance(truth_assignment).minimize() for dfa in dfa_clause) for dfa_clause in dfa_goal)

    def get_events(self):
        return self.env.get_events()

    def get_propositions(self):
        return self.env.get_propositions()

    def get_int_seq(self, dfa_int):
        int_seq = np.array([int(i) for i in str(dfa_int)])
        # int_seq = np.array([int(i) for i in str(bin(dfa_int)[2:])])
        return np.pad(int_seq, (self.per_dfa_int_seq_size - int_seq.shape[0], 0), "constant", constant_values=(0, 0))

    def get_dfa_from_int_seq(self, dfa_int_seq):
        dfa_int_str = "".join(str(int(i)) for i in dfa_int_seq.squeeze().tolist())
        dfa_int = int(dfa_int_str)
        # dfa_int = int(dfa_int_str, 2)
        dfa = DFA.from_int(dfa_int, self.propositions)
        return dfa

    def _from_int_seq(self, dfa_int_seq):
        return tuple(tuple(self.get_dfa_from_int_seq(dfa_int_seq) for dfa_int_seq in dfa_clause_int_seq) for dfa_clause_int_seq in dfa_int_seq.view(self.dfa_n_conjunctions, self.dfa_n_disjunctions, self.per_dfa_int_seq_size))

    def _to_int_seq(self, dfa_goal):
        seqs = []
        for dfa_clause in dfa_goal:
            for dfa in dfa_clause:
                seqs.append(self.get_int_seq(dfa.to_int()))
            for _ in range(self.dfa_n_disjunctions - len(dfa_clause)):
                seqs.append(np.zeros(self.per_dfa_int_seq_size))
        for _ in range(self.dfa_n_conjunctions - len(dfa_goal)):
            for _ in range(self.dfa_n_disjunctions):
                seqs.append(np.zeros(self.per_dfa_int_seq_size))
        return np.concatenate(seqs)

class NoDFAWrapper(gym.Wrapper):
    def __init__(self, env):
        """
        Removes the DFA formula from an DFAEnv
        It is useful to check the performance of off-the-shelf agents
        """
        super().__init__(env)
        self.observation_space = env.observation_space
        # self.observation_space =  env.observation_space['features']

    def reset(self):
        obs = self.env.reset()
        # obs = obs['features']
        # obs = {'features': obs}
        return obs

    def step(self, action):
        # executing the action in the environment
        obs, reward, done, info = self.env.step(action)
        # obs = obs['features']
        # obs = {'features': obs}
        return obs, reward, done, info

    def get_propositions(self):
        return list([])
