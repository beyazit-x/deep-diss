from dfa import DFA
from diss.annotated_mc import AnnotatedMarkovChain
from diss import Edge, Node, SampledPath
from diss import DemoPrefixTree as PrefixTree

import torch
import numpy as np

import gym
from stable_baselines3 import DQN



class NNPolicyWrapper:


    def __init__(self, policy: DQN, dfa_goal: DFA, env: gym.Env):
        self.policy = policy
        self.dfa_goal = dfa_goal
        self.env = env
        self.N = 1000

    def policy_probability(self, feature, a):
        obs = self.feature2obs(feature)
        print(self.policy.policy.q_net(obs))
        return self.policy.policy.q_net(obs)[a]

    def transition_probability(self, feature, a, next_feature):
        return self.env.transition_probability(feature, a, next_feature)

    def feature2obs(self, feature):
        bin_seq = self.get_binary_seq(self.dfa_goal)
        print(torch.unsqueeze(torch.from_numpy(bin_seq), dim=0).shape)
        return {'features': torch.unsqueeze(torch.from_numpy(feature), dim=0), 'dfa': torch.unsqueeze(torch.from_numpy(bin_seq), dim=0)}

    def get_binary_seq(self, dfa):
        binary_string = bin(dfa.to_int())[2:]
        binary_seq = np.array([int(i) for i in binary_string])
        return np.pad(binary_seq, (self.N - binary_seq.shape[0], 0), 'constant', constant_values=(0, 0))



class NNPlanner:

    def __init__(self, env, policy):
        self.env = env
        self.policy = policy
        self.obs_shape = None
        self.act_shape = None
        self.obs_type = None
        self.act_type = None

    def plan(self, dfa_concept, tree, rationality=None):

        policy_wrapper = NNPolicyWrapper(self.policy, dfa_concept.dfa, self.env)

        return NNMarkovChain(tree, policy_wrapper, dfa_concept.dfa, self.obs_shape, self.act_shape, self.obs_type, self.act_type)

    def to_demo(self, obs_trc, act_trc):
        # start_action, *trc = trc

        if self.obs_shape == None:
            self.obs_shape = obs_trc.shape[1:]
        if self.act_shape == None:
            self.act_shape = act_trc.shape[1:]
        if self.act_type == None:
            self.act_type = act_trc.dtype
        if self.obs_type == None:
            self.obs_type = obs_trc.dtype

        # demo = [ (None, 'env'), (start_action, 'ego')]
        demo = []
        for prev_obs, act, obs in zip(obs_trc, act_trc, obs_trc[1:]):
            byte_act = act.tobytes()
            byte_obs = obs.tobytes()
            byte_prev_obs = prev_obs.tobytes()
            # alternate env and ego
            # env -> include what action was taken and what the previous state was - information needed to compute transition probability
            # ego -> just care about the action - information needed to compute policy probability
            demo.extend([
                ((byte_prev_obs, byte_act), 'env'), # TODO double check that we can't simplify the information that needs to be here
                (byte_obs, 'ego')
                ])
        return demo

    def lift_path(self, path):
        env.lift_path(path)

class NNMarkovChain(AnnotatedMarkovChain):

    def __init__(self, tree: PrefixTree, policy: NNPolicyWrapper, dfa_goal: DFA, obs_shape, act_shape, obs_type, act_type):
        self.tree = tree
        self.policy = policy
        self.dfa_goal = dfa_goal
        self.obs_shape = obs_shape
        self.act_shape = act_shape
        self.obs_type = obs_type
        self.act_type = act_type


    def bytes2obs(self, byte):
        return np.frombuffer(byte, dtype=self.obs_type).reshape(self.obs_shape)
    def bytes2act(self, byte):
        return np.frombuffer(byte, dtype=self.act_type).reshape(self.act_shape)


    @property
    def edge_probs(self) -> dict[Edge, float]:
        """Returns the probablity of edges in the demo prefix tree."""
        edge_probs = {}
        for tree_edge in self.tree.tree.edges:
            v,w = tree_edge
            if self.tree.is_ego(v):
                byte_obs = self.tree.state(v) # this is obs
                obs = self.bytes2obs(byte_obs)
                _, next_byte_act = self.tree.state(w) # this is (obs, next_act)
                next_act = self.bytes2act(next_byte_act)

                edge_probs[tree_edge] = self.policy.policy_probability(obs, next_act[0])
            else:
                prev_byte_obs, byte_act = self.tree.state(v) # this is (prev_obs, act)
                prev_obs = self.bytes2obs(prev_byte_obs)
                act = self.bytes2act(byte_act)
                byte_obs = self.tree.state(w) # this is obs
                obs = self.bytes2obs(byte_obs)
                edge_probs[tree_edge] = self.policy.transition_probability(prev_obs, act[0], obs)

        return edge_probs

    def sample(self, pivot: Node, win: bool) -> SampledPath:
        """Sample a path conditioned on pivot and win.
        Arguments:
          - pivot: Last node in the prefix tree that the sampled path 
                   passes through.
          - win: Determines if sampled path results in ego winning.
        Returns:
           A path and corresponding log probability of sample the path OR
           None, if sampling from the empty set, e.g., want to sample
           an ego winning path, but no ego winning paths exist that pass
           through the pivot.
        """
        ATTEMPTS = 100
        for i in range(ATTEMPTS):
            result = self._sample(pivot, win)
            if result is None:
                return result
            # TODO ensure that the trace is correctly winning or losing here, otherwise, continue
            else:
                return result


    def _sample(self, pivot, win):
        assert pivot > 0

        # sample_lprob: float = 0  # Causally conditioned logprob.
        # path = list(self.tree.prefix(pivot))
        # if policy.end_of_episode(state):
        #      moves = []
        # else:
        #     prev_ego = isinstance(state[-1], str)

        #     actions = {0, 1, 3} if prev_ego else set(GW.dynamics.ACTIONS_C)
        #     neighbors = self.tree.tree.neighbors(pivot)
        #     neighbor_actions = {self.tree2policy[s][-1] for s in neighbors}
        #     actions -= neighbor_actions

        #     # HACK: Don't allow eoe counter-factuals!
        #     if prev_ego and (neighbor_actions <= {3, 4}):
        #         return None

        #     tmp = {policy.transition(state, a) for a in actions}

        #     moves = list(m for m in tmp if policy.psat(m) != float(not win))

        # if not moves:
        #     return None  # Couldn't deviate

        if win:
            # just sample from NN controller directly
            return self.sample_positive_path(pivot)
        else:
            return self.sample_negative_path(pivot)

        
        # # Sample suffix to path conditioned on win.
        # while moves:
        #     # Apply bayes rule to get Pr(s' | is_sat, s).
        #     priors = np.array([policy.prob(state, m) for m in moves])
        #     likelihoods = np.array([policy.psat(m) for m in moves])
        #     normalizer = policy.psat(state)

        #     if not win:
        #         likelihoods = 1 - likelihoods
        #         normalizer = 1 - normalizer

        #     probs =  priors * likelihoods / normalizer
        #     try:
        #         prob, state = random.choices(list(zip(probs, moves)), probs)[0]
        #     except:
        #         return None  # Numerical stability problem!

        #     if policy.policy.dag.nodes[state[0]]['kind'] == 'ego':
        #         sample_lprob += np.log(prob)

        #     # Note: win/lose are strings so the below still works...
        #     action = state[-1]
        #     path.append(action)

        #     if policy.end_of_episode(state):
        #         moves = []
        #     else:
        #         prev_ego = isinstance(action, str)
        #         # TODO: update to include ending episode action.
        #         actions = {0, 1, 3} if prev_ego else set(GW.dynamics.ACTIONS_C)
        #         moves = [policy.transition(state, a) for a in actions]
        # return path, sample_lprob


    def sample_positive_path(self, pivot):
        # build the initial observation
        feature = self.tree.state(pivot)
        initial_obs = {'features': feature, 'dfa': self.dfa_goal}

        path = list(self.tree.prefix(pivot))
        while not done:
            action = self.policy.predict(obs)
            obs, reward, done, info = env.step(action)
            path.append(action)

        return path, None

    def sample_negative_path(self, pivot):
        # build the initial observation
        feature = self.tree.state(pivot)
        negated_dfa_goal = ~self.dfa_goal
        initial_obs = {'features': feature, 'dfa': negated_dfa_goal}

        path = list(self.tree.prefix(pivot))
        while not done:
            action = self.policy.predict(obs)
            obs, reward, done, info = env.step(action)
            path.append(action)

        return path, None


