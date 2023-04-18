import sys

import numpy as np
import torch as th
import random

from dfa import DFA
from dfa.utils import dfa2dict
from dfa.utils import dict2dfa

from concept_class import PartialDFAIdentifier
from diss import LabeledExamples
from diss import diss
from diss.concept_classes import DFAConcept

from stable_baselines3.common.buffers import DictReplayBuffer
from stable_baselines3.common.type_aliases import DictReplayBufferSamples

from diss_interface import NNPlanner


DISS_ARGMAX = False
DISS_SOFTMAX_SAMPLE = not DISS_ARGMAX

class DissRelabeler():

    def __init__(self, model, env):
        self.model = model
        self.env = env
        self.propositions = env.get_propositions()
        self.replay_buffer = model.replay_buffer

        self.num_states_upper = env.num_states_upper

    def relabel(self, relabeler_name, batch_size):
        if relabeler_name == "diss":
            self.relabel_diss(batch_size)
        elif relabeler_name == "baseline":
            self.relabel_baseline(batch_size)
        else:
            raise NotImplemented

    def get_binary_seq(self, dfa):
        binary_string = bin(dfa.to_int())[2:]
        binary_seq = np.array([int(i) for i in binary_string])
        return np.pad(binary_seq, (self.env.N - binary_seq.shape[0], 0), 'constant', constant_values=(0, 0))

    def get_reward_and_done(self, dfa):
        start_state = dfa.start
        start_state_label = dfa._label(start_state)
        states = dfa.states()
        if start_state_label == True: # If starting state of dfa is accepting, then dfa_reward is 1.0.
            dfa_reward = 1.0
            dfa_done = 1.0
        elif len(states) == 1: # If starting state of dfa is rejecting and dfa has a single state, then dfa_reward is -1.0.
            dfa_reward = -1.0
            dfa_done = 1.0
        else:
            dfa_reward = 0.0 # If starting state of dfa is rejecting and dfa has a multiple states, then dfa_reward is 0.0.
            dfa_done = 0.0
        return dfa_reward, dfa_done

    def step_and_write_relabeled_dfas(self, relabeled_dfa_ints, samples):

        features, dfas = samples.observations["features"], samples.observations["dfa"]
        actions = samples.actions
        next_features, next_dfas = samples.next_observations["features"], samples.next_observations["dfa"]
        dones = samples.dones
        rewards = samples.rewards

        end_of_episode_inds, end_of_step_inds = dones.squeeze().nonzero()

        for init_dfa_int, end_of_episode_ind, end_of_step_ind in zip(relabeled_dfa_ints, end_of_episode_inds, end_of_step_inds):

            events = self.env.get_events_given_obss(features[end_of_episode_ind])

            dfa_int = init_dfa_int
            dfa = DFA.from_int(dfa_int, self.propositions)
            dfa_binary_seq = self.get_binary_seq(dfa)

            for step_ind in range(end_of_step_ind + 1):

                dfas[end_of_episode_ind][step_ind] = dfa_binary_seq

                dfa = dfa.advance(events[step_ind]).minimize()
                reward, done = self.get_reward_and_done(dfa)

                dones[end_of_episode_ind][step_ind] = done
                rewards[end_of_episode_ind][step_ind] = reward

                dfa_binary_seq = self.get_binary_seq(dfa)
                next_dfas[end_of_episode_ind][step_ind] = dfa_binary_seq

                if done: # It is guaranteed that the done signal will be 1 within the episode.
                    features[end_of_episode_ind][step_ind + 1:] = np.zeros(features[end_of_episode_ind][step_ind + 1:].shape)
                    dfas[end_of_episode_ind][step_ind + 1:] = np.zeros(dfas[end_of_episode_ind][step_ind + 1:].shape)
                    actions[end_of_episode_ind][step_ind + 1:] = np.zeros(actions[end_of_episode_ind][step_ind + 1:].shape)
                    next_features[end_of_episode_ind][step_ind + 1:] = np.zeros(next_features[end_of_episode_ind][step_ind + 1:].shape)
                    next_dfas[end_of_episode_ind][step_ind + 1:] = np.zeros(next_dfas[end_of_episode_ind][step_ind + 1:].shape)
                    dones[end_of_episode_ind][step_ind + 1:] = np.zeros(dones[end_of_episode_ind][step_ind + 1:].shape)
                    rewards[end_of_episode_ind][step_ind + 1:] = np.zeros(rewards[end_of_episode_ind][step_ind + 1:].shape)
                    break

    def relabel_baseline(self, batch_size):

        samples = self.replay_buffer.sample_traces(batch_size, self.model._vec_normalize_env)
        if samples is None:
            return

        features, dfas = samples.observations["features"], samples.observations["dfa"]
        actions = samples.actions
        next_features, next_dfas = samples.next_observations["features"], samples.next_observations["dfa"]
        dones = samples.dones
        rewards = samples.rewards

        end_of_episode_inds, end_of_step_inds = dones.squeeze().nonzero()

        relabeled_dfa_ints = []

        for end_of_episode_ind, end_of_step_ind in zip(end_of_episode_inds, end_of_step_inds):

            events = self.env.get_events_given_obss(features[end_of_episode_ind])
            events_clean = list(filter(lambda x: x != "", self.env.get_events_given_obss(features[end_of_episode_ind])))

            _, chain_length = self.env.sampler.get_concept_class()

            probs = np.array([1 for i in range(1, len(events_clean)+1)])
            probs = probs / np.sum(probs) # This is not geometric distribution, it is uniform

            rand_idxs = sorted(np.random.choice(range(len(events_clean)), size=chain_length, replace=False, p=probs))
            sampled_events = [events_clean[idx] for idx in rand_idxs]

            def transition(s, c):
                if s < chain_length and c == sampled_events[s]:
                    s = s + 1
                return s

            dfa = DFA(start=0,
                    inputs=self.propositions,
                    outputs={False, True},
                    label=lambda s: s == chain_length,
                    transition=transition)

            dfa_int = dfa.to_int()
            relabeled_dfa_ints.append(dfa_int)

        self.step_and_write_relabeled_dfas(relabeled_dfa_ints, samples)
        self.replay_buffer.relabel_traces(batch_size, samples)

    def relabel_diss(self, batch_size):

        planner = NNPlanner(self.env, self.model)

        universal = DFA(
            start=True,
            inputs=self.propositions,
            outputs={True, False},
            label=lambda s: s,
            transition=lambda s, c: True,
        )

        identifer = PartialDFAIdentifier( # possible change this identifier? to decomposed?
            partial = universal,
            base_examples = LabeledExamples(negative=[], positive=[]),
            try_reach_avoid=True, # TODO check this flag
            upperbound=self.num_states_upper,
        )

        n = batch_size
        samples = self.replay_buffer.sample_traces(n, self.model._vec_normalize_env) # This should also return actions
        if samples == None:
            return
        observations = samples.observations
        # shape of features is (n, 76, 7, 7, 13)
        features, dfas = observations["features"], observations["dfa"]
        # shape of actions is (n, 76, 1)
        actions = samples.actions

        relabeled_dfa_ints = []
        for feature, action in zip(features, actions):
            dfa_search = diss(
                demos=[planner.to_demo(feature, action)],
                to_concept=identifer,
                to_chain=planner.plan,
                competency=lambda *_: 10,
                lift_path=planner.lift_path,
                n_iters=100,
                reset_period=30,
                surprise_weight=1,
                size_weight=1/50,
                sgs_temp=1/4,
                example_drop_prob=1e-2, #1e-2,
                synth_timeout=20,
            )

            # sample as many DFAs as we can afford to without impacting the fps
            # maybe 10 to 100?
            dfa_sample_size = 10
            energies = []
            dfas = []
            """ take a hyperparameter number of dfas from dfa_search and then,
                    1) sample from metadata['energy'], or
                    2) take argmax over energy """
            for i, (data, concept, metadata) in zip(range(dfa_sample_size), dfa_search): 
                dfas.append(concept.dfa)
                energies.append(metadata['energy'])

            # print("SAMPLING")
            if DISS_ARGMAX:
                idx_min = np.argmin(energies)
                dfa_min = dfas[idx_min]
                relabeled_dfa = dfa_min
            elif DISS_SOFTMAX_SAMPLE:
                # print(energies)
                exp_energies = np.exp(energies) # TODO make this temperature tuneable
                likelihood = exp_energies / exp_energies.sum()
                # print(likelihood)
                softmax_sampled_dfa = np.random.choice(dfas, p=likelihood)
                # print("chose", dfas.index(softmax_sampled_dfa))
                relabeled_dfa = softmax_sampled_dfa
            # print('adding', relabeled_dfa)
            relabeled_dfa_int = relabeled_dfa.to_int()
            relabeled_dfa_ints.append(relabeled_dfa_int)

        self.step_and_write_relabeled_dfas(relabeled_dfa_ints, samples)
        self.replay_buffer.relabel_traces(batch_size, samples)

