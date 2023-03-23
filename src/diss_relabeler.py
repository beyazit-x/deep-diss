import sys

import numpy as np
import torch as th
import random

from dfa import DFA
from dfa.utils import dfa2dict
from dfa.utils import dict2dfa

from diss.experiment import PartialDFAIdentifier
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
        self.N = 80 # TODO get this from dfa_wrapper

    def get_state_signature(self, state_id, dfa_dict):
        state = dfa_dict[state_id]
        state_signature = ""
        other_seen_state_ids = []
        state_transitions = state[1]
        for char in state_transitions:
            next_state_id = state_transitions[char]
            if state_id == next_state_id:
                state_signature += "0"
            else:
                if next_state_id not in other_seen_state_ids:
                    other_seen_state_ids.append(next_state_id)
                state_signature += str(other_seen_state_ids.index(next_state_id) + 1)
        return state_signature

    def get_binary_seq(self, dfa):
        binary_string = bin(dfa.to_int())[2:]
        binary_seq = np.array([int(i) for i in binary_string])
        return np.pad(binary_seq, (self.N - binary_seq.shape[0], 0), 'constant', constant_values=(0, 0))

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

        relabeled_dfas = []
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

            dfa_sample_size = 10
            energies = []
            dfas = []
            """ take a hyperparameter number of dfas from dfa_search and then,
                    1) sample from metadata['energy'], or
                    2) take argmax over energy """
            for i, (data, concept, metadata) in zip(range(dfa_sample_size), dfa_search): 
                dfas.append(concept.dfa)
                energies.append(metadata['energy'])

            print("SAMPLING")
            if DISS_ARGMAX:
                idx_min = np.argmin(energies)
                dfa_min = dfas[idx_min]
                relabeled_dfa = dfa_min
            elif DISS_SOFTMAX_SAMPLE:
                print(energies)
                exp_energies = np.exp(energies) # TODO make this temperature tuneable
                likelihood = exp_energies / exp_energies.sum()
                print(likelihood)
                softmax_sampled_dfa = np.random.choice(dfas, p=likelihood)
                print("chose", dfas.index(softmax_sampled_dfa))
                relabeled_dfa = softmax_sampled_dfa
            print('adding', relabeled_dfa) 
            relabeled_dfas.append(relabeled_dfa)

        # TODO use relabeled_dfas to rewrite the buffer here


    def relabel_baseline(self, batch_size):
        n = batch_size

        samples = self.replay_buffer.sample_traces(n, self.model._vec_normalize_env) # This should also return actions
        if samples is None:
            return
        # print(samples.observations["features"].shape)

        features, dfas = samples.observations["features"], samples.observations["dfa"]
        actions = samples.actions
        next_features, next_dfas = samples.next_observations["features"], samples.next_observations["dfa"]
        dones = samples.dones
        rewards = samples.rewards

        end_of_episode_inds, end_of_step_inds = dones.squeeze().nonzero()

        for end_of_episode_ind, end_of_step_ind in zip(end_of_episode_inds, end_of_step_inds):
            # for i in range(end_of_step_ind):
            #     dfa_int = int("".join([str(int(bit)) for bit in dfas[end_of_episode_ind][i].flatten().tolist()]), 2)
            #     dfa = DFA.from_int(dfa_int, self.propositions)
            #     dfa_dict, current_state = dfa2dict(dfa)
            #     dfa = dict2dfa(dfa_dict, start=current_state).minimize()
            #     print("~~~~", i, "~~~~")
            #     print(dfa)
            # print("_______________________________")

            events = self.env.get_events_given_obss(features[end_of_episode_ind])
            events_clean = list(filter(lambda x: x != "", self.env.get_events_given_obss(features[end_of_episode_ind])))

            chain_length = 5

            if len(events_clean) < chain_length:
                continue

            # exp_base = 0.5
            # probs = np.array([exp_base**i for i in range(1, len(events_clean)+1)])
            # probs = probs / np.sum(probs)

            probs = np.array([1 for i in range(1, len(events_clean)+1)])
            probs = probs / np.sum(probs)

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

            # print(dfa)

            # print(events)

            # observed_events = events_clean[]
            # dfa_int = int("".join([str(int(bit)) for bit in dfas[end_of_episode_ind][0].flatten().tolist()]), 2)
            # dfa = DFA.from_int(dfa_int, self.propositions)
            # accepting_state = dfa.transition(events_clean)
            # dfa_dict, current_state = dfa2dict(dfa)
            # dfa_dict[accepting_state] = (True, dfa_dict[accepting_state][1])
            # dfa = dict2dfa(dfa_dict, start=current_state).minimize()

            for step_ind in range(end_of_step_ind):

                dfa = dfa.advance(events[step_ind]).minimize()
                dfa_binary_seq = self.get_binary_seq(dfa)
                dfas[end_of_episode_ind][step_ind] = dfa_binary_seq

                reward, done = self.get_reward_and_done(dfa)
                dones[end_of_episode_ind][step_ind] = done
                rewards[end_of_episode_ind][step_ind] = reward

                next_dfa = dfa.advance(events[step_ind + 1]).minimize()
                next_dfa_binary_seq = self.get_binary_seq(next_dfa)
                next_dfas[end_of_episode_ind][step_ind] = next_dfa_binary_seq

                # dfa_binary_seq = self.get_binary_seq(dfa)
                # dfas[end_of_episode_ind][step_ind] = dfa_binary_seq

                # reward, done = self.get_reward_and_done(dfa)
                # dones[end_of_episode_ind][step_ind] = done
                # rewards[end_of_episode_ind][step_ind] = reward

                # next_dfa = dfa.advance(events[step_ind]).minimize()
                # next_dfa_binary_seq = self.get_binary_seq(next_dfa)
                # next_dfas[end_of_episode_ind][step_ind] = next_dfa_binary_seq

                # dfa = next_dfa

                if done:
                    features[end_of_episode_ind][step_ind + 1:] = np.zeros(features[end_of_episode_ind][step_ind + 1:].shape)
                    dfas[end_of_episode_ind][step_ind + 1:] = np.zeros(dfas[end_of_episode_ind][step_ind + 1:].shape)
                    actions[end_of_episode_ind][step_ind + 1:] = np.zeros(actions[end_of_episode_ind][step_ind + 1:].shape)
                    next_features[end_of_episode_ind][step_ind + 1:] = np.zeros(next_features[end_of_episode_ind][step_ind + 1:].shape)
                    dones[end_of_episode_ind][step_ind + 1:] = np.zeros(dones[end_of_episode_ind][step_ind + 1:].shape)
                    rewards[end_of_episode_ind][step_ind + 1:] = np.zeros(rewards[end_of_episode_ind][step_ind + 1:].shape)
                    # print('dones', dones[end_of_episode_ind])
                    # print('rewards', rewards[end_of_episode_ind])
                    break

            
            # for i in range(end_of_step_ind):
            #     dfa_int = int("".join([str(int(bit)) for bit in dfas[end_of_episode_ind][i].flatten().tolist()]), 2)
            #     dfa = DFA.from_int(dfa_int, self.propositions)
            #     dfa_dict, current_state = dfa2dict(dfa)
            #     dfa = dict2dfa(dfa_dict, start=current_state)
            #     print("~~~~", i, "~~~~")
            #     print(dfa)
            # print('dones', samples.dones)
            # print('rewr', samples.rewards)
        self.replay_buffer.relabel_traces(n, samples)



    def relabel_naive_baseline(self, batch_size):
        # print("RUNNING RELABELER")
        n = batch_size

        samples = self.replay_buffer.sample_traces(n, self.model._vec_normalize_env) # This should also return actions
        if samples is None:
            return
        # print(samples.observations["features"].shape)

        features, dfas = samples.observations["features"], samples.observations["dfa"]
        actions = samples.actions
        next_features, next_dfas = samples.next_observations["features"], samples.next_observations["dfa"]
        dones = samples.dones
        rewards = samples.rewards

        end_of_episode_inds, end_of_step_inds = dones.squeeze().nonzero()

        for end_of_episode_ind, end_of_step_ind in zip(end_of_episode_inds, end_of_step_inds):
            # for i in range(end_of_step_ind):
            #     dfa_int = int("".join([str(int(bit)) for bit in dfas[end_of_episode_ind][i].flatten().tolist()]), 2)
            #     dfa = DFA.from_int(dfa_int, self.propositions)
            #     dfa_dict, current_state = dfa2dict(dfa)
            #     dfa = dict2dfa(dfa_dict, start=current_state).minimize()
            #     print("~~~~", i, "~~~~")
            #     print(dfa)
            # print("_______________________________")

            events = self.env.get_events_given_obss(features[end_of_episode_ind])
            events_clean = list(filter(lambda x: x != "", self.env.get_events_given_obss(features[end_of_episode_ind])))

            # print(events)

            dfa_int = int("".join([str(int(bit)) for bit in dfas[end_of_episode_ind][0].flatten().tolist()]), 2)
            dfa = DFA.from_int(dfa_int, self.propositions)
            accepting_state = dfa.transition(events_clean)
            dfa_dict, current_state = dfa2dict(dfa)
            dfa_dict[accepting_state] = (True, dfa_dict[accepting_state][1])
            dfa = dict2dfa(dfa_dict, start=current_state).minimize()

            for step_ind in range(end_of_step_ind):

                dfa = dfa.advance(events[step_ind]).minimize()
                dfa_binary_seq = self.get_binary_seq(dfa)
                dfas[end_of_episode_ind][step_ind] = dfa_binary_seq

                reward, done = self.get_reward_and_done(dfa)
                dones[end_of_episode_ind][step_ind] = done
                rewards[end_of_episode_ind][step_ind] = reward

                next_dfa = dfa.advance(events[step_ind + 1]).minimize()
                next_dfa_binary_seq = self.get_binary_seq(next_dfa)
                next_dfas[end_of_episode_ind][step_ind] = next_dfa_binary_seq

                # dfa_binary_seq = self.get_binary_seq(dfa)
                # dfas[end_of_episode_ind][step_ind] = dfa_binary_seq

                # reward, done = self.get_reward_and_done(dfa)
                # dones[end_of_episode_ind][step_ind] = done
                # rewards[end_of_episode_ind][step_ind] = reward

                # next_dfa = dfa.advance(events[step_ind]).minimize()
                # next_dfa_binary_seq = self.get_binary_seq(next_dfa)
                # next_dfas[end_of_episode_ind][step_ind] = next_dfa_binary_seq

                # dfa = next_dfa

                if done:
                    features[end_of_episode_ind][step_ind + 1:] = np.zeros(features[end_of_episode_ind][step_ind + 1:].shape)
                    dfas[end_of_episode_ind][step_ind + 1:] = np.zeros(dfas[end_of_episode_ind][step_ind + 1:].shape)
                    actions[end_of_episode_ind][step_ind + 1:] = np.zeros(actions[end_of_episode_ind][step_ind + 1:].shape)
                    next_features[end_of_episode_ind][step_ind + 1:] = np.zeros(next_features[end_of_episode_ind][step_ind + 1:].shape)
                    dones[end_of_episode_ind][step_ind + 1:] = np.zeros(dones[end_of_episode_ind][step_ind + 1:].shape)
                    rewards[end_of_episode_ind][step_ind + 1:] = np.zeros(rewards[end_of_episode_ind][step_ind + 1:].shape)
                    # print('dones', dones[end_of_episode_ind])
                    # print('rewards', rewards[end_of_episode_ind])
                    break

            
            # for i in range(end_of_step_ind):
            #     dfa_int = int("".join([str(int(bit)) for bit in dfas[end_of_episode_ind][i].flatten().tolist()]), 2)
            #     dfa = DFA.from_int(dfa_int, self.propositions)
            #     dfa_dict, current_state = dfa2dict(dfa)
            #     dfa = dict2dfa(dfa_dict, start=current_state)
            #     print("~~~~", i, "~~~~")
            #     print(dfa)
            # print('dones', samples.dones)
            # print('rewr', samples.rewards)
        self.replay_buffer.relabel_traces(n, samples)



    def relabel_old(self, batch_size):
        # TODO: Currently minimize method and the advance method changes the state names.
        # If we can make sure that these methods do not change the state names, then we
        # can easily label reached state as the accepting state.
        n = batch_size
        samples = self.replay_buffer.sample_traces(n, self.model._vec_normalize_env) # This should also return actions
        if samples is None:
            return
        # observations = samples.observations
        # features, dfas = observations["features"], observations["dfa"]
        # actions = samples.actions
        # next_observations = samples.next_observations
        # dones = samples.dones
        # rewards = samples.rewards
        # try:
        #     end_of_episode_ind = dones[0].flatten().nonzero()[0][0]
        # except:
        #     return
        # achieved_dfa_int = int("".join([str(int(bit)) for bit in dfas[0][end_of_episode_ind].flatten().tolist()]), 2)
        # achieved_dfa = DFA.from_int(achieved_dfa_int, self.propositions)
        # achieved_dfa_dict, achieved_dfa_current_state_id = dfa2dict(achieved_dfa)
        # achieved_dfa_current_state = achieved_dfa_dict[achieved_dfa_current_state_id]
        # achieved_dfa_current_state_signature = self.get_state_signature(achieved_dfa_current_state_id, achieved_dfa_dict)
        # for i in range(end_of_episode_ind + 1):
        #     dfa_int = int("".join([str(int(bit)) for bit in dfas[0][i].flatten().tolist()]), 2)
        #     dfa = DFA.from_int(dfa_int, self.propositions)
        #     dfa_dict, current_state = dfa2dict(dfa)
        #     for state_id in dfa_dict:
        #         state_signature = self.get_state_signature(state_id, dfa_dict)
        #         if state_signature == achieved_dfa_current_state_signature:
        #             break
        #     dfa_dict[state_id] = (True, dfa_dict[state_id][1])
        #     new_dfa = dict2dfa(dfa_dict, start=current_state)
        #     new_reward, new_done = self.get_reward_and_done(new_dfa)
        #     new_dfa_binary_seq = self.get_binary_seq(new_dfa)
        #     dfas[0][i] = new_dfa_binary_seq
        #     dones[0][i] = new_done
        #     rewards[0][i] = new_reward
        #     if dfa_int == achieved_dfa_int:
        #         dfas[0][i + 1:] = np.zeros(dfas[0][i + 1:].shape)
        #         dones[0][i + 1:] = np.zeros(dones[0][i + 1:].shape)
        #         rewards[0][i + 1:] = np.zeros(rewards[0][i + 1:].shape)
        #         break
        self.replay_buffer.relabel_traces(n, samples)
    

