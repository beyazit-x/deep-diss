import sys

import numpy as np
import torch as th

from dfa import DFA
from dfa.utils import dfa2dict
from dfa.utils import dict2dfa

from diss.experiment import PartialDFAIdentifier
from diss import LabeledExamples
from diss import diss

from stable_baselines3.common.buffers import DictReplayBuffer
from stable_baselines3.common.type_aliases import DictReplayBufferSamples

from diss_interface import NNPlanner

class DissRelabeler():

    def __init__(self, model, env):
        self.model = model
        self.env = env
        self.propositions = env.get_propositions()
        self.replay_buffer = model.replay_buffer
        self.N = 1000 # TODO compute this

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

    def relabel(self, env, batch_size):

        planner = NNPlanner(self.env, self.model)

        universal = DFA(
            start=True,
            inputs=self.propositions,
            outputs={True, False},
            label=lambda s: s,
            transition=lambda s, c: True,
        )

        identifer = PartialDFAIdentifier(
            partial = universal,
            base_examples = LabeledExamples(negative=[], positive=[]),
            try_reach_avoid=True, # TODO check this flag
        )

        n = 1
        samples, env_inds, eps_inds = self.replay_buffer.sample_traces(n, env) # This should also return actions
        observations = samples.observations
        print("obs <<<<<")
        print(observations['features'].shape)
        print(observations['dfa'].shape)
        # shape of features is (n, 76, 7, 7, 13)
        features, dfas = observations["features"], observations["dfa"]
        # shape of actions is (n, 76, 1)
        actions = samples.actions

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
                example_drop_prob=1/20, #1e-2,
                synth_timeout=20,
            )

            for (data, concept, metadata) in dfa_search:
                print('size', concept.size)


    def relabel_old(self, env, batch_size):
        # TODO: Currently minimize method and the advance method changes the state names.
        # If we can make sure that these methods do not change the state names, then we
        # can easily label reached state as the accepting state.
        n = 1
        samples, env_inds, eps_inds = self.replay_buffer.sample_traces(n, env) # This should also return actions
        observations = samples.observations
        features, dfas = observations["features"], observations["dfa"]
        actions = samples.actions
        next_observations = samples.next_observations
        dones = samples.dones
        rewards = samples.rewards
        # th.set_printoptions(profile="full")
        # print("------------------------------------------------------------")
        # for i in range(len(samples.observations["dfa"][0])):
        #     print(i, samples.observations["dfa"][0][i])
        #     print(i, samples.dones[0][i])
        #     print(i, samples.rewards[0][i])
        # print("------------------------------------------------------------")
        try:
            # end_of_episode_ind = dones[0].flatten().nonzero().item()
            end_of_episode_ind = dones[0].flatten().nonzero()[0][0]
        except:
            return
        achieved_dfa_int = int("".join([str(int(bit)) for bit in dfas[0][end_of_episode_ind].flatten().tolist()]), 2)
        achieved_dfa = DFA.from_int(achieved_dfa_int, self.propositions)
        achieved_dfa_dict, achieved_dfa_current_state_id = dfa2dict(achieved_dfa)
        achieved_dfa_current_state = achieved_dfa_dict[achieved_dfa_current_state_id]
        achieved_dfa_current_state_signature = self.get_state_signature(achieved_dfa_current_state_id, achieved_dfa_dict)
        for i in range(end_of_episode_ind + 1):
            dfa_int = int("".join([str(int(bit)) for bit in dfas[0][i].flatten().tolist()]), 2)
            dfa = DFA.from_int(dfa_int, self.propositions)
            dfa_dict, current_state = dfa2dict(dfa)
            for state_id in dfa_dict:
                state_signature = self.get_state_signature(state_id, dfa_dict)
                if state_signature == achieved_dfa_current_state_signature:
                    break
            dfa_dict[state_id] = (True, dfa_dict[state_id][1])
            new_dfa = dict2dfa(dfa_dict, start=current_state)
            new_reward, new_done = self.get_reward_and_done(new_dfa)
            new_dfa_binary_seq = self.get_binary_seq(new_dfa)
            dfas[0][i] = new_dfa_binary_seq
            dones[0][i] = new_done
            rewards[0][i] = new_reward
            if dfa_int == achieved_dfa_int:
                dfas[0][i + 1:] = np.zeros(dfas[0][i + 1:].shape)
                dones[0][i + 1:] = np.zeros(dones[0][i + 1:].shape)
                rewards[0][i + 1:] = np.zeros(rewards[0][i + 1:].shape)
                break
        # for i in range(len(samples.observations["dfa"][0])):
        #     print(i, samples.observations["dfa"][0][i])
        #     print(i, samples.dones[0][i])
        #     print(i, samples.rewards[0][i])
        # print("------------------------------------------------------------")
        # sys.exit()
        # # if dones[0].flatten().count_nonzero().item() > 0:
        # #     return
        # # # dfa_int = int("".join([str(int(bit)) for bit in dfas[0][-1].flatten().tolist()]), 2)
        # # # dfa = DFA.from_int(dfa_int, self.propositions)
        # # th.set_printoptions(profile="full")
        # # # print(dfas[0].shape)
        # # # print(dfas[0])
        # # # print(actions[0])
        # # th.set_printoptions(profile="default")
        # # for i in range(75):
        # #     dfa_int = int("".join([str(int(bit)) for bit in dfas[0][i].flatten().tolist()]), 2)
        # #     if dfa_int <= 0:
        # #         break
        # #     dfa = DFA.from_int(dfa_int, self.propositions)
        # #     print(i, dfa)
        # #     dfa_int = int("".join([str(int(bit)) for bit in next_observations["dfa"][0][i].flatten().tolist()]), 2)
        # #     dfa = DFA.from_int(dfa_int, self.propositions)
        # #     print("----", dones[0][i], dfa)
        # # input(">>")
        # # print(out.observations["features"].shape)
        # # actions -> (n, n_envs, 75, 1)
        # # obss -> (n, n_env, 75, 7, 7, 13)
        # # Option 1) Loop over samples at this level. Call DISS for every "single" trace and write them back in the replay buffer
        # # Option 2) Send a list of traces to DISS. DISS should loop over them and relabel them on by one. Write them back in batch.
        # # Return {"feature":..., "dfa": ...} and keep the same format as the rest of the code.
        # # dfas -> (n, n_env, 75, 100) -> (n*n_env, 75, 10) // convert to dfa from int
        # # Compute the new labels here.
        # # Try the dummy rebaleling.
        self.replay_buffer.relabel_traces(samples, env_inds, eps_inds)
    

