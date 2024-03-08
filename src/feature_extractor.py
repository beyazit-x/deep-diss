import gym
import utils
import torch
import numpy as np
import torch as th
from torch import nn
import networkx as nx
from env_model import getEnvModel
from gnns.graphs.GNN import GNNMaker
from utils.parameters import FEATURE_SIZE
from dfa import DFA
from dfa.utils import dfa2dict

########################################################################
import ring
import time
import pydot
import signal
import random
import pickle
import dill
import numpy as np
import networkx as nx
from copy import deepcopy, copy
from pysat.solvers import Solver
from pythomata.impl.simple import SimpleNFA as NFA 
from scipy.special import softmax
import dfa

from utils.parameters import TIMEOUT_SECONDS, FEATURE_SIZE
########################################################################

from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

class CustomCombinedExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.spaces.Dict, env, gnn_load_path, features_dim):
        # We do not know features-dim here before going over all the items,
        # so put something dummy for now. PyTorch requires calling
        # nn.Module.__init__ before adding modules
        super(CustomCombinedExtractor, self).__init__(observation_space, features_dim=features_dim)

        self.env = env
        self.propositions = env.get_propositions()

        self.text_embedding_size = 32
        if gnn_load_path is None:
            # TODO: Make GNN architecture a parameter
            self.gnn = GNNMaker("RGCN_8x32_ROOT_SHARED", max(FEATURE_SIZE, len(self.propositions) + 10), self.text_embedding_size)
        else:
            self.gnn = torch.load(gnn_load_path)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        obs_space = {"image": observation_space.spaces["features"].shape, "text": max(FEATURE_SIZE, len(self.env.get_propositions()) + 10)}
        self.env_model = getEnvModel(env, obs_space)

    def _get_guard_embeddings(self, guard):
        embeddings = []
        try:
            guard = guard.replace(" ", "").replace("(", "").replace(")", "").replace("\"", "")
        except:
            return embeddings
        if (guard == "true"):
            return embeddings
        guard = guard.split("&")
        cnf = []
        seen_atoms = []
        # print("guard", guard)
        for c in guard:
            atoms = c.split("|")
            clause = []
            for atom in atoms:
                try:
                    index = seen_atoms.index(atom if atom[0] != "~" else atom[1:])
                except:
                    index = len(seen_atoms)
                    seen_atoms.append(atom if atom[0] != "~" else atom[1:])
                clause.append(index + 1 if atom[0] != "~" else -(index + 1))
            cnf.append(clause)
        models = []
        with Solver(bootstrap_with=cnf) as s:
            models = list(s.enum_models())
        if len(models) == 0:
            return embeddings
        for model in models:
            temp = [0.0] * FEATURE_SIZE
            for a in model:
                if a > 0:
                    atom = seen_atoms[abs(a) - 1]
                    temp[self.propositions.index(atom)] = 1.0
            embeddings.append(temp)
        return embeddings

    def _get_onehot_guard_embeddings(self, guard):
        is_there_onehot = False
        is_there_all_zero = False
        onehot_embedding = [0.0] * FEATURE_SIZE
        onehot_embedding[-3] = 1.0 # Since it will be a temp node
        full_embeddings = self._get_guard_embeddings(guard)
        for embed in full_embeddings:
            # discard all non-onehot embeddings (a one-hot embedding must contain only a single 1)
            if embed.count(1.0) == 1:
                # clean the embedding so that it's one-hot
                is_there_onehot = True
                var_idx = embed.index(1.0)
                onehot_embedding[var_idx] = 1.0
            elif embed.count(0.0) == len(embed):
                is_there_all_zero = True
        if is_there_onehot or is_there_all_zero:
            return [onehot_embedding]
        else:
            return []

    def _is_sink_state(self, node, nxg):
        for edge in nxg.edges:
            if node == edge[0] and node != edge[1]: # If there is an outgoing edge to another node, then it is not an accepting state
                return False
        return True

    def dfa2nxg(self, mvc_dfa, minimize=False):
        """ converts a mvc format dfa into a networkx dfa """

        if minimize:
            mvc_dfa = mvc_dfa.minimize()

        dfa_dict, init_node = dfa2dict(mvc_dfa)
        init_node = str(init_node)

        nxg = nx.DiGraph()

        accepting_states = []
        for start, (accepting, transitions) in dfa_dict.items():
            # pydot_graph.add_node(nodes[start])
            start = str(start)
            nxg.add_node(start)
            if accepting:
                accepting_states.append(start)
            for action, end in transitions.items():
                if nxg.has_edge(start, str(end)):
                    existing_label = nxg.get_edge_data(start, str(end))['label']
                    nxg.add_edge(start, str(end), label='{} | {}'.format(existing_label, action))
                    # print('{} | {}'.format(existing_label, action))
                else:
                    nxg.add_edge(start, str(end), label=action)

        return init_node, accepting_states, nxg


    def _format(self, init_node, accepting_states, nxg):
        # print('init', init_node)
        # print('accepting', accepting_states)
        rejecting_states = []
        for node in nxg.nodes:
            if self._is_sink_state(node, nxg) and node not in accepting_states:
                rejecting_states.append(node)

        for node in nxg.nodes:
            nxg.nodes[node]["feat"] = np.array([[0.0] * FEATURE_SIZE])
            nxg.nodes[node]["feat"][0][-4] = 1.0
            if node in accepting_states:
                nxg.nodes[node]["feat"][0][-2] = 1.0
            if node in rejecting_states:
                nxg.nodes[node]["feat"][0][-1] = 1.0

        nxg.nodes[init_node]["feat"][0][-5] = 1.0

        edges = deepcopy(nxg.edges)

        new_node_name_base_str = "temp_"
        new_node_name_counter = 0

        for e in edges:
            # print(e, nxg.edges[e])
            guard = nxg.edges[e]["label"]
            # print(e, guard)
            nxg.remove_edge(*e)
            if e[0] == e[1]:
                continue # We define self loops below
            onehot_embedding = self._get_onehot_guard_embeddings(guard) # It is ok if we receive a cached embeddding since we do not modify it
            if len(onehot_embedding) == 0:
                continue
            new_node_name = new_node_name_base_str + str(new_node_name_counter)
            new_node_name_counter += 1
            nxg.add_node(new_node_name, feat=np.array(onehot_embedding))
            nxg.add_edge(e[0], new_node_name, type=2)
            nxg.add_edge(new_node_name, e[1], type=3)

        nx.set_node_attributes(nxg, np.array([0.0]), "is_root")
        nxg.nodes[init_node]["is_root"] = np.array([1.0]) # is_root means current state

        for node in nxg.nodes:
            nxg.add_edge(node, node, type=1)

        return nxg

    def get_dfa_from_binary_seq(self, dfa_binary_seq):
        dfa_binary_str = "".join(str(int(i)) for i in dfa_binary_seq.tolist())
        dfa_int = int(dfa_binary_str, 2)
        dfa = DFA.from_int(dfa_int, self.propositions)
        init_node, accepting_states, nxg = self.dfa2nxg(dfa)
        dfa_nxg = self._format(init_node,accepting_states,nxg)
        return dfa_nxg

    def get_obs(self, features, dfa_binary_seqs, done=None, progression_info=None):
        # dfa_nxg = self.get_dfa_from_binary_seq(dfa_binary_seqs)
        dfa_nxgs = [self.get_dfa_from_binary_seq(dfa_binary_seq) for dfa_binary_seq in dfa_binary_seqs]
        dfa_obs = {'features': features,'text': dfa_nxgs}

        preprocessed_obs = utils.my_preprocess_obss(dfa_obs, self.propositions, done=done, progression_info=progression_info, prev_preprocessed_obs=None, device=self.device)

        embedding = self.env_model(preprocessed_obs)
        embed_gnn = self.gnn(preprocessed_obs.text)
        embedding = torch.cat((embedding, embed_gnn), dim=1) if embedding is not None else embed_gnn
        return embedding

    def forward(self, observations) -> th.Tensor:
        features, dfa_binary_seqs = observations["features"], observations["dfa"]
        result = self.get_obs(features, dfa_binary_seqs)
        return result
