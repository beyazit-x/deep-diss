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

from utils.parameters import FEATURE_SIZE
########################################################################

from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

class CustomCombinedExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.spaces.Dict, env, gnn_load_path, features_dim):
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

        self.env_model = getEnvModel(env, observation_space.spaces["features"].shape)

    def preprocess_texts(self, texts, device=None):
        dfa_builder = utils.DFABuilder(self.propositions, dfa_n_conjunctions=self.env.sampler.get_n_conjunctions(), dfa_n_disjunctions=self.env.sampler.get_n_disjunctions(), device=device)
        return np.array([[dfa_builder(text).to(device)] for text in texts])

    def preprocess_obss(self, features, dfa_int_seqs, device=None):
        return utils.DictList({
            "image": features,
            "text":  self.preprocess_texts([seq for seq in dfa_int_seqs], device=device)
        })

    def get_obs(self, features, dfa_int_seqs, done=None, progression_info=None):
        preprocessed_obss = self.preprocess_obss(features, dfa_int_seqs, device=self.device)
        embedding = self.env_model(preprocessed_obss.image)
        embed_gnn = self.gnn(preprocessed_obss.text)
        embedding = torch.cat((embedding, embed_gnn), dim=1) if embedding is not None else embed_gnn
        return embedding

    def forward(self, observations) -> th.Tensor:
        features, dfa_int_seqs = observations["features"], observations["dfa"]
        result = self.get_obs(features, dfa_int_seqs)
        return result

