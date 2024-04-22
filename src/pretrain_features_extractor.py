import gym
import torch
from torch import nn
from env_model import getEnvModel
from utils.parameters import GNN_EMBEDDING_SIZE

from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

class PretrainExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.spaces.Dict, env, features_dim):
        super(PretrainExtractor, self).__init__(observation_space, features_dim=features_dim)

        self.env = env
        self.propositions = env.get_propositions()

        self.goal_embedding_size = GNN_EMBEDDING_SIZE
        self.reach_avoid_encoder = nn.Linear(2*len(self.propositions), self.goal_embedding_size)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.env_model = getEnvModel(env, observation_space.spaces["features"].shape)

    def get_obs(self, env_features, reach_avoid):
        embedding = self.env_model(env_features)
        embed_reach_avoid = self.reach_avoid_encoder(reach_avoid)
        embedding = torch.cat((embedding, embed_reach_avoid), dim=1) if embedding is not None else embed_reach_avoid
        return embedding

    def forward(self, observations) -> torch.Tensor:
        env_features, reach_avoid = observations["features"], observations["reach_avoid"]
        result = self.get_obs(env_features, reach_avoid)
        return result

