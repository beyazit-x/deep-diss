import torch
import torch.nn as nn

from envs import *
from gym.envs.classic_control import PendulumEnv
from envs.gridworld.gridworld_env import GridworldEnv
from envs.dfa_world.dfa_world import DummyEnv


def getEnvModel(env, obs_space):
    env = env.unwrapped

    if isinstance(env, LetterEnv):
        return LetterEnvModel(obs_space)
    if isinstance(env, MinigridEnv):
        return MinigridEnvModel(obs_space)
    # if isinstance(env, ZonesEnv):
    #     return ZonesEnvModel(obs_space)
    if isinstance(env, PendulumEnv):
        return PendulumEnvModel(obs_space)
    if isinstance(env, GridworldEnv):
        return GridworldEnvModel(obs_space)
    if isinstance(env, DummyEnv):
        return DummyDFAEnvModel(obs_space)
    # Add your EnvModel here...


    # The default case (No environment observations) - SimpleLTLEnv uses this
    return EnvModel(obs_space)


"""
This class is in charge of embedding the environment part of the observations.
Every environment has its own set of observations ('image', 'direction', etc) which is handeled
here by associated EnvModel subclass.

How to subclass this:
    1. Call the super().__init__() from your init
    2. In your __init__ after building the compute graph set the self.embedding_size appropriately
    3. In your forward() method call the super().forward as the default case.
    4. Add the if statement in the getEnvModel() method
"""
class EnvModel(nn.Module):
    def __init__(self, obs_space):
        super().__init__()
        self.embedding_size = 0

    def forward(self, obs):
        return None

    def size(self):
        return self.embedding_size

class DummyDFAEnvModel(EnvModel):
    def __init__(self, obs_space):
        super().__init__(obs_space)
        self.embedding_size = 0

    def forward(self, obs):
        return super().forward(obs)

    def size(self):
        return self.embedding_size

class GridworldEnvModel(EnvModel):
    def __init__(self, obs_space):
        super().__init__(obs_space)
        self.width = obs_space[0]
        self.height = obs_space[1]
        self.embedding_size =  self.width * self.height

    def forward(self, obs):
        x = obs
        x = x.reshape(x.shape[0], -1) # flatten observation
        return x

    def size(self):
        return self.embedding_size


class LetterEnvModel(EnvModel):
    def __init__(self, obs_space):
        super().__init__(obs_space)
        n = obs_space[0]
        m = obs_space[1]
        k = obs_space[2]
        self.image_conv = nn.Sequential(
            nn.Conv2d(k, 16, (2, 2)),
            nn.ReLU(),
            nn.Conv2d(16, 32, (2, 2)),
            nn.ReLU(),
            nn.Conv2d(32, 64, (2, 2)),
            nn.ReLU()
        )
        self.embedding_size = (n-3)*(m-3)*64

    def forward(self, obs):
        x = obs.transpose(1, 3).transpose(2, 3)
        x = self.image_conv(x)
        x = x.reshape(x.shape[0], -1)
        return x

class MinigridEnvModel(EnvModel):
    def __init__(self, obs_space):
        super().__init__(obs_space)
        # We need the following because Minigrid is wrapped in VecTransposeImage
        h_1, w_1, c_1 = obs_space # HxWxC format
        c_2, h_2, w_2 = obs_space # CxHxW format
        if c_1 > c_2:
            n, m, k = h_2, w_2, c_2
            self.transpose_needed = False
        else:
            n, m, k = h_1, w_1, c_1
            self.transpose_needed = True
        self.image_conv = nn.Sequential(
            nn.Conv2d(k, 16, (2, 2)),
            nn.ReLU(),
            nn.MaxPool2d((2, 2)),
            nn.Conv2d(16, 32, (2, 2)),
            nn.ReLU(),
            nn.Conv2d(32, 64, (2, 2)),
            nn.ReLU()
        )
        self.embedding_size = ((n-1)//2-2)*((m-1)//2-2)*64

    def forward(self, obs):
        x = obs
        if self.transpose_needed:
            x = x.transpose(1, 3).transpose(2, 3)
        x = self.image_conv(x)
        x = x.reshape(x.shape[0], -1)
        return x

class ZonesEnvModel(EnvModel):
    def __init__(self, obs_space):
        super().__init__(obs_space)

        n = obs_space[0]
        lidar_num_bins = 16
        self.embedding_size = 64 #(n-12)//lidar_num_bins + 4
        self.net_ = nn.Sequential(
            nn.Linear(n, 128),
            nn.ReLU(),
            nn.Linear(128, self.embedding_size),
            nn.ReLU()
        )
        # embedding_size = number of propositional lidars + 4 normal sensors

    def forward(self, obs):
        return self.net_(obs)

class PendulumEnvModel(EnvModel):
    def __init__(self, obs_space):
        super().__init__(obs_space)
        self.net_ = nn.Sequential(
            nn.Linear(obs_space[0], 3),
            nn.Tanh(),
            # nn.Linear(3, 3),
            # nn.Tanh()
        )
        self.embedding_size = 3

    def forward(self, obs):
        obs = self.net_(obs)
        return x
