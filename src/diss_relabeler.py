import sys

import numpy as np
import torch as th

from stable_baselines3.common.buffers import DictReplayBuffer
from stable_baselines3.common.type_aliases import DictReplayBufferSamples

class DissRelabeler():

    def __init__(self, replay_buffer: DictReplayBuffer):
        self.replay_buffer = replay_buffer

    def relabel(self, env, batch_size):
        obss, dfas = self.replay_buffer.sample_traces(10)
        if dfas is not None:
            self.replay_buffer.relabel_traces(dfas)
    

