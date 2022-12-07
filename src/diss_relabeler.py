import sys

import numpy as np
import torch as th

from stable_baselines3.common.buffers import DictReplayBuffer
from stable_baselines3.common.type_aliases import DictReplayBufferSamples

class DissRelabeler():

    def __init__(self, replay_buffer: DictReplayBuffer):
        self.replay_buffer = replay_buffer

    def relabel(self, env, batch_size):
        n = 1
        out = self.replay_buffer.sample_traces(n, env) # This should also return actions
        # print(out.observations["features"].shape)
        # actions -> (n, n_envs, 75, 1)
        # obss -> (n, n_env, 75, 7, 7, 13)
        # Option 1) Loop over samples at this level. Call DISS for every "single" trace and write them back in the replay buffer
        # Option 2) Send a list of traces to DISS. DISS should loop over them and relabel them on by one. Write them back in batch.
        # Return {"feature":..., "dfa": ...} and keep the same format as the rest of the code.
        # dfas -> (n, n_env, 75, 100) -> (n*n_env, 75, 10) // convert to dfa from int
        # Compute the new labels here.
        # Try the dummy rebaleling.
        if out is not None:
            self.replay_buffer.relabel_traces(out)
        return
    

