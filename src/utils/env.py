"""
This class defines the environments that we are going to use.
Note that this is the place to include the right LTL-Wrapper for each environment.
"""


import gym
import gym_minigrid
import envs.gym_letters
import dfa_wrappers
import pretrain_wrapper

def make_env(env_key, sampler, reject_reward=-1, seed=None):
    env = gym.make(env_key)
    env.seed(seed)
    return dfa_wrappers.DFAEnv(env, sampler, reject_reward)

def make_pretrain_env(env_key, sampler_mean, reject_reward=-1, seed=None):
    env = gym.make(env_key)
    env.seed(seed)
    return pretrain_wrapper.PretrainEnv(env, sampler_mean, reject_reward)
    
