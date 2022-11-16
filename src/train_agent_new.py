import argparse
from utils import make_env
from stable_baselines3 import DQN
from stable_baselines3.common.env_checker import check_env
from feature_extractor import CustomCombinedExtractor
from stable_baselines3.common.off_policy_algorithm import OffPolicyAlgorithm
from stable_baselines3.common.callbacks import BaseCallback, CallbackList, ConvertCallback, ProgressBarCallback

from stable_baselines3.common.type_aliases import MaybeCallback

from typing import Any, Dict, List, Optional, Tuple, Type, TypeVar, Union
from stable_baselines3.common.type_aliases import GymEnv, MaybeCallback, RolloutReturn, Schedule, TrainFreq, TrainFrequencyUnit

from diss_relabeler import DissRelabeler

def learn_with_diss(
        model: OffPolicyAlgorithm,
        total_timesteps: int,
        callback: MaybeCallback = None,
        log_interval: int = 4,
        eval_env: Optional[GymEnv] = None,
        eval_freq: int = -1, 
        n_eval_episodes: int = 5,
        tb_log_name: str = "run",
        eval_log_path: Optional[str] = None,
        reset_num_timesteps: bool = True,
        progress_bar: bool = False,
    ) -> OffPolicyAlgorithm:

    relabaler = DissRelabeler(model.replay_buffer)

    total_timesteps, callback = model._setup_learn(
        total_timesteps,
        eval_env,
        callback,
        eval_freq,
        n_eval_episodes,
        eval_log_path,
        reset_num_timesteps,
        tb_log_name,
        progress_bar,
    )

    callback.on_training_start(locals(), globals())

    while model.num_timesteps < total_timesteps:
        rollout = model.collect_rollouts(
            model.env,
            train_freq=model.train_freq,
            action_noise=model.action_noise,
            callback=callback,
            learning_starts=model.learning_starts,
            replay_buffer=model.replay_buffer,
            log_interval=log_interval,
        )

        if rollout.continue_training is False:
            break

        if model.num_timesteps > 0 and model.num_timesteps > model.learning_starts:
            # If no `gradient_steps` is specified,
            # do as many gradients steps as steps performed during the rollout
            gradient_steps = model.gradient_steps if model.gradient_steps >= 0 else rollout.episode_timesteps
            # Special case when the user passes `gradient_steps=0`
            if gradient_steps > 0:
                model.train(batch_size=model.batch_size, gradient_steps=gradient_steps)
                relabaler.relabel(model._vec_normalize_env, model.batch_size)

    callback.on_training_end()

    return model

parser = argparse.ArgumentParser()

parser.add_argument("--env", required=True,
                        help="name of the environment to train on (REQUIRED)")
parser.add_argument("--sampler", default="Default",
                    help="the ltl formula template to sample from (default: DefaultSampler)")
parser.add_argument("--seed", type=int, default=1,
                        help="random seed (default: 1)")

args = parser.parse_args()

env = make_env(args.env, args.sampler, seed=args.seed)

print("------------------------------------------------")
print(env)
check_env(env)
print("------------------------------------------------")

model = DQN(
    "MultiInputPolicy",
    env,
    policy_kwargs=dict(
        features_extractor_class=CustomCombinedExtractor,
        features_extractor_kwargs=dict(env=env)
        ),
    verbose=1
    )
print(model.policy)

learn_with_diss(model, total_timesteps=10000000)

model.save("dqn")

