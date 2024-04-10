#!/usr/bin/python3
import os
import dill
import asyncio
import argparse
from utils import make_env
from stable_baselines3 import DQN, SAC
from softDQN import SoftDQN
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecMonitor
from stable_baselines3.common.env_checker import check_env
from features_extractor import CustomCombinedExtractor
from stable_baselines3.common.off_policy_algorithm import OffPolicyAlgorithm
from stable_baselines3.common.callbacks import BaseCallback, CallbackList, ConvertCallback, ProgressBarCallback, CheckpointCallback

from stable_baselines3.common.type_aliases import MaybeCallback

from typing import Any, Dict, List, Optional, Tuple, Type, TypeVar, Union
from stable_baselines3.common.type_aliases import GymEnv, MaybeCallback, RolloutReturn, Schedule, TrainFreq, TrainFrequencyUnit

from diss_relabeler import DissRelabeler
from diss_replay_buffer import DissReplayBuffer
from env_model import getEnvModel

from collections import deque

from dfa_identify.concept_class_restrictions import enforce_chain, enforce_reach_avoid_seq

# import wandb
# from wandb.integration.sb3 import WandbCallback

from utils.parameters import GNN_EMBEDDING_SIZE

import torch
torch.set_num_threads(1)

class OverwriteCheckpointCallback(CheckpointCallback):
    def __init__(
        self,
        save_freq,
        save_path,
        name_prefix="rl_model",
        save_replay_buffer=False,
        save_vecnormalize=False,
        verbose=0,
    ):
        super(OverwriteCheckpointCallback, self).__init__(
            save_freq,
            save_path,
            name_prefix,
            save_replay_buffer,
            save_vecnormalize,
            verbose,
	)
    def _checkpoint_path(self, checkpoint_type="", extension=""):
        """
        Helper to get checkpoint path for each type of checkpoint.

        :param checkpoint_type: empty for the model, "replay_buffer_"
            or "vecnormalize_" for the other checkpoints.
        :param extension: Checkpoint file extension (zip for model, pkl for others)
        :return: Path to the checkpoint
        """
        return os.path.join(self.save_path, f"{self.name_prefix}_{checkpoint_type}.{extension}")


class DiscountedRewardCallback(BaseCallback):
    """
    Custom callback for plotting additional values in tensorboard.
    """

    def __init__(self, gamma, verbose=0):
        self.gamma = gamma
        self.num_episodes = 0
        self.log_interval = 4
        self.ep_rew_buffer = deque(maxlen=100)

        super(DiscountedRewardCallback, self).__init__(verbose)

    def _on_step(self):
        # Log scalar value (here a random variable)
        dones = self.locals["dones"]
        infos = self.locals["infos"]

        for idx, done in enumerate(dones):
            info = infos[idx]
            if (
                done
                and info.get("episode") is not None
            ):
                episode_info = info["episode"]
                discounted_return = episode_info['r'] * (self.gamma ** episode_info['l'])
                self.ep_rew_buffer.append(discounted_return)
                self.num_episodes += 1

        if self.num_episodes == 0 or self.num_episodes % self.log_interval != 0:
            return
        ep_disc_rew_mean = sum(self.ep_rew_buffer) / len(self.ep_rew_buffer)
        self.logger.record("rollout/ep_disc_rew_mean", ep_disc_rew_mean)

        return True

async def relabel(relabeler, relabeler_name, batch_size):
    relabeler.relabel(relabeler_name, batch_size)

async def learn(
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
):

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
        return True, model

    if model.num_timesteps > 0 and model.num_timesteps > model.learning_starts:
        # If no `gradient_steps` is specified,
        # do as many gradients steps as steps performed during the rollout
        gradient_steps = model.gradient_steps if model.gradient_steps >= 0 else rollout.episode_timesteps
        # Special case when the user passes `gradient_steps=0`
        if gradient_steps > 0:
            model.train(batch_size=model.batch_size, gradient_steps=gradient_steps)

    return False, model

async def learn_with_diss_async(
    model: OffPolicyAlgorithm,
    env,
    relabeler_name,
    save_file_name,
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
    extra_clauses = None,
):
    relabeler = DissRelabeler(model, env, extra_clauses=extra_clauses)

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

        task1 = asyncio.create_task(learn(model, total_timesteps=10000000, callback=callback))
        task2 = None
        if model.num_timesteps > model.learning_starts:
            task2 = asyncio.create_task(relabel(relabeler, relabeler_name, 2))

        await task1
        if task2 is not None:
            await task2

        done, model = task1.result()
        if done:
            break

    callback.on_training_end()

    # should probably put this saving routine in a callback
    MODEL_PATH = "model_checkpoints/gridworld"
    file_index = 0
    while os.path.exists(f"{MODEL_PATH}_{file_index}.pkl"):
        file_index += 1
    # with open(f"{MODEL_PATH}_{file_index}.pkl", 'wb') as dump_f:
    #     dill.dump(model, dump_f)
    model.save(f"{MODEL_PATH}_{file_index}.pkl", include=[])
    if args.save_gnn_path is not None:
        torch.save(model.policy.q_net.features_extractor.gnn, f"{args.save_gnn_path}")

def learn_with_diss(
    model: OffPolicyAlgorithm,
    env,
    relabeler_name,
    save_file_name,
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
    extra_clauses = None,
):
    relabeler = DissRelabeler(model, env, extra_clauses=extra_clauses)

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
                relabeler.relabel(relabeler_name, 2)

    callback.on_training_end()

    # should probably put this saving routine in a callback
    MODEL_PATH = "logs/dqn_entropy"
    file_index = 0
    while os.path.exists(f"{MODEL_PATH}_{file_index}.pkl"):
        file_index += 1
    # with open(f"{MODEL_PATH}_{file_index}.pkl", 'wb') as dump_f:
    #     dill.dump(model, dump_f)
    model.save(f"{MODEL_PATH}_{file_index}.pkl", include=[])
    if args.save_gnn_path is not None:
        torch.save(model.policy.q_net.features_extractor.gnn, f"{args.save_gnn_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--verbosity", type=int, default=1,
                            help="verbosity level passed to stable baselines model")
    parser.add_argument("--env", required=True,
                            help="name of the environment to train on (REQUIRED)")
    parser.add_argument("--sampler", default="Default",
                        help="the ltl formula template to sample from (default: DefaultSampler)")
    parser.add_argument("--relabeler", default="none",
                        help="baseline | diss | none (default)")
    parser.add_argument("--seed", type=int, default=1,
                            help="random seed (default: 1)")
    parser.add_argument("--gamma", type=float, default=0.99,
                            help="discount factor (default: 0.99)")
    parser.add_argument("--buffer-size", type=int, default=50000,
                            help="size of the regular (not HER) replay buffer (default: 50000)")
    parser.add_argument("--learning-starts", type=int, default=10000,
                            help="how many samples to collect before doing gradient updates")
    parser.add_argument("--total-timesteps", type=int, default=1000000,
                            help="how many timesteps to train for")
    parser.add_argument("--exploration-fraction", type=float, default=0.1,
                            help="fraction of entire training period over which the exploration rate is reduced")
    parser.add_argument("--batch-size", type=int, default=8,
                            help="buffer size")
    parser.add_argument("--reject-reward", type=int, default=-1,
                            help="default is -1")
    parser.add_argument("--save-gnn-path", default=None,
                            help="save the gnn model to a path after training")
    parser.add_argument("--load-gnn-path", default=None,
                            help="load a pretrained gnn model from a path")
    parser.add_argument("--enforce-clause", default=None,
                            help="enforce diss to only find dfas in a specific class")
    parser.add_argument("--async-diss", action=argparse.BooleanOptionalAction, default=False,
                            help="run diss with asyncio (default: False)")
    parser.add_argument("--mid-check", action=argparse.BooleanOptionalAction, default=False,
                            help="checkpointing during training (default: False)")
    parser.add_argument("--policy", default="SDQN",
                            help="SAC | SDQN (default)")

    args = parser.parse_args()

    # setup wandb

    # run = wandb.init(
	# config=args,
	# sync_tensorboard=True,  # automatically upload SB3's tensorboard metrics to W&B
    #     entity='nlauffer',
	# project="deep-diss",
	# monitor_gym=True,       # automatically upload gym environements' videos
	# save_code=False,
    # )

    env = make_env(args.env, args.sampler, args.reject_reward, seed=args.seed)
    # single_env = env
    # num_env = 8
    # env = DummyVecEnv([lambda: make_env(args.env, args.sampler, seed=args.seed+i) for i in range(num_env)])
    # env = VecMonitor(env)
    # single_env = make_env(args.env, args.sampler)

    print("------------------------------------------------")
    print(env)
    # check_env(env)
    print("------------------------------------------------")


    # wandb_callback=WandbCallback(
    #     gradient_save_freq=0,
    #     # model_save_path=f"models/{run.id}",
    #     verbose=2,
    #     )

    discounted_reward_callback = DiscountedRewardCallback(args.gamma)

    # checkpoint_callback = CheckpointCallback(
    #     save_freq=25000,
    #     # save_path=wandb.run.dir,
    #     name_prefix="checkpoint",
    #     save_replay_buffer=False,
    #     save_vecnormalize=False,
    # )

    # wandb.save(os.path.join(wandb.run.dir, "checkpoint*"))

    # callback_list = [discounted_reward_callback, wandb_callback, checkpoint_callback]
    callback_list = [discounted_reward_callback]
    if args.mid_check:
        callback_list.append(checkpoint_callback)

    tensorboard_dir = "./wandb_sweep_relabel_" + args.relabeler

    env_model = getEnvModel(env, env.observation_space['features'].shape)
    features_dim = env_model.embedding_size + GNN_EMBEDDING_SIZE

    policy = None
    if args.policy == "SDQN":
        policy = SoftDQN
    elif args.policy == "SAC":
        policy = SAC
    else:
        raise ValueError("Policy can only be SDQN or SAC")
    assert policy is not None

    if args.relabeler == 'none':
        model = policy(
            ent_coef=0.01,
            policy="MultiInputPolicy",
            env=env,
            policy_kwargs=dict(
                features_extractor_class=CustomCombinedExtractor,
                features_extractor_kwargs=dict(env=env, gnn_load_path=args.load_gnn_path, features_dim=features_dim),
                ),
            verbose=args.verbosity,
            tensorboard_log=tensorboard_dir,
            learning_starts=args.learning_starts,
            batch_size=args.batch_size,
            gamma=args.gamma,
            buffer_size=args.buffer_size,
            exploration_fraction=args.exploration_fraction
            )
        model.learn(total_timesteps=args.total_timesteps, callback=callback_list)
        MODEL_PATH = "checkpoint_gridworld_list"
        # file_index = 0
        # while os.path.exists(f"{MODEL_PATH}_{file_index}.pkl"):
        #     file_index += 1
        # with open(f"{MODEL_PATH}_{file_index}.pkl", 'wb') as dump_f:
        #     dill.dump(model, dump_f)
        # model.save(f"{MODEL_PATH}.pkl", include=[])
        model.save(os.path.join(wandb.run.dir, "f{MODEL_PATH}.pkl"), include=[])
    else:
        if args.enforce_clause == "chain":
            extra_clauses = enforce_chain
        elif args.enforce_clause == "reach_avoid":
            extra_clauses = enforce_reach_avoid_seq
        else:
            extra_clauses = None
        model = policy(
            ent_coef=0.01,
            policy="MultiInputPolicy",
            env=env,
            policy_kwargs=dict(
                features_extractor_class=CustomCombinedExtractor,
                features_extractor_kwargs=dict(env=env, gnn_load_path=args.load_gnn_path, features_dim=features_dim)
                ),
            replay_buffer_class=DissReplayBuffer,
            replay_buffer_kwargs=dict(
                max_episode_length=env.timeout,
                her_replay_buffer_size=args.buffer_size
                ),
            verbose=args.verbosity,
            learning_starts=args.learning_starts,
            buffer_size=args.buffer_size,
            batch_size=args.batch_size,
            gamma=args.gamma,
            tensorboard_log=tensorboard_dir,
            exploration_fraction=args.exploration_fraction
            )
        if args.async_diss:
            asyncio.run(learn_with_diss_async(model, env, args.relabeler, "dqn", callback=callback_list, total_timesteps=args.total_timesteps, extra_clauses=extra_clauses))
        else:
            learn_with_diss(model, env, args.relabeler, "dqn", callback=callback_list, total_timesteps=args.total_timesteps, extra_clauses=extra_clauses)

    run.finish()

