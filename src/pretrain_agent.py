#!/usr/bin/python3
import os
import torch
import random
import asyncio
import argparse
from utils import make_env, make_pretrain_env
from stable_baselines3 import DQN, SAC, PPO
from softDQN import SoftDQN
from stable_baselines3.common.env_checker import check_env
from pretrain_features_extractor import PretrainExtractor
from stable_baselines3.common.off_policy_algorithm import OffPolicyAlgorithm
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from stable_baselines3.common.type_aliases import GymEnv, MaybeCallback
from typing import Optional
from diss_relabeler import DissRelabeler
from diss_replay_buffer import DissReplayBuffer
from env_model import getEnvModel
from collections import deque
from dfa_identify.concept_class_restrictions import enforce_chain, enforce_reach_avoid_seq
from utils.parameters import GNN_EMBEDDING_SIZE

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

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--verbosity", type=int, default=1,
                            help="verbosity level passed to stable baselines model")
    parser.add_argument("--env", required=True,
                            help="name of the environment to train on (REQUIRED)")
    parser.add_argument("--sampler-mean", default="Default",
                        help="mean of the exponential distribution for reach/avoid sampling")
    parser.add_argument("--seed", type=int, default=None,
                            help="random seed (default: None)")
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
    parser.add_argument("--mid-check", action=argparse.BooleanOptionalAction, default=False,
                            help="checkpointing during training (default: False)")
    parser.add_argument("--policy", default="SDQN",
                            help="SAC | SDQN (default)")
    parser.add_argument("--disable-wandb", action=argparse.BooleanOptionalAction, default=False,
                            help="disable wandb logging (default: False)")


    args = parser.parse_args()

    random.seed(args.seed)

    env = make_pretrain_env(args.env, int(args.sampler_mean), args.reject_reward, seed=args.seed)

    callback_list = []

    # setup wandb
    if not args.disable_wandb:
        import wandb
        from wandb.integration.sb3 import WandbCallback

        run = wandb.init(
            config=args,
            sync_tensorboard=True,  # automatically upload SB3's tensorboard metrics to W&B
            entity='nlauffer',
            project="deep-diss",
            monitor_gym=True,       # automatically upload gym environements' videos
            save_code=False,
        )

        wandb_callback=WandbCallback(
            gradient_save_freq=0,
            verbose=2,
        )

        wandb.save(os.path.join(wandb.run.dir, "checkpoint*"))

        callback_list.append(wandb_callback)
        
        model_save_path = wandb.run.dir
    else:
        model_save_path = "./logs"

    if args.mid_check:
        checkpoint_callback = CheckpointCallback(
            save_freq=25000,
            save_path=model_save_path,
            name_prefix="checkpoint",
            save_replay_buffer=False,
            save_vecnormalize=False,
        )
        callback_list.append(checkpoint_callback)

    discounted_reward_callback = DiscountedRewardCallback(args.gamma)
    callback_list.append(discounted_reward_callback)

    tensorboard_dir = "./tensorboard"

    env_model = getEnvModel(env, env.observation_space['features'].shape)
    features_dim = env_model.embedding_size + GNN_EMBEDDING_SIZE

    policy = None
    if args.policy == "SDQN":
        policy = SoftDQN
        kwargs = {
             "ent_coef": 0.01,
             "policy": "MultiInputPolicy",
             "env": env,
             "policy_kwargs": dict(
                 features_extractor_class=PretrainExtractor,
                 features_extractor_kwargs=dict(env=env, features_dim=features_dim),
             ),
             "verbose": args.verbosity,
             "tensorboard_log": tensorboard_dir,
             "learning_starts": args.learning_starts,
             "batch_size": args.batch_size,
             "gamma": args.gamma,
             "buffer_size": args.buffer_size,
             "exploration_fraction": args.exploration_fraction
         }
    elif args.policy == "SAC":
        policy = SAC
        kwargs = {
             "ent_coef": 0.01,
             "policy": "MultiInputPolicy",
             "env": env,
             "policy_kwargs": dict(
                 features_extractor_class=PretrainExtractor,
                 features_extractor_kwargs=dict(env=env, features_dim=features_dim),
             ),
             "verbose": args.verbosity,
             "tensorboard_log": tensorboard_dir,
             "learning_starts": args.learning_starts,
             "batch_size": args.batch_size,
             "gamma": args.gamma,
             "buffer_size": args.buffer_size,
             "exploration_fraction": args.exploration_fraction
         }
    elif args.policy == "PPO":
        policy = PPO
        kwargs = {
             "ent_coef": 0.01,
             "policy": "MultiInputPolicy",
             "env": env,
             "policy_kwargs": dict(
                 features_extractor_class=PretrainExtractor,
                 features_extractor_kwargs=dict(env=env, features_dim=features_dim),
             ),
             "verbose": args.verbosity,
             "tensorboard_log": tensorboard_dir,
             "batch_size": args.batch_size,
             "gamma": args.gamma,
         }
    else:
        raise ValueError("Policy can only be SDQN or SAC")
    assert policy is not None


    model = policy(**kwargs)
    model.learn(total_timesteps=args.total_timesteps, callback=callback_list)
    
    MODEL_PATH = "checkpoint_final"
    model.save(os.path.join(model_save_path, "f{MODEL_PATH}.pkl"), include=[])
    if args.save_gnn_path is not None:
        torch.save(model.policy.q_net.features_extractor.gnn, f"{args.save_gnn_path}")

    run.finish()

