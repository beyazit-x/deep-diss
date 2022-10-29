import argparse
from utils import make_env
from stable_baselines3 import DQN
from stable_baselines3.common.env_checker import check_env
from feature_extractor import CustomCombinedExtractor

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


policy_kwargs = dict(
    features_extractor_class=CustomCombinedExtractor,
    features_extractor_kwargs=dict(env=env)
)

model = DQN("MultiInputPolicy", env, policy_kwargs=policy_kwargs)
print(model.policy)

model.learn(total_timesteps=10000000, log_interval=1)
# model.save("dqn")

# del model

# model = DQN.load("dqn")

# obs, _ = env.reset()
# while True:
#     action, _states = model.predict(obs, deterministic=True)
#     obs, reward, done, _, info = env.step(action)
#     print("\rReward: %.4f" % reward, end="\r")
#     if done:
#       obs = env.reset()
