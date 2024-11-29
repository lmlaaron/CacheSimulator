import argparse
import os

import sys
import torch
from omegaconf import OmegaConf
from tensordict import TensorDict

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from cache_guessing_game_env_impl import CacheGuessingGameEnv
from torchrl.envs.libs.gym import GymWrapper

from torchrl.envs import Compose, TransformedEnv, \
    RewardSum, StepCounter
from torchrl.envs import set_exploration_type, ExplorationType

import model_utils

HERE = os.path.dirname(os.path.abspath(__file__))


def main(cfg, num_rollouts, saved_path):
    env_config = cfg.env_config
    env_config = OmegaConf.to_container(env_config)
    env_config['verbose'] = True
    device = cfg.device

    def make_env():
        return TransformedEnv(
            GymWrapper(CacheGuessingGameEnv(env_config), device=device),
            Compose(
                RewardSum(),
                StepCounter(),
            )
        )

    env = make_env()
    train_model = model_utils.get_model(
        cfg.model_config, cfg.env_config.window_size,
        env.action_spec.space.n
    ).to(device)
    actor = train_model.get_actor()
    weights = TensorDict.load_memmap(saved_path)
    actor.load_state_dict(weights.flatten_keys("."))

    with set_exploration_type(ExplorationType.MODE):
        for i in range(num_rollouts):
            env.rollout(1000, actor)  # a traj should not be longer than 1000


if __name__ == "__main__":
    # load config
    parser = argparse.ArgumentParser()
    parser.add_argument('--saved_path')
    parser.add_argument(
        '--num_rollouts',
        '--num-rollouts',
        default=1,
        type=int
        )
    args = parser.parse_args()
    cfg = torch.load(f"{args.saved_path}/cfg.pt")
    main(cfg, args.num_rollouts, args.saved_path)
