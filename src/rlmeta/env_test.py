import json
import os
import sys

import hydra
import omegaconf

import rlmeta.envs.gym_wrappers as gym_wrappers

from cache_env_wrapper import CacheEnvWrapperFactory


@hydra.main(config_path="./conf", config_name="conf_ppo_lru_8way")
def main(cfg):
    env_config = cfg["env_config"]
    # env_config = omegaconf.OmegaConf.to_container(env_config)

    env_fac = CacheEnvWrapperFactory(env_config)
    env = env_fac(0)

    print(env.action_space.n)
    timestep = env.reset()
    print(timestep)


if __name__ == "__main__":
    main()
