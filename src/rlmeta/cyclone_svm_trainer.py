# Author: Mulong Luo
# date: 2022.6.28
# usage: to train the svm classifier of cycloen by feeding 
# the date from TextbookAgent as malicious traces 
# and spec traces for benign traces

import logging

from typing import Dict

import hydra
import torch
import torch.nn
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# sys.path.append("/home/mulong/RL_SCA/src/CacheSimulator/src")

import rlmeta.utils.nested_utils as nested_utils
import numpy as np
from rlmeta.agents.ppo.ppo_agent import PPOAgent
from rlmeta.core.types import Action
from rlmeta.envs.env import Env
from rlmeta.utils.stats_dict import StatsDict

from textbook_attacker import TextbookAgent
# from cache_guessing_game_env_impl import CacheGuessingGameEnv
# from cchunter_wrapper import CCHunterWrapper
from cache_env_wrapper import CacheEnvWrapperFactory, CacheEnvCycloneWrapperFactory 
from cyclone_wrapper import CycloneWrapper


def signal_handler(sig, frame):
    print('You pressed Ctrl+C!')
    checkpoint = trainer.save()
    print("checkpoint saved at", checkpoint)
    sys.exit(0)

@hydra.main(config_path="./config", config_name="sample_cyclone")
def main(cfg):
    #create env
    cfg.env_config['verbose'] = 1

    # generate dataset for malicious traces
    cfg.env_config['cyclone_collect_data'] = True
    cfg.env_config['cyclone_malicious_trace'] = True
    env_fac = CacheEnvCycloneWrapperFactory(cfg.env_config)
    env = env_fac(index=0)
    agent = TextbookAgent(cfg.env_config)
    episode_length = 0
    episode_return = 0.0


    for i in range(100):
        timestep = env.reset()
        num_guess = 0
        num_correct = 0
        while not timestep.done:
            # Model server requires a batch_dim, so unsqueeze here for local runs.
            timestep.observation.unsqueeze_(0)
            action, info = agent.act(timestep)
            action = Action(action, info)
            # unbatch the action

            victim_addr = env._env.victim_address
            timestep = env.step(action)
            obs, reward, done, info = timestep
            if "guess_correct" in info:
                num_guess += 1
                if info["guess_correct"]:
                    print(f"victim_address! {victim_addr} correct guess! {info['guess_correct']}")
                    num_correct += 1
                else:
                    correct = False

            agent.observe(action, timestep)
            episode_length += 1
            episode_return += timestep.reward


    env.reset(save_data=True) # save data to file

    #cfg.env_config['cyclone_malicious_trace'] = False
    #env_fac = CacheEnvCCHunterWrapperFactory(cfg.env_config)
    #env = env_fac(index=0)
 

if __name__ == "__main__":
    main()