import logging

from typing import Dict

import hydra
import torch
import torch.nn

import rlmeta_extension.nested_utils as nested_utils

from rlmeta.agents.ppo.ppo_agent import PPOAgent
from rlmeta.core.types import Action
from rlmeta.envs.env import Env
from rlmeta.utils.stats_dict import StatsDict

from cache_env_wrapper import CacheEnvWrapperFactory
from cache_ppo_model import CachePPOModel


def unbatch_action(action: Action) -> Action:
    act, info = action
    act.squeeze_(0)
    info = nested_utils.map_nested(lambda x: x.squeeze(0), info)
    return Action(act, info)


def run_loop(env: Env, agent: PPOAgent, victim_addr=-1) -> Dict[str, float]:
    episode_length = 0
    episode_return = 0.0

    if victim_addr == -1:
        timestep = env.reset()
    else:
        timestep = env.reset(victim_address=victim_addr)
    
    agent.observe_init(timestep)
    while not timestep.done:
        # Model server requires a batch_dim, so unsqueeze here for local runs.
        timestep.observation.unsqueeze_(0)
        action = agent.act(timestep)
        # Unbatch the action.
        action = unbatch_action(action)

        timestep = env.step(action)
        agent.observe(action, timestep)

        episode_length += 1
        episode_return += timestep.reward

    metrics = {
        "episode_length": episode_length,
        "episode_return": episode_return,
    }

    return metrics


def run_loops(env: Env,
              agent: PPOAgent,
              num_episodes: int,
              seed: int = 0) -> StatsDict:
    env.seed(seed)
    metrics = StatsDict()

    if env.env.allow_empty_victim_access == False:
        end_address = env.env.victim_address_max + 1
    else:
        end_address = env.env.victim_address_max + 1 + 1

    for victim_addr in range(env.env.victim_address_min, end_address):
        cur_metrics = run_loop(env, agent, victim_addr=victim_addr)
        metrics.extend(cur_metrics)

    return metrics


@hydra.main(config_path="./config", config_name="sample")
def main(cfg):
    # Create env
    cfg.env_config['verbose'] = 1
    env_fac = CacheEnvWrapperFactory(cfg.env_config)
    env = env_fac(index=0)
    
    # Load model
    cfg.model_config["output_dim"] = env.action_space.n
    params = torch.load(cfg.checkpoint)
    model = CachePPOModel(**cfg.model_config)
    model.load_state_dict(params)

    # Create agent
    agent = PPOAgent(model, deterministic_policy=cfg.deterministic_policy)

    # Run loops
    metrics = run_loops(env, agent, cfg.num_episodes, cfg.seed)
    logging.info("\n\n" + metrics.table(info="sample") + "\n")


if __name__ == "__main__":
    main()
