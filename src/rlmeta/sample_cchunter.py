import logging

from typing import Dict

import hydra
import torch
import torch.nn
import os
# append system path
import sys
sys.path.append("/media/research/yl3469/RLSCA/CacheSimulator/src")
import rlmeta_extension.nested_utils as nested_utils
import numpy as np
from rlmeta.agents.ppo.ppo_agent import PPOAgent
from rlmeta.core.types import Action
from rlmeta.envs.env import Env
from rlmeta.utils.stats_dict import StatsDict
from cache_guessing_game_env_impl import CacheGuessingGameEnv
from cchunter_wrapper import CCHunterWrapper
from cache_env_wrapper import CacheEnvWrapperFactory
from cache_ppo_model import CachePPOModel
from cache_ppo_transformer_model import CachePPOTransformerModel
import matplotlib.pyplot as plt
import pandas as pd
from cache_env_wrapper import CacheEnvCCHunterWrapperFactory


def autocorrelation_plot_forked(series, ax=None, n_lags=None, change_deno=False, change_core=False, **kwds):
    """
    Autocorrelation plot for time series.
    Parameters:
    -----------
    series: Time series
    ax: Matplotlib axis object, optional
    n_lags: maximum number of lags to show. Default is len(series)
    kwds : keywords
        Options to pass to matplotlib plotting method
    Returns:
    -----------
    class:`matplotlib.axis.Axes`
    """
    import matplotlib.pyplot as plt
    
    n_full = len(series)
    if n_full <= 2:
      raise ValueError("""len(series) = %i but should be > 2
      to maintain at least 2 points of intersection when autocorrelating
      with lags"""%n_full)
      
    # Calculate the maximum number of lags permissible
    # Subtract 2 to keep at least 2 points of intersection,
    # otherwise pandas.Series.autocorr will throw a warning about insufficient
    # degrees of freedom
    n_maxlags = n_full - 2
    
    
    # calculate the actual number of lags
    if n_lags is None:
      # Choosing a reasonable number of lags varies between datasets,
      # but if the data longer than 200 points, limit this to 100 lags as a
      # reasonable default for plotting when n_lags is not specified
      n_lags = min(n_maxlags, 100)
    else:
      if n_lags > n_maxlags:
        raise ValueError("n_lags should be < %i (i.e. len(series)-2)"%n_maxlags)
    
    if ax is None:
        ax = plt.gca(xlim=(1, n_lags), ylim=(-1.0, 1.0))

    if not change_core:
      data = np.asarray(series)
      mean = np.mean(data)
      c0 = np.sum((data - mean) ** 2) / float(n_full)
      def r(h):
          deno = n_full if not change_deno else (n_full - h)
          return ((data[:n_full - h] - mean) *
                  (data[h:] - mean)).sum() / float(deno) / c0
    else:
      def r(h):
        return series.autocorr(lag=h)
      
    x = np.arange(n_lags) + 1
    # y = lmap(r, x)
    y = np.array([r(xi) for xi in x])
    z95 = 1.959963984540054
    z99 = 2.5758293035489004
    ax.axhline(y=0.95, linestyle='--', color='grey')
    # ax.axhline(y=z95 / np.sqrt(n_full), color='grey')
    ax.axhline(y=0.0, color='black')
    # ax.axhline(y=-z95 / np.sqrt(n_full), color='grey')
    # ax.axhline(y=-z99 / np.sqrt(n_full), linestyle='--', color='grey')
    ax.set_xlabel("Lag")
    ax.set_ylabel("Autocorrelation")
    ax.plot(x, y, **kwds)
    if 'label' in kwds:
        ax.legend()
    ax.grid()
    
    return ax

def unbatch_action(action: Action) -> Action: 
    act, info = action
    act.squeeze_(0)
    info = nested_utils.map_nested(lambda x: x.squeeze(0), info)
    return Action(act, info)


def run_loop(env: Env, agent: PPOAgent, victim_addr=-1) -> Dict[str, float]:
    episode_length = 0
    episode_return = 0.0
    num_correct = 1
    num_guess = 1 # FIXME, this is not true when our training is not 100% accuracy
    hit_trace = []

    import pdb; pdb.set_trace()
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
        # import pdb; pdb.set_trace()


        timestep = env.step(action)
        obs, reward, done, info = timestep
        if 'correct_guess' in info.keys():
            num_guess += 1
            if info['correct_guess'] == True:
                print(f"victim_address! {victim_addr} correct guess! {info['correct_guess']}")
                num_correct += 1
            else:
                correct = False
        hit =  obs[0][0]
        hit_trace.append(hit)
        # add, is_guess, is_victim, is_flush, _ = env._env.parse_action(action)
        
        
            
        agent.observe(action, timestep)

        episode_length += 1
        episode_return += timestep.reward

    metrics = {
        "episode_length": episode_length,
        "episode_return": episode_return,
    }

    return metrics, hit_trace, (num_correct, num_guess)


def run_loops(env: Env,
              agent: PPOAgent,
              num_episodes: int,
              seed: int = 0) -> StatsDict:
    env.seed(seed)
    metrics = StatsDict()
    all_num_corr, all_num_guess = 0, 0
    episode_length_total = 0
    if env.env._env.allow_empty_victim_access == False:
        end_address = env.env.victim_address_max + 1
    else:
        end_address = env.env.victim_address_max + 1 + 1

    for victim_addr in range(env.env.victim_address_min, end_address):
        cur_metrics, hit_trace, (num_corr, num_guess) = run_loop(env, agent, victim_addr=victim_addr)
        # import pdb; pdb.set_trace()
        all_num_corr += num_corr
        all_num_guess += num_guess
        episode_length_total += cur_metrics["episode_length"]
        metrics.extend(cur_metrics)
        print("Episode number of guess:", num_guess)
        print("Episode number of corrects:", num_corr)
        print("correct rate:", num_corr / num_guess)
        print("bandwidth rate:", num_guess / cur_metrics["episode_length"])
        

    # plot\
        hit_trace = [int(i) for i in hit_trace]
        print(hit_trace)
        data = pd.Series(hit_trace)
        plt.figure()
        autocorrelation_plot_forked(data, n_lags=len(data)-2, change_deno=True)
        plt.savefig('cchunter_hit_trace_{}_acf.png'.format(victim_addr))
        print("Figure saved as 'cchunter_hit_trace_{}_acf.png".format(victim_addr))

    print("Total number of guess:", all_num_guess)
    print("Total number of corrects:", all_num_corr)
    print("Episode total:", episode_length_total)

    print("correct rate:", all_num_corr / all_num_guess)
    print("bandwidth rate:", all_num_guess / episode_length_total)

    return metrics


@hydra.main(config_path="./config", config_name="sample")
def main(cfg):
    global ccenv
    # Create env
    cfg.env_config['verbose'] = 1
    env_fac = CacheEnvCCHunterWrapperFactory(cfg.env_config)
    env = env_fac(index=0)
    
    # Load model
    cfg.model_config["output_dim"] = env.action_space.n
    params = torch.load(cfg.checkpoint)
    #model = CachePPOModel(**cfg.model_config)
    model = CachePPOTransformerModel(**cfg.model_config)
    # import pdb; pdb.set_trace()
    model.load_state_dict(params)
    model.eval()

    # Create agent
    agent = PPOAgent(model, deterministic_policy=True)

    # Run loops
    metrics = run_loops(env, agent, cfg.num_episodes, cfg.seed)
    logging.info("\n\n" + metrics.table(info="sample") + "\n")


if __name__ == "__main__":
    main()
