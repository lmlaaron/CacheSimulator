import copy
import logging
import os
import sys
import time

from typing import Optional

import hydra
from omegaconf import DictConfig, OmegaConf

import torch

from rich.console import Console
from rich.progress import track

import rlmeta.utils.hydra_utils as hydra_utils
import rlmeta.utils.nested_utils as nested_utils

from rlmeta.agents.agent import Agent
from rlmeta.agents.ppo.ppo_agent import PPOAgent
from rlmeta.core.callbacks import EpisodeCallbacks
from rlmeta.core.controller import Controller
from rlmeta.core.replay_buffer import ReplayBuffer
from rlmeta.core.types import Action, TimeStep
from rlmeta.envs.env import Env
from rlmeta.samplers import UniformSampler
from rlmeta.storage import TensorCircularBuffer
from rlmeta.utils.optimizer_utils import get_optimizer
from rlmeta.utils.stats_dict import StatsDict

import model_utils

from cache_env_wrapper import CacheEnvCCHunterWrapperFactory
from metric_callbacks import CCHunterMetricCallbacks
from loop_runner import LoopRunner


@hydra.main(config_path="./config", config_name="ppo_attack")
def main(cfg):
    print(f"workding_dir = {os.getcwd()}")
    metric_callbacks = CCHunterMetricCallbacks()
    logging.info(hydra_utils.config_to_json(cfg))

    env_fac = CacheEnvCCHunterWrapperFactory(OmegaConf.to_container(cfg.env_config))
    t_env = env_fac(index=0)
    e_env = env_fac(index=1)

    model = model_utils.get_model(cfg.model_config, cfg.env_config.window_size,
                                  t_env.action_space.n).to(cfg.train_device)
    optimizer = get_optimizer(cfg.optimizer.name, model.parameters(),
                              cfg.optimizer.args)
    replay_buffer = ReplayBuffer(TensorCircularBuffer(cfg.replay_buffer_size),
                                 UniformSampler())
    # A dummy Controller.
    controller = Controller()

    learner = PPOAgent(
        model,
        replay_buffer=replay_buffer,
        controller=controller,
        optimizer=optimizer,
        batch_size=cfg.batch_size,
        entropy_coeff=cfg.entropy_coeff,
        model_push_period=sys.maxsize,  # No model push
    )
    t_agent = PPOAgent(model,
                       replay_buffer=replay_buffer,
                       deterministic_policy=False)
    e_agent = PPOAgent(model, deterministic_policy=True)

    t_loop = LoopRunner(t_env,
                        t_agent,
                        replay_buffer,
                        should_update=True,
                        seed=cfg.train_seed,
                        episode_callbacks=metric_callbacks)
    e_loop = LoopRunner(e_env,
                        e_agent,
                        replay_buffer=None,
                        should_update=False,
                        seed=cfg.eval_seed,
                        episode_callbacks=metric_callbacks)

    start_time = time.perf_counter()
    for epoch in range(cfg.num_epochs):
        # Train
        stats = StatsDict()
        n_step = cfg.replay_buffer_size // cfg.batch_size
        for _ in track(range(0, cfg.steps_per_epoch, n_step),
                       description="Training..."):
            rollout_stats = t_loop.run(num_steps=cfg.replay_buffer_size)
            stats.update(rollout_stats)

            # index = torch.arange(cfg.replay_buffer_size)
            index = torch.randperm(cfg.replay_buffer_size)
            _, data = replay_buffer[index]
            data = nested_utils.map_nested(lambda x: x.to(cfg.train_device),
                                           data)

            for i in range(0, cfg.replay_buffer_size, cfg.batch_size):
                batch = nested_utils.map_nested(
                    lambda x, i=i: x[i:i + cfg.batch_size], data)
                train_stats = learner._train_step(batch)
                stats.extend(train_stats)

        cur_time = time.perf_counter() - start_time
        info = f"T Epoch {epoch}"
        if cfg.table_view:
            logging.info("\n\n" + stats.table(info, time=cur_time) + "\n")
        else:
            logging.info(
                stats.json(info, phase="Train", epoch=epoch, time=cur_time))

        # Eval
        stats = e_loop.run(num_episodes=cfg.num_eval_episodes)
        cur_time = time.perf_counter() - start_time
        info = f"E Epoch {epoch}"
        if cfg.table_view:
            logging.info("\n\n" + stats.table(info, time=cur_time) + "\n")
        else:
            logging.info(
                stats.json(info, phase="Eval", epoch=epoch, time=cur_time))

        torch.save(model.state_dict(), f"ppo_agent-{epoch}.pth")


if __name__ == "__main__":
    main()
