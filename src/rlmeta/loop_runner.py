from typing import Optional

import rlmeta.utils.nested_utils as nested_utils

from rlmeta.agents.agent import Agent
from rlmeta.core.callbacks import EpisodeCallbacks
from rlmeta.core.replay_buffer import ReplayBuffer
from rlmeta.core.types import Action, TimeStep
from rlmeta.envs.env import Env
from rlmeta.utils.stats_dict import StatsDict


class LoopRunner:
    def __init__(self,
                 env: Env,
                 agent: Agent,
                 replay_buffer: Optional[ReplayBuffer] = None,
                 should_update: bool = True,
                 seed: Optional[int] = None,
                 episode_callbacks: Optional[EpisodeCallbacks] = None) -> None:
        self._env = env
        self._agent = agent
        self._replay_buffer = replay_buffer
        self._should_update = should_update

        self._seed = seed
        if seed is not None:
            self._env.reset(seed=seed)

        self._episode_callbacks = episode_callbacks

    def run(self,
            num_episodes: Optional[int] = None,
            num_steps: Optional[int] = None,
            *args,
            **kwargs) -> StatsDict:
        assert num_episodes is not None or num_steps is not None

        if self._replay_buffer is not None:
            self._replay_buffer.clear()

        stats = StatsDict()
        if num_episodes is not None:
            for _ in range(num_episodes):
                metrics = self._run_loop(*args, **kwargs)
                stats.extend(metrics)
        else:
            while len(self._replay_buffer) < num_steps:
                metrics = self._run_loop(*args, **kwargs)
                stats.extend(metrics)

        return stats

    def _batch_obs(self, timestep: TimeStep) -> TimeStep:
        obs, reward, terminated, truncated, info = timestep
        return TimeStep(obs.unsqueeze(0), reward, truncated, terminated, info)

    def _unbatch_action(self, action: Action) -> Action:
        act, info = action
        act.squeeze_(0).cpu()
        info = nested_utils.map_nested(lambda x: x.squeeze(0), info)
        return Action(act, info)

    def _run_loop(self, *args, **kwargs) -> None:
        episode_length = 0
        episode_return = 0.0

        if self._episode_callbacks is not None:
            self._episode_callbacks.reset()
            self._episode_callbacks.on_episode_start(index=0)

        timestep = self._env.reset(*args, **kwargs)
        self._agent.observe_init(timestep)
        if self._episode_callbacks is not None:
            self._episode_callbacks.on_episode_init(index=0, timestep=timestep)

        while not (timestep.terminated or timestep.truncated):
            # Model server requires a batch_dim, so unsqueeze here for local
            # runs.
            timestep = self._batch_obs(timestep)
            action = self._agent.act(timestep)
            # Unbatch the action.
            action = self._unbatch_action(action)

            timestep = self._env.step(action)
            self._agent.observe(action, timestep)
            if self._should_update:
                self._agent.update()

            episode_length += 1
            episode_return += timestep.reward
            if self._episode_callbacks is not None:
                self._episode_callbacks.on_episode_step(index=0,
                                                        step=episode_length -
                                                        1,
                                                        action=action,
                                                        timestep=timestep)
        metrics = {
            "episode_length": episode_length,
            "episode_return": episode_return,
        }
        if self._episode_callbacks is not None:
            self._episode_callbacks.on_episode_end(index=0)
            metrics.update(self._episode_callbacks.custom_metrics)

        return metrics
