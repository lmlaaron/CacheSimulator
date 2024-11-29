import os
import sys

from typing import Any, Dict

# from rlmeta.envs.env import Env, EnvFactory
# from rlmeta.envs.gym_wrapper import GymWrapper
from torchrl.envs.libs.gym import GymWrapper
from torchrl.envs.utils import check_env_specs
import hydra
from omegaconf import DictConfig, OmegaConf

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cache_guessing_game_env_impl import CacheGuessingGameEnv
from cchunter_wrapper import CCHunterWrapper
from cyclone_wrapper import CycloneWrapper


# class CacheEnvWrapperFactory(EnvFactory):
#     def __init__(self, env_config: Dict[str, Any]) -> None:
#         self._env_config = env_config
#
#     @property
#     def env_config(self) -> Dict[str, Any]:
#         return self._env_config
#
#     def __call__(self, index: int, *args, **kwargs) -> Env:
#         env = CacheGuessingGameEnv(self.env_config)
#         env = GymWrapper(env, old_step_api=True)
#         return env
#
#
# class CacheEnvCCHunterWrapperFactory(EnvFactory):
#     def __init__(self, env_config: Dict[str, Any]) -> None:
#         self._env_config = env_config
#
#     @property
#     def env_config(self) -> Dict[str, Any]:
#         return self._env_config
#
#     def __call__(self, index: int, *args, **kwargs) -> Env:
#         # env = CacheGuessingGameEnv(self.env_config)
#         env = CCHunterWrapper(self.env_config)
#         env = GymWrapper(env, old_step_api=True)
#         return env
#
#
# class CacheEnvCycloneWrapperFactory(EnvFactory):
#     def __init__(self, env_config: Dict[str, Any]) -> None:
#         self._env_config = env_config
#
#     @property
#     def env_config(self) -> Dict[str, Any]:
#         return self._env_config
#
#     def __call__(self, index: int, *args, **kwargs) -> Env:
#         # env = CacheGuessingGameEnv(self.env_config)
#         env = CycloneWrapper(self.env_config)
#         env = GymWrapper(env, old_step_api=True)
#         return env

@hydra.main(config_path="./config", config_name="ppo_attack")
def main(cfg):
    env = CacheGuessingGameEnv(OmegaConf.to_container(cfg.env_config))
    env = GymWrapper(env)
    print(env.rollout(3))
    check_env_specs(env)


if __name__ == "__main__":
    main()
