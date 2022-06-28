# simpple SVM based detector
# based on Cyclone 
# window_size = 4
# interval_size = 20
# 1 bucket

import copy

from typing import Any, Dict, Sequence, Tuple

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

import gym

from sklearn import svm
from sklearn.model_selection import cross_val_score
from cache_guessing_game_env_impl import CacheGuessingGameEnv


class CycloneWrapper(gym.Env):
    def __init__(self,
                 env_config: Dict[str, Any],
                 keep_latency: bool = True) -> None:
        env_config["cache_state_reset"] = False

        self.reset_observation = env_config.get("reset_observation", False)
        self.keep_latency = keep_latency
        self.env_config = env_config
        self.episode_length = env_config.get("episode_length", 80)
        #self.threshold = env_config.get("threshold", 0.8)
        
        self.cyclone_window_size = env_config.get("cyclone_window_size", 4)
        self.cyclone_interval_size = env_config.get("cyclone_interval_size", 20)
        self.cyclone_num_buckets = env_config.get("cyclone_num_buckets", 1)
        self.cyclone_bucket_size = self.env_config.cache_configs.cache_1.blocks / self.cyclone_num_buckets

        #self.cyclone_counters = [[0]* self.cyclone_num_buckets ] * self.cyclone_window_size
        self.cyclone_counters = []
        for j in range(self.cyclone_num_buckets):
            temp =[]
            for i in range(self.cyclone_window_size):
                temp.append(0)
            self.cyclone_counters.append(temp)
        self.cyclone_coeff = env_config.get("cyclone_coeff", 1.0)

        # self.cc_hunter_detection_reward = env_config.get(
        #     "cc_hunter_detection_reward", -1.0)
        #self.cc_hunter_coeff = env_config.get("cc_hunter_coeff", 1.0)
        #self.cc_hunter_check_length = env_config.get("cc_hunter_check_length",
                                                    # 4)

        self._env = CacheGuessingGameEnv(env_config)
        self.validation_env = CacheGuessingGameEnv(env_config)
        self.observation_space = self._env.observation_space
        self.action_space = self._env.action_space

        self.victim_address_min = self._env.victim_address_min
        self.victim_address_max = self._env.victim_address_max
        self.attacker_address_max = self._env.attacker_address_max
        self.attacker_address_min = self._env.attacker_address_min
        self.victim_address = self._env.victim_address

        self.cnt = 0
        self.step_count = 0
        #self.cc_hunter_history = []

    def reset(self, victim_address=-1):
        # reset cyclone counter
        #self.cyclone_counters = [[0]* self.cyclone_num_buckets ] * self.cyclone_window_size
        self.cyclone_counters = []
        for j in range(self.cyclone_num_buckets):
            temp =[]
            for i in range(self.cyclone_window_size):
                temp.append(0)
            self.cyclone_counters.append(temp)
        
        self.step_count = 0
        self.cnt = 0
        #self.cc_hunter_history = []
        obs = self._env.reset(victim_address=victim_address,
                              reset_cache_state=True)
        self.victim_address = self._env.victim_address
        return obs

    ####def autocorr(self, x: np.ndarray, p: int) -> float:
    ####    if p == 0:
    ####        return 1.0
    ####    mean = x.mean()
    ####    var = x.var()
    ####    return ((x[:-p] - mean) * (x[p:] - mean)).mean() / var

    def cyclone_attack(self, cyclone_counters):
        x = np.array(cyclone_counters).reshape(-1)
        print(x)
        x_mod = np.array(cyclone_counters).reshape(-1)
        x_mod[0] = 0
        y = 1
        y_mod = 0
        X = [x, x_mod]
        Y= [y, y_mod]
        clf = svm.SVC(random_state=0)
        clf.fit(X,Y)
        y = clf.predict([x])[0]
        rew = y
        return rew

    def step(self, action):
        obs, reward, done, info = self._env.step(action)

        # is_guess = (self._env.parse_action(action)[1] == 1)
        cur_step_obs = obs[0, :]
        latency = cur_step_obs[0] if self.keep_latency else -1

        # self.cc_hunter_history.append(latency)
        # self.cc_hunter_history.append(None if latency == 2 else latency)

        # Mulong Luo
        # cyclone
        if "cyclic_set_index" in info and info["cyclic_set_index"] != -1: 
            set = int(info["cyclic_set_index"])
            if self.step_count < self.episode_length:
                self.cyclone_counters[int(set / self.cyclone_bucket_size) ][int(self.step_count / self.cyclone_interval_size) ] += 1

        self.step_count += 1
        # self.cc_hunter_history.append(info.get("cache_state_change", None))

        if done:
            self.cnt += 1 #TODO(Mulong) fix the logic so taht only guess increment the cnt
            obs = self._env.reset(victim_address=-1,
                                  reset_cache_state=False,
                                  reset_observation=self.reset_observation)
            self.victim_address = self._env.victim_address

            if self.step_count < self.episode_length:
                done = False
            else:
                #rew, cnt = self.cc_hunter_attack(self.cc_hunter_history)
                rew = self.cyclone_attack(self.cyclone_counters)
                reward += self.cyclone_coeff * rew
                info["cyclone_attack"] = self.cnt

        # done = (self.step_count >= self.episode_length)
        # if done:
        #     rew, cnt = self.cc_hunter_attack(self.cc_hunter_history)
        #     reward += self.cc_hunter_coeff * rew
        #     info["cc_hunter_attack"] = cnt
        return obs, reward, done, info
