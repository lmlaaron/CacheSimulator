import copy
import matplotlib.pyplot as plt

import pandas as pd

import gym

from cache_guessing_game_env_impl import CacheGuessingGameEnv


class CCHunterWrapper(gym.Env):  #TODO(LISA)
    def __init__(self, env_config, keep_latency=True):
        env_config["cache_state_reset"] = False

        self.keep_latency = keep_latency
        self.env_config = env_config
        self.cc_hunter_episode_scale = 20
        self._env = CacheGuessingGameEnv(env_config)
        # in this case the pattern has to be stored
        self.validation_env = CacheGuessingGameEnv(env_config)
        self.observation_space = self._env.observation_space
        self.action_space = self._env.action_space
        self.action_buffer = []
        # self.latency_buffer = []
        self.pattern_init_state = (copy.deepcopy(self._env.l1),
                                   self._env.victim_address)
        self.pattern_init_state_buffer = []
        self.victim_address_min = self._env.victim_address_min
        self.victim_address_max = self._env.victim_address_max
        self.attacker_address_max = self._env.attacker_address_max
        self.attacker_address_min = self._env.attacker_address_min
        self.victim_address = self._env.victim_address
        self.cc_hunter_episode = self.cc_hunter_episode_scale * (
            self.victim_address_max - self.victim_address_min + 1)

        self.cc_hunter_detection_reward = env_config.get(
            "cc_hunter_detection_reward", -1.0)

    def reset(self, victim_address = -1):
        self.action_buffer = []
        self.cc_hunter_length = self._env.cache_size
        self.cc_hunter_buffer = []
        self.validation_env.reset()  # Is this needed?
        # obs = self._env.reset(victim_address=-1,
        #                       if_only_reinitialize_rl_related_variables=True)
        obs = self._env.reset(victim_address=victim_address, reset_cache_state=True)
        self.pattern_init_state = (copy.deepcopy(self._env.l1),
                                   self._env.victim_address)
        # print("number of found patterns:" + str(len(self.pattern_buffer)))
        return obs

    def calculate_autocorrelation_coefficients(self, x, lags):
        """
        Calculate the autocorrelation coefficients for the given data and lags.
        """
        # n = len(x)
        series = pd.Series([i[0] for i in x])
        # print("Series is:\n", series)
        # print("series correlation:\n",series.autocorr())
        # data = np.asarray(x)
        # print(data)
        # x_mean = np.mean(data)
        # y_mean = np.mean(data)
        # rho = np.zeros(lags)
        # for lag in range(0, lags):
        #     x_m = data[:-lag]
        #     y_m = data[lag:]
        #     x_m -= x_mean
        #     y_m -= y_mean
        #     rho[lag] = np.sum(x_m * y_m) / (n - lag)
        return series.autocorr(lags)

    def autocorrelogram(self, x, plot_autocorrelogram=False):
        autocorrelogram = []
        # we may also consider a wider range of lags
        # for i in range(self._env.cache_size * 100):
        import pdb; pdb.set_trace()
        for i in range(self._env.cache_size * self.cc_hunter_episode_scale):
            autocorrelogram.append(
                self.calculate_autocorrelation_coefficients(x, i))
        # plot the autocorrelogram
        if plot_autocorrelogram:
            plt.plot(autocorrelogram)
        return autocorrelogram

    def cc_hunter_attack(self, x, threshold=0.8, plot_autocorrelogram=False):
        autocorrelogram = self.autocorrelogram(x, plot_autocorrelogram)
        # detect the attack
        cc_hunter_attack = False
        cc_hunter_attack_list = []
        for i in range(0, self._env.cache_size):
            # Note: we should not consider the auto-correlation coefficient at
            # lag 0
            if autocorrelogram[i] > threshold:
                cc_hunter_attack_list.append(i)
        # cc_hunter_attack = True if len(cc_hunter_attack_list) > 1 else False
        cc_hunter_attack = (len(cc_hunter_attack_list) > 1)
        # TODO: tune the periodicity threshold. It is currently set to 0.8,
        # which is a reasonable threshold for detecting cache conflicts pattern
        # repetition
        # print(cc_hunter_attack)
        return cc_hunter_attack, cc_hunter_attack_list

    '''
    calculate the guessing correct rate of action_buffer
    '''

    def calc_correct_seq(self, action_buffer):
        correct_guess = 0
        total_guess = 0
        for _ in range(1):
            if self.replay_action_buffer(action_buffer):
                correct_guess += 1
            total_guess += 1
        rtn = 1.0 * correct_guess / total_guess
        # print('calc_correct_seq ' + rtn)
        return rtn

    '''
    step function
    '''

    def step(self, action):
        state, reward, done, info = self._env.step(action)
        # info is an empty dict
        # TODO(LISA): when done, the agent will call reset()
        # (LISA) state gives a history of the [hit_info, addr, ****]
        # add to buffer
        # done is true if the game is over, and cache will be reset, and the
        # episode will start again
        # Ff we want to keep the cache state after the game is over, we need to
        # override the done signal

        # state = [state self._env.calc_correct_rate()]
        cur_step_obs = state[0, :]
        latency = cur_step_obs[0] if self.keep_latency else -1

        if latency == -2:
            address, is_guess, is_victim, is_flsuh, victim_addr = self._env.step(action)
            if is_victim == 1:  #victim access
                latency = info['victim_latency']
                self.cc_hunter_buffer.append((latency, cur_step_obs[2]))
        else: # attacker access
            self.cc_hunter_buffer.append((latency, cur_step_obs[2]))
        
        # import pdb; pdb.set_trace()

        # Get the past trace of (latency, addr) appended to the
        # cc_hunter_buffer

        # latnecy is part of the attack trace
        self.action_buffer.append((action, latency))
        # make sure the current existing correct guessing rate is high enough
        # before altering the reward
        if not done:
            if self._env.parse_action(action)[1] == 1:
                # if the guess is correct, we will store this in info
                info["correct_guess"] = (reward > 0)

            return state, reward, done, info
        else:  # DONE
            length = len(self.cc_hunter_buffer)
            # self._env.reset(victim_address=-1,
            #                 if_only_reinitialize_rl_related_variables=True)
            self._env.reset(victim_address=-1,
                            reset_cache_state=False,
                            reset_observation=False)
            # print('cc hunter buffer length is', length)
            if self._env.parse_action(action)[1] == 1:
                # if the guess is correct, we will store this in info
                info["correct_guess"] = (reward > 0)

            if length < self.cc_hunter_episode:
                done = False
            else:
                cc_hunter_attack, _ = self.cc_hunter_attack(
                    self.cc_hunter_buffer, self._env.flush_inst)
                if cc_hunter_attack:
                    reward += self.cc_hunter_detection_reward  # TODO
                info["cc_hunter_attack"] = cc_hunter_attack

            return state, reward, done, info
