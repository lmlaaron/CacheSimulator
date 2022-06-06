'''
 split the agent into two different agent
 P1: just generate the sequence but not the guess
 P2: just make the guess, given the memory access sequence and observations

 P1: action space: autoCAT's memory access 
 observation space: guessability

 P2: action space: NOP
 observation space: original observation space

 P1 wrapper of CacheGuessingGameEnv
 blocking the guess action or just have one guess action
 when guess is structed, calculate the guessability as the reward
 observation space becomes concatenated observations
 reward becomes agregated reward
'''
from random import random
from cache_guessing_game_env_impl import *
import sys
import signal
from sklearn import svm
from sklearn.model_selection import cross_val_score

class CacheSimulatorP1Wrapper(gym.Env):
    def __init__(self, env_config):
        # for offline training, the environment returns filler observations and zero reward 
        # until the guess
        # the step reward is also temporarily accumulated until the end
        self.offline_training = True
        self.copy = 1
        self.env_list = []
        self.env_config = env_config
        self.cache_state_reset = False # has to force no reset
        self.env = CacheGuessingGameEnv(env_config)
        self.victim_address_min = self.env.victim_address_min
        self.victim_address_max = self.env.victim_address_max
        self.window_size = self.env.window_size
        self.secret_size = self.victim_address_max - self.victim_address_min + 1
        self.max_box_value = self.env.max_box_value
        self.feature_size = self.env.feature_size
       
        # expand the observation space
        self.observation_space = spaces.Box(low=-1, high=self.max_box_value, shape=(self.window_size, self.feature_size * self.secret_size * self.copy))
        
        # merge all guessing into one action
        self.action_space_size = (self.env.action_space.n - self.secret_size+1)
        print(self.env.action_space.n)
        print(self.env.get_act_space_dim())
        self.action_space = spaces.Discrete(self.action_space_size)
        
        # instantiate the environment
        self.env_list.append(CacheGuessingGameEnv(env_config))
        self.env_config['verbose'] = False
        for _ in range(1,self.secret_size * self.copy):
            self.env_list.append(CacheGuessingGameEnv(env_config))

        # instantiate the latency_buffer
        # for each permuted secret, latency_buffer stores the latency
        self.latency_buffer = []
        for i in range(0, self.secret_size * self.copy):
            self.latency_buffer.append([])

        #permute the victim addresses
        self.victim_addr_arr = np.random.permutation(range(self.env.victim_address_min, self.env.victim_address_max+1))
        self.victim_addr_arr = []
        for i in range(self.victim_address_min, self.victim_address_max+1):
            self.victim_addr_arr.append(i)
        
        # reset the addresses
        self.env_config['verbose'] = True
        self.env_list[0].reset(self.victim_addr_arr[0])
        self.env_config['verbose'] = False
        self.reset_state = np.array([[]] * self.window_size)

        # initialize the offline_state as filler state if we use offline training
        if self.offline_training == True:
            self.offline_state = self.env.reset(seed=-1)
            self.offline_reward = 0

        for cp in range(0, self.copy):
            seed = -1#random.randint(1, 1000000)
            for i in range(0, len(self.victim_addr_arr)):
                state = self.env_list[i + cp * len(self.victim_addr_arr)].reset(victim_address = self.victim_addr_arr[i], seed= seed)
                self.reset_state = np.concatenate((self.reset_state, state), axis=1)  
            # same seed esure the initial state are teh same
    
    def reset(self):
        # permute the victim addresses
        #self.victim_addr_arr = np.random.permutation(range(self.env.victim_address_min, self.env.victim_address_max+1))
        self.victim_addr_arr = []
        for i in range(self.victim_address_min, self.victim_address_max+1):
            self.victim_addr_arr.append(i)
 
        # restore the total state
        #total_state = np.array([[]] * self.window_size)
        for i in range(len(self.env_list)):
            seed = -1#random.randint(1, 1000000)
            env = self.env_list[i]
            state = env.reset(victim_address = self.victim_addr_arr[i % len(self.victim_addr_arr)], seed = seed)
            #total_state = np.concatenate((total_state, state), axis=1) 
            
            if self.offline_training:
                state = self.offline_state 

        # reset the latency_buffer
        self.latency_buffer = []
        for i in range(0, self.secret_size * self.copy):
            self.latency_buffer.append([])

        #return total_state
        return self.reset_state

    # feed the actions to all subenv with different secret
    def step(self, action):
        early_done_reward = 0
        total_reward = 0
        total_state = [] 
        total_done = False
        done_arr = []
        total_state = np.array([[]] * self.window_size)
        #parsed_orig_action = action #self.env.parse_action(action)
        if action == self.action_space_size - 1: # guessing action
            #calculate the reward and terminate          
            for env in self.env_list:
                state, reward, done, info = env.step(action)
            total_state = self.reset_state 
            total_reward = self.P2oracle() 

            # for offline training the total_reward needs to include the history reward
            if self.offline_training == True:
               total_reward += self.offline_reward
               self.offline_reward = 0 

            total_done = True
        else:   # use the action and collect and concatenate observation
            i = 0
            for env in self.env_list:
                state, reward, done, info = env.step(action)
                latency = state[0][0]
                
                # for offline RL, we need to mask the state and accumulate reward
                if self.offline_training == True:
                    
                    # state is a n * 4 matrix
                    # r, victim_accesesd, original_action, self.step_count
                    # we only need to mask the r
                    #state[:,0] = self.offline_state[:, 0]
                    #print(state)
                    self.offline_reward += reward
                    reward = 0
                
                # length violation or other type of violation
                if done == True:
                    env.reset()
                    
                    # for offline training with stopping in the middle
                    # we still need to return the reward
                    if self.offline_training == True:
                        reward = self.offline_reward
                        self.offline_reward = 0
                    total_done = True

                self.latency_buffer[i].append(latency) #
                total_reward += reward
                total_state = np.concatenate((total_state, state), axis=1) 
                i += 1

        info = {}   
        total_reward = total_reward * 1.0 / len(self.env_list)#self.secret_size
        return total_state, total_reward, total_done, info     

    # given the existing sequence, calculate the P2 oracle reward
    # calculate the expected guessing correctness 
    def P2oracle(self):
        # score
        # calculate the total score
        # which correspond to the number of distinguishable secret
        latency_dict = {}
        for i in range(0, len(self.latency_buffer)):
            latency_dict[tuple(self.latency_buffer[i])] = 1
        score = 1.0 * len(latency_dict) / len(self.latency_buffer)
        print(' P2oracle score %f'% score)
        return score  * self.env.correct_reward + ( 1 - score ) * self.env.wrong_reward

    # use SVM to evaluate the guessability (oracle guessing correctness rate)
    def P2SVMOracle(self):
        if len(self.latency_buffer[0]) == 0:
            score = 0
        else:
            X = self.latency_buffer
            y = []
            for cp in range(0, self.copy):
                for sec in range(0, len(self.victim_addr_arr)):
                    y.append(self.victim_addr_arr[sec])
            clf = svm.SVC(random_state=0)
            print(len(X))
            print(len(y))
            #print(X)
            #print(y)
            ans = cross_val_score(clf, X, y, cv=4, scoring='accuracy')
            score = ans.mean()
        print("P2 SVM accuracy %f" % score)
        return score * self.env.correct_reward + ( 1 - score ) * self.env.wrong_reward
        

if __name__ == "__main__":
    from ray.rllib.agents.ppo import PPOTrainer
    import ray
    import ray.tune as tune
    ray.init(include_dashboard=False, ignore_reinit_error=True, num_gpus=1)
    if ray.is_initialized():
        ray.shutdown()
    #tune.register_env("cache_guessing_game_env_fix", CacheSimulatorSIMDWrapper)#
    tune.register_env("cache_guessing_game_env_fix", CacheSimulatorP1Wrapper)
    config = {
        'env': 'cache_guessing_game_env_fix', #'cache_simulator_diversity_wrapper',
        'env_config': {
            'verbose': 1,
            "force_victim_hit": False,
            'flush_inst': True,#False,
            "allow_victim_multi_access": True,#False,
            "attacker_addr_s": 0,
            "attacker_addr_e": 7,
            "victim_addr_s": 0,
            "victim_addr_e": 3,
            "reset_limit": 1,
            "cache_configs": {
                # YAML config file for cache simulaton
                "architecture": {
                  "word_size": 1, #bytes
                  "block_size": 1, #bytes
                  "write_back": True
                },
                "cache_1": {#required
                  "blocks": 4, 
                  "associativity": 1,  
                  "hit_time": 1 #cycles
                },
                "mem": {#required
                  "hit_time": 1000 #cycles
                }
            }
        }, 
        #'gamma': 0.9, 
        'num_gpus': 1, 
        'num_workers': 1, 
        'num_envs_per_worker': 1, 
        #'entropy_coeff': 0.001, 
        #'num_sgd_iter': 5, 
        #'vf_loss_coeff': 1e-05, 
        'model': {
            #'custom_model': 'test_model',#'rnn', 
            #'max_seq_len': 20, 
            #'custom_model_config': {
            #    'cell_size': 32
            #   }
        }, 
        'framework': 'torch',
    }
    #tune.run(PPOTrainer, config=config)
    trainer = PPOTrainer(config=config)
    def signal_handler(sig, frame):
        print('You pressed Ctrl+C!')
        checkpoint = trainer.save()
        print("checkpoint saved at", checkpoint)
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)
    while True:
        result = trainer.train()