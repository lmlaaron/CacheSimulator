# Author: Mulong Luo
# date 2021.12.3
# description: environment for study RL for side channel attack
import gym
from gym import spaces
import numpy as np
import random
import os
import yaml, logging
from cache_simulator import *
import sys

class CacheGuessingGameEnv(gym.Env):
  """
  Description:
    A L1 cache with total_size, num_ways 
    assume cache_line_size == 1B
  
  Observation:
    # let's book keep all obvious information in the observation space 
    # since the agent is dumb
    self.observation_space = spaces.MultiDiscrete(
      [3,                 #cache latency
      20,                 #current steps
      2,                  #whether the victim has accessed yet
      ]

  Actions:
    # action step contains four values
    # 1. access address
    # 2. whether to end and make a guess now?
    # 3. whether to invoke the victim access
    # 4. if make a guess, what is the victim's accessed address?

  Reward:

  Starting state:
    fresh cache with nolines
  
  Episode termination:
    when the attacker make a guess
    when there is length violation
    when there is guess before victim violation
    episode terminates
  """
  metadata = {'render.modes': ['human']}

  def __init__(self, env_config={
   "length_violation_reward":-10000,
   "double_victim_access_reward": -10000,
   "force_victim_hit": False,
   "victim_access_reward":-10,
   "correct_reward":200,
   "wrong_reward":-9999,
   "step_reward":-1,
   "window_size":0,
   "attacker_addr_s":4,
   "attacker_addr_e":7,
   "victim_addr_s":0,
   "victim_addr_e":3,
   "flush_inst": False,
   "allow_victim_multi_access": True,
   "verbose":0,
   "reset_limit": 1,    # specify how many reset to end an epoch?????
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
  }
):
    self.force_victim_hit =env_config["force_victim_hit"] if "force_victim_hit" in env_config else False
    self.length_violation_reward = env_config["length_violation_reward"] if "length_violation_reward" in env_config else -10000
    self.victim_access_reward = env_config["victim_access_reward"] if "victim_access_reward" in env_config else -10
    self.double_victim_access_reward = env_config["double_victim_access_reward"] if "double_victim_access_reward" in env_config else -10000
    self.allow_victim_multi_access = env_config["allow_victim_multi_access"] if "allow_victim_multi_access" in env_config else True
    self.correct_reward = env_config["correct_reward"] if "correct_reward" in env_config else 200
    self.wrong_reward = env_config["wrong_reward"] if "wrong_reward" in env_config else -9999
    self.step_reward = env_config["step_reward"] if "step_reward" in env_config else 0
    self.reset_limit = env_config["reset_limit"] if "reset_limit" in env_config else 1
    window_size = env_config["window_size"] if "window_size" in env_config else 0
    attacker_addr_s = env_config["attacker_addr_s"] if "attacker_addr_s" in env_config else 4
    attacker_addr_e = env_config["attacker_addr_e"] if "attacker_addr_e" in env_config else 7
    victim_addr_s = env_config["victim_addr_s"] if "victim_addr_s" in env_config else 0
    victim_addr_e = env_config["victim_addr_e"] if "victim_addr_e" in env_config else 3
    flush_inst = env_config["flush_inst"] if "flush_inst" in env_config else False
    self.verbose = env_config["verbose"] if "verbose" in env_config else 0
    self.logger = logging.getLogger()
    self.fh = logging.FileHandler('log')
    self.sh = logging.StreamHandler()
    self.logger.addHandler(self.fh)
    self.logger.addHandler(self.sh)
    self.fh_format = logging.Formatter('%(message)s')
    self.fh.setFormatter(self.fh_format)
    self.sh.setFormatter(self.fh_format)
    self.logger.setLevel(logging.INFO)
    if "cache_configs" in env_config:
      self.logger.info('Load config from JSON')
      self.configs = env_config["cache_configs"]
    else:
      self.config_file_name = os.path.dirname(os.path.abspath(__file__))+'/../configs/config_simple_L1'
      self.config_file = open(self.config_file_name)
      self.logger.info('Loading config from file ' + self.config_file_name)
      self.configs = yaml.load(self.config_file, yaml.CLoader)
    self.vprint(self.configs)

    self.num_ways = self.configs['cache_1']['associativity'] 
    self.cache_size = self.configs['cache_1']['blocks']
    
    if window_size == 0:
      self.window_size = self.cache_size * 2 + 8 #10 
    else:
      self.window_size = window_size
    self.hierarchy = build_hierarchy(self.configs, self.logger)
    self.state = [0, self.cache_size, 0, 0] * self.window_size
    self.attacker_address_min = attacker_addr_s
    self.attacker_address_max = attacker_addr_e
    self.attacker_address_space = range(self.attacker_address_min,
                                  self.attacker_address_max +
                                  1)  # start with one attacker cache line
    self.victim_address_min = victim_addr_s
    self.victim_address_max = victim_addr_e
    self.victim_address_space = range(self.victim_address_min,
                                self.victim_address_max + 1)  #
    self.flush_inst = flush_inst
    self.reset_time = 0
    # action step contains four values
    # 1. access address
    # 2. whether to end and make a guess now?
    # 3. whether to invoke the victim access
    # 4. if make a guess, what is the victim's accessed address?
    ####self.action_space = spaces.MultiDiscrete(
    ####  [self.cache_size,     #cache access
    ####  2,                    #whether to make a guess
    ####  2,                    #whether to invoke victim access
    ####  2,                    #whether it is a flush inst 
    ####  self.cache_size       #what is the guess of the victim's access
    ####  ])
    
    self.action_space = spaces.Discrete(
      len(self.attacker_address_space) * 2 * 2 * 2 * len(self.victim_address_space)
    )

    # let's book keep all obvious information in the observation space 
    # since the agent is dumb
    self.observation_space = spaces.MultiDiscrete(
      [
        3,                                          #cache latency
        len(self.attacker_address_space) + 1,       #attacker accessed address
        self.window_size + 2,                       #current steps
        2,                                          #whether the victim has accessed yet
      ] * self.window_size
    )
    print('Initializing...')
    self.l1 = self.hierarchy['cache_1']
    self.current_step = 0
    self.victim_accessed = False
    self.victim_address = random.randint(self.victim_address_min, self.victim_address_max + 1)
    self._randomize_cache()

    # internal guessing buffer
    # does not change after reset
    self.guess_buffer_size = 20
    self.guess_buffer = [False] * self.guess_buffer_size
    #return


  def step(self, action):
    print('Step...')
    if action.ndim > 1:  # workaround for training and predict discrepency
      action = action[0]  

    #temp_action=[]
    #temp_action.append(int(action / ( 2 * 2 * 2 * self.cache_size)) )   
    #temp_action.append(int(action / ( 2 * 2 * self.cache_size)) % 2 )   
    #temp_action.append(int(action / ( 2 * self.cache_size)) % 2 )       
    #temp_action.append(int(action / self.cache_size) % 2 )              
    #temp_action.append(action %  self.cache_size)                      
    #action = temp_action
    
    action = self.parse_action(action)
    address = str(action[0]+self.attacker_address_min)                # attacker address in attacker_address_space
    is_guess = action[1]                                              # check whether to guess or not
    is_victim = action[2]                                             # check whether to invoke victim
    is_flush = action[3]                                              # check whether to flush
    victim_addr = str(action[4] + self.victim_address_min)            # victim address

    if self.current_step > self.window_size : # if current_step is too long, terminate
      r = 2 #
      self.vprint("length violation!")
      reward = self.length_violation_reward #-10000 
      done = True
    else:
      if is_victim == True:
        if self.allow_victim_multi_access == True or self.victim_accessed == False:
          r = 2 #
          self.victim_accessed = True
          self.vprint("victim access %d" % self.victim_address)
          t = self.l1.read(str(self.victim_address), self.current_step).time
          if self.force_victim_hit == True and t > 500:   # for LRU attack, has to force victim access being hit
            self.current_step += 1
            reward = -5000
            done = False
            self.vprint("victim access has to be hit! terminate!")
          else:
            self.current_step += 1
            reward = self.victim_access_reward #-10
            done = False
        else:
          r = 2
          self.vprint("does not allow multi victim access in this config, terminate!")
          self.current_step += 1
          reward = self.double_victim_access_reward # -10000
          done = True
      else:
        if is_guess == True:
          r = 2  # 
          if self.victim_accessed and victim_addr == str(self.victim_address):
              self.vprint("correct guess " + victim_addr)
              # update the guess buffer 
              self.guess_buffer.append(True)
              self.guess_buffer.pop(0) 
              reward = self.correct_reward # 200
              done = True
          else:
              self.vprint("wrong guess " + victim_addr )
              # update the guess buffer 
              self.guess_buffer.append(False)
              self.guess_buffer.pop(0) 
              reward = self.wrong_reward #-9999
              done = True
        elif is_flush == False or self.flush_inst == False:
          if self.l1.read(address, self.current_step).time > 500: # measure the access latency
            self.vprint("acceee " + address + " miss")
            r = 1 # cache miss
          else:
            self.vprint("access " + address + " hit"  )
            r = 0 # cache hit
          self.current_step += 1
          reward = self.step_reward #-1 
          done = False
        else:    # is_flush == True
          self.l1.cflush(address, self.current_step)
          self.vprint("cflush " + address )
          r = 2
          self.current_step += 1
          reward = self.step_reward
          done = False
    #return observation, reward, done, info
    info = {}
    # the observation (r.time) in this case 
    # must be consistent with the observation space
    # return observation, reward, done?, info
    #return r, reward, done, info
    current_step = self.current_step
    if self.victim_accessed == True:
      victim_accessed = 1
    else:
      victim_accessed = 0
    self.state = [r, action[0], current_step, victim_accessed] + self.state 
    self.state = self.state[0:len(self.state)-4]
    #self.state = [r, action[0], current_step, victim_accessed]
    
    '''
    support for multiple guess per episode
    '''
    if done == True:
      self.reset_time += 1
      if self.reset_time == self.reset_limit:  # really need to end the simulation
        self.reset_time = 0
        done = True                            # reset will be called by the agent/framework
        self.vprint('correct rate:' + str(self.calc_correct_rate()))
      else:
        done = False                           # fake reset
        self._reset()                          # manually reset

    return np.array(self.state), reward, done, info

  def reset(self, victim_address=-1):
    print('Reset...')
    self.hierarchy = build_hierarchy(self.configs, self.logger)
    self.l1 = self.hierarchy['cache_1']
    self._reset(victim_address)  # fake reset
    self.state = [0, len(self.attacker_address_space), 0, 0] * self.window_size
    self.reset_time = 0
    return np.array(self.state)

  '''
  function to calculate the correctness rate
  '''
  def calc_correct_rate(self):
    return self.guess_buffer.count(True) /len(self.guess_buffer)

  '''
  evluate the correctness of an action sequence (action+ latency) 
  action_buffer: list [(action, latency)]
  '''
  def calc_correct_seq(self, action_buffer):
    last_action, _ = action_buffer[-1]
    last_action = self.parse_action(last_action)
    print(last_action)
    guess_addr = last_action[4]
    print(guess_addr)
    self.reset()
    self.total_guess = 0
    self.correct_guess = 0
    while self.total_guess < 20:
      self.reset()
      for i in range(0, len(action_buffer)):
        p = action_buffer[i]
        state, _, _, _ = self.step(p[0])
        latency = state[0]
        if latency != p[1]:
          break
      if i < len(action_buffer) - 1:
        continue
      else:
        self.total_guess += 1
        if guess_addr == self.victim_address:
          self.correct_guess += 1
    return self.correct_guess / self.total_guess

  # fake reset, just set a new victim addr 
  def _reset(self, victim_address=-1):
    self.current_step = 0
    self.victim_accessed = False
    if victim_address == -1:
      self.victim_address = random.randint(self.victim_address_min, self.victim_address_max) 
    else:
      assert(victim_address >= self.victim_address_min)
      assert(victim_address <= self.victim_address_max)
      self.victim_address = victim_address
    self.vprint("victim address %d", self.victim_address)
    self._randomize_cache()

  def render(self, mode='human'):
    return 

  def close(self):
    return

  def _randomize_cache(self, mode = "attacker"):
    #self.l1.read(str(2), -2)
    #self.l1.read(str(3), -1)
    #return
    if mode == "none":
      return
    self.current_step = -self.cache_size * 2 
    for _ in range(self.cache_size * 2):
      if mode == "victim":
        addr = random.randint(self.victim_address_min, self.victim_address_max)
      elif mode == "attacker":
        addr = random.randint(self.attacker_address_min, self.attacker_address_max)
      elif mode == "union":
        addr = random.randint(self.victim_address_min, self.victim_address_max) if random.randint(0,1) == 1 else random.randint(self.attacker_address_min, self.attacker_address_max)
      elif mode == "random":
        addr = random.randint(0, sys.maxsize)
      else:
        raise RuntimeError from None
      #if partial == True:
        #addr = random.randint(0,self.cache_size * 2)
        #if addr == self.cache_size * 2:
        #self.l1.cflush(str(addr), self.current_step)
      #else:
      self.l1.read(str(addr), self.current_step)
      self.current_step += 1

  def get_obs_space_dim(self):
    return int(np.prod(self.observation_space.shape))

  def get_act_space_dim(self):
    return int(np.prod(self.action_space.shape))

  def vprint(self, *args):
    if self.verbose == 1:
      print( " "+" ".join(map(str,args))+" ")

  '''
  parse the action 
  returns list of 5 elements representing
  address, is_guess, is_victim, is_flush, victim_addr
  '''
  def parse_action(self, action):
    temp_action=[]
    temp_action.append(int(action / ( 2 * 2 * 2 * self.cache_size)) )   
    temp_action.append(int(action / ( 2 * 2 * self.cache_size)) % 2 )   
    temp_action.append(int(action / ( 2 * self.cache_size)) % 2 )       
    temp_action.append(int(action / self.cache_size) % 2 )              
    temp_action.append(action %  self.cache_size)                      
    return temp_action