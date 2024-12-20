# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This software may be used and distributed according to the terms of the
# GNU General Public License version 2.

# a textbook prime probe attacker that serve as the agent 
# which can have high reward for the cache guessing game
# used to generate the attack sequence that can be detected by cchunter
# currently it only works for the direct-map cache (associativity=1)
import bisect
import random
class TextbookAgent():

    # the config is the same as the config cor cache_guessing_game_env_impl
    def __init__(self, env_config):
        self.local_step = 0 
        self.lat = []
        self.no_prime = False # set to true after first prime
        if "cache_configs" in env_config:
            #self.logger.info('Load config from JSON')
            self.configs = env_config["cache_configs"]
            self.num_ways = self.configs['cache_1']['associativity'] 
            self.cache_size = self.configs['cache_1']['blocks']
            attacker_addr_s = env_config["attacker_addr_s"] if "attacker_addr_s" in env_config else 4
            attacker_addr_e = env_config["attacker_addr_e"] if "attacker_addr_e" in env_config else 7
            victim_addr_s = env_config["victim_addr_s"] if "victim_addr_s" in env_config else 0
            victim_addr_e = env_config["victim_addr_e"] if "victim_addr_e" in env_config else 3
            flush_inst = env_config["flush_inst"] if "flush_inst" in env_config else False            
            self.allow_empty_victim_access = env_config["allow_empty_victim_access"] if "allow_empty_victim_access" in env_config else False
            
            assert(self.num_ways == 1) # currently only support direct-map cache
            assert(flush_inst == False) # do not allow flush instruction
            assert(attacker_addr_e - attacker_addr_s == victim_addr_e - victim_addr_s ) # address space must be shared
            #must be no shared address space
            assert( ( attacker_addr_e + 1 == victim_addr_s ) or ( victim_addr_e + 1 == attacker_addr_s ) )
            assert(self.allow_empty_victim_access == False)

    # initialize the agent with an observation
    def observe_init(self, timestep):
        # initialization doing nothing
        self.local_step = 0
        self.lat = []
        self.no_prime = False
        return


    # returns an action
    def act(self, timestep):
        info = {}
        # do prime
        if self.local_step < self.cache_size -  ( self.cache_size if self.no_prime else 0 ):#- 1:
            action = self.local_step # do prime 
            self.local_step += 1
            return action, info

        elif self.local_step == self.cache_size - (self.cache_size if self.no_prime else 0 ):#- 1: # do victim trigger
            action = self.cache_size # do victim access
            self.local_step += 1
            return action, info

        elif self.local_step < 2 * self.cache_size + 1 -(self.cache_size if self.no_prime else 0 ):#- 1 - 1:# do probe
            action = self.local_step - ( self.cache_size + 1 - (self.cache_size if self.no_prime else 0 ) )#- 1 )  
            self.local_step += 1
            #timestep,state i state
            # timestep.state[0] is [r victim_accessesd original_action self_count]
            #self.lat.append(timestep.observation[0][0][0])
            #print(timestep.observation)
            return action, info

        elif self.local_step == 2 * self.cache_size + 1 - (self.cache_size if self.no_prime else 0 ):# - 1 - 1: # do guess and terminate
            # timestep is the observation from last step
            # first timestep not useful
            action = 2 * self.cache_size # default assume that last is miss
            for addr in range(1, len(self.lat)):
                if self.lat[addr].int() == 1: # miss
                    action = addr + self.cache_size 
                    break
            self.local_step = 0
            self.lat=[]
            self.no_prime = True
            return action, info
        else:        
            assert(False)
    # is it useful for non-ML agent or not???
    def observe(self, action, timestep):
        if self.local_step < 2 * self.cache_size + 1 + 1 - (self.cache_size if self.no_prime else 0 ) and self.local_step > self.cache_size - (self.cache_size if self.no_prime else 0 ):#- 1:
        ##    self.local_step += 1
            self.lat.append(timestep.observation[0][0])
        return



class RandomAgent():

    # the config is the same as the config cor cache_guessing_game_env_impl
    def __init__(self, env_config):
        random.seed()
        self.local_step = 0
        self.lat = []
        self.no_prime = False # set to true after first prime
        if "cache_configs" in env_config:
            #self.logger.info('Load config from JSON')
            self.configs = env_config["cache_configs"]
            self.num_ways = self.configs['cache_1']['associativity'] 
            self.cache_size = self.configs['cache_1']['blocks']
            self.attacker_addr_s = env_config["attacker_addr_s"] if "attacker_addr_s" in env_config else 4
            self.attacker_addr_e = env_config["attacker_addr_e"] if "attacker_addr_e" in env_config else 7
            victim_addr_s = env_config["victim_addr_s"] if "victim_addr_s" in env_config else 0
            victim_addr_e = env_config["victim_addr_e"] if "victim_addr_e" in env_config else 3
            flush_inst = env_config["flush_inst"] if "flush_inst" in env_config else False            
            self.allow_empty_victim_access = env_config["allow_empty_victim_access"] if "allow_empty_victim_access" in env_config else False
            
    # initialize the agent with an observation
    def observe_init(self, timestep):
        # initialization doing nothing
        self.local_step = 0
        self.lat = []
        self.no_prime = False
        return


    # returns an action
    def act(self, timestep):
        info = {}
        # do prime
        action = random.randint(0, 1 + self.attacker_addr_e - self.attacker_addr_s) 
        self.local_step += 1
        return action, info

    # is it useful for non-ML agent or not???
    def observe(self, action, timestep):
        ####if self.local_step < 2 * self.cache_size + 1 + 1 - (self.cache_size if self.no_prime else 0 ) and self.local_step > self.cache_size - (self.cache_size if self.no_prime else 0 ):#- 1:
        ######    self.local_step += 1
        ####    self.lat.append(timestep.observation[0][0])
        return

 

class OccupancyAgent():

    # the config is the same as the config cor cache_guessing_game_env_impl
    def __init__(self, env_config):
        self.local_step = 0
        self.lat = []
        self.no_prime = False # set to true after first prime
        if "cache_configs" in env_config:
            #self.logger.info('Load config from JSON')
            self.configs = env_config["cache_configs"]
            self.num_ways = self.configs['cache_1']['associativity'] 
            self.cache_size = self.configs['cache_1']['blocks']
            self.attacker_addr_s = env_config["attacker_addr_s"] if "attacker_addr_s" in env_config else 4
            self.attacker_addr_e = env_config["attacker_addr_e"] if "attacker_addr_e" in env_config else 7
            victim_addr_s = env_config["victim_addr_s"] if "victim_addr_s" in env_config else 0
            victim_addr_e = env_config["victim_addr_e"] if "victim_addr_e" in env_config else 3
            flush_inst = env_config["flush_inst"] if "flush_inst" in env_config else False            
            self.allow_empty_victim_access = env_config["allow_empty_victim_access"] if "allow_empty_victim_access" in env_config else False
            
    # initialize the agent with an observation
    def observe_init(self, timestep):
        # initialization doing nothing
        self.local_step = 0
        self.lat = []
        self.no_prime = False
        return


    # returns an action
    def act(self, timestep):
        info = {}
        # do prime
        action = self.local_step % ( 1 + self.attacker_addr_e - self.attacker_addr_s )  # do prime 
        self.local_step += 1
        return action, info

    # is it useful for non-ML agent or not???
    def observe(self, action, timestep):
        ####if self.local_step < 2 * self.cache_size + 1 + 1 - (self.cache_size if self.no_prime else 0 ) and self.local_step > self.cache_size - (self.cache_size if self.no_prime else 0 ):#- 1:
        ######    self.local_step += 1
        ####    self.lat.append(timestep.observation[0][0])
        return

# single holdout method for eviction set finding
class SHMAgent():
    # the config is the same as the config cor cache_guessing_game_env_impl
    def __init__(self, env_config, name="default"):
        self.name = name
        self.local_step = -1
        self.lat = []
        self.no_prime = False # set to true after first prime
        self.cand_action = 0
        if "cache_configs" in env_config:
            #self.logger.info('Load config from JSON')
            self.configs = env_config["cache_configs"]
            self.num_ways = self.configs['cache_1']['associativity'] 
            self.cache_size = self.configs['cache_1']['blocks']
            self.attacker_addr_s = env_config["attacker_addr_s"] if "attacker_addr_s" in env_config else 4
            self.attacker_addr_e = env_config["attacker_addr_e"] if "attacker_addr_e" in env_config else 7
            victim_addr_s = env_config["victim_addr_s"] if "victim_addr_s" in env_config else 0
            victim_addr_e = env_config["victim_addr_e"] if "victim_addr_e" in env_config else 3
            flush_inst = env_config["flush_inst"] if "flush_inst" in env_config else False            
            self.allow_empty_victim_access = env_config["allow_empty_victim_access"] if "allow_empty_victim_access" in env_config else False

        self.candidates = []
        for i in  range(0,1 + self.attacker_addr_e - self.attacker_addr_s):
            self.candidates.append(i)    

        self.cand_action = 0
        act = -1
        for act in self.candidates:
          if act >= self.cand_action:
            self.cand_action = act
            break

        self.candidates.remove(self.cand_action)
        self.evset_size = 0

    def get_evset_size(self):
        return self.evset_size

    # initialize the agent with an observation
    def observe_init(self, timestep):
        # initialization doing nothing
        self.local_step = 0 
        self.lat = []
        self.no_prime = False
        self.cand_action = 0

        self.evset_size = 0
        self.candidates = []
        for i in  range(0,1 + self.attacker_addr_e - self.attacker_addr_s):
            self.candidates.append(i)    

        act = -1
        for act in self.candidates:
          if act >= self.cand_action:
            self.cand_action = act
            break
        
        self.candidates.remove(self.cand_action)

        #self.cand_action = self.candidates[0]
        return


    # returns an action
    def act(self, timestep):
        info = {}
        #####print(self.candidates)
        #####print(self.local_step)
        #####print(self.cand_action)
        ###### print(self.lat)
        # do prime
        #action = self.local_step % ( 1 + self.attacker_addr_e - self.attacker_addr_s )  # do prime 
        if self.local_step < 0:
            action =  1 + self.attacker_addr_e - self.attacker_addr_s  # victim access
            self.local_step += 1
        elif self.local_step < len(self.candidates) :
            # prime
            action = self.candidates[self.local_step]
            self.local_step += 1
        elif self.local_step == len(self.candidates) :
            action =  1 + self.attacker_addr_e - self.attacker_addr_s  # victim access
            self.local_step += 1
        else: 
            if self.lat[-1].int() == 1: # check if add to evset or 
                #print(type(self.lat[-1]))
                #exit(-1)
                act = -1
                for act in self.candidates:
                  if act >= self.cand_action:
                    self.cand_action = act
                    break

                #self.cand_action = self.candidates[0]
                #if act == -1:
                
                #else:
                #print("check")
                #print(self.candidates)
                #print(self.cand_action)
                self.candidates.remove(self.cand_action)
                
                self.local_step = 0
                action = self.candidates[self.local_step] #self.cand_action + 2 + self.attacker_addr_e - self.attacker_addr_s 
                self.local_step = 1
            else:
                action = self.cand_action + 1  + 1 + self.attacker_addr_e - self.attacker_addr_s   
                self.evset_size += 1
                #print("action " + str(action))
                #print("self.candidates")
                bisect.insort(self.candidates, self.cand_action)
                #print(self.candidates)
                act = -1
                for act in self.candidates:
                  if act > self.cand_action:
                    self.cand_action = act
                    break
                
                if act == -1:
                    exit(-1)
                else:    
                    self.candidates.remove(self.cand_action)
                
                #print(self.candidates)
                self.local_step = 0 
        ####print(self.name + ' action')
        ####print(action)
        return action, info

    # is it useful for non-ML agent or not???
    def observe(self, action, timestep):
        ####if self.local_step < 2 * self.cache_size + 1 + 1 - (self.cache_size if self.no_prime else 0 ) and self.local_step > self.cache_size - (self.cache_size if self.no_prime else 0 ):#- 1:
        ######    self.local_step += 1
        ####print(self.name + ' observe ')
        ####print(timestep.observation[0][0].int())
        self.lat.append(timestep.observation[0][0])
        return


