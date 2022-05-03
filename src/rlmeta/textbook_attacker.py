# a textbook prime probe attacker that serve as the agent 
# which can have high reward for the cache guessing game
# used to generate the attack sequence that can be detected by cchunter
# currently it only works for the direct-map cache (associativity=1)
class TextbookAgent():

    # the config is the same as the config cor cache_guessing_game_env_impl
    def __init__(self, env_config):
        self.local_step = 0
        self.lat = []
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
        return


    # returns an action
    def act(self, timestep):
        info = {}
        # do prime
        if self.local_step < self.cache_size:
            action = self.local_step # do prime 
            self.local_step += 1
            return action, info

        elif self.local_step == self.cache_size: # do victim trigger
            action = self.local_step
            self.local_step += 1
            return action, info

        elif self.local_step < 2 * self.cache_size + 1:# do probe
            action = self.local_step - ( self.cache_size + 1 )  
            self.local_step += 1
            #timestep,state i state
            # timestep.state[0] is [r victim_accessesd original_action self_count]

            self.lat.append(timestep.state[0][0])
            return action, info

        elif self.local_step == 2 * self.cache_size + 1: # do guess and terminate
            # timestep is the observation from last step
            # first timestep not useful
            action = self.cache_size + 1 # default assume that first miss
            for addr in range(1, self.lat):
                if self.lat[addr] == 1: # miss
                    action = addr + self.cache_size + 1
            self.local_step = 0
            return action, info
        else:        
            assert(False)
    # is it useful for non-ML agent or not???
    def observe(self, action, timestep):
        return


    
    