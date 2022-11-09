''' updated to work without rlmeta environment. action offsets modified for 1set-8way. Prepared for ICLR rebuttal.'''

class PrimeProbeAgent():

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
        if timestep[0][0] == -1:
            #reset the attacker
            #from IPython import embed; embed()
            self.local_step = 0
            self.lat=[]
            self.no_prime = False

        # prime phase
        if self.local_step < self.cache_size - ( self.cache_size if self.no_prime else 0 ) :
            action = self.local_step - (self.cache_size if self.no_prime else 0 ) 
            self.local_step += 1
            return action, info

        # do victim trigger
        elif self.local_step == self.cache_size - (self.cache_size if self.no_prime else 0 ) : 
            action = self.cache_size # do victim access
            self.local_step += 1
            return action, info

        # probe phase
        elif self.local_step < 2 * self.cache_size -(self.cache_size if self.no_prime else 0 ) +1 :
            action = self.local_step -9 - (self.cache_size if self.no_prime else 0)
            self.local_step += 1
            return action, info

        # do guess and terminate
        elif self.local_step == 2 * self.cache_size + 1 - (self.cache_size if self.no_prime else 0 ):
            action = self.local_step -7 
            for addr in range(1, len(self.lat)):
                if self.lat[addr] == 1: # miss
                    action = addr + 1 * self.cache_size 
                    break
            self.local_step = 0
            self.lat=[]
            #self.no_prime = True
            
            if action > 1 * self.cache_size:
                action+=1
            return action, info
        else:        
            assert(False)
            
    def observe(self, action, timestep):
        if self.local_step < 2 * self.cache_size + 1 + 1 - (self.cache_size if self.no_prime else 0 ) and self.local_step > self.cache_size - (self.cache_size if self.no_prime else 0 ):#- 1:
        ##    self.local_step += 1
            self.lat.append(timestep[0][0])
        return
