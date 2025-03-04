import logging

class PrimeVictim:
    def __init__(self, cache_hierarchy):
        self.cache = cache_hierarchy['cache_1_core_2']
        self.logger = logging.getLogger("PrimeVictim")

    def access(self):
        """
        Victim accesses a memory address that might be targeted by the attacker.
        """
        victim_address = hex(0) 
        self.logger.info(f"Victim accessing memory at {victim_address}")
        self.cache.read(victim_address, 0)
