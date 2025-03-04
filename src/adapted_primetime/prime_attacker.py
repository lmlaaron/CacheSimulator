import logging

class PrimeAttacker:
    def __init__(self, cache_hierarchy, patterns):
        self.cache = cache_hierarchy['cache_1']
        self.logger = logging.getLogger("PrimeAttacker")
        self.patterns = patterns
        self.scope_line = hex(0x04)  # Scope line always at 0x04

    def prime(self, pattern):
        """
        Prime phase: Fill the cache set with known addresses.
        The goal is to set the scope line (0x04) as the EVC in shared cache.
        """
        self.logger.info(f"Priming cache with pattern: {pattern}")
        for address in pattern:
            self.cache.read(hex(address), 0)  # Access addresses

        self.logger.info(f"Verifying scope line {self.scope_line} is present in shared cache...")
        time, _, _, _  = self.cache.read(self.scope_line, 0)
        self.logger.info(f"Scope line in cache after prime: {(time.time < 501)}")

    def scope(self):
        """
        Scope phase: Access the scope line (0x04).
        This is done twice:
        - Before victim access (ensures scope line is in attacker's private cache)
        - After victim access (checks if scope line was evicted from private cache)
        """
        self.logger.info(f"Accessing scope line {self.scope_line}")
        time, _, _, _ = self.cache.read(self.scope_line, 0)
        hit = time.time < 501
        if hit:
            self.logger.info(f"Scope line {self.scope_line} is still in cache!")
        else:
            self.logger.info(f"Scope line {self.scope_line} was evicted!")

        return hit

