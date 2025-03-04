import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from prime_attacker import PrimeAttacker
from prime_victim import PrimeVictim
from prime_patterns import generate_patterns, mutate_patterns
from prime_results import analyze_results
from cache_simulator import build_hierarchy
import logging

NUM_CYCLES = 5 

def run_attack(attacker, victim, pattern):
    """
    Execute the full Prime+Scope attack sequence.
    1. Prime: Set scope line as EVC in shared cache.
    2. Scope (Pre-Victim): Confirm scope line is in private cache.
    3. Victim: Accesses memory, evicting scope line.
    4. Scope (Post-Victim): Check if scope line was evicted.
    """
    logging.info("=== Running Attack ===")

    victim.access()
    attacker.prime(pattern)  
    pre_victim_scope = attacker.scope() 
    victim.access() 
    post_victim_scope = attacker.scope() 

    evicted = pre_victim_scope and not post_victim_scope 
    logging.info(f"Scope Line Evicted: {evicted}")

    return pattern, evicted


def main():
    logging.basicConfig(level=logging.INFO, format='%(message)s')
    logger = logging.getLogger("PrimeTime-Sim")

    config = {
        "architecture": {"word_size": 1, "block_size": 1, "write_back": True},
        "cache_1": {"blocks": 32, "associativity": 32, "hit_time": 1, "rep_policy": "plru"},
        "cache_1_core_2": {"blocks": 32, "associativity": 32, "hit_time": 1, "rep_policy": "plru"},
        "cache_2": {"blocks": 16, "associativity": 4, "hit_time": 501, "rep_policy": "plru"},
        "cache_2_core_2": {"blocks": 16, "associativity": 4, "hit_time": 501, "rep_policy": "plru"},
        "cache_3": {"blocks": 64, "associativity": 16, "hit_time": 501, "rep_policy": "brrip"},
        "mem": {"hit_time": 1000}
    }
    
    cache_hierarchy = build_hierarchy(config, logger)

    attacker = PrimeAttacker(cache_hierarchy, [])
    victim = PrimeVictim(cache_hierarchy)

    patterns = generate_patterns() 
    for cycle in range(NUM_CYCLES):
        logging.info(f"\n===== Cycle {cycle + 1}/{NUM_CYCLES} =====")

        results = [run_attack(attacker, victim, pattern) for pattern in patterns]

        top_patterns = analyze_results(results)

        patterns = mutate_patterns(top_patterns)

    logging.info("=== PrimeTime Simulation Complete ===")


if __name__ == "__main__":
    main()

