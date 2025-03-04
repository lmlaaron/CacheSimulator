import itertools
import random

def generate_base_patterns():
    """
    Generate base access patterns using multiples of 4 in hex (from 0x04 to 0x40).
    """
    base_patterns = []
    addresses = list(range(0x04, 0x44, 4))  # [0x04, 0x08, ..., 0x40]

    for i in range(2, len(addresses) + 1):
        base_patterns.append(addresses[:i])

    return base_patterns


def expand_patterns(patterns):
    """
    Expand base patterns using PrimeTime's methodology:
    - Repeat accesses
    - Reverse order variations
    - Interleave scope line (0x04)
    """
    expanded_patterns = []

    for pattern in patterns:
        expanded_patterns.append(pattern)  # Original pattern
        expanded_patterns.append(pattern * 2)  # Repeated pattern
        expanded_patterns.append(pattern[::-1])  # Reverse order

        # Insert scope line (0x04) in different ways
        for i in range(1, len(pattern) + 1):
            new_pattern = pattern[:i] + [0x04] + pattern[i:]
            expanded_patterns.append(new_pattern)

    return expanded_patterns


def mutate_patterns(patterns):
    """
    Mutate the best patterns to generate new ones for the next cycle.
    - Add repetitions
    - Swap two elements in some patterns
    - Remove one element in some patterns
    - Randomly shuffle others
    """
    mutated_patterns = []

    for pattern in patterns:
        if len(pattern) > 2:
            repeated_pattern = pattern * 2 
            mutated_patterns.append(repeated_pattern)

            swap_pattern = pattern[:]
            i, j = random.sample(range(len(pattern)), 2)
            swap_pattern[i], swap_pattern[j] = swap_pattern[j], swap_pattern[i]
            mutated_patterns.append(swap_pattern)

            remove_pattern = pattern[:]
            remove_pattern.pop(random.randint(0, len(pattern) - 1))
            mutated_patterns.append(remove_pattern)

            reversed_pattern = pattern[::-1]
            mutated_patterns.append(reversed_pattern)

        shuffle_pattern = pattern[:]
        random.shuffle(shuffle_pattern)
        mutated_patterns.append(shuffle_pattern)

    return mutated_patterns


def generate_patterns():
    """
    Complete PrimeTime-style pattern generation.
    """
    base_patterns = generate_base_patterns()
    expanded_patterns = expand_patterns(base_patterns)
    mutated_patterns = mutate_patterns(expanded_patterns)

    return base_patterns + expanded_patterns + mutated_patterns


if __name__ == "__main__":
    patterns = generate_patterns()
    for i, pattern in enumerate(patterns[:20]):
        print(f"Pattern {i+1}: {[hex(addr) for addr in pattern]}")

