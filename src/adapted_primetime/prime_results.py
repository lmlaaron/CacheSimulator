import logging

def analyze_results(results):
    """
    Analyze attack results, rank patterns based on eviction success,
    and return only the best patterns for the next cycle.
    """
    logger = logging.getLogger("PrimeResults")

    # Sort by eviction success (True > False)
    sorted_results = sorted(results, key=lambda x: x[1], reverse=True)

    # Only keep patterns that successfully evicted the scope line at least ONCE
    top_patterns = [pattern for pattern, evicted in sorted_results if evicted]

    if not top_patterns:
        logger.info("No successful eviction patterns found. Keeping top 5 failed patterns for mutation.")
        top_patterns = [pattern for pattern, evicted in sorted_results[:5]]  # Keep top 5 for mutation

    logger.info("\n===== Best Eviction Patterns =====")
    for pattern in top_patterns:
        logger.info(f"Pattern {pattern}")

    return top_patterns  # Return best patterns for mutation



