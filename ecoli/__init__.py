import numpy as np

from vivarium.core.registry import (
    divider_registry,
    updater_registry,
)

from ecoli.processes.cell_division import (
    divide_by_domain,
    divide_RNAs_by_domain,
    divide_unique,
)


def divide_binomial(state):
    """Binomial Divider
    """
    try:
        counts_1 = np.random.binomial(state, 0.5)
        counts_2 = state - counts_1
    except:
        print(f"binomial_divider can not divide {state}.")
        counts_1 = state
        counts_2 = state

    return [counts_1, counts_2]

divider_registry.register('binomial_ecoli', divide_binomial)
divider_registry.register('by_domain', divide_by_domain)
divider_registry.register('rna_by_domain', divide_RNAs_by_domain)
divider_registry.register('divide_unique', divide_unique)