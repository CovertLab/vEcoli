from vivarium.core.registry import (
    divider_registry,
    updater_registry,
)

from ecoli.processes.cell_division import (
    divide_by_domain,
    divide_RNAs_by_domain,
    divide_unique,
)

from ecoli.library.registry import divide_binomial, dict_value_updater, make_dict_value_updater


divider_registry.register('binomial_ecoli', divide_binomial)
divider_registry.register('by_domain', divide_by_domain)
divider_registry.register('rna_by_domain', divide_RNAs_by_domain)
divider_registry.register('divide_unique', divide_unique)

updater_registry.register('dict_value', dict_value_updater)

updater_registry.register('rnap_updater', make_dict_value_updater(
    unique_index=0,
    domain_index=0,
    coordinates=0,
    direction=True)
)