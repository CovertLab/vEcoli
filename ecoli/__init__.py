from vivarium.core.registry import (
    divider_registry,
    updater_registry,
)
from ecoli.library.schema import (
    UNIQUE_DEFAULTS,
    divide_binomial,
    dict_value_updater,
    make_dict_value_updater, divide_by_domain, divide_unique, divide_RNAs_by_domain, divide_domain, empty_dict_divider,
    divide_ribosomes
)

# register :term:`updaters`
updater_registry.register('dict_value', dict_value_updater)
for unique_mol, defaults in UNIQUE_DEFAULTS.items():
    updater_registry.register(f'{unique_mol}_updater',
                              make_dict_value_updater(defaults))

# register :term:`dividers`
divider_registry.register('binomial_ecoli', divide_binomial)
divider_registry.register('by_domain', divide_by_domain)
divider_registry.register('divide_domain', divide_domain)
divider_registry.register('rna_by_domain', divide_RNAs_by_domain)
divider_registry.register('divide_unique', divide_unique)
divider_registry.register('empty_dict', empty_dict_divider)
divider_registry.register('divide_ribosomes', divide_ribosomes)
