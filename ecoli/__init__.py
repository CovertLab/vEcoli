from vivarium.core.registry import (
    divider_registry,
    updater_registry,
    serializer_registry,
)
from ecoli.library.schema import (
    UNIQUE_DEFAULTS,
    divide_binomial,
    make_dict_value_updater,
    divide_by_domain,
    divide_unique,
    divide_RNAs_by_domain,
    divide_domain,
    empty_dict_divider,
    divide_ribosomes,
    divide_set_none,
)
from ecoli.library.updaters import (
    inverse_updater_registry,
    inverse_update_accumulate,
    inverse_update_set,
    inverse_update_null,
    inverse_update_merge,
    inverse_update_nonnegative_accumulate,
    inverse_update_dictionary,
)
from ecoli.library.serialize import UnumSerializer

# register :term:`updaters`
for unique_mol, defaults in UNIQUE_DEFAULTS.items():
    updater_registry.register(f'{unique_mol}_updater',
                              make_dict_value_updater(defaults))
    # TODO: Handle defaults in inverse updater.
    inverse_updater_registry.register(
        f'{unique_mol}_updater', inverse_update_dictionary)

inverse_updater_registry.register(
    'accumulate', inverse_update_accumulate)
inverse_updater_registry.register('set', inverse_update_set)
inverse_updater_registry.register('null', inverse_update_null)
inverse_updater_registry.register('merge', inverse_update_merge)
inverse_updater_registry.register(
    'nonnegative_accumulate', inverse_update_nonnegative_accumulate)
# inverse_updater_registry.register(
#     'dict_value', inverse_update_dictionary)

# register :term:`dividers`
divider_registry.register('binomial_ecoli', divide_binomial)
divider_registry.register('by_domain', divide_by_domain)
divider_registry.register('divide_domain', divide_domain)
divider_registry.register('rna_by_domain', divide_RNAs_by_domain)
divider_registry.register('divide_unique', divide_unique)
divider_registry.register('empty_dict', empty_dict_divider)
divider_registry.register('divide_ribosomes', divide_ribosomes)
divider_registry.register('set_none', divide_set_none)

# register serializers
serializer_registry.register('unum', UnumSerializer())
