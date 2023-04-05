from vivarium.core.registry import (
    divider_registry,
    updater_registry,
    serializer_registry,
)
from ecoli.library.schema import (
    divide_binomial,
    divide_by_domain,
    divide_RNAs_by_domain,
    empty_dict_divider,
    divide_ribosomes_by_RNA,
    divide_set_none,
    bulk_numpy_updater,
    unique_numpy_updater,
    divide_bulk
)
from ecoli.library.updaters import (
    inverse_updater_registry,
    inverse_update_accumulate,
    inverse_update_set,
    inverse_update_null,
    inverse_update_merge,
    inverse_update_nonnegative_accumulate,
    inverse_update_bulk_numpy,
    inverse_update_unique_numpy
)
from ecoli.library.serialize import UnumSerializer, ParameterSerializer

# register :term:`updaters`
updater_registry.register('bulk_numpy', bulk_numpy_updater)
updater_registry.register('unique_numpy', unique_numpy_updater)

inverse_updater_registry.register(
    'accumulate', inverse_update_accumulate)
inverse_updater_registry.register('set', inverse_update_set)
inverse_updater_registry.register('null', inverse_update_null)
inverse_updater_registry.register('merge', inverse_update_merge)
inverse_updater_registry.register(
    'nonnegative_accumulate', inverse_update_nonnegative_accumulate)
inverse_updater_registry.register(
    'bulk_numpy', inverse_update_bulk_numpy)
inverse_updater_registry.register(
    'unique_numpy', inverse_update_unique_numpy)


# register :term:`dividers`
divider_registry.register('binomial_ecoli', divide_binomial)
divider_registry.register('bulk_binomial', divide_bulk)
divider_registry.register('by_domain', divide_by_domain)
divider_registry.register('rna_by_domain', divide_RNAs_by_domain)
divider_registry.register('empty_dict', empty_dict_divider)
divider_registry.register('ribosome_by_RNA', divide_ribosomes_by_RNA)
divider_registry.register('set_none', divide_set_none)

# register serializers
for serializer_cls in (UnumSerializer, ParameterSerializer):
    serializer = serializer_cls()
    serializer_registry.register(
        serializer.name, serializer)
