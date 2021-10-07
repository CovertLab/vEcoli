from vivarium.core.registry import divider_registry
from vivarium.core.registry import updater_registry

from ecoli.library.registry import divide_binomial, dict_value_updater, make_dict_value_updater

divider_registry.register('binomial_ecoli', divide_binomial)

updater_registry.register('dict_value', dict_value_updater)

updater_registry.register('rnap_updater', make_dict_value_updater(
    unique_index=0,
    domain_index=0,
    coordinates=0,
    direction=True)
)
