import numpy as np

from vivarium.core.registry import divider_registry
from vivarium.core.registry import updater_registry


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


def dict_value_updater(current, update):
    '''
    Updater which translates add_item and delete_item -style updates
    into operations on a dictionary.

    Expects current to be a dictionary, with no restriction on the types of objects
    stored within it, and no defaults. For enforcing expectations/defaults, try
    make_dict_value_updater(**defaults).
    '''
    result = current

    if update.get("add_items"):
        for operation in update["add_items"]:
            result[operation["key"]] = operation["state"]

    for k in update.get("remove_items", {}):
        result.pop(k)

    return result

updater_registry.register('dict_value', dict_value_updater)
