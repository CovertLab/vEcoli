import numpy as np

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


def make_dict_value_updater(**defaults):
    '''
    Returns an updater which translates add_item and delete_item -style updates
    into operations on a dictionary.

    The returned updater expects current to be a dictionary. Each added item
    can have a subset of the provided defaults as its keys;
    entries not provided will have values supplied by the defaults.
    '''

    def dict_value_updater(current, update):
        result = current

        for operation in update.get("add_items", {}):
            state = operation["state"]
            if not set(state.keys()).issubset(defaults.keys()):
                raise Exception(f"Attempted to write state with keys not included in defaults")
            state = {**defaults, **state}
            result[operation["key"]] = state

        for k in update.get("remove_items", {}):
            result.pop(k)

        return result

    return dict_value_updater