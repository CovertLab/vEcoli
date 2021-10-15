import copy
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

    for key, value in update.items():
        if key == "_add":
            for added_value in value:
                added_key = added_value["key"]
                added_state = added_value["state"]
                result[added_key] = added_state
        elif key == "_delete":
            for k in value:
                del result[k]
        elif key in result:
            result[key].update(value)
        else:
            raise Exception(f"Invalid dict_value_updater key: {key}")
    return result


def make_dict_value_updater(defaults):
    '''
    Returns an updater which translates add_item and delete_item -style updates
    into operations on a dictionary.

    The returned updater expects current to be a dictionary. Each added item
    can have a subset of the provided defaults as its keys;
    entries not provided will have values supplied by the defaults.
    '''

    def custom_dict_value_updater(current, update):
        result = current
        for key, value in update.items():
            if key == "_add":
                assert isinstance(value, list)
                for added_value in value:
                    added_key = added_value["key"]
                    added_state = added_value["state"]
                    if added_key in current:
                        raise Exception(f"Cannot add {added_key}, already in state")
                    elif not added_state.keys() <= defaults.keys():
                        raise Exception(f"State has keys not in defaults: "
                                        f"{added_state.keys() - defaults.keys()}")
                    result[added_key] = {**defaults, **added_state}
            elif key == "_delete":
                assert isinstance(value, list)
                for k in value:
                    del result[k]
            elif key in result:
                result[key].update(value)
            else:
                raise Exception(f"Invalid dict_value_updater key: {key}")
        return result

    return custom_dict_value_updater


UNIQUE_DEFAULTS = {
    'active_ribosome': {
        'protein_index': 0,
        'peptide_length': 0,
        'mRNA_index': 0,
        'unique_index': 0,
        'pos_on_mRNA': 0,
        'submass': np.zeros(9)
    },
    'full_chromosomes': {
        'domain_index': 0,
        'unique_index': 0,
        'division_time': 0,
        'has_triggered_division': 0,
        'submass': np.zeros(9)
    },
    'chromosome_domains': {
        'domain_index': 0,
        'child_domains': 0,
        'unique_index': 0,
        'submass': np.zeros(9)
    },
    'active_replisomes': {
        'domain_index': 0,
        'coordinates': 0,
        'unique_index': 0,
        'right_replichore': 0,
        'submass': np.zeros(9)
    },
    'oriCs': {
        'domain_index': 0,
        'unique_index': 0,
        'submass': np.zeros(9)
    },
    'promoters': {
        'TU_index': 0,
        'coordinates': 0,
        'domain_index': 0,
        'bound_TF': 0,
        'unique_index': 0,
        'submass': np.zeros(9)
    },
    'chromosomal_segments': {
        'unique_index': 0,
        'submass': np.zeros(9)
    },
    'DnaA_boxes': {
        'domain_index': 0,
        'coordinates': 0,
        'DnaA_bound': 0,
        'unique_index': 0,
        'submass': np.zeros(9)
    },
    'active_RNAPs': {
        'unique_index': 0,
        'domain_index': 0,
        'coordinates': 0,
        'direction': 0,
        'submass': np.zeros(9)
    },
    'RNAs': {
        'unique_index': 0,
        'TU_index': 0,
        'transcript_length': 0,
        'RNAP_index': 0,
        'is_mRNA': 0,
        'is_full_transcript': 0,
        'can_translate': 0,
        'submass': np.zeros(9)
    },
}
