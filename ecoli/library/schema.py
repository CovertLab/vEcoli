import random

import numpy as np


UNIQUE_DIVIDERS = {
    'active_ribosome': {
        'divider': 'divide_ribosomes',
        'topology': {'RNA': ('..', 'RNA'),
                     'active_RNAP': ('..', 'active_RNAP',),
                     'chromosome_domain': ('..', 'chromosome_domain')}
    },
    'full_chromosomes': {
        'divider': 'by_domain',
        'topology': {'chromosome_domain': ('..', 'chromosome_domain')}
    },
    'chromosome_domains': 'divide_domain',
    'active_replisomes': {
        'divider': 'by_domain',
        'topology': {'chromosome_domain': ('..', 'chromosome_domain')}
    },
    'oriCs': {
        'divider': 'by_domain',
        'topology': {'chromosome_domain': ('..', 'chromosome_domain')}
    },
    'promoters': {
        'divider': 'by_domain',
        'topology': {'chromosome_domain': ('..', 'chromosome_domain')}
    },
    'chromosomal_segments': {
        'divider': 'by_domain',
        'topology': {'chromosome_domain': ('..', 'chromosome_domain')}
    },
    'DnaA_boxes': {
        'divider': 'by_domain',
        'topology': {'chromosome_domain': ('..', 'chromosome_domain')}
    },
    'active_RNAPs': {
        'divider': 'by_domain',
        'topology': {'chromosome_domain': ('..', 'chromosome_domain')}
    },
    'RNAs': {
        'divider': 'rna_by_domain',
        'topology': {'active_RNAP': ('..', 'active_RNAP',),
                     'chromosome_domain': ('..', 'chromosome_domain')}
    },
}

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

def create_unique_indexes(n_indexes, random_state):
    """Creates a list of unique indexes by making them random.

    Args:
        n_indexes: Number of indexes to generate.
        random_state: A Numpy :py:class:`np.random.RandomState` object
            to use as a PRNG.

    Returns:
        List of indexes. Each index is a string representing a number in
        the range :math:`[0, 2^{63})`.
    """
    return [str(num) for num in random_state.randint(0, 2**63, n_indexes)]

def array_from(d):
    """Returns an array with the dictionary values"""
    return np.array(list(d.values()))

def key_array_from(d):
    """Returns an array with the dictionary keys"""
    return np.array(list(d.keys()))

def array_to(keys, array):
    return {
        key: array[index]
        for index, key in enumerate(keys)}

def array_to_nonzero(keys, array):
    return {
        key: array[index]
        for index, key in enumerate(keys)
        if array[index] != 0}

def type_of(array):
    if len(array) == 0:
        return None

    head = array[0]
    if isinstance(head, (list, np.ndarray)):
        return type_of(head)
    else:
        return type(head)

def arrays_from(ds, keys):
    if not ds:
        return np.array([])

    arrays = {
        key: []
        for key in keys}

    for d in ds:
        for key, value in d.items():
            if key in arrays:
                arrays[key].append(value)

    return tuple([
        np.array(array, dtype=type_of(array))
        for array in arrays.values()])

def arrays_to(n, attrs):
    ds = []
    for index in np.arange(n):
        d = {}
        for attr in attrs.keys():
            d[attr] = attrs[attr][index]
        ds.append(d)

    return ds

def bulk_schema(
        elements,
        updater=None,
        partition=True,
):
    schema = {
        '_default': 0,
        '_divider': 'binomial_ecoli',
        '_emit': True}
    if partition:
        schema['_properties'] = {'bulk': True}
    if updater:
        schema['_updater'] = updater
    return {
        element: schema
        for element in elements}

def mw_schema(mass_dict):
    return {
        element: {
            '_properties': {
                'mw': mw}}
        for element, mw in mass_dict.items()}

def listener_schema(elements):
    return {
        element: {
            '_default': default,
            '_updater': 'set',
            '_emit': True}
        for element, default in elements.items()}

def add_elements(elements, id):
    return {
        '_add': [{
            'key': str(element[id]),
            'state': element}
            for element in elements]}

def submass_schema():
    return {
        '_default': np.zeros(9),
        '_emit': True}

def dict_value_schema(name):
    return {
        '_default': {},
        '_updater': f'{name}_updater',
        '_divider': UNIQUE_DIVIDERS[name],
        '_emit': True
    }



# :term:`dividers`
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


def make_dict_value_updater(defaults):
    '''
    Returns an updater which translates _add and _delete -style updates
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
                    if k in result:
                        del result[k]
                    else:
                        pass
                        # TODO -- fix processes to not delete invalid keys
                        # print(f"Invalid delete key: {k}")
            elif key in result:
                result[key].update(value)
            else:
                pass
                # TODO -- fix processes to not updater invalid keys
                # print(f"Invalid update key: {key}")
        return result

    return custom_dict_value_updater


def get_cell_for_index(index_to_children, domain_index_to_add, root_index):
    if domain_index_to_add == root_index:  # If the root index:
        return -1
    if domain_index_to_add in index_to_children[root_index]:  # If a daughter cell index:
        return domain_index_to_add
    for domain_index in index_to_children:
        children = index_to_children[domain_index]
        if domain_index_to_add in children:
            cell = get_cell_for_index(index_to_children, domain_index, root_index)
    return cell


def get_domain_index_to_daughter(chromosome_domain):
    """
    Creates a dictionary linking domain indexes to their respective cells.
    If the index does not belong to a daughter cell, it is assigned a value of -1.
    """

    index_to_children = {}
    for domain_key in chromosome_domain:
        domain = chromosome_domain[domain_key]
        index_to_children[domain['domain_index']] = domain['child_domains']

    root_index = -1
    for root_candidate in index_to_children:
        root = True
        for domain_index_to_check in index_to_children:
            if root_candidate in index_to_children[domain_index_to_check]:
                root = False
        if root:
            root_index = root_candidate

    index_to_daughter = {}
    for domain_index_to_add in index_to_children:
        index_to_daughter[domain_index_to_add] = get_cell_for_index(
            index_to_children, domain_index_to_add, root_index)

    # check that there are 2 daughter indices, and return them
    daughter_ids = set(index_to_daughter.values())
    daughter_ids.remove(-1)
    daughter_ids = list(daughter_ids)
    assert len(daughter_ids) == 2
    daughter1_index = daughter_ids[0]
    daughter2_index = daughter_ids[1]

    return index_to_daughter, daughter1_index, daughter2_index


def get_full_transcript_rnas_to_daughter(full_transcript_rnas):
    """Make a mapping from all full transcript RNA indices to a daughter index."""
    random_state = np.random.RandomState(seed=len(full_transcript_rnas))  # TODO(Matt): pass in random_state from topology when available
    sorted_indexes = np.array(sorted(full_transcript_rnas))
    bitmap = random_state.choice([True, False], len(full_transcript_rnas))
    daughter_1_indexes = sorted_indexes[bitmap]
    daughter_2_indexes = sorted_indexes[~bitmap]
    return daughter_1_indexes, daughter_2_indexes


def divide_ribosomes(ribosomes, state):
    """divide ribosomes according to the rna they are attached to"""
    daughter1 = {}
    daughter2 = {}

    # get domain_index-to-daughter_index mapping
    index_to_daughter, d1_index, d2_index = get_domain_index_to_daughter(state['chromosome_domain'])

    full_transcript_rnas = [index for index in state['RNA'] if state['RNA'][index]['is_full_transcript']]
    daughter_1_indexes, daughter_2_indexes = get_full_transcript_rnas_to_daughter(full_transcript_rnas)

    for ribo_index, specs in ribosomes.items():
        ribo_mrna = ribosomes[ribo_index]['mRNA_index']
        if ribo_mrna in daughter_1_indexes:
            daughter1[ribo_index] = specs
        elif ribo_mrna in daughter_2_indexes:
            daughter2[ribo_index] = specs
        else:  # Else the ribosome is on a partial mRNA
            rnap_index = state['RNA'][ribo_mrna]['RNAP_index']
            domain_index = state['active_RNAP'][rnap_index]['domain_index']
            # Some ribosomes go to neither daughter as they are on domain_index 0 in forced-early division
            if index_to_daughter[domain_index] == d1_index:
                daughter1[ribo_index] = specs
            elif index_to_daughter[domain_index] == d2_index:
                daughter2[ribo_index] = specs

    return [daughter1, daughter2]


def divide_by_domain(values, state):
    """
    divide a dictionary into two daughters based on their domain_index
    """
    daughter1 = {}
    daughter2 = {}

    # get domain_index-to-daughter_index mapping
    index_to_daughter, d1_index, d2_index = get_domain_index_to_daughter(state['chromosome_domain'])

    for state_id, value in values.items():
        domain_index = value['domain_index']
        if index_to_daughter[domain_index] == d1_index:
            daughter1[state_id] = value
        elif index_to_daughter[domain_index] == d2_index:
            daughter2[state_id] = value
    return [daughter1, daughter2]


def divide_domain(values):
    """
    Divides the chromosome domains between two cells. The left daughter of the
    root index becomes the new root of the chromosome_domain tree for daughter
    cell 1, and likewise for the right daughter for daughter cell 2.
    """
    daughter1 = {}
    daughter2 = {}
    index_to_daughter, d1_index, d2_index = get_domain_index_to_daughter(values)
    for key in values:
        key_domain_index = values[key]['domain_index']
        key_daughter_cell = index_to_daughter[key_domain_index]
        if key_daughter_cell == d1_index:
            daughter1[key] = values[key]
        elif key_daughter_cell == d2_index:
            daughter2[key] = values[key]
    return [daughter1, daughter2]


def divide_unique(unique_molecules, **args):
    """divide unique molecules binomially"""
    # TODO (Matt): Set a seed
    n_unique_molecules = len(unique_molecules)
    unique_molecule_ids = list(unique_molecules.keys())

    daughter1_counts = np.random.binomial(n_unique_molecules, 0.5)
    daughter1_ids = random.sample(unique_molecule_ids, daughter1_counts)

    daughter1 = {}
    daughter2 = {}
    for unique_id in unique_molecule_ids:
        specs = unique_molecules[unique_id]
        if unique_id in daughter1_ids:
            daughter1[unique_id] = specs
        else:
            daughter2[unique_id] = specs
    return [daughter1, daughter2]


def divide_RNAs_by_domain(values, state):
    """
    divide a dictionary of unique RNAs into two daughters,
    with partial RNAs divided along with their domain index
    """
    daughter1 = {}
    daughter2 = {}
    full_transcript_ids = []

    # get domain_index-to-daughter_index mapping
    index_to_daughter, d1_index, d2_index = get_domain_index_to_daughter(state['chromosome_domain'])

    # divide partial transcripts by domain_index
    for unique_id, specs in values.items():
        associated_rnap_key = str(values[unique_id]['RNAP_index'])
        if not specs['is_full_transcript']:
            domain_index = state['active_RNAP'][associated_rnap_key]['domain_index']
            if index_to_daughter[domain_index] == d1_index:
                daughter1[unique_id] = specs
            elif index_to_daughter[domain_index] == d2_index:
                daughter2[unique_id] = specs
        else:
            # save full transcript ids
            full_transcript_ids.append(unique_id)

    # divide full transcripts with get_full_transcript_rnas_to_daughter
    daughter_1_indexes, daughter_2_indexes = get_full_transcript_rnas_to_daughter(full_transcript_ids)
    for index in daughter_1_indexes:
        daughter1[index] = values[index]
    for index in daughter_2_indexes:
        daughter2[index] = values[index]

    return [daughter1, daughter2]


def empty_dict_divider(values):
    return [{}, {}]


def divide_set_none(values):
    return [None, None]


