from typing import List

import numpy as np

RAND_MAX = 2**31 - 1

UNIQUE_DIVIDERS = {
    'active_ribosome': {
        'divider': 'ribosome_by_RNA',
        'topology': {'RNA': ('..', 'RNA'),
            'full_chromosome': ('..', 'full_chromosome'),
            'chromosome_domain': ('..', 'chromosome_domain')}
    },
    'full_chromosomes': {
        'divider': 'by_domain',
        'topology': {
            'full_chromosome': (),
            'chromosome_domain': ('..', 'chromosome_domain')}
    },
    'chromosome_domains': {
        'divider': 'by_domain',
        'topology': {
            'full_chromosome': ('..', 'full_chromosome'),
            'chromosome_domain': ()}
    },
    'active_replisomes': {
        'divider': 'by_domain',
        'topology': {
            'full_chromosome': ('..', 'full_chromosome'),
            'chromosome_domain': ('..', 'chromosome_domain')}
    },
    'oriCs': {
        'divider': 'by_domain',
        'topology': {
            'full_chromosome': ('..', 'full_chromosome'),
            'chromosome_domain': ('..', 'chromosome_domain')}
    },
    'promoters': {
        'divider': 'by_domain',
        'topology': {
            'full_chromosome': ('..', 'full_chromosome'),
            'chromosome_domain': ('..', 'chromosome_domain')}
    },
    'chromosomal_segments': {
        'divider': 'by_domain',
        'topology': {
            'full_chromosome': ('..', 'full_chromosome'),
            'chromosome_domain': ('..', 'chromosome_domain')}
    },
    'DnaA_boxes': {
        'divider': 'by_domain',
        'topology': {
            'full_chromosome': ('..', 'full_chromosome'),
            'chromosome_domain': ('..', 'chromosome_domain')}
    },
    'active_RNAPs': {
        'divider': 'by_domain',
        'topology': {
            'full_chromosome': ('..', 'full_chromosome'),
            'chromosome_domain': ('..', 'chromosome_domain')}
    },
    'RNAs': {
        'divider': 'rna_by_domain',
        'topology': {'active_RNAP': ('..', 'active_RNAP',),
            'full_chromosome': ('..', 'full_chromosome'),
            'chromosome_domain': ('..', 'chromosome_domain')}
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
    return [num for num in random_state.randint(0, 2**63, n_indexes)]


def array_from(d):
    """Returns an array with the dictionary values"""
    return np.array(list(d.values()))


def counts(states, idx):
    # idx is an integer array so advancing indexing applies (implicit copy)
    if len(states.dtype) > 1:
        return states['count'][idx]
    return states[idx]


class get_bulk_counts():
    def serialize(bulk):
        return np.ascontiguousarray(bulk['count'])


class get_unique_fields():
    def serialize(unique):
        return [np.ascontiguousarray(unique[field])
            for field in unique.dtype.names]


def numpy_schema(name, partition=True, divider=None):
    schema = {
        '_default': [],
        '_emit': True
    }
    if divider:
        schema['_divider'] = divider
    if name == 'bulk':
        if partition:
            schema['_properties'] = {'bulk': True}
        schema['_updater'] = bulk_numpy_updater
        # Only pull out counts to be serialized (save space and time)
        schema['_serializer'] = get_bulk_counts
        schema['_divider'] = 'bulk_binomial'
    else:
        schema['_updater'] = unique_numpy_updater
        # These are some big and slow emits
        schema['_emit'] = False
        # Convert to list of contiguous Numpy arrays for faster and more
        # efficient serialization (still do not recommend emitting unique)
        schema['_serializer'] = get_unique_fields
        schema['_divider'] = UNIQUE_DIVIDERS[name]
    return schema


def bulk_name_to_idx(names, bulk_names):
    # Convert from string names to indices in bulk array
    if isinstance(names, np.ndarray) or isinstance(names, list):
        # Big brain solution from https://stackoverflow.com/a/32191125
        sorter = np.argsort(bulk_names)
        return sorter[np.searchsorted(bulk_names, names, sorter=sorter)]
    else:
        return np.where(bulk_names == names)[0][0]


def bulk_numpy_updater(current, update):
    # Bulk updates are lists of tuples, where first value
    # in each tuple is an array of indices to update and
    # second value is array of updates to apply
    result = current
    for (idx, value) in update:
        result['count'][idx] += value
    return result


def attrs(states, attributes):
    # _entryState has dtype int8 so this works
    mol_mask = states['_entryState'].view(np.bool_)
    # mol_mask is a boolean array so advanced indexing applies (implicit copy)
    return [states[attribute][mol_mask] for attribute in attributes]


def get_free_indices(result, n_objects):
    # Find inactive rows for new molecules and expand array
    # by at least 10% to create more rows when necessary
    free_indices = np.where(result['_entryState'] == 0)[0]
    n_free_indices = free_indices.size

    if n_free_indices < n_objects:
        old_size = result.size
        n_new_entries = max(
            np.int64(old_size * 0.1),
            n_objects - n_free_indices
            )

        result = np.append(
            result,
            np.zeros(int(n_new_entries), dtype=result.dtype)
        )

        free_indices = np.concatenate((
            free_indices,
            old_size + np.arange(n_new_entries)
        ))

    return result, free_indices[:n_objects]


def unique_numpy_updater(current, update):
    if len(update) == 0:
        return current

    result = current
    current_time = current['time'][0]
    update_time = update['time']

    # If new timestep, update time and _cached_entryState
    if current_time != update_time:
        result['time'] = update_time
        result['_cached_entryState'] = current['_entryState']
    
    cached_state = current['_cached_entryState']

    if 'set' in update.keys():
        # Set updates are dictionaries where each key is a column and
        # each value is an array. They are designed to apply to all rows
        # (molecules) that were active at the beginning of a timestep (e.g.
        # length of update arrays == # of rows where _cached_entryState > 0)
        set_update = update['set']
        # Skip updates for rows with molecules that were initially active
        # (cached_state > 0) but deleted in same timestep (cached_state == 2)
        safe_rows = (cached_state[cached_state > 0] == 1)
        # Only apply updates to rows that were initially and still are active
        initial_still_active = (cached_state == 1)
        for col, col_values in set_update.items():
            result[col][initial_still_active] = col_values[safe_rows]

    if 'delete' in update.keys():
        # Delete updates are arrays of row indices to delete in unique
        # molecule array after filtering to only include initially active rows
        delete_indices = update['delete']
        # Convert from indices for filtered (active) array to indices for
        # complete (active + inactive) internal array
        initially_active_idx = np.nonzero(cached_state > 0)[0]
        rows_to_delete = initially_active_idx[delete_indices]
        # Only delete rows that were not previously deleted in same timestep
        rows_to_delete = rows_to_delete[cached_state[rows_to_delete]==1]
        result[rows_to_delete] = np.zeros(1, dtype=result.dtype)
        # Ensure that additional updates in same timestep skip deleted rows
        result['_cached_entryState'][rows_to_delete] = 2
        # Do not overwrite time with zeros
        result['time'][rows_to_delete] = update_time

    if 'add' in update.keys():
        # Add updates are dictionaries where each key is a column and
        # each value is an array. The nth element of each array is the value
        # for the corresponding column of the nth new molecule to be added.
        add_update = update['add']
        # All arrays have same length so can get length of first to figure
        # out how many new unique molecules are being added
        n_new_molecules = len(next(iter(add_update.values())))
        result, free_indices = get_free_indices(result, n_new_molecules)
        for col, col_values in add_update.items():
            result[col][free_indices] = col_values
        result['_entryState'][free_indices] = 1

    return result


def listener_schema(elements):
    return {
        element: {
            '_default': default,
            '_updater': 'set',
            '_emit': True}
        for element, default in elements.items()}


# :term:`dividers`
def divide_binomial(state: float) -> List[float]:
    """Binomial Divider

    Args:
        state: The value to divide.
        config: Must contain a ``seed`` key with an integer seed. This
            seed will be added to ``int(state)`` to seed a random number
            generator used to calculate the binomial.

    Returns:
        The divided values.
    """
    seed = int(state) % RAND_MAX
    random_state = np.random.RandomState(seed=seed)
    counts_1 = random_state.binomial(state, 0.5)
    counts_2 = state - counts_1
    return [counts_1, counts_2]


def divide_bulk(state):
    counts = state['count']
    seed = counts.sum() % RAND_MAX
    # TODO: Random state/seed in store?
    random_state = np.random.RandomState(seed=seed)
    daughter_1 = state.copy()
    daughter_2 = state.copy()
    daughter_1['count'] = random_state.binomial(counts, 0.5)
    daughter_2['count'] = counts - daughter_1['count']
    return [daughter_1, daughter_2]

# TODO: Remove once all processes have been converted
def bulk_schema():
    return

def dict_value_schema():
    return

def arrays_from():
    return

# TODO: Create a store for growth rate noise simulation parameter

def divide_ribosomes_by_RNA(values, state):
    mRNA_index, = attrs(values, ['mRNA_index'])
    n_molecules = len(mRNA_index)
    if n_molecules > 0:
        # Divide ribosomes based on their mRNA index
        d1_rnas, d2_rnas = divide_RNAs_by_domain(state['RNA'], state)
        d1_bool = np.isin(mRNA_index, d1_rnas['unique_index'])
        d2_bool = np.isin(mRNA_index, d2_rnas['unique_index'])

        # Binomially divide indexes of mRNAs that are degraded but still
        # has bound ribosomes. This happens because mRNA degradation does
        # not abort ongoing translation of the mRNA
        degraded_mRNA_indexes = np.unique(mRNA_index[
            np.logical_not(np.logical_or(d1_bool, d2_bool))])
        n_degraded_mRNA = len(degraded_mRNA_indexes)

        if n_degraded_mRNA > 0:
            # TODO: Random state/seed in store?
            random_state = np.random.RandomState(seed=n_molecules)
            n_degraded_mRNA_d1 = random_state.binomial(
                n_degraded_mRNA, p=0.5)
            degraded_mRNA_indexes_d1 = random_state.choice(
                degraded_mRNA_indexes, size=n_degraded_mRNA_d1, replace=False)
            degraded_mRNA_indexes_d2 = np.setdiff1d(
                degraded_mRNA_indexes, degraded_mRNA_indexes_d1)

            # Divide "lost" ribosomes based on how these mRNAs were divided
            lost_ribosomes_d1 = np.isin(mRNA_index, degraded_mRNA_indexes_d1)
            lost_ribosomes_d2 = np.isin(mRNA_index, degraded_mRNA_indexes_d2)

            d1_bool[lost_ribosomes_d1] = True
            d2_bool[lost_ribosomes_d2] = True

        n_d1 = np.count_nonzero(d1_bool)
        n_d2 = np.count_nonzero(d2_bool)

        assert n_molecules == n_d1 + n_d2
        assert np.count_nonzero(np.logical_and(d1_bool, d2_bool)) == 0

        ribosomes = values[values['_entryState'].view(np.bool_)]
        return [ribosomes[d1_bool], ribosomes[d2_bool]]
    
    return [np.zeros(0, dtype=values.dtype), np.zeros(0, dtype=values.dtype)]



def divide_domains(state):
    """
    Divides the chromosome domains between two cells.
    """
    domain_index_full_chroms, = attrs(state['full_chromosome'],
        ['domain_index'])
    domain_index_domains, child_domains = attrs(state['chromosome_domain'],
        ['domain_index', 'child_domains'])

    # TODO: Random state/seed in store?
    # d1_gets_first_chromosome = randomState.rand() < 0.5
    # index = not d1_gets_first_chromosome
    # d1_domain_index_full_chroms = domain_index_full_chroms[index::2]
    # d2_domain_index_full_chroms = domain_index_full_chroms[not index::2]

    d1_domain_index_full_chroms = domain_index_full_chroms[0::2]
    d2_domain_index_full_chroms = domain_index_full_chroms[1::2]
    d1_all_domain_indexes = get_descendent_domains(
        d1_domain_index_full_chroms, domain_index_domains,
        child_domains, -1
    )
    d2_all_domain_indexes = get_descendent_domains(
        d2_domain_index_full_chroms, domain_index_domains,
        child_domains, -1
    )

    # Check that the domains are being divided correctly
    assert np.intersect1d(d1_all_domain_indexes,
        d2_all_domain_indexes).size == 0

    return {
        'd1_all_domain_indexes': d1_all_domain_indexes,
        'd2_all_domain_indexes': d2_all_domain_indexes,
    }


def divide_by_domain(values, state):
    domain_division = divide_domains(state)
    values = values[values['_entryState'].view(np.bool_)]
    d1_bool = np.isin(values['domain_index'],
        domain_division['d1_all_domain_indexes'])
    d2_bool = np.isin(values['domain_index'],
        domain_division['d2_all_domain_indexes'])
    assert d1_bool.sum() + d2_bool.sum() == len(values)
    return [values[d1_bool], values[d2_bool]]


def divide_RNAs_by_domain(values, state):
    is_full_transcript, RNAP_index = attrs(values,
        ["is_full_transcript", "RNAP_index"])

    n_molecules = len(is_full_transcript)

    if n_molecules > 0:
        # Figure out which RNAPs went to each daughter cell
        domain_division = divide_domains(state)
        rnaps = state['active_RNAP']
        rnaps = rnaps[rnaps['_entryState'].view(np.bool_)]
        d1_rnap_bool = np.isin(rnaps['domain_index'],
            domain_division['d1_all_domain_indexes'])
        d2_rnap_bool = np.isin(rnaps['domain_index'],
            domain_division['d2_all_domain_indexes'])
        d1_rnap_indexes = rnaps['unique_index'][d1_rnap_bool]
        d2_rnap_indexes = rnaps['unique_index'][d2_rnap_bool]

        d1_bool = np.zeros(n_molecules, dtype=np.bool)
        d2_bool = np.zeros(n_molecules, dtype=np.bool)

        # Divide full transcripts binomially
        full_transcript_indexes = np.where(is_full_transcript)[0]
        if len(full_transcript_indexes) > 0:
            # TODO: Random state/seed in store?
            random_state = np.random.RandomState(seed=n_molecules)
            n_full_d1 = random_state.binomial(
                np.count_nonzero(is_full_transcript), p=0.5)
            full_d1_indexes = random_state.choice(
                full_transcript_indexes, size=n_full_d1,
                replace=False)
            full_d2_indexes = np.setdiff1d(full_transcript_indexes,
                full_d1_indexes)

            d1_bool[full_d1_indexes] = True
            d2_bool[full_d2_indexes] = True

        # Divide partial transcripts based on how their associated
        # RNAPs were divided
        partial_transcript_indexes = np.where(
            np.logical_not(is_full_transcript))[0]
        RNAP_index_partial_transcripts = RNAP_index[
            partial_transcript_indexes]

        partial_d1_indexes = partial_transcript_indexes[
            np.isin(RNAP_index_partial_transcripts, d1_rnap_indexes)]
        partial_d2_indexes = partial_transcript_indexes[
            np.isin(RNAP_index_partial_transcripts, d2_rnap_indexes)]

        d1_bool[partial_d1_indexes] = True
        d2_bool[partial_d2_indexes] = True

        n_d1 = np.count_nonzero(d1_bool)
        n_d2 = np.count_nonzero(d2_bool)

        assert n_molecules == n_d1 + n_d2
        assert np.count_nonzero(np.logical_and(d1_bool, d2_bool)) == 0

        rnas = values[values['_entryState'].view(np.bool_)]
        return [rnas[d1_bool], rnas[d2_bool]]
    
    return [np.zeros(0, dtype=values.dtype), np.zeros(0, dtype=values.dtype)]


def empty_dict_divider(values):
    return [{}, {}]


def divide_set_none(values):
    return [None, None]


def remove_properties(schema, properties):
    if isinstance(schema, dict):
        for property in properties:
            schema.pop(property, None)
        for key, value in schema.items():
            schema[key] = remove_properties(value, properties)
    return schema


def flatten(l):
    """
    Flattens a nested list into a single list.
    """
    return [item for sublist in l for item in sublist]


def follow_domain_tree(domain, domain_index, child_domains, place_holder):
    """
    Recursive function that returns all the descendents of a single node in
    the domain tree, including itself.
    """
    children_nodes = child_domains[np.where(domain_index == domain)[0][0]]

    if children_nodes[0] != place_holder:
        # If the node has children, recursively run function on each of the
        # node's two children
        branches = flatten([
            follow_domain_tree(child, domain_index, child_domains, place_holder)
            for child in children_nodes])

        # Append index of the node itself
        branches.append(domain)
        return branches

    else:
        # If the node has no children, return the index of itself
        return [domain]


def get_descendent_domains(root_domains, domain_index, child_domains, place_holder):
    """
    Returns an array of domain indexes that are descendents of the indexes
    listed in root_domains, including the indexes in root_domains themselves.
    """
    return np.array(flatten([
        follow_domain_tree(root_domain, domain_index, child_domains, place_holder)
        for root_domain in root_domains]))
