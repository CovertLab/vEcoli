"""
===========================
Simulation Helper Functions
===========================

This is a collection of helper functions used thoughout our code base.
"""

from typing import List, Tuple, Dict, Any

import numpy as np
from vivarium.core.store import Store
from vivarium.core.registry import Serializer

RAND_MAX = 2**31 - 1

UNIQUE_DIVIDERS = { 
    'active_ribosome': {
        'divider': 'ribosome_by_RNA',
        'topology': {'RNA': ('..', 'RNA'),
            'full_chromosome': ('..', 'full_chromosome'),
            'chromosome_domain': ('..', 'chromosome_domain'),
            'active_RNAP': ('..', 'active_RNAP',)}
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
    'genes': {
        'divider': 'by_domain',
        'topology': {
            'full_chromosome': ('..', 'full_chromosome'),
            'chromosome_domain': ('..', 'chromosome_domain')}
    },
}
"""A mapping of unique molecules to the names of their divider functions ars they are registered 
in the ``divider_registry`` in ``ecoli/__init__.py``

:meta hide-value:
"""


def array_from(d: dict) -> np.ndarray:
    """Makes a Numpy array from dictionary values.
    
    Args:
        d: Dictionary whose values are to be converted
    
    Returns:
        Array of all values in d.
    """
    return np.array(list(d.values()))


def create_unique_indexes(
    n_indexes: int, 
    random_state: np.random.RandomState) -> List[int]:
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


def not_a_process(value):
    """Returns `True` if not a :py:class:vivarium.core.Process instance.
    """
    return not (isinstance(value, Store) and value.topology)


def counts(states: np.ndarray, idx: int | np.ndarray) -> np.ndarray:
    """Helper function to pull out counts at given indices.

    Args:
        states: Either a Numpy structured array with a `'count'` field or a 1D
            Numpy array of counts.
        idx: Indices for the counts of interest.

    Returns:
        Counts of molecules at specified indices (copy so can be safely mutated)
    """
    if len(states.dtype) > 1:
        return states['count'][idx]
    # evolve_state reads from ('allocate', process_name, 'bulk')
    # which is a simple Numpy array (not structured)
    return states[idx].copy()


class get_bulk_counts(Serializer):
    """Serializer for bulk molecules that saves counts without IDs or masses.
    """
    def serialize(bulk: np.ndarray) -> np.ndarray:
        """
        Args:
            bulk: Numpy structured array with a `count` field
        
        Returns:
            Contiguous (required by orjson) array of bulk molecule counts
        """
        return np.ascontiguousarray(bulk['count'])


class get_unique_fields(Serializer):
    """Serializer for unique molecules.
    """
    def serialize(unique: np.ndarray) -> np.ndarray:
        """
        Args:
            unique: Numpy structured array of attributes for one unique molecule
        
        Returns:
            List of contiguous (required by orjson) arrays, one for each attribute
        """
        return [np.ascontiguousarray(unique[field])
            for field in unique.dtype.names]


def numpy_schema(name: str, partition: bool=True, emit: bool=True
                 ) -> Dict[str, Any]:
    """Helper function used in ports schemas for bulk and unique molecules

    Args:
        name: `bulk` for bulk molecules or one of the keys in :py:data:`UNIQUE_DIVIDERS`
            for unique molecules
        partition: `True` if bulk molecule counts should be partitioned (default)
        emit: `True` if should be emitted (default)
    
    Returns:
        Fully configured ports schema for molecules of type `name`
    """
    schema = {
        '_default': [],
        '_emit': emit
    }
    if name == 'bulk':
        if partition:
            schema['_properties'] = {'bulk': True}
        schema['_updater'] = bulk_numpy_updater
        # Only pull out counts to be serialized (save space and time)
        schema['_serializer'] = get_bulk_counts
        schema['_divider'] = 'bulk_binomial'
    else:
        # Since vivarium-core ensures that each store will only have a single
        # updater, it's OK to create new UniqueNumpyUpdater objects each time
        schema['_updater'] = UniqueNumpyUpdater().updater
        # Convert to list of contiguous Numpy arrays for faster and more
        # efficient serialization (still do not recommend emitting unique)
        schema['_serializer'] = get_unique_fields
        schema['_divider'] = UNIQUE_DIVIDERS[name]
    return schema


def bulk_name_to_idx(
    names: str | (List | np.ndarray),
    bulk_names: List | np.ndarray
) -> int | np.ndarray:
    """Primarily used to retrieve indices for groups of bulk molecules (e.g. NTPs)
    in the first run of a process and cache for future runs

    Args:
        names: List or array of things to find. Can also be single string.
        bulk_names: List of array of things to search
    
    Returns:
        Index or indices such that ``bulk_names[indices] == names``
    """
    # Convert from string names to indices in bulk array
    if isinstance(names, np.ndarray) or isinstance(names, list):
        # Big brain solution from https://stackoverflow.com/a/32191125
        # One downside: all values in names MUST be in bulk_names
        # Can mask missing values with bulk_names[return value] == names
        sorter = np.argsort(bulk_names)
        return np.take(sorter, np.searchsorted(
            bulk_names, names, sorter=sorter), mode='clip')
    else:
        return np.where(bulk_names == names)[0][0]


def bulk_numpy_updater(
        current: np.ndarray,
        update: List[Tuple[int|np.ndarray, int|np.ndarray]]
    ) -> np.ndarray: 
    """Updater function for bulk molecule structured array.

    Args:
        current: Bulk molecule structured array
        update: List of tuples ``(mol_idx, add_val)``, where
            ``mol_idx`` is the index (or array of indices) for 
            the molecule(s) to be updated and ``add_val`` is the 
            count (or array of counts) to be added to the current 
            count(s) for the specified molecule(s).
        
    Returns:
        Updated bulk molecule structured array
    """
    # Bulk updates are lists of tuples, where first value
    # in each tuple is an array of indices to update and
    # second value is array of updates to apply
    result = current
    # Numpy arrays are read-only outside of updater
    result.flags.writeable = True
    for (idx, value) in update:
        result['count'][idx] += value
    result.flags.writeable = False
    return result


def attrs(states: np.ndarray, attributes: List[str]) -> List[np.ndarray]:
    """Helper function to pull out arrays for unique molecule attributes

    Args:
        states: Structured Numpy array for all unique molecules of a given 
            type (e.g. RNA, active RNAP, etc.)
        attributes: List of field names (attributes) whose data should be 
            retrieved for all active unique molecules in ``states``

    Returns:
        List of arrays, one for each attribute. nth entry in each array 
        corresponds to the value of that attribute for the nth active 
        unique molecule in ``states`` 
    """
    # _entryState has dtype int8 so this works
    mol_mask = states['_entryState'].view(np.bool_)
    return [states[attribute][mol_mask] for attribute in attributes]


def get_free_indices(result: np.ndarray, n_objects: int
                     ) -> Tuple[np.ndarray, np.ndarray]:
    """Find inactive rows for new molecules and expand array if needed

    Args:
        result: Structured Numpy array for all unique molecules of a given 
            type (e.g. RNA, active RNAP, etc.)
        n_objects: Number of new unique molecules to be added
    
    Returns:
        A tuple ``(result, free_idx)``. ``result`` is the same as the 
        input argument unless ``n_objects`` is greater than the number 
        of inactive rows in ``result``. In this case, ``result`` is 
        grown by at least 10% by concatenating new rows (all zeros). 
        ``free_idx`` is an array of size ``n_objects`` that contains 
        the indices of rows in ``result`` that are inactive (``_entryState`` 
        field is 0).
    """
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

class UniqueNumpyUpdater:
    """Updates that set attributes of currently active unique molecules 
    must be applied before any updates that add or delete molecules. If this 
    is not enforced, in a single timestep, an update might delete a molecule 
    and allow a subsequent update to add a new molecule in the same row. Then, 
    an update that intends to modify an attribute of the original molecule 
    in that row will actually corrupt the data for the new molecule. 

    To fix this, this unique molecule updater is a bound method with access 
    to instance attributes that allow it to accumulate updates until given 
    the signal to apply the accumulated updates in the proper order. The 
    signal to apply these updates is given by a special process (
    :py:class:`ecoli.processes.unique_update.UniqueUpdate`) that is 
    automatically configured in 
    :py:meth:`ecoli.composites.ecoli_master.Ecoli._generate_processes_and_steps`
    """
    def __init__(self):
        self.add_updates = []
        self.set_updates = []
        self.delete_updates = []

    def updater(self, current, update):
        if len(update) == 0:
            return current
        
        # Store updates in class instance variables until all
        # evolvers have finished running. The UniqueUpdate process
        # then signals for all the updates to be applied in the
        # following order: set, add, delete (prevents overwriting)
        for update_type, update_val in update.items():
            if update_type == 'add':
                self.add_updates.append(update_val)
            elif update_type == 'set':
                self.set_updates.append(update_val)
            elif update_type == 'delete':
                self.delete_updates.append(update_val)
        
        if not update.get('update', False):
            return current

        result = current
        # Numpy arrays are read-only outside of updater
        result.flags.writeable = True
        active_mask = result['_entryState'].view(np.bool_)
        # Generate array of active indices for delete updates only
        if len(self.delete_updates) > 0:
            initially_active_idx = np.nonzero(active_mask)[0]
        for set_update in self.set_updates:
            # Set updates are dictionaries where each key is a column and
            # each value is an array. They are designed to apply to all rows
            # (molecules) that were active at the beginning of a timestep
            for col, col_values in set_update.items():
                result[col][active_mask] = col_values
        for add_update in self.add_updates:
            # Add updates are dictionaries where each key is a column and
            # each value is an array. The nth element of each array is the value
            # for the corresponding column of the nth new molecule to be added.
            n_new_molecules = len(next(iter(add_update.values())))
            result, free_indices = get_free_indices(result, n_new_molecules)
            for col, col_values in add_update.items():
                result[col][free_indices] = col_values
            result['_entryState'][free_indices] = 1
        for delete_indices in self.delete_updates:
            # Delete updates are arrays of active row indices to delete
            rows_to_delete = initially_active_idx[delete_indices]
            result[rows_to_delete] = np.zeros(1, dtype=result.dtype)
        
        self.add_updates = []
        self.delete_updates = []
        self.set_updates = []
        result.flags.writeable = False
        return result

def listener_schema(elements):
    basic_schema = {'_updater': 'set', '_emit': True}
    schema = {}
    for element, default in elements.items():
        # Assume that tuples contain (default, metadata) in that order
        if isinstance(default, tuple):
            schema[element] = {**basic_schema,
                '_default': default[0],
                '_properties': {'metadata': default[1]}}
        else:
            schema[element] = {**basic_schema, '_default': default}
    return schema


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
    daughter_1.flags.writeable = False
    daughter_2.flags.writeable = False
    return [daughter_1, daughter_2]

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
    # Some chromosome domains may be left behind because
    # they no longer exist after chromosome division. Skip
    # this assert when checking division of domains
    if 'child_domains' not in values.dtype.names:
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

        d1_bool = np.zeros(n_molecules, dtype=np.bool_)
        d2_bool = np.zeros(n_molecules, dtype=np.bool_)

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
