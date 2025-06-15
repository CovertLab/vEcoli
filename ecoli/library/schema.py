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
    "active_ribosome": {
        "divider": "ribosome_by_RNA",
        "topology": {
            "RNA": ("..", "RNA"),
            "full_chromosome": ("..", "full_chromosome"),
            "chromosome_domain": ("..", "chromosome_domain"),
            "active_RNAP": (
                "..",
                "active_RNAP",
            ),
        },
    },
    "full_chromosomes": {
        "divider": "by_domain",
        "topology": {
            "full_chromosome": (),
            "chromosome_domain": ("..", "chromosome_domain"),
        },
    },
    "chromosome_domains": {
        "divider": "by_domain",
        "topology": {
            "full_chromosome": ("..", "full_chromosome"),
            "chromosome_domain": (),
        },
    },
    "active_replisomes": {
        "divider": "by_domain",
        "topology": {
            "full_chromosome": ("..", "full_chromosome"),
            "chromosome_domain": ("..", "chromosome_domain"),
        },
    },
    "oriCs": {
        "divider": "by_domain",
        "topology": {
            "full_chromosome": ("..", "full_chromosome"),
            "chromosome_domain": ("..", "chromosome_domain"),
        },
    },
    "promoters": {
        "divider": "by_domain",
        "topology": {
            "full_chromosome": ("..", "full_chromosome"),
            "chromosome_domain": ("..", "chromosome_domain"),
        },
    },
    "chromosomal_segments": {
        "divider": "by_domain",
        "topology": {
            "full_chromosome": ("..", "full_chromosome"),
            "chromosome_domain": ("..", "chromosome_domain"),
        },
    },
    "DnaA_boxes": {
        "divider": "by_domain",
        "topology": {
            "full_chromosome": ("..", "full_chromosome"),
            "chromosome_domain": ("..", "chromosome_domain"),
        },
    },
    "active_RNAPs": {
        "divider": "by_domain",
        "topology": {
            "full_chromosome": ("..", "full_chromosome"),
            "chromosome_domain": ("..", "chromosome_domain"),
        },
    },
    "RNAs": {
        "divider": "rna_by_domain",
        "topology": {
            "active_RNAP": (
                "..",
                "active_RNAP",
            ),
            "full_chromosome": ("..", "full_chromosome"),
            "chromosome_domain": ("..", "chromosome_domain"),
        },
    },
    "genes": {
        "divider": "by_domain",
        "topology": {
            "full_chromosome": ("..", "full_chromosome"),
            "chromosome_domain": ("..", "chromosome_domain"),
        },
    },
}
"""A mapping of unique molecules to the names of their divider functions ars they are registered 
in the ``divider_registry`` in ``ecoli/__init__.py``

:meta hide-value:
"""


class MetadataArray(np.ndarray):
    """Subclass of Numpy array that allows for metadata to be stored with the array.
    Currently used to store next unique molecule index for unique molecule arrays."""

    def __new__(cls, input_array, metadata=None):
        # Input array should be an array instance
        obj = np.asarray(input_array).view(cls)
        # Ensure unique_index field exists and is unique
        if "unique_index" in obj.dtype.names:
            if "_entryState" in obj.dtype.names:
                unique_indices = obj["unique_index"][obj["_entryState"].view(np.bool_)]
                if len(unique_indices) != len(set(unique_indices)):
                    raise ValueError(
                        "All elements in the 'unique_index' field must be unique."
                    )
            else:
                raise ValueError("Input array must have an '_entryState' field.")
        else:
            raise ValueError("Input array must have a 'unique_index' field.")
        obj.metadata = metadata
        return obj

    def __array_finalize__(self, obj):
        # metadata is set in __new__ when creating new array
        if obj is None:
            return
        # Views should inherit metadata from parent
        self.metadata = getattr(obj, "metadata", None)

    def __array_wrap__(self, out_arr, context=None, return_scalar=False):
        # If the result is a scalar, return it as a base scalar type
        if out_arr.shape == ():
            return out_arr.item()
        else:
            return super(MetadataArray, self).__array_wrap__(out_arr, context)


def array_from(d: dict) -> np.ndarray:
    """Makes a Numpy array from dictionary values.

    Args:
        d: Dictionary whose values are to be converted

    Returns:
        Array of all values in d.
    """
    return np.array(list(d.values()))


def create_unique_indices(
    n_indexes: int, unique_molecules: MetadataArray
) -> np.ndarray:
    """We strongly recommend letting
    :py:meth:`UniqueNumpyUpdater.updater` generate unique indices
    for new unique molecules. If that is not possible, this function
    can be used to generate unique indices that should not conflict
    with any existing unique indices.

    Args:
        n_indexes: Number of indexes to generate.
        unique_molecules: Structured Numpy array of unique molecules.

    Returns:
        List of unique indexes for new unique molecules.
    """
    next_unique_index = unique_molecules.metadata
    unique_indices = np.arange(
        next_unique_index, int(next_unique_index + n_indexes), dtype=int
    )
    unique_molecules.metadata += n_indexes
    return unique_indices


def zero_listener(listener: dict[str, Any]) -> dict[str, Any]:
    """
    Takes a listener dictionary and creates a zeroed version of it.
    """
    new_listener = {}
    for key, value in listener.items():
        if isinstance(value, dict):
            new_listener[key] = zero_listener(value)
        else:
            zeros = np.zeros_like(value)
            if zeros.shape == ():
                zeros = zeros.item()
            new_listener[key] = zeros
    return new_listener


def not_a_process(value):
    """Returns ``True`` if not a :py:class:`vivarium.core.process.Process` instance."""
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
        return states["count"][idx]
    # evolve_state reads from ('allocate', process_name, 'bulk')
    # which is a simple Numpy array (not structured)
    return states[idx].copy()


class get_bulk_counts(Serializer):
    """Serializer for bulk molecules that saves counts without IDs or masses."""

    def serialize(bulk: np.ndarray) -> np.ndarray:
        """
        Args:
            bulk: Numpy structured array with a `count` field

        Returns:
            Contiguous (required by orjson) array of bulk molecule counts
        """
        return np.ascontiguousarray(bulk["count"])


class get_unique_fields(Serializer):
    """Serializer for unique molecules."""

    def serialize(unique: np.ndarray) -> list[np.ndarray]:
        """
        Args:
            unique: Numpy structured array of attributes for one unique molecule

        Returns:
            List of contiguous (required by orjson) arrays, one for each attribute
        """
        return [np.ascontiguousarray(unique[field]) for field in unique.dtype.names]


def numpy_schema(name: str, emit: bool = True) -> Dict[str, Any]:
    """Helper function used in ports schemas for bulk and unique molecules

    Args:
        name: ``bulk`` for bulk molecules or one of the keys in :py:data:`UNIQUE_DIVIDERS`
            for unique molecules
        emit: ``True`` if should be emitted (default)

    Returns:
        Fully configured ports schema for molecules of type ``name``
    """
    schema = {"_default": [], "_emit": emit}
    if name == "bulk":
        schema["_updater"] = bulk_numpy_updater
        # Only pull out counts to be serialized (save space and time)
        schema["_serializer"] = get_bulk_counts
        schema["_divider"] = "bulk_binomial"
    else:
        # Since vivarium-core ensures that each store will only have a single
        # updater, it's OK to create new UniqueNumpyUpdater objects each time
        schema["_updater"] = UniqueNumpyUpdater().updater
        # Convert to list of contiguous Numpy arrays for faster and more
        # efficient serialization (still do not recommend emitting unique)
        schema["_serializer"] = get_unique_fields
        schema["_divider"] = UNIQUE_DIVIDERS[name]
    return schema


def bulk_name_to_idx(
    names: str | (List | np.ndarray), bulk_names: List | np.ndarray
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
        return np.take(
            sorter, np.searchsorted(bulk_names, names, sorter=sorter), mode="clip"
        )
    else:
        return np.where(np.array(bulk_names) == names)[0][0]


def bulk_numpy_updater(
    current: np.ndarray, update: List[Tuple[int | np.ndarray, int | np.ndarray]]
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
    for idx, value in update:
        result["count"][idx] += value
    result.flags.writeable = False
    return result


def attrs(states: MetadataArray, attributes: List[str]) -> List[np.ndarray]:
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
    mol_mask = states["_entryState"].view(np.bool_)
    return [np.asarray(states[attribute][mol_mask]) for attribute in attributes]


def get_free_indices(
    result: MetadataArray, n_objects: int
) -> Tuple[MetadataArray, np.ndarray]:
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
    free_indices = np.where(result["_entryState"] == 0)[0]
    n_free_indices = free_indices.size

    if n_free_indices < n_objects:
        old_size = result.size
        n_new_entries = max(int(old_size * 0.1), n_objects - n_free_indices)

        result = MetadataArray(
            np.append(result, np.zeros(int(n_new_entries), dtype=result.dtype)),
            result.metadata,
        )

        free_indices = np.concatenate(
            (free_indices, old_size + np.arange(n_new_entries))
        )

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
    signal to apply these updates is given by a special process
    (:py:class:`~ecoli.processes.unique_update.UniqueUpdate`) that is
    automatically added to the simulation by
    :py:meth:`~ecoli.composites.ecoli_master.Ecoli.generate_processes_and_steps`.
    """

    def __init__(self):
        """Sets up instance attributes to accumulate updates.

        Attributes:
            add_updates: List of updates that add unique molecules
            set_updates: List of updates that modify existing unique molecules
            delete_updates: List of updates that delete unique molecules
        """
        self.add_updates = []
        self.set_updates = []
        self.delete_updates = []

    def updater(self, current: MetadataArray, update: Dict[str, Any]) -> MetadataArray:
        """Accumulates updates in instance attributes until given signal to
        apply all updates in the following order: ``set``, ``add``, ``delete``

        Args:
            current: Structured Numpy array for a given unique molecule
            update: Dictionary of updates to apply that can contain any
                combination of the following keys:

                - ``set``: Dictionary or list of dictionaries
                    Each key is an attribute of the given unique molecule
                    and each value is an array. Each array contains
                    the new attribute values for all active unique
                    molecules in a givne timestep. Can have multiple such
                    dictionaries in a list to apply multiple ``set`` updates.

                - ``add``: Dictionary or list of dictionaries
                    Each key is an attribute of the given unique moleucle
                    and each value is an array. The nth element of
                    each array is the value for the corresponding
                    attribute for the nth unique molecule to be added. If not
                    provided, unique indices for the ``unique_index`` attribute
                    are automatically generated for each new molecule. If
                    you need to reference the unique indices of new molecules in
                    the same process and time step in which you generated them,
                    you MUST use :py:func:`~ecoli.library.schema.create_unique_indices`
                    to generate the indices and supply them under the ``unique_index``
                    key of your ``add`` update. Can have multiple such
                    dictionaries in a list to apply multiple ``add`` updates.

                - ``delete``: List or 1D Numpy array of integers, or list of those
                    List of **active** molecule indices to delete. Note that
                    ``current`` may have rows that are marked as inactive, so
                    deleting the 10th active molecule may not equate to
                    deleting the value in the 10th row of ``current``. Can have
                    multiple such lists in a list to apply multiple ``delete`` updates.

                - ``update``: Boolean
                    Special key that should only be included in the update of
                    :py:class:`~ecoli.processes.unique_update.UniqueUpdate`.
                    Tells updater to apply all cached updates at the
                    end of an "execution layer" (see :ref:`partitioning`).

        Returns:
            Updated unique molecule structured Numpy array.
        """
        if len(update) == 0:
            return current

        # Store updates in class instance variables until all
        # evolvers have finished running. The UniqueUpdate process
        # then signals for all the updates to be applied in the
        # following order: set, add, delete (prevents overwriting)
        for update_type, update_val in update.items():
            if update_type == "add":
                if isinstance(update_val, list):
                    self.add_updates.extend(update_val)
                elif isinstance(update_val, dict):
                    self.add_updates.append(update_val)
                else:
                    raise ValueError(
                        "Add updates must be dictionaries or lists of dictionaries"
                    )
            elif update_type == "set":
                if isinstance(update_val, list):
                    self.set_updates.extend(update_val)
                elif isinstance(update_val, dict):
                    self.set_updates.append(update_val)
                else:
                    raise ValueError(
                        "Add updates must be dictionaries or lists of dictionaries"
                    )
            elif update_type == "delete":
                if isinstance(update_val, list):
                    if len(update_val) == 0:
                        continue
                    elif isinstance(update_val[0], list) or isinstance(
                        update_val[0], np.ndarray
                    ):
                        self.delete_updates.extend(update_val)
                    elif isinstance(update_val[0], int):
                        self.delete_updates.append(update_val)
                    elif isinstance(update_val[0], np.integer):
                        self.delete_updates.append(update_val)
                    else:
                        raise ValueError(
                            "Delete updates must be lists/arrays of integers "
                            "OR lists of lists/arrays of integers"
                        )
                elif isinstance(update_val, np.ndarray) and np.issubdtype(
                    update_val.dtype, np.integer
                ):
                    self.delete_updates.append(update_val)
                else:
                    raise ValueError(
                        "Delete updates must be lists/arrays of integers "
                        "OR lists of lists/arrays of integers"
                    )

        if not update.get("update", False):
            return current

        result = current
        # Numpy arrays are read-only outside of updater
        result.flags.writeable = True
        active_mask = result["_entryState"].view(np.bool_)
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
            if "unique_index" not in add_update:
                result["unique_index"][free_indices] = (
                    np.arange(n_new_molecules) + result.metadata
                )
                result.metadata += n_new_molecules
            for col, col_values in add_update.items():
                result[col][free_indices] = col_values
            result["_entryState"][free_indices] = 1
        for delete_indices in self.delete_updates:
            # Delete updates are arrays of active row indices to delete
            rows_to_delete = initially_active_idx[delete_indices]
            result[rows_to_delete] = np.zeros(1, dtype=result.dtype)

        self.add_updates = []
        self.delete_updates = []
        self.set_updates = []
        result.flags.writeable = False
        return result


def listener_schema(elements: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    """Helper function that can be used in ``ports_schema`` to create generic
    schema for a collection of listeners.

    Args:
        elements: Dictionary where keys are listener names and values are the
            defaults for each listener. Alternatively, if the value is a
            tuple, assume that the first element is the default and the second
            is metadata that will be emitted at the beginning of a simulation
            (see :py:meth:`~ecoli.experiments.ecoli_master_sim.EcoliSim.output_metadata`).
            This metadata can then be retrieved later to aid in interpreting
            listener values (see :py:func:`~ecoli.library.parquet_emitter.field_metadata`).
            As an example, this metadata might be an array of molecule names
            for a listener whose emits are arrays of counts, where the nth
            molecule name in the metadata corresponds to the nth value in the
            counts that are emitted.

    Returns:
        Ports schemas for all listeners in ``elements``.
    """
    basic_schema = {"_updater": "set", "_emit": True}
    schema = {}
    for element, default in elements.items():
        # Assume that tuples contain (default, metadata) in that order
        if isinstance(default, tuple):
            schema[element] = {
                **basic_schema,
                "_default": default[0],
                "_properties": {"metadata": default[1]},
            }
        else:
            schema[element] = {**basic_schema, "_default": default}
    return schema


# :term:`dividers`
def divide_binomial(state: int) -> tuple[int, int]:
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
    return counts_1, counts_2


def divide_bulk(state: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Divider function for bulk molecules. Automatically added to bulk
    molecule ports schemas by :py:func:`ecoli.library.schema.numpy_schema`
    when ``name == 'bulk'``. Uses binomial distribution with ``p=0.5`` to
    randomly partition counts.

    Args:
        state: Structured Numpy array of bulk molecule data.

    Returns:
        List of two structured Numpy arrays, each representing the bulk
        molecule state of a daughter cell.
    """
    counts = state["count"]
    seed = counts.sum() % RAND_MAX
    # TODO: Random state/seed in store?
    random_state = np.random.RandomState(seed=seed)
    daughter_1 = state.copy()
    daughter_2 = state.copy()
    daughter_1["count"] = random_state.binomial(counts, 0.5)
    daughter_2["count"] = counts - daughter_1["count"]
    daughter_1.flags.writeable = False
    daughter_2.flags.writeable = False
    return daughter_1, daughter_2


# TODO: Create a store for growth rate noise simulation parameter


def divide_ribosomes_by_RNA(
    values: MetadataArray, state: Dict[str, Any]
) -> tuple[np.ndarray, np.ndarray]:
    """Divider function for active ribosome unique molecules. Automatically
    added to ports schema by :py:func:`ecoli.library.schema.numpy_schema` when
    ``name == 'active_ribosome'``. Ensures that ribosomes are divided the same
    way that their associated mRNAs are.

    Args:
        values: Structured Numpy array of active ribosome unique molecule state
        state: View into relevant unique molecule states according to the
            topology defined under the ``active_ribosome`` key in
            :py:data:`ecoli.library.schema.UNIQUE_DIVIDERS`.

    Returns:
        List of two structured Numpy arrays, each containing the active
        ribosome unique molecule state of a daughter cell
    """
    (mRNA_index,) = attrs(values, ["mRNA_index"])
    n_molecules = len(mRNA_index)
    if n_molecules > 0:
        # Divide ribosomes based on their mRNA index
        d1_rnas, d2_rnas = divide_RNAs_by_domain(state["RNA"], state)
        d1_bool = np.isin(mRNA_index, d1_rnas["unique_index"])
        d2_bool = np.isin(mRNA_index, d2_rnas["unique_index"])

        # Binomially divide indexes of mRNAs that are degraded but still
        # has bound ribosomes. This happens because mRNA degradation does
        # not abort ongoing translation of the mRNA
        degraded_mRNA_indexes = np.unique(
            mRNA_index[np.logical_not(np.logical_or(d1_bool, d2_bool))]
        )
        n_degraded_mRNA = len(degraded_mRNA_indexes)

        if n_degraded_mRNA > 0:
            # TODO: Random state/seed in store?
            random_state = np.random.RandomState(seed=n_molecules)
            n_degraded_mRNA_d1 = random_state.binomial(n_degraded_mRNA, p=0.5)
            degraded_mRNA_indexes_d1 = random_state.choice(
                degraded_mRNA_indexes, size=n_degraded_mRNA_d1, replace=False
            )
            degraded_mRNA_indexes_d2 = np.setdiff1d(
                degraded_mRNA_indexes, degraded_mRNA_indexes_d1
            )

            # Divide "lost" ribosomes based on how these mRNAs were divided
            lost_ribosomes_d1 = np.isin(mRNA_index, degraded_mRNA_indexes_d1)
            lost_ribosomes_d2 = np.isin(mRNA_index, degraded_mRNA_indexes_d2)

            d1_bool[lost_ribosomes_d1] = True
            d2_bool[lost_ribosomes_d2] = True

        n_d1 = np.count_nonzero(d1_bool)
        n_d2 = np.count_nonzero(d2_bool)

        assert n_molecules == n_d1 + n_d2
        assert np.count_nonzero(np.logical_and(d1_bool, d2_bool)) == 0

        ribosomes = values[values["_entryState"].view(np.bool_)]
        return ribosomes[d1_bool], ribosomes[d2_bool]

    return np.zeros(0, dtype=values.dtype), np.zeros(0, dtype=values.dtype)


def divide_domains(state: dict[str, MetadataArray]) -> dict[str, np.ndarray]:
    """Divider function for chromosome domains. Ensures that all chromosome
    domains associated with a full chromosome go to the same daughter cell
    that the full chromosome does.

    Args:
        state: Structured Numpy array of chromosome domain unique molecule
            state.

    Returns:
        List of two structured Numpy arrays, each containing the chromosome
        domain unique molecule state for a daughter cell.
    """
    (domain_index_full_chroms,) = attrs(state["full_chromosome"], ["domain_index"])
    domain_index_domains, child_domains = attrs(
        state["chromosome_domain"], ["domain_index", "child_domains"]
    )

    # TODO: Random state/seed in store?
    # d1_gets_first_chromosome = randomState.rand() < 0.5
    # index = not d1_gets_first_chromosome
    # d1_domain_index_full_chroms = domain_index_full_chroms[index::2]
    # d2_domain_index_full_chroms = domain_index_full_chroms[not index::2]

    d1_domain_index_full_chroms = domain_index_full_chroms[0::2]
    d2_domain_index_full_chroms = domain_index_full_chroms[1::2]
    d1_all_domain_indexes = get_descendent_domains(
        d1_domain_index_full_chroms, domain_index_domains, child_domains, -1
    )
    d2_all_domain_indexes = get_descendent_domains(
        d2_domain_index_full_chroms, domain_index_domains, child_domains, -1
    )

    # Check that the domains are being divided correctly
    assert np.intersect1d(d1_all_domain_indexes, d2_all_domain_indexes).size == 0

    return {
        "d1_all_domain_indexes": d1_all_domain_indexes,
        "d2_all_domain_indexes": d2_all_domain_indexes,
    }


def divide_by_domain(
    values: np.ndarray, state: Dict[str, Any]
) -> tuple[np.ndarray, np.ndarray]:
    """Divider function for unique molecules that are attached to the
    chromsome. Ensures that these molecules are divided in accordance
    with the way that chromosome domains are divided.

    Args:
        values: Structured Numpy array of unique molecule state
        state: View of ``full_chromosome`` and ``chromosome_domain``
            state as configured under any of the unique molecules with
            a divider of `by_domain` in
            :py:data:`ecoli.library.schema.UNIQUE_DIVIDERS`.

    Returns:
        List of two structured Numpy arrays, each containing the
        unique molecule state of a daughter cell.
    """
    domain_division = divide_domains(state)
    values = values[values["_entryState"].view(np.bool_)]
    d1_bool = np.isin(values["domain_index"], domain_division["d1_all_domain_indexes"])
    d2_bool = np.isin(values["domain_index"], domain_division["d2_all_domain_indexes"])
    # Some chromosome domains may be left behind because
    # they no longer exist after chromosome division. Skip
    # this assert when checking division of domains
    if "child_domains" not in values.dtype.names:
        assert d1_bool.sum() + d2_bool.sum() == len(values)
    return values[d1_bool], values[d2_bool]


def divide_RNAs_by_domain(
    values: MetadataArray, state: Dict[str, Any]
) -> tuple[np.ndarray, np.ndarray]:
    """Divider function for RNA unique molecules. Ensures that incomplete
    transcripts are divided in accordance with how active RNAPs are
    divided (which themselves depend on how chromosome domains are divided).

    Args:
        values: Structured Numpy array of RNA unique molecule state
        state: View of relevant unique molecule states according to the
            topology under the ``RNAs`` key in
            :py:data:`ecoli.library.schema.UNIQUE_DIVIDERS`.

    Returns:
        List of two structured Numpy arrays, each containing the RNA
        unique molecule state of a daughter cell.
    """
    is_full_transcript, RNAP_index = attrs(values, ["is_full_transcript", "RNAP_index"])

    n_molecules = len(is_full_transcript)

    if n_molecules > 0:
        # Figure out which RNAPs went to each daughter cell
        domain_division = divide_domains(state)
        rnaps = state["active_RNAP"]
        rnaps = rnaps[rnaps["_entryState"].view(np.bool_)]
        d1_rnap_bool = np.isin(
            rnaps["domain_index"], domain_division["d1_all_domain_indexes"]
        )
        d2_rnap_bool = np.isin(
            rnaps["domain_index"], domain_division["d2_all_domain_indexes"]
        )
        d1_rnap_indexes = rnaps["unique_index"][d1_rnap_bool]
        d2_rnap_indexes = rnaps["unique_index"][d2_rnap_bool]

        d1_bool = np.zeros(n_molecules, dtype=np.bool_)
        d2_bool = np.zeros(n_molecules, dtype=np.bool_)

        # Divide full transcripts binomially
        full_transcript_indexes = np.where(is_full_transcript)[0]
        if len(full_transcript_indexes) > 0:
            # TODO: Random state/seed in store?
            random_state = np.random.RandomState(seed=n_molecules)
            n_full_d1 = random_state.binomial(
                np.count_nonzero(is_full_transcript), p=0.5
            )
            full_d1_indexes = random_state.choice(
                full_transcript_indexes, size=n_full_d1, replace=False
            )
            full_d2_indexes = np.setdiff1d(full_transcript_indexes, full_d1_indexes)

            d1_bool[full_d1_indexes] = True
            d2_bool[full_d2_indexes] = True

        # Divide partial transcripts based on how their associated
        # RNAPs were divided
        partial_transcript_indexes = np.where(np.logical_not(is_full_transcript))[0]
        RNAP_index_partial_transcripts = RNAP_index[partial_transcript_indexes]

        partial_d1_indexes = partial_transcript_indexes[
            np.isin(RNAP_index_partial_transcripts, d1_rnap_indexes)
        ]
        partial_d2_indexes = partial_transcript_indexes[
            np.isin(RNAP_index_partial_transcripts, d2_rnap_indexes)
        ]

        d1_bool[partial_d1_indexes] = True
        d2_bool[partial_d2_indexes] = True

        n_d1 = np.count_nonzero(d1_bool)
        n_d2 = np.count_nonzero(d2_bool)

        assert n_molecules == n_d1 + n_d2
        assert np.count_nonzero(np.logical_and(d1_bool, d2_bool)) == 0

        rnas = values[values["_entryState"].view(np.bool_)]
        return rnas[d1_bool], rnas[d2_bool]

    return np.zeros(0, dtype=values.dtype), np.zeros(0, dtype=values.dtype)


def empty_dict_divider(values):
    """Divider function that sets both daughter cell states to empty dicts."""
    return {}, {}


def divide_set_none(values):
    """Divider function that sets both daughter cell states to ``None``."""
    return None, None


def remove_properties(schema: Dict[str, Any], properties: List[str]) -> Dict[str, Any]:
    """Helper function to recursively remove certain properties from a
    ports schema.

    Args:
        schema: Ports schema to remove properties from
        properties: List of properties to remove

    Returns:
        Ports schema with all properties in ``properties`` recursively removed.
    """
    if isinstance(schema, dict):
        for property in properties:
            schema.pop(property, None)
        for key, value in schema.items():
            schema[key] = remove_properties(value, properties)
    return schema


def flatten(nested_list: List[List[Any]]) -> List[Any]:
    """
    Flattens a nested list into a single list.

    Args:
        l: Nested list to flatten.
    """
    return [item for sublist in nested_list for item in sublist]


def follow_domain_tree(
    domain: int, domain_index: np.ndarray, child_domains: np.ndarray, place_holder: int
) -> List[int]:
    """
    Recursive function that returns all the descendents of a single node in
    the domain tree, including itself.

    Args:
        domain: Domain index to find all descendents for
        domain_index: Array of all domain indices
        child_domains: Array of child domains for each index in ``domain_index``
        place_holder: Placeholder domain index (e.g. used in ``child_domains``
            for domain indices that do not have child domains)
    """
    children_nodes = child_domains[np.where(domain_index == domain)[0][0]]

    if children_nodes[0] != place_holder:
        # If the node has children, recursively run function on each of the
        # node's two children
        branches = flatten(
            [
                follow_domain_tree(child, domain_index, child_domains, place_holder)
                for child in children_nodes
            ]
        )

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

    Args:
        root_domains: List of domains to get descendents of
        domain_index: Array of all domain indices for chromosome domains
        child_domains: Array of child domains for each index in ``domain_index``
        place_holder: Placeholder domain index (e.g. used in ``child_domains``
            for domain indices that do not have any child domains)
    """
    return np.array(
        flatten(
            [
                follow_domain_tree(
                    root_domain, domain_index, child_domains, place_holder
                )
                for root_domain in root_domains
            ]
        )
    )
