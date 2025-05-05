import numpy as np
from vivarium.core.registry import Serializer
from typing import List, Tuple, Dict, Any


RAND_MAX = 2**31 - 1

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
    # result.flags.writeable = True
    for idx, value in update:
        result["count"][idx] += value
    # result.flags.writeable = False
    return result


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



class get_bulk_counts(Serializer):
    """Serializer for bulk molecules that saves counts without IDs or masses."""

    def serialize(self, bulk: np.ndarray) -> np.ndarray:
        """
        Args:
            bulk: Numpy structured array with a `count` field

        Returns:
            Contiguous (required by orjson) array of bulk molecule counts
        """
        return np.ascontiguousarray(bulk["count"])