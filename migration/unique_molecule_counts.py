"""
tests that vivarium-ecoli unique_molecule_counts process update is the same as saved wcEcoli updates
"""

import pytest

from ecoli.processes.listeners.unique_molecule_counts import UniqueMoleculeCounts
from migration.migration_utils import run_and_compare


@pytest.mark.master
def test_unique_molecule_counts_migration():
    times = [0, 1870]
    for initial_time in times:
        run_and_compare(initial_time, UniqueMoleculeCounts, partition=False, post=True)


if __name__ == "__main__":
    test_unique_molecule_counts_migration()
