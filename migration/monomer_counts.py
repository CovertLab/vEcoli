"""
tests that vivarium-ecoli monomer_counts process update is the same as saved wcEcoli updates
"""

import pytest

from ecoli.processes.listeners.monomer_counts import MonomerCounts
from migration.migration_utils import run_and_compare


@pytest.mark.master
def test_monomer_counts_migration():
    times = [0, 1870]
    for initial_time in times:
        run_and_compare(initial_time, MonomerCounts, partition=False, post=True)


if __name__ == "__main__":
    test_monomer_counts_migration()
