"""
tests that vivarium-ecoli ribosome_data process update is the same as saved wcEcoli updates
"""

import pytest

from ecoli.processes.listeners.ribosome_data import RibosomeData
from migration.migration_utils import run_and_compare


@pytest.mark.master
def test_ribosome_data_migration():
    times = [0, 1870]
    for initial_time in times:
        run_and_compare(initial_time, RibosomeData, partition=False, post=True)


if __name__ == "__main__":
    test_ribosome_data_migration()
