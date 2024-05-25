"""
tests that vivarium-ecoli rnap_data process update is the same as saved wcEcoli updates
"""

import pytest

from ecoli.processes.listeners.rnap_data import RnapData
from migration.migration_utils import run_and_compare


@pytest.mark.master
def test_rnap_data_migration():
    times = [0, 1870]
    for initial_time in times:
        run_and_compare(initial_time, RnapData, partition=False, post=True)


if __name__ == "__main__":
    test_rnap_data_migration()
