"""
tests that vivarium-ecoli replication_data process update is the same as saved wcEcoli updates
"""

import pytest

from ecoli.processes.listeners.replication_data import ReplicationData
from migration.migration_utils import run_and_compare


@pytest.mark.master
def test_replication_data_migration():
    times = [0, 1870]
    for initial_time in times:
        run_and_compare(initial_time, ReplicationData, partition=False, post=True)


if __name__ == "__main__":
    test_replication_data_migration()
