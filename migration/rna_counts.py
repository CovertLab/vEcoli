"""
tests that vivarium-ecoli mrna_counts process update is the same as saved wcEcoli updates
"""

import pytest

from ecoli.processes.listeners.RNA_counts import RNACounts
from migration.migration_utils import run_and_compare


@pytest.mark.master
def test_rna_counts_migration():
    times = [0, 1870]
    for initial_time in times:
        run_and_compare(initial_time, RNACounts, partition=False, post=True)


if __name__ == "__main__":
    test_rna_counts_migration()
