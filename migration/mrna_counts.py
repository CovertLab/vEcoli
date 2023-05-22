"""
tests that vivarium-ecoli mrna_counts process update is the same as saved wcEcoli updates
"""
import pytest

from ecoli.processes.listeners.mRNA_counts import mRNACounts
from migration.migration_utils import run_and_compare

@pytest.mark.master
def test_mrna_counts_migration():
    times = [0, 2104]
    for initial_time in times:
        run_and_compare(initial_time, mRNACounts, partition=False, post=True)

if __name__ == "__main__":
    test_mrna_counts_migration()
