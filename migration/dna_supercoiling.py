"""
tests that vivarium-ecoli dna_supercoiling process update is the same as saved wcEcoli updates
"""
import pytest

from ecoli.processes.listeners.dna_supercoiling import DnaSupercoiling
from migration.migration_utils import run_and_compare

@pytest.mark.master
def test_dna_supercoiling_migration():
    times = [0, 1870]
    for initial_time in times:
        run_and_compare(initial_time, DnaSupercoiling, partition=False, post=True)

if __name__ == "__main__":
    test_dna_supercoiling_migration()
