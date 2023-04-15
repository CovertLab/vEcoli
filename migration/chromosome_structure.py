"""
tests that vivarium-ecoli chromosome_structure process update is the same as saved wcEcoli updates
"""
import pytest

from ecoli.processes.chromosome_structure import ChromosomeStructure
from migration.migration_utils import run_and_compare

@pytest.mark.master
def test_chromosome_structure_migration():
    times = [0, 2104]
    for initial_time in times:
        run_and_compare(initial_time, ChromosomeStructure,
            partition=False, layer=1)

if __name__ == "__main__":
    test_chromosome_structure_migration()
