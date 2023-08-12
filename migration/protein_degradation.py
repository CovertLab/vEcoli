"""
tests that vivarium-ecoli protein_degradation process update is the same as saved wcEcoli updates
"""
import pytest

from ecoli.processes.protein_degradation import ProteinDegradation
from migration.migration_utils import run_and_compare

@pytest.mark.master
def test_protein_degradation_migration():
    times = [0, 1870]
    for initial_time in times:
        run_and_compare(initial_time, ProteinDegradation, layer=3)

if __name__ == "__main__":
    test_protein_degradation_migration()
