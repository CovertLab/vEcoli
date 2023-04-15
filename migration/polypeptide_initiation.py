"""
tests that vivarium-ecoli polypeptide_initiation process update is the same as saved wcEcoli updates
"""
import pytest

from ecoli.processes.polypeptide_initiation import PolypeptideInitiation
from migration.migration_utils import run_and_compare

@pytest.mark.master
def test_polypeptide_initiation_migration():
    times = [0, 2132]
    # Activating too many new active ribosomes at 2132 for some reason
    for initial_time in times:
        run_and_compare(initial_time, PolypeptideInitiation)

if __name__ == "__main__":
    test_polypeptide_initiation_migration()
