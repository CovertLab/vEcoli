"""
tests that vivarium-ecoli complexation process update is the same as saved wcEcoli updates
"""
import pytest

from ecoli.processes.complexation import Complexation
from migration.migration_utils import run_and_compare

@pytest.mark.master
def test_complexation_migration():
    times = [0, 2104]
    for initial_time in times:
        run_and_compare(initial_time, Complexation)

if __name__ == "__main__":
    test_complexation_migration()
