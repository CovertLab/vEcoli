"""
tests that vivarium-ecoli mass_listener process update is the same as saved wcEcoli updates
"""
import pytest

from ecoli.processes.listeners.mass_listener import MassListener
from migration.migration_utils import run_and_compare

@pytest.mark.master
def test_mass_listener_migration():
    times = [0, 2132]
    for initial_time in times:
        run_and_compare(initial_time, MassListener, partition=False, post=True)

if __name__ == "__main__":
    test_mass_listener_migration()
