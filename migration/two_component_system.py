"""
tests that vivarium-ecoli two_component_system process update is the same as saved wcEcoli updates
"""
import pytest

from ecoli.processes.two_component_system import TwoComponentSystem
from migration.migration_utils import run_and_compare

@pytest.mark.master
def test_two_component_system_migration():
    times = [0, 2104]
    for initial_time in times:
        run_and_compare(initial_time, TwoComponentSystem)

if __name__ == "__main__":
    test_two_component_system_migration()
