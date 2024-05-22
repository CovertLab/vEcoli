"""
tests that vivarium-ecoli tf_binding process update is the same as saved wcEcoli updates
"""

import pytest

from ecoli.processes.tf_binding import TfBinding
from migration.migration_utils import run_and_compare


@pytest.mark.master
def test_tf_binding_migration():
    times = [0, 1870]
    for initial_time in times:
        run_and_compare(initial_time, TfBinding, partition=False, layer=2)


if __name__ == "__main__":
    test_tf_binding_migration()
