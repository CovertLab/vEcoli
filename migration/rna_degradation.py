"""
tests that vivarium-ecoli RnaDegradation process update is the same as saved wcEcoli updates
"""

import pytest

from ecoli.processes.rna_degradation import RnaDegradation
from migration.migration_utils import run_and_compare


@pytest.mark.master
def test_rna_degradation_migration():
    times = [0, 1870]
    for initial_time in times:
        run_and_compare(initial_time, RnaDegradation, layer=3)


if __name__ == "__main__":
    test_rna_degradation_migration()
