"""
tests that vivarium-ecoli polypeptide_elongation process update is the same as saved wcEcoli updates
"""

import pytest

from ecoli.processes.polypeptide_elongation import PolypeptideElongation
from migration.migration_utils import run_and_compare


@pytest.mark.master
def test_polypeptide_elongation_migration():
    times = [0, 1870]
    for initial_time in times:
        run_and_compare(initial_time, PolypeptideElongation, layer=4)


if __name__ == "__main__":
    test_polypeptide_elongation_migration()
