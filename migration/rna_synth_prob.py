"""
tests that vivarium-ecoli rna_synth_prob process update is the same as saved wcEcoli updates
"""

import pytest

from ecoli.processes.listeners.rna_synth_prob import RnaSynthProb
from migration.migration_utils import run_and_compare


@pytest.mark.master
def test_rna_synth_prob_migration():
    times = [0, 1870]
    for initial_time in times:
        run_and_compare(initial_time, RnaSynthProb, partition=False, post=True)


if __name__ == "__main__":
    test_rna_synth_prob_migration()
