"""
tests that vivarium-ecoli transcript_elongation process update is the same as saved wcEcoli updates
"""
import pytest

from ecoli.processes.transcript_elongation import TranscriptElongation
from migration.migration_utils import run_and_compare

@pytest.mark.master
def test_transcript_elongation_migration():
    times = [0, 1870]
    for initial_time in times:
        run_and_compare(initial_time, TranscriptElongation, layer=4)

if __name__ == "__main__":
    test_transcript_elongation_migration()
