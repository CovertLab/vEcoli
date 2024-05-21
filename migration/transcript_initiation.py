"""
tests that vivarium-ecoli transcript_initiation process update is the same as saved wcEcoli updates
"""

import pytest

from ecoli.processes.transcript_initiation import TranscriptInitiation
from migration.migration_utils import run_and_compare


@pytest.mark.master
def test_transcript_initiation_migration():
    times = [0, 1870]
    for initial_time in times:
        run_and_compare(initial_time, TranscriptInitiation, layer=3)


if __name__ == "__main__":
    test_transcript_initiation_migration()
