# -- tests -- #

import tempfile
from pathlib import Path
from textwrap import dedent

import duckdb
import polars as pl
import pytest

from ecoli.library.transform.data_transformer import load_outputs
from ecoli.library.transform.models import AvailablePartitions
from ecoli.library.transform.tests.fixtures import lazyframe_fixture


def test_load_outputs():
    conn = duckdb.connect()
    query = dedent(""" \
        SELECT * FROM (
                FROM read_parquet(
                    [f'{REPO_ROOT}'/out/sms_multiseed_multigen/history/*/*/*/*/*/*.pq'],
                    hive_partitioning = true,
                    hive_types = {
                        'experiment_id': VARCHAR,
                        'variant': BIGINT,
                        'lineage_seed': BIGINT,
                        'generation': BIGINT,
                        'agent_id': VARCHAR,
                    }
                )
        ) WHERE experiment_id='sms_multiseed_multigen' AND variant=0 AND lineage_seed=0 AND generation=4 AND agent_id='0000'
    """)
    loaded = load_outputs(query, conn)
    print(loaded)


@pytest.mark.asyncio
async def test_export_parquet() -> None:
    n_rows = int(1e4)
    tmp = tempfile.TemporaryDirectory()
    pq_path = Path(tmp.name) / "66_000.parquet"
    lf: pl.LazyFrame | None = lazyframe_fixture(n_rows).sink_parquet(pq_path)
    print(lf.collect_schema() if lf else "Lf not yet populated!")
    tmp.cleanup()


def test_available_partitions():
    expid = "sms_multiseed_multigen"
    available_partitions = AvailablePartitions(experiment_id=expid)
    print(available_partitions.combinations)