import os
from typing import Any

import polars as pl
import hvplot

from ecoli.analysis.template import get_field_metadata, named_idx

def plot(
    params: dict[str, Any],
    config_lf: pl.LazyFrame,
    history_lf: pl.LazyFrame,
    sim_data_path: list[str],
    validation_data_path: list[str],
    outdir: str
):
    molecules_of_interest = ['GUANOSINE-5DP-3DP[c]', 'WATER[c]', 'PROTON[c]']
    bulk_names = get_field_metadata(config_lf, 'bulk')
    bulk_idx = {}
    for mol in molecules_of_interest:
        bulk_idx[mol] = bulk_names.index(mol)

    advanced_col_projection = {
        'time': pl.col('time'),
        **named_idx('bulk', bulk_idx.keys(), bulk_idx.values())
    }
    history_lf = history_lf.select(**advanced_col_projection)

    # NOTE: Must explicitly sort by ``time`` column or rows may be out of order
    history_lf = history_lf.sort('time')

    # When satisfied with your query, call ``collect`` on your LazyFrame
    history_df = history_lf.collect(streaming=True)

    plot = history_df.plot.scatter(x='time')
    os.chdir(outdir)
    hvplot.save(plot, 'test.html')
    history_df.write_parquet('test.pq')
    