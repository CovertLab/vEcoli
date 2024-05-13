from typing import Any

import polars as pl
import hvplot

from ecoli.analysis.template import get_field_metadata, named_idx

def plot(
    params: dict[str, Any],
    config_lf: pl.LazyFrame,
    history_lf: pl.LazyFrame,
    sim_data_path: str,
    validation_data_path: str
):
    pass
