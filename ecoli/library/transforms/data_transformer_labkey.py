"""
Data Relabeling/Aggregation Transformation (Eco/BioCyc)
"""

import pandas as pd
import polars as pl

from ecoli.library.transforms.data_transformer import DataTransformer


class DataTransformerLabkey(DataTransformer):
    def _transform(
            self,
            experiment_id: str,
            outputs_loaded: pd.DataFrame,
            observable_ids: list[str] | None = None,
            lazy: bool = True,
            **kwargs
    ) -> pl.DataFrame | pl.LazyFrame:
        pass
