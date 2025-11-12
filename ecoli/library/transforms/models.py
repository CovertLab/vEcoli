import abc
import dataclasses
from abc import ABC
from enum import StrEnum
from pathlib import Path
from typing import Any, _T, Callable

import duckdb
import polars as pl

from wholecell.utils.unit_struct_array import UnitStructArray


@dataclasses.dataclass
class SimulationConfigData:
    _df: pl.DataFrame

    def __init__(self, query: str):
        self._df = duckdb.sql(query).pl()

    def __getattr__(self, attr):
        if attr != "get":
            return getattr(self._df, attr)
        return getattr(self, attr)

    def get(self, attr: str) -> Any:
        value = self._df[[attr]].to_numpy().flatten()
        if len(value) != 1:
            raise ValueError(f"There is more than one configuration value for {attr} somehow!! I don't know how to handle this yet.")
        return value[0]


@dataclasses.dataclass
class DatasetLabels:
    bulk_ids: list[str]
    bulk_ids_biocyc: list[str]
    bulk_names_unique: list[str]
    bulk_common_names: list[str]
    reaction_ids: list[str]
    mrna_cistron_ids: list[str]
    mrna_cistron_names: list[str]
    cistron_data: UnitStructArray
    _common_names: list[str] | None = None

    def __post_init__(self):
        self._common_names = None

    @property
    def common_names(self):
        return self._common_names

    @common_names.setter
    def common_names(self, names: list[str] | None):
        self._common_names = names


class DataTransformation(ABC):
    _x: pl.DataFrame | None = None
    _y: pl.DataFrame | None = None

    @abc.abstractmethod
    def _f(self, x: pl.DataFrame, **kwargs) -> pl.DataFrame:
        pass

    def f(self, x: pl.DataFrame, **kwargs) -> None:
        y = self._f(x, **kwargs)
        self.x = x
        self.y = y

    def __call__(self, x: pl.DataFrame, **kwargs):
        return self.f(x, **kwargs)


class DataTransformationMean(DataTransformation):
    def _f(self, x: pl.DataFrame, *kwargs):
        new_data = {}

        for col in x.iter_columns():
            col_data = col.sum()




    @property
    def x(self):
        return self._x

    @x.setter
    def x(self, new: pl.DataFrame | None):
        self._x = new

    @property
    def y(self):
        return self._x

    @y.setter
    def y(self, new: pl.DataFrame | None):
        self._y = new

    def cardinality(self) -> tuple[float, float]:
        nx, ny = list(map(lambda df: len(df.rows()), [self.x, self.y]))
        dx, dy = list(map(lambda df: len(df.columns), [self.x, self.y]))
        return (
            (ny / nx), (dy / dx)
        )



class DataTransformExportFormat(StrEnum):
    PARQUET = "parquet"
    NDJSON = "ndjson"
    CSV = "csv"