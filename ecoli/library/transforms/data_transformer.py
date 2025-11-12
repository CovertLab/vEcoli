"""
Base DataTransformation ABC that is inherited by any data transformer required for a given post-processing transformation analysis module
"""


import abc
import itertools
import math
import subprocess
import tempfile
from pathlib import Path
from typing import Any, LiteralString

import duckdb
import pandas as pd
import polars as pl
import numpy as np
import pytest
from duckdb import DuckDBPyConnection

from ecoli.library.transforms import PARTITION_GROUPS
from ecoli.library.parquet_emitter import dataset_sql
from ecoli.library.sim_data import LoadSimData
from ecoli.library.transforms.models import DatasetLabels, DataTransformExportFormat
from reconstruction.ecoli.simulation_data import SimulationDataEcoli


class DataTransformer(abc.ABC):
    sim_data_path: str
    sim_data: SimulationDataEcoli
    data_labels: DatasetLabels

    def __init__(self, sim_data_path: Path):
        self.sim_data_path = str(sim_data_path)
        self.sim_data = LoadSimData(str(self.sim_data_path)).sim_data
        self.data_labels = self._get_data_labels()

    @abc.abstractmethod
    def _transform(
            self,
            experiment_id: str,
            outputs_loaded: pd.DataFrame,
            observable_ids: list[str] | None = None,
            lazy: bool = True,
            **kwargs
    ) -> pl.DataFrame | pl.LazyFrame:
        pass

    def transform(
            self,
            experiment_id: str,
            simulation_outdir: str,
            observable_ids: list[str] | None = None,
            variant: int = 0,
            seed: int = 0,
            generation: int = 0,
            agent_id: str = "0",
            history_sql: str | None = None,
            lazy: bool = True,
            conn: DuckDBPyConnection | None = None,
            **kwargs
    ) -> pl.DataFrame | pl.LazyFrame:
        history_sql_filtered = history_sql
        if history_sql_filtered is None:
            dbf_dict = partitions_dict("single", experiment_id, variant, seed, generation, agent_id)
            db_filter = get_db_filter(dbf_dict)
            history_sql_filtered = history_sql or get_filtered_query(simulation_outdir, experiment_id, db_filter)

        output_loaded: pd.DataFrame = load_outputs(sql=history_sql_filtered, conn=conn)
        return self._transform(experiment_id=experiment_id, outputs_loaded=output_loaded, observable_ids=observable_ids, lazy=lazy, **kwargs)

    def export(self, genes_df: pl.LazyFrame, outdir: str, filename: str, io_format: DataTransformExportFormat | str | None = None) -> pl.LazyFrame | None:
        export_format = io_format or DataTransformExportFormat.PARQUET
        exporter = getattr(genes_df, f"sink_{export_format}")
        out_path = Path(outdir) / f"{filename}.{export_format}"
        lf = exporter(out_path)
        print(f'Successfully exported {filename} to {outdir}!')
        return lf

    def _get_data_labels(self) -> DatasetLabels:
        bulk_ids = self._get_bulk_ids()
        bulk_ids_biocyc = [bulk_id[:-3] for bulk_id in bulk_ids]
        bulk_names_unique = list(np.unique(bulk_ids_biocyc))
        bulk_common_names = self._get_common_names(bulk_names_unique)
        rxn_ids = self._get_rxn_ids()
        cistron_data = self.sim_data.process.transcription.cistron_data
        mrna_cistron_ids = cistron_data["id"][cistron_data["is_mRNA"]].tolist()
        mrna_cistron_names = [self.sim_data.common_names.get_common_name(cistron_id) for cistron_id in mrna_cistron_ids]
        return DatasetLabels(
            bulk_ids,
            bulk_ids_biocyc,
            bulk_names_unique,
            bulk_common_names,
            rxn_ids,
            mrna_cistron_ids,
            mrna_cistron_names,
            cistron_data
        )

    def _get_bulk_ids(self):
        bulk_ids = self.sim_data.internal_state.bulk_molecules.bulk_data["id"].tolist()
        return bulk_ids

    def _get_rxn_ids(self):
        rxn_ids = self.sim_data.process.metabolism.base_reaction_ids
        return rxn_ids

    def _get_common_names(self, bulk_names: list[str]):
        bulk_common_names = [self.sim_data.common_names.get_common_name(name) for name in bulk_names]
        duplicates = []
        for item in bulk_common_names:
            if bulk_common_names.count(item) > 1 and item not in duplicates:
                duplicates.append(item)

        for dup in duplicates:
            sp_idxs = [index for index, item in enumerate(bulk_common_names) if item == dup]
            for sp_idx in sp_idxs:
                bulk_rename = str(bulk_common_names[sp_idx]) + f"[{bulk_names[sp_idx]}]"
                bulk_common_names[sp_idx] = bulk_rename

        return bulk_common_names


# -- utils methods -- #


def partitions_dict(analysis_type, exp_select, variant_select, seed_select, gen_select, agent_select) -> dict[str, Any]:
    partitions_req = PARTITION_GROUPS[analysis_type]
    partitions_all = read_partitions(exp_select, variant_select, seed_select, gen_select, agent_select)
    partitions_dict = {}
    for partition in partitions_req:
        partitions_dict[partition] = partitions_all[partition]
    partitions_dict["experiment_id"] = f"'{partitions_dict['experiment_id']}'"
    return partitions_dict


def read_partitions(
        exp_select: str, variant_select: int, seed_select: int, gen_select: int, agent_select: str
) -> dict:
    partitions_selected = {
        "experiment_id": exp_select,
        "variant": variant_select,
        "lineage_seed": seed_select,
        "generation": gen_select,
        "agent_id": agent_select,
    }
    return partitions_selected


def get_db_filter(partitions_dict) -> LiteralString:
    db_filter_list = []
    for key, value in partitions_dict.items():
        db_filter_list.append(str(key) + "=" + str(value))
    db_filter = " AND ".join(db_filter_list)
    return db_filter


def get_filtered_query(output_dir, experiment_id, db_filter):
    pq_columns = [
        "bulk",
        "listeners__fba_results__base_reaction_fluxes",
        "listeners__rna_counts__full_mRNA_cistron_counts",
    ]
    history_sql_base, _, _ = dataset_sql(output_dir, experiment_ids=[experiment_id])
    return f"SELECT {','.join(pq_columns)},time FROM ({history_sql_base}) WHERE {db_filter} ORDER BY time"


def load_outputs(sql: str, conn: DuckDBPyConnection | None = None) -> pd.DataFrame:
    kwargs = {
        "query": sql
    }
    if conn is not None:
        kwargs["connection"] = conn

    outputs_df = duckdb.sql(**kwargs).df()
    outputs_df = outputs_df.groupby("time", as_index=False).sum()
    outputs_df = outputs_df.copy()
    return outputs_df


def downsample_eager(df_long: pd.DataFrame | pl.DataFrame) -> pd.DataFrame | pl.DataFrame:
    tp_all = np.unique(df_long["time"]).astype(int)
    ds_ratio = int(np.ceil(np.shape(df_long)[0] / 20000))
    tp_ds = list(itertools.islice(tp_all, 0, max(tp_all), ds_ratio))
    condition: list[bool] = np.isin(df_long["time"], tp_ds).tolist()
    df_ds = df_long.filter([condition]) if isinstance(df_long, pl.DataFrame) \
        else df_long[condition]
    # df_ds = df_long.filter([np.isin(df_long["time"], tp_ds)]) if isinstance(df_long, pl.DataFrame) \
    #     else df_long[np.isin(df_long["time"], tp_ds)]
    return df_ds


def downsample(df_long: pl.LazyFrame):
    tp_all = (
        df_long
        .select(pl.col("time").unique().sort())
        .collect()
        .get_column("time")
        .to_numpy()
        .astype(int)
    )
    # Total row count (scalar)
    n_rows = df_long.select(pl.count().alias("n")).collect().item()
    # Downsampling ratio
    ds_ratio = int(math.ceil(n_rows / 20_000))
    # Compute the sampled time values exactly as before
    tp_ds = list(itertools.islice(tp_all, 0, tp_all.max(), ds_ratio))
    # --- Lazy filter based on membership ---
    df_ds = df_long.filter(pl.col("time").is_in(tp_ds))
    return df_ds


def downsample_dataframe(df_long: pd.DataFrame) -> pd.DataFrame:
    tp_all = np.unique(df_long["time"]).astype(int)
    ds_ratio = int(np.ceil(np.shape(df_long)[0] / 20000))
    tp_ds = list(itertools.islice(tp_all, 0, max(tp_all), ds_ratio))
    df_ds = df_long[np.isin(df_long["time"], tp_ds)]
    return df_ds


def get_config_value(config_df: pl.DataFrame, col: str):
    return config_df[[col]].to_numpy().flatten()[0]


# -- tests -- #

def lazyframe_fixture(upper: int = 1111) -> pl.LazyFrame:
    def fake_data(upper: int = 1111) -> dict[str, list[float]]:
        return {
            "time": np.arange(0, upper).tolist(),
            "x": list(map(lambda i: (i ** 0.3) / (2.2 ** 11.11), list(range(upper)))),
            "y": list(map(lambda i: (-i ** 0.3 ** 0.2) / (2.2 ** 11.11), list(range(upper)))),
            "z": list(map(lambda i: (i ** 0.3) / (2.2 ** 11.11 ** 2 / 3), list(range(upper))))
        }
    return pl.LazyFrame(fake_data(upper))


def test_downsample(sim_data_path: Path) -> None:
    def downsample_eager(df_long: pl.DataFrame) -> pl.DataFrame:
        tp_all = np.unique(df_long["time"].to_numpy()).astype(int)
        ds_ratio = int(math.ceil(len(df_long) / 20000))
        tp_ds = list(itertools.islice(tp_all, 0, tp_all.max(), ds_ratio))
        return df_long.filter(pl.col("time").is_in(tp_ds))

    def downsample_pd(df_long: pd.DataFrame) -> pd.DataFrame:
        tp_all = np.unique(df_long["time"]).astype(int)
        ds_ratio = int(np.ceil(np.shape(df_long)[0] / 20000))
        tp_ds = list(itertools.islice(tp_all, 0, max(tp_all), ds_ratio))
        df_ds = df_long[np.isin(df_long["time"], tp_ds)]
        return df_ds

    pddf = pd.DataFrame({
        "time": np.arange(0, 100),
        "bulk": np.random.rand(100),
    })

    pldf = pl.from_pandas(pddf)
    lf = pldf.lazy()

    eager_out = downsample_eager(pldf)
    lazy_out = downsample(lf).collect()
    pd_out = downsample_pd(pddf)

    assert eager_out.equals(lazy_out)
    assert pd_out.equals(lazy_out.to_pandas())


def test_run() -> None:
    subprocess.run("rm -rf /Users/alexanderpatrie/sms/vecoli_repo/out/multiseed_multigen_analysis && uv run --env-file .env runscripts/analysis.py --config configs/data_transformation.json".split(" "))


@pytest.mark.asyncio
async def test_export_parquet() -> None:
    n_rows = int(1e4)
    tmp = tempfile.TemporaryDirectory()
    pq_path = Path(tmp.name) / "66_000.parquet"
    lf: pl.LazyFrame | None = lazyframe_fixture(n_rows).sink_parquet(pq_path)
    print(lf.collect_schema() if lf else "Lf not yet populated!")
    tmp.cleanup()
