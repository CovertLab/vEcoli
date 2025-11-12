"""
Data Relabeling/Aggregation Transformation (Eco/BioCyc)
"""


import dataclasses
import itertools
import math
import warnings
from pathlib import Path
from typing import Any, LiteralString

import duckdb
import pandas as pd
from duckdb import DuckDBPyConnection
import polars as pl
import numpy as np

from ecoli.library.parquet_emitter import read_stacked_columns, dataset_sql
from ecoli.library.sim_data import LoadSimData
from reconstruction.ecoli.simulation_data import SimulationDataEcoli
from wholecell.utils.unit_struct_array import UnitStructArray


REPO_ROOT = Path(__file__).parent.parent.parent.parent
PARTITION_GROUPS = {
    "multivariant": ["experiment_id"],
    "multiseed": ["experiment_id", "variant"],
    "multigeneration": ["experiment_id", "variant", "lineage_seed"],
    "multidaughter": ["experiment_id", "variant", "lineage_seed", "generation"],
    "single": [
        "experiment_id",
        "variant",
        "lineage_seed",
        "generation",
        "agent_id",
    ],
}


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


class DataTransformer(object):
    sim_data_path: str
    sim_data: SimulationDataEcoli
    data_labels: DatasetLabels

    def __init__(self, sim_data_path: Path):
        self.sim_data_path = str(sim_data_path)
        self.sim_data = LoadSimData(str(self.sim_data_path)).sim_data
        self.data_labels = self._get_data_labels()

    def _get_bulk_ids(self):
        bulk_ids = self.sim_data.internal_state.bulk_molecules.bulk_data["id"].tolist()
        return bulk_ids

    def _get_rxn_ids(self):
        rxn_ids = self.sim_data.process.metabolism.base_reaction_ids
        return rxn_ids

    def _get_common_names(self, bulk_names: list[str]):
        bulk_common_names = [self.sim_data.common_names.get_common_name(name) for name in self.data_labels.bulk_names_unique]
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

    # -- utils methods -- #

    @classmethod
    def _partitions_dict(cls, analysis_type, exp_select, variant_select, seed_select, gen_select, agent_select) -> dict[str, Any]:
        partitions_req = PARTITION_GROUPS[analysis_type]
        partitions_all = cls._read_partitions(exp_select, variant_select, seed_select, gen_select, agent_select)

        partitions_dict = {}
        for partition in partitions_req:
            partitions_dict[partition] = partitions_all[partition]
        partitions_dict["experiment_id"] = f"'{partitions_dict['experiment_id']}'"
        return partitions_dict

    @classmethod
    def _read_partitions(
            cls, exp_select: str, variant_select: int, seed_select: int, gen_select: int, agent_select: str
    ) -> dict:
        partitions_selected = {
            "experiment_id": exp_select,
            "variant": variant_select,
            "lineage_seed": seed_select,
            "generation": gen_select,
            "agent_id": agent_select,
        }
        return partitions_selected

    @classmethod
    def _get_db_filter(cls, partitions_dict) -> LiteralString:
        db_filter_list = []
        for key, value in partitions_dict.items():
            db_filter_list.append(str(key) + "=" + str(value))
        db_filter = " AND ".join(db_filter_list)

        return db_filter

    @classmethod
    def _get_filtered_query(cls, output_dir, experiment_id, db_filter):
        pq_columns = [
            "bulk",
            "listeners__fba_results__base_reaction_fluxes",
            "listeners__rna_counts__full_mRNA_cistron_counts",
        ]

        history_sql_base, _, _ = dataset_sql(output_dir, experiment_ids=[experiment_id])
        return f"SELECT {','.join(pq_columns)},time FROM ({history_sql_base}) WHERE {db_filter} ORDER BY time"

    @classmethod
    def _load_outputs(cls, sql: str) -> pd.DataFrame:
        outputs_df = duckdb.sql(sql).df()
        outputs_df = outputs_df.groupby("time", as_index=False).sum()

        return outputs_df

    @classmethod
    def _downsample_eager(cls, df_long: pd.DataFrame | pl.DataFrame) -> pd.DataFrame | pl.DataFrame:
        tp_all = np.unique(df_long["time"]).astype(int)
        ds_ratio = int(np.ceil(np.shape(df_long)[0] / 20000))
        tp_ds = list(itertools.islice(tp_all, 0, max(tp_all), ds_ratio))
        condition: list[bool] = np.isin(df_long["time"], tp_ds).tolist()
        df_ds = df_long.filter([condition]) if isinstance(df_long, pl.DataFrame) \
            else df_long[condition]
        # df_ds = df_long.filter([np.isin(df_long["time"], tp_ds)]) if isinstance(df_long, pl.DataFrame) \
        #     else df_long[np.isin(df_long["time"], tp_ds)]
        return df_ds

    @classmethod
    def _downsample_dataframe(cls, df_long: pl.LazyFrame):
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

    @classmethod
    def _downsample(cls, df_long: pd.DataFrame) -> pd.DataFrame:
        tp_all = np.unique(df_long["time"]).astype(int)
        ds_ratio = int(np.ceil(np.shape(df_long)[0] / 20000))
        tp_ds = list(itertools.islice(tp_all, 0, max(tp_all), ds_ratio))
        df_ds = df_long[np.isin(df_long["time"], tp_ds)]
        return df_ds

    @classmethod
    def _get_config_value(cls, config_df: pl.DataFrame, col: str):
        return config_df[[col]].to_numpy().flatten()[0]

    # -- public methods -- #

    def get_genes(
        self,
        experiment_id: str,
        simulation_outdir: str,
        observable_ids: list[str] | None = None,
        variant: int = 0,
        seed: int = 0,
        generation: int = 0,
        agent_id: str = "0",
        history_sql: str | None = None,
        lazy: bool = True
    ) -> pl.DataFrame | pl.LazyFrame:
        # (
        #     bulk_ids,
        #     bulk_ids_biocyc,
        #     bulk_names_unique,
        #     bulk_common_names,
        #     rxn_ids,
        #     cistron_data,
        #     mrna_cistron_ids,
        #     mrna_cistron_names,
        # ) = DataTransformer._get_ids(simdata_path, sim_data)

        history_sql_filtered = history_sql
        if history_sql_filtered is None:
            dbf_dict = DataTransformer._partitions_dict("single", experiment_id, variant, seed, generation, agent_id)
            db_filter = DataTransformer._get_db_filter(dbf_dict)
            history_sql_filtered = history_sql or DataTransformer._get_filtered_query(simulation_outdir, experiment_id, db_filter)

        output_loaded: pd.DataFrame = DataTransformer._load_outputs(history_sql_filtered)

        mrna_select = self.data_labels.mrna_cistron_names

        mrna_mtx = np.stack(output_loaded["listeners__rna_counts__full_mRNA_cistron_counts"])

        mrna_idxs = [self.data_labels.mrna_cistron_names.index(gene_id) for gene_id in mrna_select]

        mrna_trajs = [mrna_mtx[:, mrna_idx] for mrna_idx in mrna_idxs]

        mrna_plot_dict = {key: val for (key, val) in zip(mrna_select, mrna_trajs)}

        mrna_plot_dict["time"] = output_loaded["time"]

        # mrna_plot_df = pd.DataFrame(mrna_plot_dict)
        # mrna_df_long = mrna_plot_df.melt(
        #     id_vars=["time"],  # Columns to keep as identifier variables
        #     var_name="gene names",  # Name for the new column containing original column headers
        #     value_name="counts",  # Name for the new column containing original column values
        # )
        mrna_df_long = pl.LazyFrame(
            pd.DataFrame(mrna_plot_dict).melt(
                id_vars=["time"],  # Columns to keep as identifier variables
                var_name="gene names",  # Name for the new column containing original column headers
                value_name="counts",  # Name for the new column containing original column values
            )
        )

        # mrna_df = DataTransformer._downsample(mrna_df_long)
        mrna_df: pl.LazyFrame = DataTransformer._downsample_dataframe(mrna_df_long)

        # return mrna_df[mrna_df["gene names"].isin(observable_ids)] if observable_ids is not None else mrna_df
        genes_data: pl.LazyFrame = mrna_df.filter(
            pl.col("gene_names").is_in(observable_ids)
        )
        return genes_data if lazy else genes_data.collect()

    def export_parquet(self, genes_df: pl.LazyFrame, outdir: str, filename: str) -> pl.LazyFrame | None:
        lf = genes_df.sink_parquet(f"{outdir}/{filename}.parquet")
        print(f'Successfully exported {filename} to {outdir}!')
        return lf


# -- primary export -- #

def plot(
        params: dict[str, Any],
        conn: DuckDBPyConnection,
        history_sql: str,
        config_sql: str,
        success_sql: str,
        sim_data_paths: dict[str, dict[int, str]],
        validation_data_paths: list[str],
        outdir: str,
        variant_metadata: dict[str, dict[int, Any]],
        variant_names: dict[str, str],
) -> None:
    # get config params
    # config_df = SimulationConfigData(query=config_sql)   # duckdb.sql(config_sql).pl()
    # experiment_id, variant, seed, generation, agent_id, sim_outdir = list(map(
    #     lambda column: config_df.get(column),
    #     ["experiment_id", "variant", "lineage_seed", "generation", "agent_id", "emitter_arg__out_dir"]
    # ))
    # # log
    # print(f'# ======================================================================================= #\n>> Experiment_id: {experiment_id}, Variant: {variant}, Seed: {seed}, Generation: {generation}, AgentID: {agent_id}')
    # # set up transformer
    # simdata_path = Path(sim_data_paths[experiment_id][variant])
    # transformer = DataTransformer(sim_data_path=simdata_path)
    # # generate genes_df
    # bulk_df: pl.LazyFrame = transformer.get_genes(
    #     experiment_id=experiment_id,
    #     simulation_outdir=sim_outdir,
    #     observable_ids=params['observable_ids'],
    #     variant=variant,
    #     seed=seed,
    #     generation=generation,
    #     agent_id=agent_id,
    #     history_sql=history_sql,
    #     lazy=True
    # )
    # print(f'GENES DF:\n{bulk_df.collect_schema()}')
    # # # export lazyframe to parquet
    # filename = f"genes_{experiment_id}"
    # # export_parquet(genes_df, outdir, filename)
    pass

