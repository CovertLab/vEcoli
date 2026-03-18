# TODO: Implement exchange fluxes analysis
# Goal: Access and aggregate all relevant exchange fluxes, generating a summary table
# Minimally, this must capture glucose uptake and violacein production.
# Ideally a general solution will return all exchange fluxes (or all nonzero).
# Aggregation TBD -- either a single time average or time intervals.
# IDEA - consider multiple output tables: a verbose one with all fluxes, and a tightly focused one with vio KPIs (e.g. vio, glucose fluxes, vio yield)

import os
from typing import Any

from duckdb import DuckDBPyConnection
import polars as pl
import fnmatch

from ecoli.library.parquet_emitter import read_stacked_columns


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
):
    filter_clause = ""
    if params.get("generation_lower_bound", None):
        filter_clause += f"WHERE generation >= {params['generation_lower_bound']}"

    if params.get("time_lower_bound", None):
        filter_clause += (
            " AND " if filter_clause else "WHERE "
        ) + f"time >= {params['time_lower_bound']}"

    all_col_names = (
        conn.sql(f"SELECT column_name FROM (DESCRIBE ({history_sql}))")
        .pl()["column_name"]
        .to_list()
    )
    pattern = "listeners__fba_results__external_exchange_fluxes__*"
    flux_col_names = fnmatch.filter(all_col_names, pattern)
    metabolite_dict = {col: col.split("__")[-1].split("[")[0] for col in flux_col_names}
    columns = [
        "listeners__mass__instantaneous_growth_rate * 3600 AS growth_rate_h",
    ]
    avg_fluxes = []
    for k, v in metabolite_dict.items():
        columns.append(f'"{k}" AS "{v}"')
        avg_fluxes.append(f'AVG("{v}") AS "{v}"')
    flux_subquery = read_stacked_columns(history_sql, columns, order_results=False)
    id_cols = [
        "experiment_id",
        "variant",
        "lineage_seed",
        "generation",
        "agent_id",
    ]

    flux_data = conn.sql(
        f"""
        SELECT {", ".join(avg_fluxes)},
            avg(growth_rate_h) AS growth_rate_h,
            concat('Cell: ', lineage_seed, '_', agent_id) AS cell_id
        FROM ({flux_subquery})
        {filter_clause}
        GROUP BY {", ".join(id_cols)}
        """
    ).pl()

    # Transpose: cell_ids become columns, metabolites + growth_rate_h become rows
    cell_ids = flux_data["cell_id"].to_list()
    wide_table = flux_data.drop("cell_id").transpose(
        include_header=True, header_name="EcoCyc Compound ID", column_names=cell_ids
    )

    # Calculate summary statistics
    value_cols = [col for col in wide_table.columns if col != "EcoCyc Compound ID"]
    wide_table = wide_table.with_columns(
        [
            pl.mean_horizontal(value_cols).alias("mean"),
            pl.concat_list(value_cols).list.std().alias("std"),
        ]
    )

    # Reorder columns: EcoCyc Compound ID, mean, std, then all cell columns
    wide_table = wide_table.select(["EcoCyc Compound ID", "mean", "std"] + value_cols)

    wide_table.write_csv(
        os.path.join(outdir, "exchange_fluxes.tsv"), separator="\t", include_header=True
    )
