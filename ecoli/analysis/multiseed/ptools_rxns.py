import os
from typing import Any

from duckdb import DuckDBPyConnection
import numpy as np
import pandas as pd

from ecoli.library.sim_data import LoadSimData


def build_query(
    columns, history_sql
):  # generates sql query for user specified parquet columns
    query_sql = f"""
        SELECT {",".join(columns)}, time FROM ({history_sql})
        ORDER BY time
    """

    return query_sql


def read_outputs(
    history_sql: str,
    conn: DuckDBPyConnection,
    columns=["bulk", "listeners__rna_counts__full_mRNA_counts"],
):
    # retrieves specifc columns from parquet outputs and returns a dataframe
    query_sql = build_query(columns, history_sql)

    outputs_df = conn.sql(query_sql).df()

    outputs_df = outputs_df.groupby("time", as_index=False).sum()

    return outputs_df


def consolidate_timepoints(state_mtx, n_tp, normalized=False):
    # generate consolidated relative time points
    checkpoints = np.linspace(0, np.shape(state_mtx)[0], n_tp, dtype=int)

    if normalized:
        denom = [
            len(state_mtx[checkpoints[i] : checkpoints[i + 1]])
            for i in range(len(checkpoints) - 1)
        ]

        block_sums = [
            state_mtx[checkpoints[i] : checkpoints[i + 1]].sum(axis=0) / denom[i]
            for i in range(len(checkpoints) - 1)
        ]

    else:
        block_sums = [
            state_mtx[checkpoints[i] : checkpoints[i + 1]].sum(axis=0)
            for i in range(len(checkpoints) - 1)
        ]

    block_sums = np.stack(block_sums, axis=0)
    block_sums_final = np.insert(block_sums, 0, state_mtx[0], axis=0)

    return block_sums_final


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
    exp_id = list(sim_data_paths.keys())[0]

    sim_data_path = list(sim_data_paths[exp_id].values())[0]

    sim_data = LoadSimData(sim_data_path).sim_data

    output_columns = ["bulk", "listeners__fba_results__base_reaction_fluxes"]

    output_df = read_outputs(history_sql, conn, output_columns)

    rxn_mtx = np.stack(output_df["listeners__fba_results__base_reaction_fluxes"].values)

    rxn_ids_base = sim_data.process.metabolism.base_reaction_ids

    n_tp = params["n_tp"]

    tp_columns = ["t" + str(i) for i in range(n_tp)]

    rxn_blocksum = consolidate_timepoints(rxn_mtx, n_tp, normalized=True)

    ptools_rxns = pd.DataFrame(
        data=np.abs(rxn_blocksum.transpose()), index=rxn_ids_base, columns=tp_columns
    )

    ptools_rxns.index.name = "$"

    ptools_rxns.to_csv(
        os.path.join(outdir, "ptools_rxns_multiseed.txt"),
        sep="\t",
        index=True,
        header=True,
        float_format="%.4f",
    )
