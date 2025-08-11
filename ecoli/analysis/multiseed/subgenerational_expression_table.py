"""
Generates a table of genes that are subgenerationally expressed, with their
expression frequencies and average/maximum mRNA/protein counts.
"""

import pickle
import os
from typing import Any

# noinspection PyUnresolvedReferences
from duckdb import DuckDBPyConnection
import polars as pl

from ecoli.library.parquet_emitter import (
    field_metadata,
    ndidx_to_duckdb_expr,
    num_cells,
    open_arbitrary_sim_data,
    read_stacked_columns,
    skip_n_gens,
)


IGNORE_FIRST_N_GENS = 8


def plot(
    params: dict[str, Any],
    conn: DuckDBPyConnection,
    history_sql: str,
    config_sql: str,
    success_sql: str,
    sim_data_dict: dict[str, dict[int, str]],
    validation_data_paths: list[str],
    outdir: str,
    variant_metadata: dict[str, dict[int, Any]],
    variant_names: dict[str, str],
):
    with open_arbitrary_sim_data(sim_data_dict) as f:
        sim_data = pickle.load(f)

    ignore_first_n_gens = params.get("ignore_first_n_gens", IGNORE_FIRST_N_GENS)

    # Ignore first N generations
    history_sql = skip_n_gens(history_sql, ignore_first_n_gens)
    config_sql = skip_n_gens(config_sql, ignore_first_n_gens)

    if num_cells(conn, config_sql) == 0:
        print("Skipping analysis - not enough generations run.")
        return

    # Get list of cistron IDs from sim_data
    cistron_data = sim_data.process.transcription.cistron_data
    cistron_ids = cistron_data["id"]

    # Filter list for cistron IDs with associated protein ids
    cistron_id_to_protein_id = {
        protein["cistron_id"]: protein["id"]
        for protein in sim_data.process.translation.monomer_data
    }
    mRNA_cistron_ids = [
        cistron_id
        for cistron_id in cistron_ids
        if cistron_id in cistron_id_to_protein_id
    ]

    # Get IDs of associated monomers and genes
    monomer_ids = [
        cistron_id_to_protein_id[cistron_id] for cistron_id in mRNA_cistron_ids
    ]
    cistron_id_to_gene_id = {
        cistron["id"]: cistron["gene_id"] for cistron in cistron_data
    }
    gene_ids = [cistron_id_to_gene_id[cistron_id] for cistron_id in mRNA_cistron_ids]

    # Get subcolumn for mRNA cistron IDs in RNA counts table
    mRNA_cistron_ids_rna_counts_table = field_metadata(
        conn, config_sql, "listeners__rna_counts__mRNA_cistron_counts"
    )

    # Get indexes of mRNA cistrons in this subcolums (DuckDB lists are 1-indexed)
    mRNA_cistron_id_to_index = {
        cistron_id: i + 1
        for (i, cistron_id) in enumerate(mRNA_cistron_ids_rna_counts_table)
    }
    mRNA_cistron_indexes = [
        mRNA_cistron_id_to_index[cistron_id] for cistron_id in mRNA_cistron_ids
    ]

    # Get subcolumn for monomer IDs in monomer counts table
    monomer_ids_monomer_counts_table = field_metadata(
        conn, config_sql, "listeners__monomer_counts"
    )

    # Get indexes of monomers in this subcolumn (DuckDB lists are 1-indexed)
    monomer_id_to_index = {
        monomer_id: i + 1
        for (i, monomer_id) in enumerate(monomer_ids_monomer_counts_table)
    }
    monomer_indexes = [monomer_id_to_index[monomer_id] for monomer_id in monomer_ids]

    monomer_expr = ndidx_to_duckdb_expr("listeners__monomer_counts", [monomer_indexes])
    cistron_expr = ndidx_to_duckdb_expr(
        "listeners__rna_counts__mRNA_cistron_counts", [mRNA_cistron_indexes]
    )
    subquery = read_stacked_columns(
        history_sql,
        [monomer_expr, cistron_expr],
        order_results=False,
    )
    out_df = conn.sql(
        f"""
        -- Unnest monomer and mRNA count columns, labelling with
        -- index so we can later calculate per-cistron aggregates
        WITH unnested_counts AS (
            SELECT lineage_seed, generation, agent_id,
                unnest(listeners__monomer_counts) AS monomer_counts,
                unnest(listeners__rna_counts__mRNA_cistron_counts) AS mrna_counts,
                generate_subscripts(listeners__monomer_counts, 1) AS cistron_idx
            FROM ({subquery})
        ),
        -- Group by cell and cistron to get existence of each mRNA per cell
        cell_aggregate AS (
            SELECT
                SUM(mrna_counts) > 0 AS exists,
                MAX(monomer_counts) AS max_monomer_counts,
                MAX(mrna_counts) AS max_mRNA_counts,
                cistron_idx
            FROM unnested_counts
            GROUP BY lineage_seed, generation, agent_id, cistron_idx
        ),
        full_aggregate AS (
            SELECT
                -- Calculate probability that mRNA exists per cell cycle
                AVG(exists::INTEGER) AS p_expressed,
                -- Get maximum mRNA and monomer counts across all cells and times
                MAX(max_monomer_counts) AS max_monomer_counts,
                MAX(max_mRNA_counts) AS max_mRNA_counts,
                cistron_idx
            FROM cell_aggregate
            GROUP BY cistron_idx
        )
        SELECT * FROM full_aggregate
        -- Filter to only include sub-generational genes
        WHERE p_expressed > 0 AND p_expressed < 1
        """
    ).pl()

    # Add gene, cistron, and protein names (DuckDB lists are 1-indexed so
    # must subtract one before using to index Numpy arrays)
    out_df = out_df.with_columns(
        gene_name=pl.Series(gene_ids)[out_df["cistron_idx"] - 1],
        cistron_name=pl.Series(mRNA_cistron_ids)[out_df["cistron_idx"] - 1],
        protein_name=pl.Series([i[:-3] for i in monomer_ids])[
            out_df["cistron_idx"] - 1
        ],
    )
    out_df.write_csv(os.path.join(outdir, "subgen.tsv"), separator="\t")
