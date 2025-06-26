"""
Anaylsis Script for saving all protein counts impacted by new genes. Note that
this file only works for saving data for simulations that contain at least two
variants (one control and one experimental variant).
"""

import pickle
import os
import csv

# noinspection PyUnresolvedReferences
from duckdb import DuckDBPyConnection
import numpy as np
import polars as pl
from typing import Any, cast

from ecoli.library.parquet_emitter import (
    read_stacked_columns,
    ndlist_to_ndarray,
    open_arbitrary_sim_data,
)
from reconstruction.ecoli.fit_sim_data_1 import SimulationDataEcoli

IGNORE_FIRST_N_GENS = 1
"""
Indicate which generation the data should start being collected from (sometimes 
this number should be greater than 0 because the first few generations may not 
be representative of the true dynamics occuring in the cell).
"""


def save_file(out_dir, filename, columns, values):
    output_file = os.path.join(out_dir, filename)
    print(f"Saving data to {output_file}")
    with open(output_file, "w") as f:
        writer = csv.writer(f)
        # Header for columns
        writer.writerow(columns)
        # Data rows
        for i in range(values.shape[0]):
            writer.writerow(values[i, :])


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
    """
    Calculates average monomer counts per variant and saves them as separate
    files under ``{outdir}/saved_data/unfiltered_data``. Filters data to only
    include monomers whose counts are greater than ``params["filter_num"]``
    (default 0) and saves under ``{outdir}/saved_data/filtered_data``.
    """
    # Create saving paths
    save_dir = os.path.join(outdir, "saved_data")
    unfiltered_dir = os.path.join(save_dir, "unfiltered_data")
    filtered_dir = os.path.join(save_dir, "filtered_data")
    os.makedirs(unfiltered_dir, exist_ok=True)
    os.makedirs(filtered_dir, exist_ok=True)

    with open_arbitrary_sim_data(sim_data_dict) as f:
        sim_data: "SimulationDataEcoli" = pickle.load(f)
    mRNA_sim_data = sim_data.process.transcription.cistron_data.struct_array
    monomer_sim_data = sim_data.process.translation.monomer_data.struct_array
    new_gene_mRNA_ids = mRNA_sim_data[mRNA_sim_data["is_new_gene"]]["id"].tolist()
    mRNA_monomer_id_dict = dict(
        zip(monomer_sim_data["cistron_id"], monomer_sim_data["id"])
    )
    new_gene_monomer_ids = [
        cast(str, mRNA_monomer_id_dict.get(mRNA_id)) for mRNA_id in new_gene_mRNA_ids
    ]
    all_monomer_ids = monomer_sim_data["id"]
    original_monomer_ids = all_monomer_ids[
        ~np.isin(all_monomer_ids, new_gene_monomer_ids)
    ]
    monomer_idx_dict = {monomer: i for i, monomer in enumerate(all_monomer_ids)}
    original_monomer_idx = [
        cast(int, monomer_idx_dict.get(monomer_id))
        for monomer_id in original_monomer_ids
    ]

    subquery = read_stacked_columns(
        history_sql, ["listeners__monomer_counts"], order_results=False
    )
    avg_monomer_per_variant = conn.sql(f"""
        WITH unnested_counts AS (
            SELECT unnest(listeners__monomer_counts) AS monomer_counts,
                generate_subscripts(listeners__monomer_counts, 1)
                    AS monomer_idx, variant
            FROM ({subquery})
        ),
        -- Get average counts per monomer per variant
        average_counts AS (
            SELECT avg(monomer_counts) AS avg_counts, variant, monomer_idx
            FROM unnested_counts
            GROUP BY variant, monomer_idx
        )
        -- Organize average counts into lists, one row per variant
        SELECT list(avg_counts ORDER BY monomer_idx)
            AS avg_monomer_counts, variant
        FROM average_counts
        GROUP BY variant
        ORDER BY variant
        """).pl()

    # Extract average counts that are greater than some threshold (default: 0)
    filter_num = params.get("filter_num", 0)
    control_variant = avg_monomer_per_variant["variant"][0]
    # For each non-baseline variant, save two CSV files, each with three columns:
    # monomer IDs, baseline average monomer count, variant average monomer count.
    # First CSV file includes all monomers. Second removes new genes and filters
    # out monomers whose average count < params["filter_num"] in either variant.
    for exp_variant in avg_monomer_per_variant["variant"][1:]:
        file_suffix = f"var_{exp_variant}_startGen_{IGNORE_FIRST_N_GENS}.csv"
        variant_pair = avg_monomer_per_variant.filter(
            pl.col("variant").is_in([control_variant, exp_variant])
        ).sort("variant")
        avg_monomer_counts = ndlist_to_ndarray(variant_pair["avg_monomer_counts"])
        # Save unfiltered data
        col_labels = ["all_monomer_ids", "var_0_avg_PCs", f"var_{exp_variant}_avg_PCs"]
        values = np.concatenate(
            (all_monomer_ids[:, np.newaxis], avg_monomer_counts.T), axis=1
        )
        save_file(
            unfiltered_dir, f"wcm_full_monomers_{file_suffix}", col_labels, values
        )
        # Do not include new genes in filtered counts
        avg_monomer_counts = avg_monomer_counts[:, original_monomer_idx]
        var0_filter_PCs_idxs = np.nonzero(avg_monomer_counts[0] > filter_num)
        var1_filter_PCs_idxs = np.nonzero(avg_monomer_counts[1] > filter_num)
        shared_filtered_PC_idxs = np.intersect1d(
            var0_filter_PCs_idxs, var1_filter_PCs_idxs
        )
        avg_monomer_counts = avg_monomer_counts[:, shared_filtered_PC_idxs]
        filtered_ids = original_monomer_ids[shared_filtered_PC_idxs]
        # Save filtered data
        col_labels = [
            "filtered_monomer_ids",
            "var_0_avg_PCs",
            f"var_{exp_variant}_avg_PCs",
        ]
        values = np.concatenate(
            (filtered_ids[:, np.newaxis], avg_monomer_counts.T), axis=1
        )
        save_file(
            filtered_dir, f"wcm_filter_monomers_{file_suffix}", col_labels, values
        )
