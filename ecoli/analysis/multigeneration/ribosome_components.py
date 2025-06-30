"""
Record the 30S and 50S component count vs time
"""

import altair as alt
import os
from typing import Any, Dict

from duckdb import DuckDBPyConnection
import pickle
import polars as pl

from ecoli.library.parquet_emitter import (
    field_metadata,
    open_arbitrary_sim_data,
    named_idx,
    read_stacked_columns,
)

# ----------------------------------------- #


def plot(
    params: Dict[str, Any],
    conn: DuckDBPyConnection,
    history_sql: str,
    config_sql: str,
    success_sql: str,
    sim_data_dict: Dict[str, Dict[int, str]],
    validation_data_paths: list[str],
    outdir: str,
    variant_metadata: Dict[str, Dict[int, Any]],
    variant_names: Dict[str, str],
):
    # Load simulation data
    with open_arbitrary_sim_data(sim_data_dict) as f:
        sim_data = pickle.load(f)

    # Extract molecule IDs for ribosomal subunits
    s30_protein_ids = sim_data.molecule_groups.s30_proteins
    s30_16s_rRNA_ids = sim_data.molecule_groups.s30_16s_rRNA
    s30_full_complex_id = sim_data.molecule_ids.s30_full_complex
    s50_protein_ids = sim_data.molecule_groups.s50_proteins
    s50_23s_rRNA_ids = sim_data.molecule_groups.s50_23s_rRNA
    s50_5s_rRNA_ids = sim_data.molecule_groups.s50_5s_rRNA
    s50_full_complex_id = sim_data.molecule_ids.s50_full_complex

    # Retrieve stoichiometry for each protein subunit
    complexation = sim_data.process.complexation
    s30_info = complexation.get_monomers(s30_full_complex_id)
    s50_info = complexation.get_monomers(s50_full_complex_id)
    s30_stoich = dict(zip(s30_info["subunitIds"], s30_info["subunitStoich"]))
    s50_stoich = dict(zip(s50_info["subunitIds"], s50_info["subunitStoich"]))

    # Map bulk IDs to SQL column indices
    bulk_ids = field_metadata(conn, config_sql, "bulk")
    bulk_index = {mid: idx for idx, mid in enumerate(bulk_ids)}

    # Determine column indexes in SQL for rRNAs and complexes
    s30_16s_idx = [bulk_index[i] for i in s30_16s_rRNA_ids if i in bulk_index]
    s50_23s_idx = [bulk_index[i] for i in s50_23s_rRNA_ids if i in bulk_index]
    s50_5s_idx = [bulk_index[i] for i in s50_5s_rRNA_ids if i in bulk_index]
    s30_complex_idx = bulk_index[s30_full_complex_id]
    s50_complex_idx = bulk_index[s50_full_complex_id]

    # Map monomer counts IDs to SQL column indices
    mono_ids = field_metadata(conn, config_sql, "listeners__monomer_counts")
    mono_index = {mid: idx for idx, mid in enumerate(mono_ids)}
    s30_protein_idx = [mono_index[i] for i in s30_protein_ids if i in mono_index]
    s50_protein_idx = [mono_index[i] for i in s50_protein_ids if i in mono_index]

    # Build named_idx spec for reading
    bulk_cols = [
        named_idx("bulk", s30_16s_rRNA_ids, [s30_16s_idx]),
        named_idx("bulk", s50_23s_rRNA_ids, [s50_23s_idx]),
        named_idx("bulk", s50_5s_rRNA_ids, [s50_5s_idx]),
        named_idx("bulk", [s30_full_complex_id], [[s30_complex_idx]]),
        named_idx("bulk", [s50_full_complex_id], [[s50_complex_idx]]),
    ]
    protein_cols = [
        named_idx("listeners__monomer_counts", [pid], [[idx]])
        for pid, idx in zip(
            s30_protein_ids + s50_protein_ids, s30_protein_idx + s50_protein_idx
        )
    ]
    additional = ["listeners__unique_molecule_counts__active_ribosome", "time"]
    cols = bulk_cols + protein_cols + additional

    # Read time-series data
    data = read_stacked_columns(history_sql, cols, conn=conn)
    df = pl.DataFrame(data).with_columns(Time_min=pl.col("time") / 60)

    # Sum rRNA counts horizontally
    s30_16s = pl.sum_horizontal([pl.col(i) for i in s30_16s_rRNA_ids])
    s50_23s = pl.sum_horizontal([pl.col(i) for i in s50_23s_rRNA_ids])
    s50_5s = pl.sum_horizontal([pl.col(i) for i in s50_5s_rRNA_ids])

    # Extract complex and active ribosome counts
    s30_complex = pl.col(s30_full_complex_id)
    s50_complex = pl.col(s50_full_complex_id)
    active_ribo = pl.col("listeners__unique_molecule_counts__active_ribosome")

    # Adjust protein counts by stoichiometry
    for pid in s30_protein_ids:
        df = df.with_columns(**{f"adj_s30_{pid}": pl.col(pid) / s30_stoich[pid]})
    for pid in s50_protein_ids:
        df = df.with_columns(**{f"adj_s50_{pid}": pl.col(pid) / s50_stoich[pid]})

    # Determine limiting protein across subunits
    s30_lim = pl.min_horizontal([pl.col(f"adj_s30_{pid}") for pid in s30_protein_ids])
    s50_lim = pl.min_horizontal([pl.col(f"adj_s50_{pid}") for pid in s50_protein_ids])

    # Calculate total rRNA including complexes and active ribosomes
    df = df.with_columns(
        s30_16s_total=s30_16s + s30_complex + active_ribo,
        s50_23s_total=s50_23s + s50_complex + active_ribo,
        s50_5s_total=s50_5s + s50_complex + active_ribo,
        s30_limiting=s30_lim,
        s50_limiting=s50_lim,
        s30_total=s30_complex + active_ribo,
        s50_total=s50_complex + active_ribo,
    )

    # ----------------------------------------- #

    plot_cols_30 = ["s30_limiting", "s30_16s_total", "s30_total"]
    plot_cols_50 = ["s50_limiting", "s50_23s_total", "s50_5s_total", "s50_total"]

    melt_30 = df.select(["Time_min"] + plot_cols_30).melt(
        id_vars="Time_min", variable_name="component", value_name="count"
    )
    melt_50 = df.select(["Time_min"] + plot_cols_50).melt(
        id_vars="Time_min", variable_name="component", value_name="count"
    )

    chart_30 = (
        alt.Chart(melt_30)
        .mark_line()
        .encode(
            x="Time_min",
            y="count",
            color=alt.Color("component", title="30S Components"),
        )
        .properties(title="30S Component Counts", width=600)
    )

    chart_50 = (
        alt.Chart(melt_50)
        .mark_line()
        .encode(
            x="Time_min",
            y="count",
            color=alt.Color("component", title="50S Components"),
        )
        .properties(title="50S Component Counts", width=600)
    )

    combined = (
        alt.vconcat(chart_30, chart_50)
        .resolve_scale(color="independent")
        .resolve_legend(color="independent")
    )
    combined.save(os.path.join(outdir, "ribosome_components.html"))
