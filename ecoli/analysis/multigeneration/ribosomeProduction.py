import altair as alt
import os
from typing import Any
import pickle
import polars as pl
import numpy as np
from duckdb import DuckDBPyConnection

from ecoli.library.parquet_emitter import (
    field_metadata,
    open_arbitrary_sim_data,
)


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
    # Load sim_data
    with open_arbitrary_sim_data(sim_data_dict) as f:
        sim_data = pickle.load(f)

    # Get expected doubling time
    expected_doubling_time = sim_data.doubling_time

    # Get ribosomal RNA IDs
    ids_16s = []
    ids_16s.extend(sim_data.molecule_groups.s30_16s_rRNA)
    ids_16s.append(sim_data.molecule_ids.s30_full_complex)

    ids_23s = []
    ids_23s.extend(sim_data.molecule_groups.s50_23s_rRNA)
    ids_23s.append(sim_data.molecule_ids.s50_full_complex)

    ids_5s = []
    ids_5s.extend(sim_data.molecule_groups.s50_5s_rRNA)
    ids_5s.append(sim_data.molecule_ids.s50_full_complex)

    # Get indices for ribosomal RNAs
    bulk_molecule_ids = field_metadata(conn, config_sql, "listeners__bulk_molecules")
    ids_16s_indexes = [
        bulk_molecule_ids.index(mol_id)
        for mol_id in ids_16s
        if mol_id in bulk_molecule_ids
    ]
    ids_23s_indexes = [
        bulk_molecule_ids.index(mol_id)
        for mol_id in ids_23s
        if mol_id in bulk_molecule_ids
    ]
    ids_5s_indexes = [
        bulk_molecule_ids.index(mol_id)
        for mol_id in ids_5s
        if mol_id in bulk_molecule_ids
    ]

    # Get unique molecule index for active ribosome
    unique_molecule_ids = field_metadata(
        conn, config_sql, "listeners__unique_molecules"
    )
    ribosome_index = (
        unique_molecule_ids.index("active_ribosome")
        if "active_ribosome" in unique_molecule_ids
        else None
    )

    # Get cistron indices for rRNA
    cistron_ids = [
        cistron["id"] for cistron in sim_data.process.transcription.cistron_data
    ]

    idx_16s = []
    for id16s in sim_data.molecule_groups.s30_16s_rRNA:
        cistron_id = id16s[:-3]  # Remove _rna suffix
        if cistron_id in cistron_ids:
            idx_16s.append(cistron_ids.index(cistron_id))

    idx_23s = []
    for id23s in sim_data.molecule_groups.s50_23s_rRNA:
        cistron_id = id23s[:-3]  # Remove _rna suffix
        if cistron_id in cistron_ids:
            idx_23s.append(cistron_ids.index(cistron_id))

    idx_5s = []
    for id5s in sim_data.molecule_groups.s50_5s_rRNA:
        cistron_id = id5s[:-3]  # Remove _rna suffix
        if cistron_id in cistron_ids:
            idx_5s.append(cistron_ids.index(cistron_id))

    # Calculate expected initiation probabilities
    condition = sim_data.condition
    cistron_synth_prob = sim_data.process.transcription.cistron_tu_mapping_matrix.dot(
        sim_data.process.transcription.rna_synth_prob[condition]
    )

    rrn16s_fit_init_prob = cistron_synth_prob[idx_16s].sum() if idx_16s else 0
    rrn23s_fit_init_prob = cistron_synth_prob[idx_23s].sum() if idx_23s else 0
    rrn5s_fit_init_prob = cistron_synth_prob[idx_5s].sum() if idx_5s else 0

    # Define columns to read
    columns_to_read = [
        "time",
        "listeners__mass__instantaneous_growth_rate",
        "listeners__main__timeStepSec",
        "listeners__ribosome_data__rRNA16S_initiated",
        "listeners__ribosome_data__rRNA23S_initiated",
        "listeners__ribosome_data__rRNA5S_initiated",
        "listeners__ribosome_data__rRNA16S_init_prob",
        "listeners__ribosome_data__rRNA23S_init_prob",
        "listeners__ribosome_data__rRNA5S_init_prob",
        "listeners__ribosome_data__total_rna_init",
        "listeners__ribosome_data__effectiveElongationRate",
    ]

    # Add bulk molecule columns
    if ids_16s_indexes:
        for idx in ids_16s_indexes:
            columns_to_read.append(f"listeners__bulk_molecules__{idx}")
    if ids_23s_indexes:
        for idx in ids_23s_indexes:
            columns_to_read.append(f"listeners__bulk_molecules__{idx}")
    if ids_5s_indexes:
        for idx in ids_5s_indexes:
            columns_to_read.append(f"listeners__bulk_molecules__{idx}")

    # Add unique molecule column
    if ribosome_index is not None:
        columns_to_read.append(f"listeners__unique_molecules__{ribosome_index}")

    # Read data
    data_df = conn.execute(f"""
        SELECT {", ".join(columns_to_read)}
        FROM ({history_sql})
        ORDER BY variant_idx, generation, agent_id, time
    """).pl()

    # Group by first cell of each generation (assuming agent_id=0 is first cell)
    first_cell_data = data_df.filter(pl.col("agent_id") == 0)

    # Calculate derived metrics
    time_min = first_cell_data["time"] / 60

    # Growth rate and doubling time
    growth_rate = first_cell_data["listeners__mass__instantaneous_growth_rate"]
    doubling_time = np.log(2) / growth_rate

    # Calculate rRNA counts
    rrn16s_bulk = (
        sum(
            [
                first_cell_data[f"listeners__bulk_molecules__{idx}"]
                for idx in ids_16s_indexes
            ]
        )
        if ids_16s_indexes
        else pl.lit(0)
    )
    rrn23s_bulk = (
        sum(
            [
                first_cell_data[f"listeners__bulk_molecules__{idx}"]
                for idx in ids_23s_indexes
            ]
        )
        if ids_23s_indexes
        else pl.lit(0)
    )
    rrn5s_bulk = (
        sum(
            [
                first_cell_data[f"listeners__bulk_molecules__{idx}"]
                for idx in ids_5s_indexes
            ]
        )
        if ids_5s_indexes
        else pl.lit(0)
    )

    if ribosome_index is not None:
        ribosome_count = first_cell_data[
            f"listeners__unique_molecules__{ribosome_index}"
        ]
        rrn16s_count = rrn16s_bulk + ribosome_count
        rrn23s_count = rrn23s_bulk + ribosome_count
        rrn5s_count = rrn5s_bulk + ribosome_count
    else:
        rrn16s_count = rrn16s_bulk
        rrn23s_count = rrn23s_bulk
        rrn5s_count = rrn5s_bulk

    # Calculate rRNA doubling times
    time_step = first_cell_data["listeners__main__timeStepSec"]

    rrn16s_produced = first_cell_data["listeners__ribosome_data__rRNA16S_initiated"]
    rrn23s_produced = first_cell_data["listeners__ribosome_data__rRNA23S_initiated"]
    rrn5s_produced = first_cell_data["listeners__ribosome_data__rRNA5S_initiated"]

    # Avoid division by zero
    rrn16s_doubling_time = (
        pl.when(rrn16s_produced > 0)
        .then(np.log(2) / ((1 / time_step) * (rrn16s_produced / rrn16s_count)))
        .otherwise(None)
        / 60
    )  # Convert to minutes

    rrn23s_doubling_time = (
        pl.when(rrn23s_produced > 0)
        .then(np.log(2) / ((1 / time_step) * (rrn23s_produced / rrn23s_count)))
        .otherwise(None)
        / 60
    )

    rrn5s_doubling_time = (
        pl.when(rrn5s_produced > 0)
        .then(np.log(2) / ((1 / time_step) * (rrn5s_produced / rrn5s_count)))
        .otherwise(None)
        / 60
    )

    # Prepare plotting dataframe
    plot_data = first_cell_data.with_columns(
        [
            time_min.alias("Time (min)"),
            doubling_time.alias("Doubling Time (min)"),
            rrn16s_doubling_time.alias("16S Doubling Time (min)"),
            rrn23s_doubling_time.alias("23S Doubling Time (min)"),
            rrn5s_doubling_time.alias("5S Doubling Time (min)"),
            (
                first_cell_data["listeners__ribosome_data__rRNA16S_init_prob"]
                / first_cell_data["listeners__ribosome_data__total_rna_init"]
            ).alias("16S Init Prob"),
            (
                first_cell_data["listeners__ribosome_data__rRNA23S_init_prob"]
                / first_cell_data["listeners__ribosome_data__total_rna_init"]
            ).alias("23S Init Prob"),
            (
                first_cell_data["listeners__ribosome_data__rRNA5S_init_prob"]
                / first_cell_data["listeners__ribosome_data__total_rna_init"]
            ).alias("5S Init Prob"),
            first_cell_data["listeners__ribosome_data__effectiveElongationRate"].alias(
                "Elongation Rate (aa/s)"
            ),
            pl.lit(expected_doubling_time.as_number() / 60).alias(
                "Expected Doubling Time (min)"
            ),
            pl.lit(rrn16s_fit_init_prob).alias("Expected 16S Init Prob"),
            pl.lit(rrn23s_fit_init_prob).alias("Expected 23S Init Prob"),
            pl.lit(rrn5s_fit_init_prob).alias("Expected 5S Init Prob"),
        ]
    )

    # Create plots
    base = alt.Chart(plot_data).add_selection(alt.selection_interval(bind="scales"))

    # Doubling time plot
    doubling_plot = base.mark_line(color="blue").encode(
        x=alt.X("Time (min):Q"),
        y=alt.Y("Doubling Time (min):Q", title="Doubling Time (min)"),
    ) + base.mark_line(color="red", strokeDash=[5, 5]).encode(
        x=alt.X("Time (min):Q"), y=alt.Y("Expected Doubling Time (min):Q")
    )

    # 16S doubling time plot
    rrn16s_plot = base.mark_line(color="blue").encode(
        x=alt.X("Time (min):Q"),
        y=alt.Y("16S Doubling Time (min):Q", title="16S Doubling Time (min)"),
    ) + base.mark_line(color="red", strokeDash=[5, 5]).encode(
        x=alt.X("Time (min):Q"), y=alt.Y("Expected Doubling Time (min):Q")
    )

    # 23S doubling time plot
    rrn23s_plot = base.mark_line(color="blue").encode(
        x=alt.X("Time (min):Q"),
        y=alt.Y("23S Doubling Time (min):Q", title="23S Doubling Time (min)"),
    ) + base.mark_line(color="red", strokeDash=[5, 5]).encode(
        x=alt.X("Time (min):Q"), y=alt.Y("Expected Doubling Time (min):Q")
    )

    # 5S doubling time plot
    rrn5s_plot = base.mark_line(color="blue").encode(
        x=alt.X("Time (min):Q"),
        y=alt.Y("5S Doubling Time (min):Q", title="5S Doubling Time (min)"),
    ) + base.mark_line(color="red", strokeDash=[5, 5]).encode(
        x=alt.X("Time (min):Q"), y=alt.Y("Expected Doubling Time (min):Q")
    )

    # Initiation probability plots
    init_16s_plot = base.mark_line(color="blue").encode(
        x=alt.X("Time (min):Q"), y=alt.Y("16S Init Prob:Q", title="16S Init Prob")
    ) + base.mark_line(color="red", strokeDash=[5, 5]).encode(
        x=alt.X("Time (min):Q"), y=alt.Y("Expected 16S Init Prob:Q")
    )

    init_23s_plot = base.mark_line(color="blue").encode(
        x=alt.X("Time (min):Q"), y=alt.Y("23S Init Prob:Q", title="23S Init Prob")
    ) + base.mark_line(color="red", strokeDash=[5, 5]).encode(
        x=alt.X("Time (min):Q"), y=alt.Y("Expected 23S Init Prob:Q")
    )

    init_5s_plot = base.mark_line(color="blue").encode(
        x=alt.X("Time (min):Q"), y=alt.Y("5S Init Prob:Q", title="5S Init Prob")
    ) + base.mark_line(color="red", strokeDash=[5, 5]).encode(
        x=alt.X("Time (min):Q"), y=alt.Y("Expected 5S Init Prob:Q")
    )

    # Elongation rate plot
    elongation_plot = base.mark_line(color="blue").encode(
        x=alt.X("Time (min):Q"),
        y=alt.Y(
            "Elongation Rate (aa/s):Q", title="Average Ribosome Elongation Rate (aa/s)"
        ),
    )

    # Combine all plots vertically
    combined_plot = alt.vconcat(
        doubling_plot.properties(title="Cell Doubling Time", width=600, height=100),
        rrn16s_plot.properties(title="16S rRNA Doubling Time", width=600, height=100),
        rrn23s_plot.properties(title="23S rRNA Doubling Time", width=600, height=100),
        rrn5s_plot.properties(title="5S rRNA Doubling Time", width=600, height=100),
        init_16s_plot.properties(
            title="16S rRNA Initiation Probability", width=600, height=100
        ),
        init_23s_plot.properties(
            title="23S rRNA Initiation Probability", width=600, height=100
        ),
        init_5s_plot.properties(
            title="5S rRNA Initiation Probability", width=600, height=100
        ),
        elongation_plot.properties(
            title="Ribosome Elongation Rate", width=600, height=100
        ),
        resolve=alt.Resolve(scale=alt.ScaleResolve(y="independent")),
    )

    # Save the plot
    combined_plot.save(os.path.join(outdir, "ribosome_production.html"))
