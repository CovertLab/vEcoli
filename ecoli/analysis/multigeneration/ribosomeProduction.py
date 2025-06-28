"""
Record several things:
1. normalised dry mass over time
2. cell, 5S RNA, 16S RNA, and 23S rRNA doubling time (be calculated use the `log(2)` formulation)
3. 5S RNA, 16S RNA, and 23S rRNA initiation probability
4. Ribosome elongation rate
"""

import altair as alt
import os
from typing import Any
import pickle
import polars as pl
import numpy as np
from duckdb import DuckDBPyConnection
import pandas as pd

from ecoli.library.parquet_emitter import (
    open_arbitrary_sim_data,
)

# ----------------------------------------- #


def make_get_bulk_counts(sim_data):
    """
    Create a function to extract counts of specified bulk molecules using sim_data indices.

    Args:
        sim_data: Simulation data object containing molecule IDs and related information.

    Returns:
        A function that takes a DataFrame and list of molecule IDs and returns their total counts.
    """
    # Get Molecular ID from bulk_molecules
    try:
        molecule_ids_list = sim_data.internal_state.bulk_molecules.bulk_data[
            "id"
        ].tolist()
    except AttributeError:
        raise ValueError("[ERROR] Check the structure of `sim_data`")

    mol_id_to_index = {mol_id: idx for idx, mol_id in enumerate(molecule_ids_list)}

    def get_bulk_counts(df, molecule_ids):
        """
        Extract total counts of specified molecule IDs from the 'bulk' column.

        Args:
            df: Polars DataFrame with a 'bulk' column containing Series of counts.
            molecule_ids: List of molecule IDs to sum (e.g., s30_16s_rRNA).

        Returns:
            Polars Series with total counts for each row.
        """
        indices = []
        for mol_id in molecule_ids:
            if mol_id in mol_id_to_index:
                indices.append(mol_id_to_index[mol_id])
            else:
                print(f"warning: molecular ID '{mol_id}' is missing")

        return (
            df["bulk"]
            .map_elements(
                lambda counts_series: (
                    sum(counts_series[i] for i in indices if i < len(counts_series))
                    if isinstance(counts_series, pl.Series)
                    else 0
                ),
                return_dtype=pl.Float64,
            )
            .fill_null(0)
        )

    return get_bulk_counts


def get_unique_counts(df, molecule_type):
    """Get counts of unique molecules (e.g., active ribosomes) from listeners."""
    col_name = f"listeners__unique_molecule_counts__{molecule_type}"
    if col_name in df.columns:
        return df[col_name].fill_null(0)
    return pl.Series(np.zeros(len(df), dtype=np.int64))


# Calculate the RNA doubling times
def calc_rna_doubling_time(produced_col, count_col, borderline):
    production_rate = pl.col(produced_col) / pl.col("time_step_sec")
    growth_rate = production_rate / pl.col(count_col)
    doubling_time_min = np.log(2) / growth_rate / 60.0

    # data sanitation
    valid_condition = (
        (pl.col(produced_col) >= 0)
        & (pl.col(count_col) > 0)
        & (growth_rate > 0)
        & doubling_time_min.is_finite()
        & (doubling_time_min > 0)
        & (doubling_time_min < 2 * borderline)
    )

    return pl.when(valid_condition).then(doubling_time_min).otherwise(pl.lit(None))


# ----------------------------------------- #


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
    """Visualize ribosome production metrics for E. coli simulation."""
    # Load sim_data
    with open_arbitrary_sim_data(sim_data_dict) as f:
        sim_data = pickle.load(f)

    # Get expected doubling time in minutes
    sim_doubling_time_min = sim_data.doubling_time.asNumber()

    required_columns = [
        "time",
        "variant",
        "generation",
        "agent_id",
        "experiment_id",
        "lineage_seed",
        "listeners__mass__instantaneous_growth_rate",
        "listeners__mass__dry_mass",
        "listeners__ribosome_data__rRNA16S_initiated",
        "listeners__ribosome_data__rRNA23S_initiated",
        "listeners__ribosome_data__rRNA5S_initiated",
        "listeners__ribosome_data__rRNA16S_init_prob",
        "listeners__ribosome_data__rRNA23S_init_prob",
        "listeners__ribosome_data__rRNA5S_init_prob",
        "listeners__ribosome_data__total_rna_init",
        "listeners__ribosome_data__effective_elongation_rate",
        "listeners__unique_molecule_counts__active_ribosome",
        "bulk",
    ]

    s30_16s_rRNA = list(sim_data.molecule_groups.s30_16s_rRNA) + [
        sim_data.molecule_ids.s30_full_complex
    ]
    s50_23s_rRNA = list(sim_data.molecule_groups.s50_23s_rRNA) + [
        sim_data.molecule_ids.s50_full_complex
    ]
    s50_5s_rRNA = list(sim_data.molecule_groups.s50_5s_rRNA) + [
        sim_data.molecule_ids.s50_full_complex
    ]

    # Check available columns
    available_columns = (
        conn.sql(f"DESCRIBE ({history_sql})").pl()["column_name"].to_list()
    )
    data_columns = [col for col in required_columns if col in available_columns]

    print(
        f"[INFO] Loading {len(data_columns)} columns for ribosome production analysis"
    )

    df = conn.sql(f"""
        SELECT {", ".join(data_columns)}
        FROM ({history_sql})
        WHERE agent_id = 0
        ORDER BY variant, generation, time
    """).pl()
    df = df.rename({"variant": "variant_id", "generation": "generation_index"})

    # Convert time from seconds to minutes
    df = df.with_columns((pl.col("time") / 60).alias("time_min"))

    # Calculate mass doubling time
    if "listeners__mass__instantaneous_growth_rate" in df.columns:
        df = df.with_columns(
            doubling_time_min=(
                np.log(2) / pl.col("listeners__mass__instantaneous_growth_rate")
            )
            / 60
        )

    # Create get_bulk_counts function with sim_data
    get_bulk_counts_func = make_get_bulk_counts(sim_data)

    # Calculate rRNA counts
    df = df.with_columns(
        [
            get_bulk_counts_func(df, s30_16s_rRNA).alias("bulk_16s_count"),
            get_bulk_counts_func(df, s50_23s_rRNA).alias("bulk_23s_count"),
            get_bulk_counts_func(df, s50_5s_rRNA).alias("bulk_5s_count"),
            get_unique_counts(df, "active_ribosome").alias("ribosome_count"),
        ]
    )

    # Total rRNA = bulk rRNA + rRNA in active ribosomes
    df = df.with_columns(
        [
            (pl.col("bulk_16s_count") + pl.col("ribosome_count")).alias("rrn16s_count"),
            (pl.col("bulk_23s_count") + pl.col("ribosome_count")).alias("rrn23s_count"),
            (pl.col("bulk_5s_count") + pl.col("ribosome_count")).alias("rrn5s_count"),
        ]
    )

    # Calculate time step
    df = df.with_columns(
        pl.col("time")
        .diff()
        .over(["variant_id", "generation_index", "agent_id"])
        .alias("time_step_sec")
    )
    df = df.with_columns(
        time_step_sec=pl.when(pl.col("time_step_sec").is_null())
        .then(pl.col("time"))
        .otherwise(pl.col("time_step_sec"))
    )

    if "listeners__ribosome_data__rRNA16S_initiated" in df.columns:
        df = df.with_columns(
            rrn16S_doubling_time_min=calc_rna_doubling_time(
                "listeners__ribosome_data__rRNA16S_initiated",
                "rrn16s_count",
                sim_doubling_time_min,
            )
        )
    if "listeners__ribosome_data__rRNA23S_initiated" in df.columns:
        df = df.with_columns(
            rrn23S_doubling_time_min=calc_rna_doubling_time(
                "listeners__ribosome_data__rRNA23S_initiated",
                "rrn23s_count",
                sim_doubling_time_min,
            )
        )
    if "listeners__ribosome_data__rRNA5S_initiated" in df.columns:
        df = df.with_columns(
            rrn5S_doubling_time_min=calc_rna_doubling_time(
                "listeners__ribosome_data__rRNA5S_initiated",
                "rrn5s_count",
                sim_doubling_time_min,
            )
        )

    # Calculate initiation probabilities
    if "listeners__ribosome_data__rRNA16S_init_prob" in df.columns:
        df = df.with_columns(
            rrn16S_init_prob_normalized=pl.col(
                "listeners__ribosome_data__rRNA16S_init_prob"
            )
        )
    if "listeners__ribosome_data__rRNA23S_init_prob" in df.columns:
        df = df.with_columns(
            rrn23S_init_prob_normalized=pl.col(
                "listeners__ribosome_data__rRNA23S_init_prob"
            )
        )
    if "listeners__ribosome_data__rRNA5S_init_prob" in df.columns:
        df = df.with_columns(
            rrn5S_init_prob_normalized=pl.col(
                "listeners__ribosome_data__rRNA5S_init_prob"
            )
        )

    # Calculate expected initiation probabilities
    condition = sim_data.condition
    transcription = sim_data.process.transcription
    cistron_synth_prob = transcription.cistron_tu_mapping_matrix.dot(
        transcription.rna_synth_prob[condition]
    )

    def get_cistron_prob(ids):
        indices = []
        for rna_id in ids:
            cistron_id = rna_id[:-3]  # Remove RNA suffix
            idx = np.where(transcription.cistron_data["id"] == cistron_id)[0]
            if len(idx) > 0:
                indices.append(idx[0])
        return cistron_synth_prob[indices].sum() if indices else 0.0

    rrn16s_fit_init_prob = get_cistron_prob(sim_data.molecule_groups.s30_16s_rRNA)
    rrn23s_fit_init_prob = get_cistron_prob(sim_data.molecule_groups.s50_23s_rRNA)
    rrn5s_fit_init_prob = get_cistron_prob(sim_data.molecule_groups.s50_5s_rRNA)

    # Select columns for plotting
    plot_columns = ["time_min", "variant_id", "generation_index"]

    # Add other columns
    for col in [
        "listeners__mass__dry_mass",
        "doubling_time_min",
        "rrn16S_doubling_time_min",
        "rrn23S_doubling_time_min",
        "rrn5S_doubling_time_min",
        "rrn16S_init_prob_normalized",
        "rrn23S_init_prob_normalized",
        "rrn5S_init_prob_normalized",
        "listeners__ribosome_data__effective_elongation_rate",
    ]:
        if col in df.columns:
            plot_columns.append(col)

    plot_df = df.select(plot_columns)

    # Calculate initial dry mass at time=0 for each variant and generation
    initial_dry_mass = (
        plot_df.filter(pl.col("time_min") == 0)
        .select(["variant_id", "listeners__mass__dry_mass"])
        .rename({"listeners__mass__dry_mass": "initial_dry_mass"})
    )

    plot_df = plot_df.join(initial_dry_mass, on=["variant_id"], how="left")

    plot_df = plot_df.with_columns(
        (pl.col("listeners__mass__dry_mass") / pl.col("initial_dry_mass")).alias(
            "dry_mass_normalized"
        )
    )

    # ----------------------------------------- #

    def create_line_chart(y_field, title, y_title, reference=None):
        base = alt.Chart(plot_df.to_pandas())
        line = base.mark_line().encode(
            x=alt.X("time_min:Q", title="Time (min)"),
            y=alt.Y(f"{y_field}:Q", title=y_title),
            color=alt.Color("variant_id:N", legend=alt.Legend(title="Variant")),
        )
        chart = line.properties(title=title, width=600, height=120)
        if reference is not None:
            ref_line = (
                alt.Chart(pd.DataFrame({"y": [reference]}))
                .mark_rule(color="red", strokeDash=[5, 5])
                .encode(y="y:Q")
            )
            return chart + ref_line
        return chart

    # ----------------------------------------- #
    plots = []

    if "dry_mass_normalized" in plot_df.columns:
        plots.append(
            create_line_chart(
                "dry_mass_normalized",
                "Normalized Dry Mass Over Time",
                "Dry mass (relative to t=0)",
            )
        )

    if "doubling_time_min" in plot_df.columns:
        plots.append(
            create_line_chart(
                "doubling_time_min",
                "Cell Doubling Time",
                "Doubling Time (min)",
                sim_doubling_time_min,
            )
        )

    rna_types = ["16S", "23S", "5S"]
    for rna in rna_types:
        col_name = f"rrn{rna}_doubling_time_min"
        if col_name in plot_df.columns:
            plots.append(
                create_line_chart(
                    col_name,
                    f"{rna} rRNA Doubling Time",
                    "Doubling Time (min)",
                    sim_doubling_time_min,
                )
            )

    init_probs = {
        "16S": rrn16s_fit_init_prob,
        "23S": rrn23s_fit_init_prob,
        "5S": rrn5s_fit_init_prob,
    }
    for rna, ref_prob in init_probs.items():
        col_name = f"rrn{rna}_init_prob_normalized"
        if col_name in plot_df.columns:
            plots.append(
                create_line_chart(
                    col_name,
                    f"{rna} rRNA Initiation Probability",
                    "Probability",
                    ref_prob,
                )
            )

    if "listeners__ribosome_data__effective_elongation_rate" in plot_df.columns:
        plots.append(
            create_line_chart(
                "listeners__ribosome_data__effective_elongation_rate",
                "Ribosome Elongation Rate",
                "Amino acids/s",
            )
        )

    if not plots:
        fallback_df = pl.DataFrame(
            {
                "message": ["No data available for ribosome production visualization"],
                "x": [0],
                "y": [0],
            }
        )
        fallback_plot = (
            alt.Chart(fallback_df.to_pandas())
            .mark_text(size=20, color="red")
            .encode(x="x:Q", y="y:Q", text="message:N")
            .properties(
                width=600,
                height=400,
                title="Ribosome Production Metrics - No Data Available",
            )
        )
        plots.append(fallback_plot)

    combined_plot = (
        alt.vconcat(*plots)
        .resolve_scale(x="shared", y="independent")
        .properties(title="Ribosome Production Metrics")
    )

    output_path = os.path.join(outdir, "ribosome_production_report.html")
    combined_plot.save(output_path)
    print(f"Saved visualization to: {output_path}")

    return combined_plot
