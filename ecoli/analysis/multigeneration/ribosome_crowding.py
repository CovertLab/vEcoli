"""
Record the translation probability comparison on Gene EG10184
"""

import altair as alt
import os
from typing import Any

from duckdb import DuckDBPyConnection
import pickle
import polars as pl
import numpy as np

from ecoli.library.parquet_emitter import (
    field_metadata,
    open_arbitrary_sim_data,
    named_idx,
    read_stacked_columns,
)

# ----------------------------------------- #

# Set this to ensure maximum figure size is not exceeded
MAX_NUMBER_OF_MONOMERS_TO_PLOT = 300


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
    Comparison of target translation probabilities vs actual translation
    probabilities for mRNAs whose translation probabilities exceeded the limit
    set by the physical size and the elongation rates of ribosomes.
    """

    # Load sim_data to get monomer information
    with open_arbitrary_sim_data(sim_data_dict) as f:
        sim_data = pickle.load(f)

    # Get monomer IDs and mappings
    mRNA_sim_data = sim_data.process.transcription.cistron_data.struct_array
    monomer_sim_data = sim_data.process.translation.monomer_data.struct_array
    monomer_ids = monomer_sim_data["id"].tolist()

    # Build mappings: monomer_id -> mRNA_id -> gene_id
    monomer_to_mRNA_id_dict = dict(
        zip(monomer_sim_data["id"], monomer_sim_data["cistron_id"])
    )
    mRNA_to_gene_id_dict = dict(zip(mRNA_sim_data["id"], mRNA_sim_data["gene_id"]))

    # Get field metadata for ribosome data
    try:
        target_field_names = field_metadata(
            conn,
            config_sql,
            "listeners__ribosome_data__target_prob_translation_per_transcript",
        )
        actual_field_names = field_metadata(
            conn,
            config_sql,
            "listeners__ribosome_data__actual_prob_translation_per_transcript",
        )
    except Exception as e:
        print(f"Error getting field metadata: {e}")
        print("Trying alternative listener names...")
        try:
            target_field_names = field_metadata(
                conn, config_sql, "listeners__ribosome_data"
            )
            actual_field_names = target_field_names  # Assume same structure
        except Exception as e2:
            print(f"Alternative approach also failed: {e2}")
            return

    # Find indices for each monomer in the field metadata
    target_monomer_indices = []
    actual_monomer_indices = []
    valid_monomer_ids = []

    for i, monomer_id in enumerate(monomer_ids):
        if monomer_id in target_field_names and monomer_id in actual_field_names:
            target_idx = target_field_names.index(monomer_id)
            actual_idx = actual_field_names.index(monomer_id)
            target_monomer_indices.append(target_idx)
            actual_monomer_indices.append(actual_idx)
            valid_monomer_ids.append(monomer_id)

    if not valid_monomer_ids:
        print("No valid monomer IDs found in ribosome data fields.")
        return

    print(f"[INFO] Found {len(valid_monomer_ids)} valid monomer IDs")

    # Create named indices for data reading
    target_named = named_idx(
        "listeners__ribosome_data__target_prob_translation_per_transcript",
        valid_monomer_ids,
        [target_monomer_indices],
    )
    actual_named = named_idx(
        "listeners__ribosome_data__actual_prob_translation_per_transcript",
        valid_monomer_ids,
        [actual_monomer_indices],
    )

    # Read target and actual data separately to ensure proper structure
    try:
        # Read target data
        target_data = read_stacked_columns(
            history_sql,
            [target_named],
            conn=conn,
        )
        target_df = pl.DataFrame(target_data).with_columns(
            **{"Time (min)": pl.col("time") / 60}
        )

        # Read actual data
        actual_data = read_stacked_columns(
            history_sql,
            [actual_named],
            conn=conn,
        )
        actual_df = pl.DataFrame(actual_data).with_columns(
            **{"Time (min)": pl.col("time") / 60}
        )

        # Get the probability columns
        target_prob_cols = [
            col for col in target_df.columns if col in valid_monomer_ids
        ]
        actual_prob_cols = [
            col for col in actual_df.columns if col in valid_monomer_ids
        ]

        if not target_prob_cols or not actual_prob_cols:
            print("Could not find probability columns in datasets")
            return

        # Create arrays for calculation
        target_prob_array = target_df.select(target_prob_cols).to_numpy()
        actual_prob_array = actual_df.select(actual_prob_cols).to_numpy()
        time_min = target_df["Time (min)"].to_numpy()

        print("[INFO] Successfully read target and actual data")
        print(
            f"[INFO] Target shape: {target_prob_array.shape}, Actual shape: {actual_prob_array.shape}"
        )

    except Exception as e:
        print(f"Failed to read separate datasets: {e}")
        return

    # Calculate differences to find overcrowded mRNAs
    prob_differences = target_prob_array - actual_prob_array
    overcrowded_monomer_indexes = np.where(prob_differences.max(axis=0) > 0)[0]
    n_overcrowded_monomers = len(overcrowded_monomer_indexes)

    print(f"[INFO] Found {n_overcrowded_monomers} overcrowded monomers")

    if n_overcrowded_monomers == 0:
        print("No overcrowded mRNAs found in the simulation.")
        return

    # Get gene IDs for overcrowded monomers
    overcrowded_monomer_ids = [
        valid_monomer_ids[i] for i in overcrowded_monomer_indexes
    ]
    overcrowded_gene_ids = [
        mRNA_to_gene_id_dict.get(monomer_to_mRNA_id_dict.get(monomer_id), "unknown")
        for monomer_id in overcrowded_monomer_ids
    ]

    n_overcrowded_monomers_to_plot = min(
        n_overcrowded_monomers, MAX_NUMBER_OF_MONOMERS_TO_PLOT
    )

    # ----------------------------------------- #

    plot_data = []
    for i, monomer_index in enumerate(overcrowded_monomer_indexes):
        if i >= MAX_NUMBER_OF_MONOMERS_TO_PLOT:
            break

        gene_id = overcrowded_gene_ids[i]

        # Get the data for this monomer
        target_probs = target_prob_array[:, monomer_index]
        actual_probs = actual_prob_array[:, monomer_index]

        # Add target probabilities
        for j, time_val in enumerate(time_min):
            plot_data.append(
                {
                    "Time_min": float(time_val),
                    "Gene_ID": str(gene_id),
                    "Probability_Type": "target",
                    "Translation_Probability": float(target_probs[j]),
                    "Plot_Order": i,
                }
            )

        # Add actual probabilities
        for j, time_val in enumerate(time_min):
            plot_data.append(
                {
                    "Time_min": float(time_val),
                    "Gene_ID": str(gene_id),
                    "Probability_Type": "actual",
                    "Translation_Probability": float(actual_probs[j]),
                    "Plot_Order": i,
                }
            )

    if not plot_data:
        print("No data prepared for plotting")
        return

    plot_df = pl.DataFrame(plot_data)

    # Create individual plots for each overcrowded gene
    charts = []
    for i in range(n_overcrowded_monomers_to_plot):
        gene_data = plot_df.filter(pl.col("Plot_Order") == i)

        if gene_data.height == 0:
            continue

        gene_id = gene_data["Gene_ID"][0]

        # Create chart with simplified encoding and proper tooltip
        chart = (
            alt.Chart(gene_data)
            .mark_line(point=False, strokeWidth=2)
            .encode(
                x=alt.X("Time_min:Q", title="Time (min)", scale=alt.Scale(nice=True)),
                y=alt.Y(
                    "Translation_Probability:Q",
                    title=f"{gene_id} translation probability",
                    scale=alt.Scale(nice=True),
                ),
                color=alt.Color(
                    "Probability_Type:N",
                    scale=alt.Scale(
                        domain=["target", "actual"], range=["#1f77b4", "#ff7f0e"]
                    ),
                    legend=alt.Legend(title="Type") if i == 0 else None,
                ),
                strokeDash=alt.StrokeDash(
                    "Probability_Type:N",
                    scale=alt.Scale(
                        domain=["target", "actual"], range=[[1, 0], [5, 5]]
                    ),
                    legend=None,
                ),
                tooltip=[
                    alt.Tooltip("Time_min:Q", title="Time (min)", format=".2f"),
                    alt.Tooltip(
                        "Translation_Probability:Q", title="Probability", format=".4f"
                    ),
                    alt.Tooltip("Probability_Type:N", title="Type"),
                    alt.Tooltip("Gene_ID:N", title="Gene"),
                ],
            )
            .properties(
                width=600,
                height=150,
                title=alt.TitleParams(
                    text=[
                        f"Gene {gene_id} - Translation Probability Comparison",
                        f"Total overcrowded proteins: {n_overcrowded_monomers}"
                        + (
                            f" (showing first {MAX_NUMBER_OF_MONOMERS_TO_PLOT})"
                            if n_overcrowded_monomers > MAX_NUMBER_OF_MONOMERS_TO_PLOT
                            else ""
                        )
                        if i == 0
                        else "",
                    ],
                    fontSize=12,
                    anchor="start",
                ),
            )
        )

        charts.append(chart)

    if charts:
        combined_chart = (
            alt.vconcat(*charts)
            .resolve_scale(color="independent")
            .add_params(alt.selection_interval(bind="scales"))
        )

        alt.data_transformers.enable("json")

        output_path = os.path.join(outdir, "ribosome_crowding.html")
        combined_chart.save(output_path)

        print(
            f"[INFO] Generated ribosome crowding plot for {len(charts)} overcrowded proteins"
        )
        print(f"[INFO] Plot saved to: {output_path}")

        # Also save as JSON for debugging if needed
        json_path = os.path.join(outdir, "ribosome_crowding.json")
        combined_chart.save(json_path)
        print(f"[INFO] Chart specification saved to: {json_path}")

    else:
        print("[INFO] No charts created - no data to plot")
