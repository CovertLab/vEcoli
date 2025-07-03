"""
Record the translation probability comparison on Gene EG10184
"""

import altair as alt
import os
from typing import Any
import pickle
import polars as pl
from duckdb import DuckDBPyConnection

from ecoli.library.parquet_emitter import (
    field_metadata,
    open_arbitrary_sim_data,
    named_idx,
)

MAX_NUMBER_OF_MONOMERS_TO_PLOT = 300

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
    """
    Comparison of target vs actual translation probabilities for overcrowded mRNAs.
    """

    # Load sim_data for monomer mappings
    with open_arbitrary_sim_data(sim_data_dict) as f:
        sim_data = pickle.load(f)

    # Get monomer and gene mappings
    mRNA_data = sim_data.process.transcription.cistron_data.struct_array
    monomer_data = sim_data.process.translation.monomer_data.struct_array

    monomer_to_gene = {}
    for mono_id, cistron_id in zip(monomer_data["id"], monomer_data["cistron_id"]):
        gene_id = next(
            (
                g
                for c, g in zip(mRNA_data["id"], mRNA_data["gene_id"])
                if c == cistron_id
            ),
            "Unknown",
        )
        monomer_to_gene[mono_id] = gene_id

    # Get field metadata
    try:
        field_names = field_metadata(
            conn,
            config_sql,
            "listeners__ribosome_data__target_prob_translation_per_transcript",
        )
    except Exception as e:
        print(f"[ERROR] Error getting field metadata: {e}")
        return

    # First pass: Find overcrowded monomer indices
    overcrowded_query = f"""
    WITH unnested AS (
        SELECT 
            unnest(listeners__ribosome_data__actual_prob_translation_per_transcript) as actual,
            unnest(listeners__ribosome_data__target_prob_translation_per_transcript) as target,
            generate_subscripts(listeners__ribosome_data__target_prob_translation_per_transcript, 1) as idx
        FROM ({history_sql})
    )
    SELECT DISTINCT idx
    FROM unnested
    WHERE target > actual
    ORDER BY idx
    LIMIT {MAX_NUMBER_OF_MONOMERS_TO_PLOT}
    """

    overcrowded_indices = [
        row[0] - 1 for row in conn.execute(overcrowded_query).fetchall()
    ]  # Convert to 0-based

    if not overcrowded_indices:
        print("[INFO] No overcrowded monomers found.")
        return

    print(f"[INFO] Found {len(overcrowded_indices)} overcrowded monomers")

    # Second pass: Get data for overcrowded monomers only
    actual_expr = named_idx(
        "listeners__ribosome_data__actual_prob_translation_per_transcript",
        [f"actual_{i}" for i in range(len(overcrowded_indices))],
        [overcrowded_indices],
    )

    target_expr = named_idx(
        "listeners__ribosome_data__target_prob_translation_per_transcript",
        [f"target_{i}" for i in range(len(overcrowded_indices))],
        [overcrowded_indices],
    )

    data_query = f"SELECT {actual_expr}, {target_expr}, time FROM ({history_sql})"
    df = conn.execute(data_query).fetchdf()

    # ----------------------------------------- #
    # Prepare plot data following original format
    plot_data = []
    n_overcrowded_monomers = len(overcrowded_indices)
    n_overcrowded_monomers_to_plot = min(
        n_overcrowded_monomers, MAX_NUMBER_OF_MONOMERS_TO_PLOT
    )

    for i, idx in enumerate(overcrowded_indices):
        if i >= MAX_NUMBER_OF_MONOMERS_TO_PLOT:
            break

        if idx < len(field_names):
            monomer_id = field_names[idx]
            gene_id = monomer_to_gene.get(monomer_id, "Unknown")

            # Add target probabilities
            for _, row in df.iterrows():
                plot_data.append(
                    {
                        "Time_min": float(row["time"]),
                        "Gene_ID": str(gene_id),
                        "Probability_Type": "target",
                        "Translation_Probability": float(row[f"target_{i}"]),
                        "Plot_Order": i,
                    }
                )

            # Add actual probabilities
            for _, row in df.iterrows():
                plot_data.append(
                    {
                        "Time_min": float(row["time"]),
                        "Gene_ID": str(gene_id),
                        "Probability_Type": "actual",
                        "Translation_Probability": float(row[f"actual_{i}"]),
                        "Plot_Order": i,
                    }
                )

    if not plot_data:
        print("[INFO] No data prepared for plotting")
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
