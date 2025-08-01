"""
Visualize FBA reaction fluxes over time for specified reactions.
You can specify the reactions to visualize using the \'BioCyc_ID\' parameter in params:
    \"fba_flux\": {
        \"BioCyc_ID\": [\"Name1\", \"Name2\", ...],
        # Optional: specify generations to visualize
        # If not specified, all generations will be used
        \"generation\": [1, 2, ...]
        }
For each reaction in params.BioCyc_ID, plot flux vs time (for all generation),
and with average (for all generation) flux marked.
"""

import altair as alt
import os
from typing import Any

import polars as pl
from duckdb import DuckDBPyConnection
import pandas as pd

from ecoli.library.parquet_emitter import field_metadata

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
    """Visualize FBA reaction fluxes over time for specified BioCyc reactions."""

    # Get BioCyc IDs from params
    biocyc_ids = params.get("BioCyc_ID", [])
    if not biocyc_ids:
        print(
            "[ERROR] No BioCyc_ID found in params. Please specify reaction IDs to visualize."
        )
        return None

    if isinstance(biocyc_ids, str):
        biocyc_ids = [biocyc_ids]

    print(f"[INFO] Visualizing fluxes for {len(biocyc_ids)} reactions: {biocyc_ids}")

    # Required columns for the query
    required_columns = [
        "time",
        "variant",
        "generation",
        "listeners__fba_results__reaction_fluxes",
    ]

    # Build SQL query
    sql = f"""
    SELECT
        {", ".join(required_columns)}
    FROM ({history_sql})
    ORDER BY variant, generation, time
    """

    # Execute query
    try:
        df = conn.sql(sql).pl()
    except Exception as e:
        print(f"[ERROR] Error executing SQL query: {e}")
        return None

    if df.is_empty():
        print("[ERROR] No data found")
        return None

    # Configuration parameters for filtering
    target_generations = params.get(
        "generation", None
    )  # List of generation IDs or None for all

    # Filter by specified generations
    if target_generations is not None:
        print(f"[INFO] Target generations: {target_generations}")
        df = df.filter(pl.col("generation").is_in(target_generations))

    # Convert time to minutes
    if "time" in df.columns:
        df = df.with_columns((pl.col("time") / 60).alias("time_min"))

    print(f"[INFO] Loaded data with {df.height} time steps")

    # Load the reaction IDs from the config - this is the array that maps to flux matrix columns
    try:
        reaction_ids = field_metadata(
            conn, config_sql, "listeners__fba_results__reaction_fluxes"
        )
        print(f"[INFO] Total reactions in sim_data: {len(reaction_ids)}")
    except Exception as e:
        print(f"[ERROR] Error loading reaction IDs: {e}")
        return None

    # Expand the flux matrix and name columns using reaction_ids
    flux_expressions = []
    for idx, reaction_id in enumerate(reaction_ids):
        flux_expressions.append(
            pl.col("listeners__fba_results__reaction_fluxes")
            .list.get(idx)
            .alias(reaction_id)
        )

    # Add all flux columns to the dataframe
    flux_df = df.with_columns(flux_expressions)
    print(f"[INFO] Expanded flux matrix to {len(reaction_ids)} named columns")

    # Extract flux data for specified BioCyc IDs by column name
    flux_data = {}

    for biocyc_id in biocyc_ids:
        if biocyc_id in flux_df.columns:
            flux_data[biocyc_id] = biocyc_id
            print(f"[INFO] Found flux data for {biocyc_id}")
        else:
            print(f"[WARNING] BioCyc ID '{biocyc_id}' not found in reaction data")
            print(
                f"[INFO] Available reactions: {reaction_ids[:10]}..."
                if len(reaction_ids) > 10
                else f"[INFO] Available reactions: {reaction_ids}"
            )

    if not flux_data:
        print(
            "[ERROR] No flux data could be extracted for any of the specified reactions"
        )
        return None

    # Calculate average fluxes for each reaction
    avg_fluxes = {}
    for biocyc_id in flux_data:
        avg_flux = flux_df[biocyc_id].mean()
        avg_fluxes[biocyc_id] = avg_flux
        print(f"[INFO] Average flux for {biocyc_id}: {avg_flux:.6f}")

    # ---------------------------------------------- #

    def create_flux_chart(biocyc_id, flux_col, avg_flux):
        """Create a line chart for a single reaction flux with average line."""
        data = flux_df.select(
            ["time_min", "variant", "generation", flux_col]
        ).to_pandas()

        # Remove any null values
        data = data.dropna(subset=[flux_col])

        if data.empty:
            print(f"[WARNING] No valid data for reaction {biocyc_id}")
            return None

        # Main flux line chart
        flux_chart = (
            alt.Chart(data)
            .mark_line(strokeWidth=2)
            .encode(
                x=alt.X("time_min:Q", title="Time (min)"),
                y=alt.Y(f"{flux_col}:Q", title="Flux (mmol/gDW/hr)"),
                color=alt.Color("generation:N", legend=alt.Legend(title="Generation")),
                tooltip=["time_min:Q", f"{flux_col}:Q", "generation:N"],
            )
        )

        # Average flux horizontal line
        avg_line_data = pd.DataFrame(
            {"avg_flux": [avg_flux], "label": [f"Avg: {avg_flux:.4f}"]}
        )

        avg_line = (
            alt.Chart(avg_line_data)
            .mark_rule(color="red", strokeDash=[5, 5], strokeWidth=2)
            .encode(y=alt.Y("avg_flux:Q"), tooltip=["label:N"])
        )

        # Combine flux line and average line
        combined_chart = (
            (flux_chart + avg_line)
            .properties(title=f"Flux vs Time: {biocyc_id}", width=600, height=300)
            .resolve_scale(y="shared")
        )

        return combined_chart

    # ---------------------------------------------- #

    # Create charts for each reaction
    charts = []

    for biocyc_id in flux_data:
        avg_flux = avg_fluxes.get(biocyc_id, 0)
        chart = create_flux_chart(biocyc_id, biocyc_id, avg_flux)
        if chart is not None:
            charts.append(chart)

    if not charts:
        print("[ERROR] No valid charts could be created")
        return None

    # Arrange charts vertically
    if len(charts) == 1:
        combined_plot = charts[0]
    else:
        combined_plot = alt.vconcat(*charts).resolve_scale(x="shared", y="independent")

    # Add overall title
    combined_plot = combined_plot.properties(
        title=alt.TitleParams(
            text=f"FBA Reaction Fluxes Over Time ({len(charts)} reactions)",
            fontSize=16,
            anchor="start",
        )
    )

    # Save the plot
    output_path = os.path.join(outdir, "fba_flux_report.html")
    combined_plot.save(output_path)
    print(f"[INFO] Saved visualization to: {output_path}")

    # Also save a summary CSV with average fluxes
    summary_df = pl.DataFrame(
        {
            "BioCyc_ID": list(avg_fluxes.keys()),
            "Average_Flux": list(avg_fluxes.values()),
        }
    )

    summary_path = os.path.join(outdir, "fba_flux_summary.csv")
    summary_df.write_csv(summary_path)
    print(f"[INFO] Saved flux summary to: {summary_path}")

    return combined_plot
