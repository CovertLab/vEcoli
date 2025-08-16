"""
Visualize FBA reaction fluxes over time for specified reactions with net flux calculation across multiple variants.

Layout: Each specified BioCyc reaction ID will be visualized as a separate line chart,
and different variants will be shown as different colored lines.

You can specify the reactions to visualize using the 'BioCyc_ID' parameter in params:
    "fba_flux": {
        # Required: specify BioCyc IDs of reactions to visualize
        "BioCyc_ID": ["Name1", "Name2", ...],
        # Optional: specify variants to visualize
        # If not specified, all variants will be used
        "variant": ["variant1", "variant2", ...],
        # Optional: specify generation for visualization
        # If not specified, all generations will be used
        "generation": [1, 2, 3, ...]
        }

This script uses the base reaction ID to extended reaction mapping to efficiently
find forward and reverse reactions, then calculates net flux using SQL for
optimal memory usage and performance.

For each reaction in params.BioCyc_ID, plot net flux vs time for all variants,
with different variants shown as different colored lines,
and with average net flux marked for each variant.
"""

import altair as alt
import os
from typing import Any

import polars as pl
from duckdb import DuckDBPyConnection
import pandas as pd

from ecoli.library.parquet_emitter import field_metadata
from ecoli.analysis.utils import (
    create_base_to_extended_mapping,
    build_flux_calculation_sql,
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
    """Visualize FBA reaction net fluxes over time for specified BioCyc reactions across multiple variants."""

    # Get parameters
    biocyc_ids = params.get("BioCyc_ID", [])
    if not biocyc_ids:
        print(
            "[ERROR] No BioCyc_ID found in params. Please specify reaction IDs to visualize."
        )
        return None

    if isinstance(biocyc_ids, str):
        biocyc_ids = [biocyc_ids]

    print(
        f"[INFO] Visualizing net fluxes for {len(biocyc_ids)} reactions: {biocyc_ids}"
    )

    # Create base to extended reaction mapping
    base_to_extended_mapping = create_base_to_extended_mapping(sim_data_dict)
    if not base_to_extended_mapping:
        print("[ERROR] Could not create base to extended reaction mapping")
        return None

    # Load reaction IDs from config
    try:
        all_reaction_ids = field_metadata(
            conn, config_sql, "listeners__fba_results__reaction_fluxes"
        )
        print(f"[INFO] Total reactions in sim_data: {len(all_reaction_ids)}")
    except Exception as e:
        print(f"[ERROR] Error loading reaction IDs: {e}")
        return None

    # Build SQL query for efficient flux calculation
    flux_calculation_sql, valid_biocyc_ids = build_flux_calculation_sql(
        biocyc_ids, base_to_extended_mapping, all_reaction_ids, history_sql
    )

    if not flux_calculation_sql or not valid_biocyc_ids:
        print("[ERROR] Could not build flux calculation SQL")
        return None

    print(f"[INFO] Processing {len(valid_biocyc_ids)} valid BioCyc IDs")

    # Execute the optimized SQL query
    try:
        df = conn.sql(flux_calculation_sql).pl()
        print(f"[INFO] Loaded data with {df.height} time steps")
    except Exception as e:
        print(f"[ERROR] Error executing flux calculation SQL: {e}")
        return None

    if df.is_empty():
        print("[ERROR] No data found")
        return None

    # Filter by specified variants and generations if provided
    target_variants = params.get("variant", None)
    target_generations = params.get("generation", None)

    if target_variants is not None:
        print(f"[INFO] Filtering for variants: {target_variants}")
        df = df.filter(pl.col("variant").is_in(target_variants))

    if target_generations is not None:
        print(f"[INFO] Filtering for generations: {target_generations}")
        df = df.filter(pl.col("generation").is_in(target_generations))

    # Print variant information
    unique_variants = sorted(df["variant"].unique().to_list())
    print(f"[INFO] Found {len(unique_variants)} variants: {unique_variants}")

    # Print average net flux information for each reaction
    for biocyc_id in valid_biocyc_ids:
        net_flux_col = f"{biocyc_id}_net_flux"
        if net_flux_col in df.columns:
            avg_flux = df[net_flux_col].mean()
            print(
                f"[INFO] Average net flux for {biocyc_id}: {avg_flux:.6f} mmol/gDW/hr"
            )

    # Calculate average net fluxes for each reaction and variant combination
    avg_net_fluxes = {}
    for biocyc_id in valid_biocyc_ids:
        net_flux_col = f"{biocyc_id}_net_flux"
        if net_flux_col in df.columns:
            # Calculate average net flux by variant
            variant_avgs = df.group_by("variant").agg(
                pl.col(net_flux_col).mean().alias("avg_net_flux")
            )
            avg_net_fluxes[biocyc_id] = variant_avgs

    # --------------------------- #

    # Create visualization functions
    def create_net_flux_chart(biocyc_id, net_flux_col, variant_avgs):
        """Create a line chart for a single reaction net flux with average lines for each variant."""

        # Check if the column exists in dataframe
        if net_flux_col not in df.columns:
            print(f"[WARNING] Column {net_flux_col} not found in dataframe")
            return None

        # Select only the columns we need to minimize data transfer
        data = df.select(["time_min", "generation", "variant", net_flux_col])

        # Remove any null values
        data = data.filter(pl.col(net_flux_col).is_not_null())

        if data.height == 0:
            print(f"[WARNING] No valid data for reaction {biocyc_id}")
            return None

        # Main net flux line chart (different variants as different colored lines)
        net_flux_chart = (
            alt.Chart(data)
            .mark_line(strokeWidth=2)
            .encode(
                x=alt.X("time_min:Q", title="Time (min)"),
                y=alt.Y(f"{net_flux_col}:Q", title="Net Flux (mmol/gDW/hr)"),
                color=alt.Color("variant:N", legend=alt.Legend(title="Variant")),
                tooltip=[
                    "time_min:Q",
                    f"{net_flux_col}:Q",
                    "variant:N",
                    "generation:N",
                ],
            )
        )

        # Create average lines for each variant
        avg_line_data = []
        for row in variant_avgs.iter_rows(named=True):
            variant_name = row["variant"]
            avg_value = row["avg_net_flux"]
            avg_line_data.append(
                {
                    "variant": variant_name,
                    "avg_net_flux": avg_value,
                    "label": f"{variant_name} Avg: {avg_value:.4f}",
                }
            )

        if avg_line_data:
            avg_line_df = pd.DataFrame(avg_line_data)

            avg_lines = (
                alt.Chart(avg_line_df)
                .mark_rule(strokeDash=[5, 5], strokeWidth=2)
                .encode(
                    y=alt.Y("avg_net_flux:Q"),
                    color=alt.Color(
                        "variant:N", legend=None
                    ),  # Use same color scale as main chart
                    tooltip=["label:N"],
                )
            )
        else:
            avg_lines = alt.Chart().mark_point()  # Empty chart

        # Zero line for reference
        zero_line = (
            alt.Chart(pd.DataFrame({"zero": [0], "label": ["Zero"]}))
            .mark_rule(color="gray", strokeDash=[2, 2], strokeWidth=1, opacity=0.7)
            .encode(y=alt.Y("zero:Q"), tooltip=["label:N"])
        )

        # Combine net flux line, average lines, and zero line
        combined_chart = (
            (net_flux_chart + avg_lines + zero_line)
            .properties(title=f"Net Flux vs Time: {biocyc_id}", width=600, height=300)
            .resolve_scale(y="shared", color="shared")
        )

        return combined_chart

    # --------------------------- #

    # Create charts for each reaction
    charts = []

    for biocyc_id in valid_biocyc_ids:
        net_flux_col = f"{biocyc_id}_net_flux"
        variant_avgs = avg_net_fluxes.get(biocyc_id)
        if variant_avgs is not None:
            chart = create_net_flux_chart(biocyc_id, net_flux_col, variant_avgs)
            if chart is not None:
                charts.append(chart)

    if not charts:
        print("[ERROR] No valid charts could be created")
        return None

    # Arrange charts vertically
    if len(charts) == 1:
        combined_plot = charts[0]
    else:
        combined_plot = alt.vconcat(*charts).resolve_scale(
            x="shared", y="independent", color="shared"
        )

    # Add overall title
    combined_plot = combined_plot.properties(
        title=alt.TitleParams(
            text=f"FBA Reaction Net Fluxes Over Time - Multi-Variant Analysis ({len(charts)} reactions)",
            fontSize=16,
            anchor="start",
        )
    )

    # Save the plot
    output_path = os.path.join(outdir, "fba_net_flux_multivariant_analysis.html")
    combined_plot.save(output_path)
    print(f"[INFO] Saved visualization to: {output_path}")

    return combined_plot
