"""
Visualize FBA reaction fluxes over time for specified reactions with net flux calculation across multiple variants.

Layout: Each row represents a variant, each column represents a reaction.
Within each subplot, different generations are shown as different colored lines.

You can specify the reactions to visualize using the 'BioCyc_ID' parameter in params:
    "fba_flux": {
        # Required: specify BioCyc reaction IDs to visualize
        "BioCyc_ID": ["Name1", "Name2", ...],
        # Optional: specify variants to visualize
        # If not specified, all variants will be used
        "variant": [1, 2, ...],
        # Optional: specify generations to visualize
        # If not specified, all generations will be used
        "generation": [1, 2, ...]
        }

This script uses the base reaction ID to extended reaction mapping to efficiently
find forward and reverse reactions, then calculates net flux using SQL for
optimal memory usage and performance.
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
    """Visualize FBA reaction net fluxes"""

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

    # Print variant and generation information
    unique_variants = sorted(df["variant"].unique().to_list())
    unique_generations = sorted(df["generation"].unique().to_list())
    print(f"[INFO] Found {len(unique_variants)} variants: {unique_variants}")
    print(f"[INFO] Found {len(unique_generations)} generations: {unique_generations}")

    # Print average net flux information
    for biocyc_id in valid_biocyc_ids:
        net_flux_col = f"{biocyc_id}_net_flux"
        if net_flux_col in df.columns:
            avg_flux = df[net_flux_col].mean()
            print(
                f"[INFO] Average net flux for {biocyc_id}: {avg_flux:.6f} mmol/gDW/hr"
            )

    # Calculate average net fluxes for each variant-reaction-generation combination
    avg_data = []
    for biocyc_id in valid_biocyc_ids:
        net_flux_col = f"{biocyc_id}_net_flux"
        if net_flux_col in df.columns:
            # Calculate average net flux by variant and generation
            variant_gen_avgs = df.group_by(["variant", "generation"]).agg(
                pl.col(net_flux_col).mean().alias("avg_net_flux")
            )

            for row in variant_gen_avgs.iter_rows(named=True):
                avg_data.append(
                    {
                        "biocyc_id": biocyc_id,
                        "variant": row["variant"],
                        "generation": row["generation"],
                        "avg_net_flux": row["avg_net_flux"],
                    }
                )

    avg_df = pl.DataFrame(avg_data)

    # --------------------------- #

    # Create visualization functions
    def create_subplot_chart(variant, biocyc_id):
        """Create a single subplot for a specific variant-reaction combination."""
        net_flux_col = f"{biocyc_id}_net_flux"

        # Check if the column exists in dataframe
        if net_flux_col not in df.columns:
            print(f"[WARNING] Column {net_flux_col} not found in dataframe")
            return None

        # Filter data for this variant and reaction
        subplot_data = (
            df.filter(pl.col("variant") == variant)
            .select(["time_min", "generation", net_flux_col])
            .filter(pl.col(net_flux_col).is_not_null())
        )

        if subplot_data.height == 0:
            print(f"[WARNING] No data for variant {variant}, reaction {biocyc_id}")
            return None

        # Main line chart with generations as different colors
        line_chart = (
            alt.Chart(subplot_data)
            .mark_line(strokeWidth=1.5)
            .encode(
                x=alt.X(
                    "time_min:Q",
                    title="Time (min)" if variant == unique_variants[-1] else "",
                ),
                y=alt.Y(
                    f"{net_flux_col}:Q",
                    title="Net Flux (mmol/gDW/hr)"
                    if biocyc_id == valid_biocyc_ids[0]
                    else "",
                ),
                color=alt.Color(
                    "generation:N",
                    legend=alt.Legend(title="Generation")
                    if variant == unique_variants[0]
                    and biocyc_id == valid_biocyc_ids[0]
                    else None,
                ),
                tooltip=["time_min:Q", f"{net_flux_col}:Q", "generation:N"],
            )
        )

        # Average lines for each generation in this variant
        avg_subset = avg_df.filter(
            (pl.col("variant") == variant) & (pl.col("biocyc_id") == biocyc_id)
        )

        if avg_subset.height > 0:
            avg_lines = (
                alt.Chart(avg_subset)
                .mark_rule(strokeDash=[3, 3], strokeWidth=1.5)
                .encode(
                    y=alt.Y("avg_net_flux:Q"),
                    color=alt.Color("generation:N", legend=None),
                    tooltip=[
                        alt.Tooltip("avg_net_flux:Q", format=".4f"),
                        "generation:N",
                    ],
                )
            )
        else:
            avg_lines = alt.Chart().mark_point()  # Empty chart

        # Zero line for reference
        zero_line = (
            alt.Chart(pd.DataFrame({"zero": [0]}))
            .mark_rule(color="gray", strokeDash=[1, 1], strokeWidth=1, opacity=0.5)
            .encode(y=alt.Y("zero:Q"))
        )

        # Combine all elements
        combined = (line_chart + avg_lines + zero_line).resolve_scale(color="shared")

        # Add title only for top row
        if variant == unique_variants[0]:
            combined = combined.properties(title=f"{biocyc_id}")

        combined = combined.properties(width=400, height=300)

        return combined

    # --------------------------- #

    def create_empty_subplot():
        """Create an empty placeholder subplot."""
        return (
            alt.Chart(pd.DataFrame({"x": [0], "y": [0]}))
            .mark_point(opacity=0)
            .properties(width=200, height=150)
        )

    # Create subplot grid: rows = variants, columns = reactions
    subplot_grid = []

    for variant in unique_variants:
        variant_row = []
        for biocyc_id in valid_biocyc_ids:
            subplot = create_subplot_chart(variant, biocyc_id)
            if subplot is not None:
                variant_row.append(subplot)
            else:
                # Create empty placeholder if no data
                variant_row.append(create_empty_subplot())

        if variant_row:
            # Add variant label on the left
            variant_label = (
                alt.Chart(pd.DataFrame({"label": [f"Variant {variant}"]}))
                .mark_text(
                    align="center", baseline="middle", fontSize=12, fontWeight="bold"
                )
                .encode(text="label:N")
                .properties(width=160, height=300)
            )

            # Combine variant label with row of subplots
            row_with_label = alt.hconcat(variant_label, *variant_row, spacing=10)
            subplot_grid.append(row_with_label)

    if not subplot_grid:
        print("[ERROR] No valid subplots could be created")
        return None

    # Combine all rows
    combined_plot = alt.vconcat(*subplot_grid, spacing=20)

    # Add overall title
    combined_plot = combined_plot.properties(
        title=alt.TitleParams(
            text=f"FBA Net Flux Analysis: {len(unique_variants)} Variants × {len(valid_biocyc_ids)} Reactions",
            fontSize=16,
            anchor="start",
        )
    ).resolve_scale(color="shared")

    # Save the plot
    output_path = os.path.join(outdir, "fba_net_flux_grid_analysis.html")
    combined_plot.save(output_path)
    print(f"[INFO] Saved visualization to: {output_path}")

    return combined_plot
