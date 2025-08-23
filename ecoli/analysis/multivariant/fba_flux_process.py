"""
Visualize FBA reaction fluxes over time for a biological process by aggregating multiple reactions
across multiple variants.

Layout: The specified biological process will be visualized as a single line chart,
with different variants shown as different colored lines. The total flux is calculated
as the sum of net fluxes from all reactions specified in the BioCyc_ID list.

You can specify the biological process to visualize using parameters in params:
    "fba_process_flux": {
        # Required: specify BioCyc IDs of reactions involved in the biological process
        "BioCyc_ID": ["Name1", "Name2", ...],
        # Required: name of the biological process for visualization
        "process_name": "Glucose Transport",
        # Optional: specify variants to visualize
        # If not specified, all variants will be used
        "variant": ["variant1", "variant2", ...],
        # Optional: specify generation for visualization
        # If not specified, all generations will be used
        "generation": [1, 2, 3, ...]
        }

This script uses the base reaction ID to extended reaction mapping to efficiently
find forward and reverse reactions, then calculates total process flux using SQL for
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


def build_process_flux_sql(
    biocyc_ids, base_to_extended_mapping, all_reaction_ids, history_sql, process_name
):
    """
    Build SQL query to calculate total biological process flux by summing net fluxes from all reactions.

    Args:
        biocyc_ids (list): List of BioCyc IDs (base reaction IDs) for the biological process
        base_to_extended_mapping (dict): Mapping from base to extended reactions
        all_reaction_ids (list): List of all reaction IDs from field_metadata
        history_sql (str): SQL query for historical data
        process_name (str): Name of the biological process

    Returns:
        tuple: (sql_query, valid_biocyc_ids, process_flux_column_name)
    """
    # First, use the existing function to get individual reaction flux calculations
    individual_flux_sql, valid_biocyc_ids = build_flux_calculation_sql(
        biocyc_ids, base_to_extended_mapping, all_reaction_ids, history_sql
    )

    if not individual_flux_sql or not valid_biocyc_ids:
        print("[ERROR] Could not build individual flux calculations for process")
        return None, [], ""

    # Extract just the flux calculations from the individual SQL
    # Parse the SELECT clause to get individual flux expressions
    flux_calculations = []
    for biocyc_id in valid_biocyc_ids:
        flux_calculations.append(f'"{biocyc_id}_net_flux"')

    # Create safe process column name
    safe_process_col = f'"{process_name.replace(" ", "_")}_total_flux"'

    # Build the process flux calculation SQL
    # This sums all individual net fluxes to get total process flux
    process_flux_expr = f"({' + '.join(flux_calculations)}) AS {safe_process_col}"

    # Build complete SQL query that first calculates individual fluxes, then sums them
    sql = f"""
    WITH individual_fluxes AS (
        {individual_flux_sql}
    )
    SELECT 
        time,
        generation,
        variant,
        time_min,
        {process_flux_expr}
    FROM individual_fluxes
    ORDER BY variant, generation, time
    """

    return sql, valid_biocyc_ids, safe_process_col.strip('"')


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
    """Visualize FBA biological process flux over time by aggregating multiple reactions across variants."""

    # Get parameters
    biocyc_ids = params.get("BioCyc_ID", [])
    process_name = params.get("process_name", "Biological Process")

    if not biocyc_ids:
        print(
            "[ERROR] No BioCyc_ID found in params. Please specify reaction IDs for the biological process."
        )
        return None

    if isinstance(biocyc_ids, str):
        biocyc_ids = [biocyc_ids]

    print(
        f"[INFO] Analyzing biological process '{process_name}' with {len(biocyc_ids)} reactions: {biocyc_ids}"
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

    # Build SQL query for efficient process flux calculation
    process_flux_sql, valid_biocyc_ids, process_flux_col = build_process_flux_sql(
        biocyc_ids,
        base_to_extended_mapping,
        all_reaction_ids,
        history_sql,
        process_name,
    )

    if not process_flux_sql or not valid_biocyc_ids:
        print("[ERROR] Could not build process flux calculation SQL")
        return None

    print(
        f"[INFO] Processing {len(valid_biocyc_ids)} valid BioCyc IDs for {process_name}"
    )

    # Execute the optimized SQL query
    try:
        df = conn.sql(process_flux_sql).pl()
        print(f"[INFO] Loaded data with {df.height} time steps")
    except Exception as e:
        print(f"[ERROR] Error executing process flux calculation SQL: {e}")
        return None

    if df.is_empty():
        print("[ERROR] No data found")
        return None

    # Filter by specified variants and generations if provided
    target_variants = params.get("variant", None)
    target_generation = params.get("generation", None)

    if target_variants is not None:
        print(f"[INFO] Filtering for variants: {target_variants}")
        df = df.filter(pl.col("variant").is_in(target_variants))

    if target_generation is not None:
        print(f"[INFO] Filtering for generations: {target_generation}")
        df = df.filter(pl.col("generation").is_in(target_generation))

    # Print variant information
    unique_variants = df["variant"].unique().to_list()
    print(f"[INFO] Found {len(unique_variants)} variants: {unique_variants}")

    # Calculate average process flux for each variant
    variant_averages = df.group_by("variant").agg(
        pl.col(process_flux_col).mean().alias("avg_process_flux")
    )

    print(f"[INFO] Average {process_name} flux by variant:")
    for row in variant_averages.iter_rows(named=True):
        print(f"  {row['variant']}: {row['avg_process_flux']:.6f} mmol/gDW/hr")

    # --------------------------- #
    # Create visualization
    def create_process_flux_chart():
        """Create a line chart for the biological process total flux."""

        # Select only needed columns
        chart_data = df.select(["time_min", "generation", "variant", process_flux_col])

        # Remove any null values
        chart_data = chart_data.filter(pl.col(process_flux_col).is_not_null())

        if chart_data.height == 0:
            print(f"[WARNING] No valid data for process {process_name}")
            return None

        # Main process flux line chart
        main_chart = (
            alt.Chart(chart_data)
            .mark_line(strokeWidth=2)
            .encode(
                x=alt.X("time_min:Q", title="Time (min)"),
                y=alt.Y(f"{process_flux_col}:Q", title="Process Flux (mmol/gDW/hr)"),
                color=alt.Color("variant:N", legend=alt.Legend(title="Variant")),
                tooltip=[
                    "time_min:Q",
                    f"{process_flux_col}:Q",
                    "variant:N",
                    "generation:N",
                ],
            )
        )

        # Create average lines for each variant
        avg_line_data = []
        for row in variant_averages.iter_rows(named=True):
            variant_name = row["variant"]
            avg_value = row["avg_process_flux"]
            avg_line_data.append(
                {
                    "variant": variant_name,
                    "avg_flux": avg_value,
                    "label": f"{variant_name} Avg: {avg_value:.4f}",
                }
            )

        avg_line_df = pd.DataFrame(avg_line_data)

        avg_lines = (
            alt.Chart(avg_line_df)
            .mark_rule(strokeDash=[5, 5], strokeWidth=2)
            .encode(
                y=alt.Y("avg_flux:Q"),
                color=alt.Color("variant:N", legend=None),
                tooltip=["label:N"],
            )
        )

        # Zero line for reference
        zero_line_data = pd.DataFrame({"zero": [0], "label": ["Zero"]})
        zero_line = (
            alt.Chart(zero_line_data)
            .mark_rule(color="gray", strokeDash=[2, 2], strokeWidth=1, opacity=0.7)
            .encode(y=alt.Y("zero:Q"), tooltip=["label:N"])
        )

        # Combine all elements
        combined_chart = (
            (main_chart + avg_lines + zero_line)
            .properties(
                title=f"Total Flux vs Time: {process_name}", width=800, height=400
            )
            .resolve_scale(y="shared", color="shared")
        )

        return combined_chart

    # --------------------------- #

    # Create the chart
    chart = create_process_flux_chart()
    if chart is None:
        print("[ERROR] Could not create chart")
        return None

    # Add overall title
    final_chart = chart.properties(
        title=alt.TitleParams(
            text=f"Biological Process Flux Analysis: {process_name} ({len(valid_biocyc_ids)} reactions)",
            fontSize=16,
            anchor="start",
        )
    )

    # Save the plot
    output_path = os.path.join(
        outdir, f"fba_process_flux_{process_name.replace(' ', '_').lower()}.html"
    )
    final_chart.save(output_path)
    print(f"[INFO] Saved visualization to: {output_path}")

    return final_chart
