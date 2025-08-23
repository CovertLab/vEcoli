"""
Visualize FBA reaction net fluxes over time for a single generation with specified time window.

You can specify the reactions to visualize using the 'BioCyc_ID' parameter in params:
    "single_gen_fba_flux": {
        "BioCyc_ID": ["Name1", "Name2", ...],
        # Optional: specify time window to analyze
        # If not specified, all time points will be used
        "time_window": [start_time, end_time]  # in seconds
        }

This script uses the base reaction ID to extended reaction mapping to efficiently
find forward and reverse reactions, then calculates net flux using SQL for
optimal memory usage and performance.

For each reaction in params.BioCyc_ID, plot net flux vs time with each reaction having its own subplot.
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
    """Visualize FBA reaction net fluxes over time for a single generation."""

    # Get parameters
    biocyc_ids = params.get("BioCyc_ID", [])
    if not biocyc_ids:
        print(
            "[ERROR] No BioCyc_ID found in params. Please specify reaction IDs to visualize."
        )
        return None

    if isinstance(biocyc_ids, str):
        biocyc_ids = [biocyc_ids]

    print(f"[INFO] Analyzing for {len(biocyc_ids)} reactions: {biocyc_ids}")

    # Get time window (optional)
    time_window = params.get("time_window", None)
    if time_window is not None:
        if len(time_window) != 2:
            print(
                "[ERROR] time_window must be a list of [start_time, end_time] in seconds."
            )
            return None
        start_time, end_time = time_window
        print(f"[INFO] Time window: {start_time}s to {end_time}s")
    else:
        print("[INFO] Using full time range")

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

    # Apply time window filter if specified
    if time_window is not None:
        start_time_min = start_time / 60
        end_time_min = end_time / 60
        df = df.filter(
            (pl.col("time_min") >= start_time_min)
            & (pl.col("time_min") <= end_time_min)
        )
        print(
            f"[INFO] Filtered to time window: {start_time_min:.2f} - {end_time_min:.2f} minutes"
        )

    if df.height == 0:
        print("[ERROR] No data points in specified time window")
        return None

    print(f"[INFO] Final dataset has {df.height} time steps")

    # Calculate statistics for each reaction
    flux_stats = {}
    for biocyc_id in valid_biocyc_ids:
        net_flux_col = f"{biocyc_id}_net_flux"
        if net_flux_col in df.columns:
            stats = {
                "avg": df[net_flux_col].mean(),
                "min": df[net_flux_col].min(),
                "max": df[net_flux_col].max(),
                "std": df[net_flux_col].std(),
            }
            flux_stats[biocyc_id] = stats

            print(
                f"[INFO] Net flux stats for {biocyc_id}: "
                f"avg={stats['avg']:.6f}, min={stats['min']:.6f}, "
                f"max={stats['max']:.6f} (mmol/gDW/hr)"
            )

    # --------------------------- #

    # Create visualization functions
    def create_individual_net_flux_chart(biocyc_id, net_flux_col, stats):
        """Create an individual chart for a single reaction net flux."""

        # Check if the column exists in dataframe
        if net_flux_col not in df.columns:
            print(f"[WARNING] Column {net_flux_col} not found in dataframe")
            return None

        # Select only the columns we need
        data_pl = df.select(["time_min", net_flux_col])

        # Remove any null values using Polars syntax
        data_pl = data_pl.filter(pl.col(net_flux_col).is_not_null())

        if data_pl.height == 0:
            print(f"[WARNING] No valid data for reaction {biocyc_id}")
            return None

        # Convert to pandas for Altair
        data = data_pl.to_pandas()

        # Main net flux line chart
        net_flux_chart = (
            alt.Chart(data)
            .mark_line(strokeWidth=2, color="steelblue")
            .encode(
                x=alt.X("time_min:Q", title="Time (min)"),
                y=alt.Y(f"{net_flux_col}:Q", title="Net Flux (mmol/gDW/hr)"),
                tooltip=["time_min:Q", f"{net_flux_col}:Q"],
            )
        )

        # Average net flux horizontal line
        avg_line = (
            alt.Chart(
                pd.DataFrame(
                    {
                        "avg_net_flux": [stats["avg"]],
                        "label": [f"Avg: {stats['avg']:.4f}"],
                    }
                )
            )
            .mark_rule(color="red", strokeDash=[5, 5], strokeWidth=2)
            .encode(y=alt.Y("avg_net_flux:Q"), tooltip=["label:N"])
        )

        # Zero line for reference
        zero_line = (
            alt.Chart(pd.DataFrame({"zero": [0], "label": ["Zero"]}))
            .mark_rule(color="gray", strokeDash=[2, 2], strokeWidth=1, opacity=0.7)
            .encode(y=alt.Y("zero:Q"), tooltip=["label:N"])
        )

        # Combine all elements
        combined_chart = (
            (net_flux_chart + avg_line + zero_line)
            .properties(
                title=f"{biocyc_id} (Avg={stats['avg']:.4f}, Range=[{stats['min']:.4f}, {stats['max']:.4f}])",
                width=600,
                height=200,
            )
            .resolve_scale(y="shared")
        )

        return combined_chart

    # --------------------------- #

    # Create individual charts for each reaction
    charts = []

    for biocyc_id in valid_biocyc_ids:
        net_flux_col = f"{biocyc_id}_net_flux"
        stats = flux_stats.get(biocyc_id, {})
        if stats:  # Only create chart if we have valid stats
            chart = create_individual_net_flux_chart(biocyc_id, net_flux_col, stats)
            if chart is not None:
                charts.append(chart)

    if not charts:
        print("[ERROR] No valid charts could be created")
        return None

    # Arrange charts vertically with shared x-axis
    if len(charts) == 1:
        combined_plot = charts[0]
    else:
        combined_plot = alt.vconcat(*charts).resolve_scale(x="shared", y="independent")

    # Add overall title
    time_window_str = (
        f" (Time: {time_window[0] / 60:.1f}-{time_window[1] / 60:.1f} min)"
        if time_window
        else ""
    )
    combined_plot = combined_plot.properties(
        title=alt.TitleParams(
            text=f"Single Generation FBA Net Fluxes{time_window_str}",
            fontSize=16,
            anchor="start",
        )
    )

    # Save the plot
    output_path = os.path.join(outdir, "single_generation_fba_net_flux.html")
    combined_plot.save(output_path)
    print(f"[INFO] Saved visualization to: {output_path}")

    return combined_plot
