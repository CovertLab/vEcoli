"""
Visualize FBA reaction flux dynamics using PCA trajectory analysis.

You can specify the reactions and time window using parameters:
    "fba_flux_pca": {
        # Required: specify BioCyc IDs of reactions to analyze
        "BioCyc_ID": ["Name1", "Name2", ...],  # Reactions of interest
        # Optional: specify time window to analyze
        # If not specified, all time points will be used
        "time_window": [start_time, end_time]  # in seconds
        }

This script uses the base reaction ID to extended reaction mapping to efficiently
find forward and reverse reactions, then calculates net flux using SQL for
optimal memory usage and performance.

For the specified reactions, each timestep forms a vector of net flux values.
PCA is applied to reduce dimensionality to 2D and visualize the metabolic trajectory.
"""

import altair as alt
import os
from typing import Any
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

import polars as pl
from duckdb import DuckDBPyConnection
import pandas as pd

from ecoli.library.parquet_emitter import field_metadata
from ecoli.analysis.utils import (
    create_base_to_extended_mapping,
    build_flux_calculation_sql,
)


def perform_pca_analysis(net_flux_df, valid_biocyc_ids, time_points):
    """
    Perform PCA analysis on the net flux data.

    Args:
        net_flux_df: Polars DataFrame with net flux columns
        valid_biocyc_ids: List of BioCyc IDs (reaction names)
        time_points: Time points corresponding to each row

    Returns:
        dict: PCA results including transformed data, components, and explained variance
    """

    print("[INFO] Performing PCA analysis...")

    # Select only the net flux columns for PCA
    net_flux_cols = [f"{biocyc_id}_net_flux" for biocyc_id in valid_biocyc_ids]

    # Check that all columns exist
    missing_cols = [col for col in net_flux_cols if col not in net_flux_df.columns]
    if missing_cols:
        print(f"[ERROR] Missing net flux columns: {missing_cols}")
        return None

    # Extract net flux matrix (n_timepoints x n_reactions)
    net_flux_matrix = net_flux_df.select(net_flux_cols).to_numpy()

    print(f"[INFO] Net flux matrix shape for PCA: {net_flux_matrix.shape}")

    # Print basic statistics for each reaction
    for i, biocyc_id in enumerate(valid_biocyc_ids):
        flux_values = net_flux_matrix[:, i]
        print(
            f"[INFO] Net flux stats for {biocyc_id}: "
            f"mean={flux_values.mean():.6f}, std={flux_values.std():.6f} (mmol/gDW/hr)"
        )

    # Standardize the data (important for PCA)
    scaler = StandardScaler()
    scaled_matrix = scaler.fit_transform(net_flux_matrix)

    # Perform PCA to reduce to 2 dimensions
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(scaled_matrix)

    # Calculate explained variance
    explained_variance = pca.explained_variance_ratio_
    cumulative_variance = np.cumsum(explained_variance)

    print(
        f"[INFO] PCA explained variance: PC1={explained_variance[0]:.3f}, PC2={explained_variance[1]:.3f}"
    )
    print(f"[INFO] Cumulative explained variance: {cumulative_variance[1]:.3f}")

    # Get component loadings (contribution of each reaction to each PC)
    components = pca.components_

    # Create loading data for visualization
    loadings_data = []
    for i, biocyc_id in enumerate(valid_biocyc_ids):
        loadings_data.append(
            {
                "BioCyc_ID": biocyc_id,
                "PC1_loading": components[0, i],
                "PC2_loading": components[1, i],
                "PC1_abs": abs(components[0, i]),
                "PC2_abs": abs(components[1, i]),
            }
        )

    pca_results = {
        "pca_coordinates": pca_result,
        "time_points": time_points,
        "explained_variance": explained_variance,
        "cumulative_variance": cumulative_variance[1],
        "components": components,
        "loadings_data": loadings_data,
        "scaler": scaler,
        "pca_model": pca,
    }

    return pca_results


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
    """Visualize FBA flux dynamics using PCA trajectory analysis."""

    # Get parameters
    biocyc_ids = params.get("BioCyc_ID", [])
    if not biocyc_ids:
        print(
            "[ERROR] No BioCyc_ID found in params. Please specify reaction IDs for PCA analysis."
        )
        return None

    if isinstance(biocyc_ids, str):
        biocyc_ids = [biocyc_ids]

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

    print(f"[INFO] PCA analysis with {len(biocyc_ids)} reactions: {biocyc_ids}")

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

    if len(valid_biocyc_ids) < 2:
        print("[ERROR] Need at least 2 valid reactions for PCA analysis")
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

    if df.height < 3:
        print("[ERROR] Need at least 3 time points for meaningful PCA analysis")
        return None

    print(f"[INFO] Final dataset has {df.height} time steps")

    # Get time points for trajectory
    time_points = df.select("time_min").to_numpy().flatten()

    # Perform PCA analysis
    pca_results = perform_pca_analysis(df, valid_biocyc_ids, time_points)

    if pca_results is None:
        print("[ERROR] PCA analysis failed")
        return None

    # Create visualization functions
    def create_pca_trajectory_chart(pca_results):
        """Create PCA trajectory visualization."""

        # Prepare trajectory data
        pca_coords = pca_results["pca_coordinates"]
        time_points = pca_results["time_points"]

        trajectory_data = pd.DataFrame(
            {
                "PC1": pca_coords[:, 0],
                "PC2": pca_coords[:, 1],
                "Time_min": time_points,
                "Point_idx": range(len(time_points)),
            }
        )

        # Create trajectory line
        trajectory_line = (
            alt.Chart(trajectory_data)
            .mark_line(strokeWidth=2, color="steelblue")
            .encode(
                x=alt.X(
                    "PC1:Q",
                    title=f"PC1 ({pca_results['explained_variance'][0]:.1%} variance)",
                ),
                y=alt.Y(
                    "PC2:Q",
                    title=f"PC2 ({pca_results['explained_variance'][1]:.1%} variance)",
                ),
                order="Point_idx:O",
            )
        )

        # Add points with time color coding
        trajectory_points = (
            alt.Chart(trajectory_data)
            .mark_circle(size=60, stroke="white", strokeWidth=1)
            .encode(
                x=alt.X("PC1:Q"),
                y=alt.Y("PC2:Q"),
                color=alt.Color(
                    "Time_min:Q", title="Time (min)", scale=alt.Scale(scheme="viridis")
                ),
                tooltip=["Time_min:Q", "PC1:Q", "PC2:Q"],
            )
        )

        # Mark start and end points
        start_point = (
            alt.Chart(trajectory_data.head(1))
            .mark_circle(size=100, color="green", stroke="white", strokeWidth=2)
            .encode(x=alt.X("PC1:Q"), y=alt.Y("PC2:Q"), tooltip=alt.value("Start"))
        )

        end_point = (
            alt.Chart(trajectory_data.tail(1))
            .mark_circle(size=100, color="red", stroke="white", strokeWidth=2)
            .encode(x=alt.X("PC1:Q"), y=alt.Y("PC2:Q"), tooltip=alt.value("End"))
        )

        # Combine all elements
        pca_chart = (
            trajectory_line + trajectory_points + start_point + end_point
        ).properties(
            title=f"PCA Trajectory ({pca_results['cumulative_variance']:.1%} variance explained)",
            width=500,
            height=400,
        )

        return pca_chart

    def create_loadings_chart(pca_results):
        """Create PCA loadings visualization."""

        loadings_df = pd.DataFrame(pca_results["loadings_data"])

        # Create biplot showing variable loadings
        loadings_chart = (
            alt.Chart(loadings_df)
            .mark_circle(size=100, stroke="black", strokeWidth=1)
            .encode(
                x=alt.X("PC1_loading:Q", title="PC1 Loading"),
                y=alt.Y("PC2_loading:Q", title="PC2 Loading"),
                color=alt.Color("BioCyc_ID:N", title="Reaction"),
                tooltip=["BioCyc_ID:N", "PC1_loading:Q", "PC2_loading:Q"],
            )
        )

        # Add reaction labels
        loadings_text = (
            alt.Chart(loadings_df)
            .mark_text(dx=10, dy=-10, fontSize=10)
            .encode(
                x=alt.X("PC1_loading:Q"), y=alt.Y("PC2_loading:Q"), text="BioCyc_ID:N"
            )
        )

        # Add reference lines
        zero_line_x = (
            alt.Chart(pd.DataFrame({"x": [0]}))
            .mark_rule(color="gray", strokeDash=[2, 2])
            .encode(x="x:Q")
        )
        zero_line_y = (
            alt.Chart(pd.DataFrame({"y": [0]}))
            .mark_rule(color="gray", strokeDash=[2, 2])
            .encode(y="y:Q")
        )

        combined_loadings = (
            zero_line_x + zero_line_y + loadings_chart + loadings_text
        ).properties(
            title="PCA Loadings - Reaction Contributions", width=500, height=400
        )

        return combined_loadings

    # Create visualizations
    trajectory_chart = create_pca_trajectory_chart(pca_results)
    loadings_chart = create_loadings_chart(pca_results)

    # Combine charts side by side
    combined_plot = alt.hconcat(trajectory_chart, loadings_chart).resolve_scale(
        color="independent"
    )

    # Add overall title
    time_window_str = (
        f" (Time: {time_window[0] / 60:.1f}-{time_window[1] / 60:.1f} min)"
        if time_window
        else ""
    )
    combined_plot = combined_plot.properties(
        title=alt.TitleParams(
            text=f"FBA Flux PCA Trajectory Analysis{time_window_str}",
            fontSize=16,
            anchor="start",
        )
    )

    # Save the plot
    output_path = os.path.join(outdir, "fba_flux_pca_trajectory.html")
    combined_plot.save(output_path)
    print(f"[INFO] Saved PCA visualization to: {output_path}")

    return combined_plot
