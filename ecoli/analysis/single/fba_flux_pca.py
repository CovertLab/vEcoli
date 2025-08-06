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


# ----------------------------------------- #
# Helper functions
def find_matching_reactions(reaction_ids, reaction_name, reverse_flag=False):
    """
    Find all reaction IDs that match the given reaction name pattern.
    This is done once per BioCyc ID to avoid repeated string matching.

    Args:
        reaction_ids (list): List of all reaction IDs in the model
        reaction_name (str): Root reaction name to search for
        reverse_flag (bool): If True, search for reverse reactions;
                           If False, search for forward reactions

    Returns:
        list: List of matching reaction IDs and their indices
    """

    matching_reactions = []

    if reverse_flag:
        # For reverse reactions, we look for reactions with "(reverse)" suffix
        # Step 1: Try to find exact reverse name
        reverse_name = reaction_name + " (reverse)"
        if reverse_name in reaction_ids:
            idx = reaction_ids.index(reverse_name)
            matching_reactions.append((reverse_name, idx))

        # Step 2: Search for extended reverse names with delimiters
        delimiters = ["_", "[", "-", "/"]
        for delimiter in delimiters:
            extend_name = reaction_name + delimiter
            for idx, reaction_id in enumerate(reaction_ids):
                if (
                    extend_name in reaction_id
                    and "(reverse)" in reaction_id
                    and reaction_id not in [r[0] for r in matching_reactions]
                ):
                    matching_reactions.append((reaction_id, idx))

    else:
        # For forward reactions, we look for reactions WITHOUT "(reverse)" suffix
        # Step 1: Try to find exact root name (forward)
        if reaction_name in reaction_ids and "(reverse)" not in reaction_name:
            idx = reaction_ids.index(reaction_name)
            matching_reactions.append((reaction_name, idx))

        # Step 2: Search for extended forward names with delimiters
        delimiters = ["_", "[", "-", "/"]
        for delimiter in delimiters:
            extend_name = reaction_name + delimiter
            for idx, reaction_id in enumerate(reaction_ids):
                if (
                    extend_name in reaction_id
                    and "(reverse)" not in reaction_id
                    and reaction_id not in [r[0] for r in matching_reactions]
                ):
                    matching_reactions.append((reaction_id, idx))

    return matching_reactions


def precompute_reaction_mappings(reaction_ids, biocyc_ids):
    """
    Precompute all reaction mappings for forward and reverse reactions.
    This avoids repeated string matching during flux calculation.

    Args:
        reaction_ids (list): List of all reaction IDs in the model
        biocyc_ids (list): List of BioCyc IDs to analyze

    Returns:
        dict: Mapping of BioCyc ID to forward/reverse reaction indices
    """

    reaction_mappings = {}

    for biocyc_id in biocyc_ids:
        print(f"[INFO] Preprocessing reaction mappings for {biocyc_id}...")

        # Find forward reactions
        forward_reactions = find_matching_reactions(
            reaction_ids, biocyc_id, reverse_flag=False
        )
        forward_indices = [idx for _, idx in forward_reactions]

        # Find reverse reactions
        reverse_reactions = find_matching_reactions(
            reaction_ids, biocyc_id, reverse_flag=True
        )
        reverse_indices = [idx for _, idx in reverse_reactions]

        reaction_mappings[biocyc_id] = {
            "forward_indices": forward_indices,
            "reverse_indices": reverse_indices,
            "forward_reactions": [name for name, _ in forward_reactions],
            "reverse_reactions": [name for name, _ in reverse_reactions],
        }

        print(
            f"[INFO] Found {len(forward_reactions)} forward and {len(reverse_reactions)} reverse reactions for {biocyc_id}"
        )

        if not forward_reactions and not reverse_reactions:
            print(f"[WARNING] No reactions found for {biocyc_id}")

    return reaction_mappings


def calculate_net_flux_matrix(flux_df, reaction_mappings, biocyc_ids):
    """
    Calculate net flux matrix for PCA analysis.
    Each row represents a timestep, each column represents a reaction's net flux.

    Args:
        flux_df: Polars DataFrame with flux data
        reaction_mappings: Precomputed reaction mappings
        biocyc_ids: List of BioCyc IDs to include in the matrix

    Returns:
        numpy.ndarray: Net flux matrix (n_timepoints x n_reactions)
        list: List of valid BioCyc IDs (columns of the matrix)
    """

    # Convert flux matrix to numpy array for faster computation
    flux_matrix = flux_df.select("listeners__fba_results__reaction_fluxes").to_numpy()

    # Stack all flux arrays into a 2D matrix (n_timepoints x n_reactions)
    flux_array = np.vstack([row[0] for row in flux_matrix])

    print(f"[INFO] Flux array shape: {flux_array.shape}")

    # Calculate net flux for each BioCyc ID
    net_flux_vectors = []
    valid_biocyc_ids = []

    for biocyc_id in biocyc_ids:
        if biocyc_id not in reaction_mappings:
            continue

        mappings = reaction_mappings[biocyc_id]
        forward_indices = mappings["forward_indices"]
        reverse_indices = mappings["reverse_indices"]

        # Skip if no reactions found
        if not forward_indices and not reverse_indices:
            print(f"[WARNING] No reactions found for {biocyc_id}, skipping...")
            continue

        # Calculate forward flux sum
        if forward_indices:
            forward_flux = flux_array[:, forward_indices].sum(axis=1)
        else:
            forward_flux = np.zeros(flux_array.shape[0])

        # Calculate reverse flux sum
        if reverse_indices:
            reverse_flux = flux_array[:, reverse_indices].sum(axis=1)
        else:
            reverse_flux = np.zeros(flux_array.shape[0])

        # Calculate net flux
        net_flux = forward_flux - reverse_flux

        net_flux_vectors.append(net_flux)
        valid_biocyc_ids.append(biocyc_id)

        print(
            f"[INFO] Net flux stats for {biocyc_id}: mean={net_flux.mean():.6f}, std={net_flux.std():.6f} (mmol/gDW/hr)"
        )

    if not net_flux_vectors:
        return None, []

    # Stack net flux vectors to form the matrix (n_timepoints x n_reactions)
    net_flux_matrix = np.column_stack(net_flux_vectors)

    print(f"[INFO] Net flux matrix shape for PCA: {net_flux_matrix.shape}")

    return net_flux_matrix, valid_biocyc_ids


def perform_pca_analysis(net_flux_matrix, valid_biocyc_ids, time_points):
    """
    Perform PCA analysis on the net flux matrix.

    Args:
        net_flux_matrix: Net flux matrix (n_timepoints x n_reactions)
        valid_biocyc_ids: List of BioCyc IDs (reaction names)
        time_points: Time points corresponding to each row

    Returns:
        dict: PCA results including transformed data, components, and explained variance
    """

    print("[INFO] Performing PCA analysis...")

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
    """Visualize FBA flux dynamics using PCA trajectory analysis."""

    # Get BioCyc IDs from params
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

    # Required columns for the query
    required_columns = [
        "time",
        "listeners__fba_results__reaction_fluxes",
    ]

    # Build SQL query for single generation
    sql = f"""
    SELECT
        {", ".join(required_columns)}
    FROM ({history_sql})
    ORDER BY time
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

    # Convert time to minutes
    if "time" in df.columns:
        df = df.with_columns((pl.col("time") / 60).alias("time_min"))

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

    print(f"[INFO] Loaded data with {df.height} time steps")

    if df.height < 3:
        print("[ERROR] Need at least 3 time points for meaningful PCA analysis")
        return None

    # Load the reaction IDs from the config
    try:
        reaction_ids = field_metadata(
            conn, config_sql, "listeners__fba_results__reaction_fluxes"
        )
        print(f"[INFO] Total reactions in sim_data: {len(reaction_ids)}")
    except Exception as e:
        print(f"[ERROR] Error loading reaction IDs: {e}")
        return None

    # Precompute reaction mappings for all BioCyc IDs
    reaction_mappings = precompute_reaction_mappings(reaction_ids, biocyc_ids)

    # Calculate net flux matrix for PCA
    net_flux_matrix, valid_biocyc_ids = calculate_net_flux_matrix(
        df, reaction_mappings, biocyc_ids
    )

    if net_flux_matrix is None or len(valid_biocyc_ids) < 2:
        print("[ERROR] Need at least 2 valid reactions for PCA analysis")
        return None

    # Get time points for trajectory
    time_points = df.select("time_min").to_numpy().flatten()

    # Perform PCA analysis
    pca_results = perform_pca_analysis(net_flux_matrix, valid_biocyc_ids, time_points)

    # ---------------------------------------------- #
    # Create visualizations

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
            title=f"PCA Trajectory ({pca_results['cumulative_variance']:.1%} variance explained",
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

    # ---------------------------------------------- #

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
            text=f"FBA Flux PCA Trajectory Analysis - {time_window_str}",
            fontSize=16,
            anchor="start",
        )
    )

    # Save the plot
    output_path = os.path.join(outdir, "single_fba_flux_pca.html")
    combined_plot.save(output_path)
    print(f"[INFO] Saved PCA visualization to: {output_path}")

    return combined_plot
