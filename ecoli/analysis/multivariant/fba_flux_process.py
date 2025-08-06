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

This script will:
1. Find all reactions matching each specified BioCyc ID (including variants with delimiters)
2. Calculate net flux for each reaction (forward_flux - reverse_flux)
3. Sum all net fluxes to get the total biological process flux
4. Plot total process flux vs time for all variants
"""

import altair as alt
import os
from typing import Any
import numpy as np

import polars as pl
from duckdb import DuckDBPyConnection
import pandas as pd

from ecoli.library.parquet_emitter import field_metadata

# ----------------------------------------- #
# Helper functions (reused from your existing code)


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


def calculate_process_flux_vectorized(flux_df, reaction_mappings, process_name):
    """
    Calculate total biological process flux by summing net fluxes from all reactions.

    Args:
        flux_df: Polars DataFrame with expanded flux columns
        reaction_mappings: Precomputed reaction mappings for all BioCyc IDs
        process_name: Name of the biological process

    Returns:
        Polars DataFrame with process flux column added
    """

    # Convert flux matrix to numpy array for faster computation
    flux_matrix = flux_df.select("listeners__fba_results__reaction_fluxes").to_numpy()

    # Stack all flux arrays into a 2D matrix (n_timepoints x n_reactions)
    flux_array = np.vstack([row[0] for row in flux_matrix])

    print(f"[INFO] Flux array shape: {flux_array.shape}")

    # Initialize total process flux array
    total_process_flux = np.zeros(flux_array.shape[0])

    # Track individual reaction contributions for detailed analysis
    reaction_contributions = {}

    # Sum net fluxes from all reactions in the process
    for biocyc_id, mappings in reaction_mappings.items():
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

        # Calculate net flux for this BioCyc ID
        net_flux = forward_flux - reverse_flux

        # Add to total process flux
        total_process_flux += net_flux

        # Store individual contribution
        reaction_contributions[biocyc_id] = net_flux

        print(
            f"[INFO] {biocyc_id} contribution to {process_name}: avg = {net_flux.mean():.6f} mmol/gDW/hr"
        )

    # Add process flux column to dataframe
    process_flux_col_name = f"{process_name.replace(' ', '_')}_total_flux"
    flux_df = flux_df.with_columns(
        pl.Series(name=process_flux_col_name, values=total_process_flux)
    )

    print(
        f"[INFO] Total {process_name} flux: avg = {total_process_flux.mean():.6f} mmol/gDW/hr"
    )

    return flux_df, process_flux_col_name, reaction_contributions


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

    # Required columns for the query
    required_columns = [
        "time",
        "generation",
        "variant",
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
    target_variants = params.get("variant", None)
    target_generation = params.get("generation", None)

    # Filter by specified variants and generations
    if target_variants is not None:
        print(f"[INFO] Target variants: {target_variants}")
        df = df.filter(pl.col("variant").is_in(target_variants))

    if target_generation is not None:
        print(f"[INFO] Target generations: {target_generation}")
        df = df.filter(pl.col("generation").is_in(target_generation))

    # Convert time to minutes
    if "time" in df.columns:
        df = df.with_columns((pl.col("time") / 60).alias("time_min"))

    print(f"[INFO] Loaded data with {df.height} time steps")

    # Print variant information
    unique_variants = df["variant"].unique().to_list()
    print(f"[INFO] Found {len(unique_variants)} variants: {unique_variants}")

    # Load reaction IDs from config
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

    # Filter out BioCyc IDs that have no matching reactions
    valid_biocyc_ids = [
        biocyc_id
        for biocyc_id, mappings in reaction_mappings.items()
        if mappings["forward_indices"] or mappings["reverse_indices"]
    ]

    if not valid_biocyc_ids:
        print("[ERROR] No valid reactions found for any of the specified BioCyc IDs")
        return None

    print(
        f"[INFO] Processing {len(valid_biocyc_ids)} valid BioCyc IDs for {process_name}"
    )

    # Filter reaction mappings to only include valid BioCyc IDs
    valid_reaction_mappings = {
        biocyc_id: reaction_mappings[biocyc_id] for biocyc_id in valid_biocyc_ids
    }

    # Calculate total process flux
    try:
        flux_df, process_flux_col, reaction_contributions = (
            calculate_process_flux_vectorized(df, valid_reaction_mappings, process_name)
        )
        print(f"[INFO] Successfully calculated total flux for {process_name}")
    except Exception as e:
        print(f"[ERROR] Failed to calculate process flux: {e}")
        return None

    # Calculate average process flux for each variant
    variant_averages = flux_df.group_by("variant").agg(
        pl.col(process_flux_col).mean().alias("avg_process_flux")
    )

    # ------------------------------------ #

    # Create the main line chart
    def create_process_flux_chart():
        """Create a line chart for the biological process total flux."""

        # Select only needed columns
        chart_data = flux_df.select(
            ["time_min", "generation", "variant", process_flux_col]
        )

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

    # ------------------------------------ #

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
