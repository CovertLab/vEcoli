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

This script will view each BioCyc_ID as root name,
and find all reactions matching the specified BioCyc IDs,
including those with delimiters like '_', '[', '-', '/' after the root name,
and those with '(reverse)' for reverse reactions.
It will then calculate the net flux for each reaction as:
    net_flux = forward_flux - reverse_flux

For each reaction in params.BioCyc_ID, plot net flux vs time for all variants,
with different variants shown as different colored lines,
and with average net flux marked for each variant.
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
# Optimized helper functions


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


def calculate_net_flux_vectorized(flux_df, reaction_mappings):
    """
    Calculate net flux for all BioCyc IDs using vectorized operations.
    This is much faster than row-by-row calculation.

    Args:
        flux_df: Polars DataFrame with expanded flux columns
        reaction_mappings: Precomputed reaction mappings

    Returns:
        Polars DataFrame with net flux columns added
    """

    # Convert flux matrix to numpy array for faster computation
    flux_matrix = flux_df.select("listeners__fba_results__reaction_fluxes").to_numpy()

    # Stack all flux arrays into a 2D matrix (n_timepoints x n_reactions)
    flux_array = np.vstack([row[0] for row in flux_matrix])

    print(f"[INFO] Flux array shape: {flux_array.shape}")

    # Calculate net flux for each BioCyc ID
    net_flux_columns = {}

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

        # Calculate net flux
        net_flux = forward_flux - reverse_flux

        net_flux_col_name = f"{biocyc_id}_net_flux"
        net_flux_columns[net_flux_col_name] = net_flux

        print(
            f"[INFO] Calculated net flux for {biocyc_id}: avg = {net_flux.mean():.6f} mmol/gDW/hr"
        )

    # Add all net flux columns to the dataframe at once
    for col_name, values in net_flux_columns.items():
        flux_df = flux_df.with_columns(pl.Series(name=col_name, values=values))

    return flux_df, net_flux_columns


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
    """Visualize FBA reaction net fluxes over time for specified BioCyc reactions across multiple variants."""

    # Get BioCyc IDs from params
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

    # Required columns for the query (added variant)
    required_columns = [
        "time",
        "generation",
        "variant",
        "listeners__fba_results__reaction_fluxes",
    ]

    # Build SQL query (ordered by variant first, then generation, then time)
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
    target_variants = params.get(
        "variant", None
    )  # List of variant names or None for all

    target_generation = params.get(
        "generation", None
    )  # List of generations or None for all

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

    # Load the reaction IDs from the config - this is the array that maps to flux matrix columns
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

    print(f"[INFO] Processing {len(valid_biocyc_ids)} valid BioCyc IDs")

    # Calculate net flux using vectorized operations
    try:
        flux_df, net_flux_columns = calculate_net_flux_vectorized(df, reaction_mappings)
        print("[INFO] Successfully calculated all net fluxes")
    except Exception as e:
        print(f"[ERROR] Failed to calculate net fluxes: {e}")
        return None

    # Calculate average net fluxes for each reaction and variant combination
    avg_net_fluxes = {}
    for biocyc_id in valid_biocyc_ids:
        net_flux_col = f"{biocyc_id}_net_flux"
        if net_flux_col in flux_df.columns:
            # Calculate average net flux by variant
            variant_avgs = flux_df.group_by("variant").agg(
                pl.col(net_flux_col).mean().alias("avg_net_flux")
            )
            avg_net_fluxes[biocyc_id] = variant_avgs

    # ---------------------------------------------- #

    def create_net_flux_chart(biocyc_id, net_flux_col, variant_avgs):
        """Create a line chart for a single reaction net flux with average lines for each variant."""
        # Select only the columns we need to minimize data transfer
        data = flux_df.select(["time_min", "generation", "variant", net_flux_col])

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

        # Zero line for reference
        zero_line_data = pd.DataFrame({"zero": [0], "label": ["Zero"]})
        zero_line = (
            alt.Chart(zero_line_data)
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

    # ---------------------------------------------- #

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
    output_path = os.path.join(outdir, "union_fba_net_flux_multivariant_report.html")
    combined_plot.save(output_path)
    print(f"[INFO] Saved visualization to: {output_path}")

    # Also save a summary CSV with average net fluxes by variant and reaction details
    summary_data = []
    for biocyc_id in valid_biocyc_ids:
        mappings = reaction_mappings[biocyc_id]
        variant_avgs = avg_net_fluxes.get(biocyc_id)

        if variant_avgs is not None:
            for row in variant_avgs.iter_rows(named=True):
                variant_name = row["variant"]
                avg_flux = row["avg_net_flux"]

                summary_data.append(
                    {
                        "BioCyc_ID": biocyc_id,
                        "Variant": variant_name,
                        "Average_Net_Flux": avg_flux,
                        "Forward_Reactions": "; ".join(mappings["forward_reactions"]),
                        "Reverse_Reactions": "; ".join(mappings["reverse_reactions"]),
                        "Num_Forward": len(mappings["forward_reactions"]),
                        "Num_Reverse": len(mappings["reverse_reactions"]),
                    }
                )

    summary_df = pl.DataFrame(summary_data)
    summary_path = os.path.join(outdir, "union_fba_net_flux_multivariant_summary.csv")
    summary_df.write_csv(summary_path)
    print(f"[INFO] Saved net flux summary to: {summary_path}")

    # Save detailed flux data for further analysis (only net flux columns to save space)
    net_flux_cols = [f"{biocyc_id}_net_flux" for biocyc_id in valid_biocyc_ids]
    detailed_columns = ["time_min", "generation", "variant"] + net_flux_cols
    detailed_df = flux_df.select(detailed_columns)
    detailed_path = os.path.join(outdir, "union_fba_net_flux_multivariant_detailed.csv")
    detailed_df.write_csv(detailed_path)
    print(f"[INFO] Saved detailed net flux data to: {detailed_path}")

    return combined_plot
