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

This script will view each BioCyc_ID as root name,
and find all reactions matching the specified BioCyc IDs,
including those with delimiters like '_', '[', '-', '/' after the root name,
and those with '(reverse)' for reverse reactions.
It will then calculate the net flux for each reaction as:
    net_flux = forward_flux - reverse_flux
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


def calculate_net_flux_optimized(flux_df, reaction_mappings):
    """
    Calculate net flux for all BioCyc IDs using optimized operations.
    Only extracts and processes the reaction columns we actually need.

    Args:
        flux_df: Polars DataFrame with expanded flux columns
        reaction_mappings: Precomputed reaction mappings

    Returns:
        Polars DataFrame with net flux columns added
    """

    print("[INFO] Starting optimized net flux calculation...")

    # Get all unique reaction indices we need (to minimize memory usage)
    all_needed_indices = set()
    for mappings in reaction_mappings.values():
        all_needed_indices.update(mappings["forward_indices"])
        all_needed_indices.update(mappings["reverse_indices"])

    needed_indices = sorted(list(all_needed_indices))
    print(
        f"[INFO] Only processing {len(needed_indices)} out of {len(flux_df)} total reactions"
    )

    # Convert flux matrix to numpy array, but only extract needed columns
    flux_matrix = flux_df.select("listeners__fba_results__reaction_fluxes").to_numpy()
    flux_array = np.vstack([row[0] for row in flux_matrix])

    # Extract only the columns we need
    flux_array_subset = flux_array[:, needed_indices]

    # Create mapping from original indices to subset indices
    index_mapping = {
        orig_idx: new_idx for new_idx, orig_idx in enumerate(needed_indices)
    }

    print(f"[INFO] Reduced flux array shape: {flux_array_subset.shape}")

    # Calculate net flux for each BioCyc ID
    net_flux_data = {}

    for biocyc_id, mappings in reaction_mappings.items():
        forward_indices = mappings["forward_indices"]
        reverse_indices = mappings["reverse_indices"]

        # Skip if no reactions found
        if not forward_indices and not reverse_indices:
            print(f"[WARNING] No reactions found for {biocyc_id}, skipping...")
            continue

        # Map original indices to subset indices
        forward_subset_indices = [
            index_mapping[idx] for idx in forward_indices if idx in index_mapping
        ]
        reverse_subset_indices = [
            index_mapping[idx] for idx in reverse_indices if idx in index_mapping
        ]

        # Calculate forward flux sum
        if forward_subset_indices:
            forward_flux = flux_array_subset[:, forward_subset_indices].sum(axis=1)
        else:
            forward_flux = np.zeros(flux_array_subset.shape[0])

        # Calculate reverse flux sum
        if reverse_subset_indices:
            reverse_flux = flux_array_subset[:, reverse_subset_indices].sum(axis=1)
        else:
            reverse_flux = np.zeros(flux_array_subset.shape[0])

        # Calculate net flux
        net_flux = forward_flux - reverse_flux
        net_flux_data[biocyc_id] = net_flux

        print(
            f"[INFO] Calculated net flux for {biocyc_id}: avg = {net_flux.mean():.6f} mmol/gDW/hr"
        )

    # Add all net flux columns to the dataframe at once
    for biocyc_id, net_flux_values in net_flux_data.items():
        net_flux_col_name = f"{biocyc_id}_net_flux"
        flux_df = flux_df.with_columns(
            pl.Series(name=net_flux_col_name, values=net_flux_values)
        )

    return flux_df, net_flux_data


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
    """Visualize FBA reaction net fluxes: rows=variants, columns=reactions, colors=generations."""

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

    # Required columns for the query
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
    target_variants = params.get("variant", None)
    target_generations = params.get("generation", None)

    # Filter by specified variants
    if target_variants is not None:
        print(f"[INFO] Target variants: {target_variants}")
        df = df.filter(pl.col("variant").is_in(target_variants))

    # Filter by specified generations
    if target_generations is not None:
        print(f"[INFO] Target generations: {target_generations}")
        df = df.filter(pl.col("generation").is_in(target_generations))

    # Convert time to minutes
    if "time" in df.columns:
        df = df.with_columns((pl.col("time") / 60).alias("time_min"))

    print(f"[INFO] Loaded data with {df.height} time steps")

    # Print variant and generation information
    unique_variants = sorted(df["variant"].unique().to_list())
    unique_generations = sorted(df["generation"].unique().to_list())
    print(f"[INFO] Found {len(unique_variants)} variants: {unique_variants}")
    print(f"[INFO] Found {len(unique_generations)} generations: {unique_generations}")

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

    # Calculate net flux using optimized operations
    try:
        flux_df, net_flux_data = calculate_net_flux_optimized(df, reaction_mappings)
        print("[INFO] Successfully calculated all net fluxes")
    except Exception as e:
        print(f"[ERROR] Failed to calculate net fluxes: {e}")
        return None

    # Calculate average net fluxes for each variant-reaction-generation combination
    avg_data = []
    for biocyc_id in valid_biocyc_ids:
        net_flux_col = f"{biocyc_id}_net_flux"
        if net_flux_col in flux_df.columns:
            # Calculate average net flux by variant and generation
            variant_gen_avgs = flux_df.group_by(["variant", "generation"]).agg(
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

    # ---------------------------------------------- #

    def create_subplot_chart(variant, biocyc_id):
        """Create a single subplot for a specific variant-reaction combination."""
        net_flux_col = f"{biocyc_id}_net_flux"

        # Filter data for this variant and reaction
        subplot_data = (
            flux_df.filter(pl.col("variant") == variant)
            .select(["time_min", "generation", net_flux_col])
            .filter(pl.col(net_flux_col).is_not_null())
        )

        if subplot_data.height == 0:
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

        # Add y-axis label only for leftmost column
        if biocyc_id == valid_biocyc_ids[0]:
            combined = combined.properties(width=200, height=150)
        else:
            combined = combined.properties(width=200, height=150)

        return combined

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
                empty_chart = alt.Chart(pd.DataFrame({"x": [0], "y": [0]})).mark_point(
                    opacity=0
                )
                variant_row.append(empty_chart)

        if variant_row:
            # Add variant label on the left
            variant_label = (
                alt.Chart(pd.DataFrame({"label": [f"variant {variant}"]}))
                .mark_text(
                    align="center", baseline="middle", fontSize=12, fontWeight="bold"
                )
                .encode(text="label:N")
                .properties(width=80, height=150)
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
    output_path = os.path.join(outdir, "separate_fba_net_flux_grid_report.html")
    combined_plot.save(output_path)
    print(f"[INFO] Saved visualization to: {output_path}")

    # Save summary CSV with average net fluxes
    summary_data = []
    for biocyc_id in valid_biocyc_ids:
        mappings = reaction_mappings[biocyc_id]

        # Get averages for this reaction across all variants and generations
        reaction_avgs = avg_df.filter(pl.col("biocyc_id") == biocyc_id)

        for row in reaction_avgs.iter_rows(named=True):
            summary_data.append(
                {
                    "BioCyc_ID": biocyc_id,
                    "Variant": row["variant"],
                    "Generation": row["generation"],
                    "Average_Net_Flux": row["avg_net_flux"],
                    "Forward_Reactions": "; ".join(mappings["forward_reactions"]),
                    "Reverse_Reactions": "; ".join(mappings["reverse_reactions"]),
                    "Num_Forward": len(mappings["forward_reactions"]),
                    "Num_Reverse": len(mappings["reverse_reactions"]),
                }
            )

    summary_df = pl.DataFrame(summary_data)
    summary_path = os.path.join(outdir, "separate_fba_net_flux_grid_summary.csv")
    summary_df.write_csv(summary_path)
    print(f"[INFO] Saved net flux summary to: {summary_path}")

    # Save detailed flux data for further analysis (memory-optimized)
    net_flux_cols = [f"{biocyc_id}_net_flux" for biocyc_id in valid_biocyc_ids]
    detailed_columns = ["time_min", "generation", "variant"] + net_flux_cols
    detailed_df = flux_df.select(detailed_columns)
    detailed_path = os.path.join(outdir, "separate_fba_net_flux_grid_detailed.csv")
    detailed_df.write_csv(detailed_path)
    print(f"[INFO] Saved detailed net flux data to: {detailed_path}")

    return combined_plot
