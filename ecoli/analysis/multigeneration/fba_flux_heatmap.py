"""
Visualize FBA reaction fluxes as a heatmap across multiple generations.

You can specify the reactions to visualize using parameters in params:
    "fba_flux_heatmap": {
        "BioCyc_ID": ["Name1", "Name2", ...],  # Required: reactions to analyze
        "normalized_reaction": "NormalizationReactionName",  # Optional: reaction for normalization
        "generation": [1, 2, ...]  # Optional: generations to analyze (default: all)
    }

This script will:
1. Find all reactions matching the specified BioCyc IDs
2. Calculate net flux for each reaction (forward - reverse)
3. Calculate time-averaged net flux for each generation
4. Optionally normalize fluxes relative to a reference reaction (flux/reference_flux * 100)
5. Create a heatmap with generations on y-axis and reactions on x-axis

Normalization formula: normalized_flux = (reaction_flux / reference_reaction_flux) * 100
If normalized_reaction is not specified, raw flux values are used.
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


def calculate_net_flux_optimized(flux_df, reaction_mappings):
    """
    Calculate net flux for all BioCyc IDs using optimized operations.
    Only extracts and processes the reaction columns we actually need.

    Args:
        flux_df: Polars DataFrame with expanded flux columns
        reaction_mappings: Precomputed reaction mappings

    Returns:
        Polars DataFrame with net flux columns added, dict of net flux data
    """

    print("[INFO] Starting optimized net flux calculation...")

    # Get all unique reaction indices we need (to minimize memory usage)
    all_needed_indices = set()
    for mappings in reaction_mappings.values():
        all_needed_indices.update(mappings["forward_indices"])
        all_needed_indices.update(mappings["reverse_indices"])

    needed_indices = sorted(list(all_needed_indices))
    print(f"[INFO] Only processing {len(needed_indices)} out of total reactions")

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
    """Create a heatmap visualization of FBA reaction net fluxes across multiple generations."""

    # Get parameters
    biocyc_ids = params.get("BioCyc_ID", [])
    normalized_reaction = params.get("normalized_reaction", None)
    target_generations = params.get("generation", None)

    if not biocyc_ids:
        print(
            "[ERROR] No BioCyc_ID found in params. Please specify reaction IDs to visualize."
        )
        return None

    if isinstance(biocyc_ids, str):
        biocyc_ids = [biocyc_ids]

    print(f"[INFO] Creating heatmap for {len(biocyc_ids)} reactions: {biocyc_ids}")
    if normalized_reaction:
        print(f"[INFO] Normalizing relative to: {normalized_reaction}")

    # All reactions we need to analyze (including normalization reaction if specified)
    all_reactions = biocyc_ids.copy()
    if normalized_reaction and normalized_reaction not in all_reactions:
        all_reactions.append(normalized_reaction)

    # Required columns for the query
    required_columns = [
        "time",
        "generation",
        "listeners__fba_results__reaction_fluxes",
    ]

    # Build SQL query (ordered by generation, then time)
    sql = f"""
    SELECT
        {", ".join(required_columns)}
    FROM ({history_sql})
    ORDER BY generation, time
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

    # Filter by specified generations
    if target_generations is not None:
        print(f"[INFO] Target generations: {target_generations}")
        df = df.filter(pl.col("generation").is_in(target_generations))

    # Convert time to minutes
    if "time" in df.columns:
        df = df.with_columns((pl.col("time") / 60).alias("time_min"))

    print(f"[INFO] Loaded data with {df.height} time steps")

    # Print generation information
    unique_generations = sorted(df["generation"].unique().to_list())
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

    # Precompute reaction mappings for all reactions (including normalization reaction)
    reaction_mappings = precompute_reaction_mappings(reaction_ids, all_reactions)

    # Filter out reactions that have no matching reactions
    valid_reactions = [
        reaction_id
        for reaction_id, mappings in reaction_mappings.items()
        if mappings["forward_indices"] or mappings["reverse_indices"]
    ]

    if not valid_reactions:
        print("[ERROR] No valid reactions found for any of the specified BioCyc IDs")
        return None

    # Check if normalization reaction is valid
    if normalized_reaction and normalized_reaction not in valid_reactions:
        print(
            f"[WARNING] Normalization reaction {normalized_reaction} not found. Proceeding without normalization."
        )
        normalized_reaction = None

    # Filter valid BioCyc IDs (exclude normalization reaction from visualization if it's not in the original list)
    valid_biocyc_ids = [rxn for rxn in biocyc_ids if rxn in valid_reactions]

    if not valid_biocyc_ids:
        print("[ERROR] No valid BioCyc IDs found for visualization")
        return None

    print(f"[INFO] Processing {len(valid_reactions)} valid reactions")
    print(f"[INFO] Visualizing {len(valid_biocyc_ids)} reactions in heatmap")

    # Calculate net flux using optimized operations
    try:
        flux_df, net_flux_data = calculate_net_flux_optimized(df, reaction_mappings)
        print("[INFO] Successfully calculated all net fluxes")
    except Exception as e:
        print(f"[ERROR] Failed to calculate net fluxes: {e}")
        return None

    # Calculate time-averaged net flux for each generation and reaction
    print("[INFO] Calculating time-averaged fluxes for each generation...")

    heatmap_data = []

    for generation in unique_generations:
        generation_data = flux_df.filter(pl.col("generation") == generation)

        generation_averages = {}
        for reaction_id in valid_reactions:
            net_flux_col = f"{reaction_id}_net_flux"
            if net_flux_col in generation_data.columns:
                avg_flux = generation_data[net_flux_col].mean()
                generation_averages[reaction_id] = avg_flux
            else:
                generation_averages[reaction_id] = 0.0

        # Apply normalization if specified
        if normalized_reaction and normalized_reaction in generation_averages:
            norm_flux = generation_averages[normalized_reaction]
            if abs(norm_flux) > 1e-10:  # Avoid division by zero
                print(
                    f"[INFO] Gen {generation}: Normalizing by {normalized_reaction} = {norm_flux:.6f}"
                )
                for reaction_id in valid_biocyc_ids:
                    if reaction_id in generation_averages:
                        normalized_value = (
                            generation_averages[reaction_id] / norm_flux
                        ) * 100
                        heatmap_data.append(
                            {
                                "Generation": str(generation),
                                "Reaction": reaction_id,
                                "Net_Flux": normalized_value,
                                "Raw_Flux": generation_averages[reaction_id],
                                "Normalization_Flux": norm_flux,
                            }
                        )
            else:
                print(
                    f"[WARNING] Gen {generation}: Normalization reaction flux is zero, using raw values"
                )
                for reaction_id in valid_biocyc_ids:
                    if reaction_id in generation_averages:
                        heatmap_data.append(
                            {
                                "Generation": str(generation),
                                "Reaction": reaction_id,
                                "Net_Flux": generation_averages[reaction_id],
                                "Raw_Flux": generation_averages[reaction_id],
                                "Normalization_Flux": 0.0,
                            }
                        )
        else:
            # No normalization
            for reaction_id in valid_biocyc_ids:
                if reaction_id in generation_averages:
                    heatmap_data.append(
                        {
                            "Generation": str(generation),
                            "Reaction": reaction_id,
                            "Net_Flux": generation_averages[reaction_id],
                            "Raw_Flux": generation_averages[reaction_id],
                            "Normalization_Flux": None,
                        }
                    )

    if not heatmap_data:
        print("[ERROR] No heatmap data could be generated")
        return None

    heatmap_df = pd.DataFrame(heatmap_data)
    print(f"[INFO] Generated heatmap data with {len(heatmap_data)} data points")

    # Create the heatmap using Altair
    print("[INFO] Creating heatmap visualization...")

    # Determine color scale based on data range
    flux_min = heatmap_df["Net_Flux"].min()
    flux_max = heatmap_df["Net_Flux"].max()
    flux_abs_max = max(abs(flux_min), abs(flux_max))

    # Use diverging color scheme if data crosses zero, sequential otherwise
    if flux_min < 0 and flux_max > 0:
        color_scheme = "redblue"
        color_scale = alt.Scale(
            scheme=color_scheme, domain=[-flux_abs_max, flux_abs_max], type="linear"
        )
    else:
        color_scheme = "viridis"
        color_scale = alt.Scale(scheme=color_scheme, type="linear")

    # Create heatmap
    flux_title = (
        "Net Flux (% of norm)" if normalized_reaction else "Net Flux (mmol/gDW/hr)"
    )
    chart_title = f"FBA Net Flux Heatmap: {len(unique_generations)} Generations × {len(valid_biocyc_ids)} Reactions"
    if normalized_reaction:
        chart_title += f" (Normalized by {normalized_reaction})"

    # Build tooltip list conditionally
    tooltip_list = [
        "Generation:N",
        "Reaction:N",
        alt.Tooltip("Net_Flux:Q", format=".4f", title="Net Flux"),
        alt.Tooltip("Raw_Flux:Q", format=".6f", title="Raw Flux"),
    ]
    if normalized_reaction:
        tooltip_list.append(
            alt.Tooltip("Normalization_Flux:Q", format=".6f", title="Norm Flux")
        )

    heatmap = (
        alt.Chart(heatmap_df)
        .mark_rect(stroke="white", strokeWidth=1)
        .encode(
            x=alt.X(
                "Reaction:N",
                title="BioCyc Reaction ID",
                sort=valid_biocyc_ids,
                axis=alt.Axis(labelAngle=-45),
            ),
            y=alt.Y(
                "Generation:N",
                title="Generation",
                sort=alt.SortField(field="Generation", order="ascending"),
            ),
            color=alt.Color("Net_Flux:Q", title=flux_title, scale=color_scale),
            tooltip=tooltip_list,
        )
        .properties(
            # Square size grid cells
            width=len(valid_biocyc_ids) * 50,
            height=len(unique_generations) * 50,
            title=alt.TitleParams(text=chart_title, fontSize=14, anchor="start"),
        )
    )

    # Add text annotations on the heatmap cells
    text_annotations = (
        alt.Chart(heatmap_df)
        .mark_text(baseline="middle", fontSize=10, fontWeight="bold")
        .encode(
            x=alt.X("Reaction:N", sort=valid_biocyc_ids),
            y=alt.Y(
                "Generation:N",
                sort=alt.SortField(field="Generation", order="ascending"),
            ),
            text=alt.Text("Net_Flux:Q", format=".2f"),
            color=alt.condition(
                f"datum.Net_Flux > {flux_abs_max * 0.5}",
                alt.value("white"),
                alt.value("black"),
            ),
        )
    )

    # Combine heatmap and text
    final_chart = heatmap + text_annotations

    # Save the plot
    output_path = os.path.join(outdir, "fba_flux_heatmap.html")
    final_chart.save(output_path)
    print(f"[INFO] Saved heatmap visualization to: {output_path}")

    # Save heatmap data as CSV
    heatmap_csv_path = os.path.join(outdir, "fba_flux_heatmap_data.csv")
    heatmap_df.to_csv(heatmap_csv_path, index=False)
    print(f"[INFO] Saved heatmap data to: {heatmap_csv_path}")

    # Save reaction mapping summary
    summary_data = []
    for biocyc_id in valid_biocyc_ids:
        mappings = reaction_mappings[biocyc_id]
        summary_data.append(
            {
                "BioCyc_ID": biocyc_id,
                "Forward_Reactions": "; ".join(mappings["forward_reactions"]),
                "Reverse_Reactions": "; ".join(mappings["reverse_reactions"]),
                "Num_Forward": len(mappings["forward_reactions"]),
                "Num_Reverse": len(mappings["reverse_reactions"]),
            }
        )

    summary_df = pd.DataFrame(summary_data)
    summary_path = os.path.join(outdir, "fba_flux_heatmap_summary.csv")
    summary_df.to_csv(summary_path, index=False)
    print(f"[INFO] Saved reaction summary to: {summary_path}")

    return final_chart
