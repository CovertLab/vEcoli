"""
Visualize FBA reaction fluxes as a heatmap across multiple generations.

You can specify the reactions to visualize using parameters in params:
    "fba_flux_heatmap": {
        "BioCyc_ID": ["Name1", "Name2", ...],  # Required: reactions to analyze
        "normalized_reaction": "NormalizationReactionName",  # Optional: reaction for normalization
        "generation": [1, 2, ...]  # Optional: generations to analyze (default: all)
    }

This script will:
1. Find all reactions matching the specified BioCyc IDs using efficient SQL-based approach
2. Calculate net flux for each reaction (forward - reverse) directly in SQL
3. Calculate time-averaged net flux for each generation
4. Optionally normalize fluxes relative to a reference reaction (flux/reference_flux * 100)
5. Create a heatmap with generations on y-axis and reactions on x-axis

Normalization formula: normalized_flux = (reaction_flux / reference_reaction_flux) * 100
If normalized_reaction is not specified, raw flux values are used.
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
    flux_calculation_sql, valid_reactions = build_flux_calculation_sql(
        all_reactions, base_to_extended_mapping, all_reaction_ids, history_sql
    )

    if not flux_calculation_sql or not valid_reactions:
        print("[ERROR] Could not build flux calculation SQL")
        return None

    print(f"[INFO] Processing {len(valid_reactions)} valid reactions")

    # Check if normalization reaction is valid
    if normalized_reaction and normalized_reaction not in valid_reactions:
        print(
            f"[WARNING] Normalization reaction {normalized_reaction} not found. Proceeding without normalization."
        )
        normalized_reaction = None

    # Filter valid BioCyc IDs for visualization
    # Exclude normalization reaction if it's not in the original list
    valid_biocyc_ids = [rxn for rxn in biocyc_ids if rxn in valid_reactions]

    if not valid_biocyc_ids:
        print("[ERROR] No valid BioCyc IDs found for visualization")
        return None

    print(f"[INFO] Visualizing {len(valid_biocyc_ids)} reactions in heatmap")

    # Execute the optimized SQL query
    try:
        df = conn.sql(flux_calculation_sql).pl()
        print(f"[INFO] Loaded flux data with {df.height} time steps")
    except Exception as e:
        print(f"[ERROR] Error executing flux calculation SQL: {e}")
        return None

    if df.height == 0:
        print("[ERROR] No data found")
        return None

    # Filter by specified generations if provided
    if target_generations is not None:
        print(f"[INFO] Target generations: {target_generations}")
        df = df.filter(pl.col("generation").is_in(target_generations))

    if df.height == 0:
        print("[ERROR] No data found for specified generations")
        return None

    # Print generation information
    unique_generations = sorted(df["generation"].unique().to_list())
    print(f"[INFO] Found {len(unique_generations)} generations: {unique_generations}")

    # Calculate time-averaged net flux for each generation and reaction
    print("[INFO] Calculating time-averaged fluxes for each generation...")

    heatmap_data = []

    for generation in unique_generations:
        generation_data = df.filter(pl.col("generation") == generation)

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

    # Print some statistics about the flux data
    flux_stats = {
        "min": heatmap_df["Net_Flux"].min(),
        "max": heatmap_df["Net_Flux"].max(),
        "mean": heatmap_df["Net_Flux"].mean(),
        "std": heatmap_df["Net_Flux"].std(),
    }
    print(
        f"[INFO] Flux statistics: min={flux_stats['min']:.6f}, max={flux_stats['max']:.6f}, "
        f"mean={flux_stats['mean']:.6f}, std={flux_stats['std']:.6f}"
    )

    # Create the heatmap using Altair
    print("[INFO] Creating heatmap visualization...")

    # Determine color scale based on data range
    flux_min = heatmap_df["Net_Flux"].min()
    flux_max = heatmap_df["Net_Flux"].max()
    flux_abs_max = max(abs(flux_min), abs(flux_max))

    # Use diverging color scheme if data crosses zero, sequential otherwise
    if flux_min < 0 and flux_max > 0:
        color_scale = alt.Scale(
            scheme="redblue", domain=[-flux_abs_max, flux_abs_max], type="linear"
        )
        print("[INFO] Using diverging color scheme (data crosses zero)")
    else:
        color_scale = alt.Scale(scheme="viridis", type="linear")
        print("[INFO] Using sequential color scheme")

    # Create heatmap
    flux_title = (
        "Net Flux (% of norm)" if normalized_reaction else "Net Flux (mmol/gDW/hr)"
    )
    chart_title = f"FBA Net Flux Heatmap: {len(unique_generations)} Generations x {len(valid_biocyc_ids)} Reactions"
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

    # Calculate appropriate chart dimensions
    chart_width = max(400, len(valid_biocyc_ids) * 60)
    chart_height = max(300, len(unique_generations) * 50)

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
            width=chart_width,
            height=chart_height,
            title=alt.TitleParams(text=chart_title, fontSize=14, anchor="start"),
        )
    )

    # Add text annotations on the heatmap cells (only if not too many cells)
    total_cells = len(valid_biocyc_ids) * len(unique_generations)
    if total_cells <= 100:  # Only add text for smaller heatmaps to avoid clutter
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
        print("[INFO] Added text annotations to heatmap")
    else:
        final_chart = heatmap
        print("[INFO] Skipped text annotations (too many cells for readability)")

    # Save the plot
    output_path = os.path.join(outdir, "fba_flux_heatmap.html")
    final_chart.save(output_path)
    print(f"[INFO] Saved heatmap visualization to: {output_path}")

    # Save the underlying data as CSV for reference
    csv_path = os.path.join(outdir, "fba_flux_heatmap_data.csv")
    heatmap_df.to_csv(csv_path, index=False)
    print(f"[INFO] Saved heatmap data to: {csv_path}")

    # Print summary of results
    print("[INFO] Heatmap visualization completed successfully!")
    print(f"[INFO] - Generations analyzed: {len(unique_generations)}")
    print(f"[INFO] - Reactions visualized: {len(valid_biocyc_ids)}")
    print(
        f"[INFO] - Normalization: {'Yes (' + normalized_reaction + ')' if normalized_reaction else 'No'}"
    )
    print(f"[INFO] - Total data points: {len(heatmap_data)}")

    return final_chart
