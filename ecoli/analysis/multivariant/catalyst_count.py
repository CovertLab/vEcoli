"""
Visualize catalyst counts over time for specified BioCyc reactions across generations.
For each specific BioCyc ID reaction, this scripts will add all the catalysts which catalyse it:
```number of catalysts = sum(number of catalysts[i])```

Supports two visualization modes:
1. 'grid' mode: Each row represents a variant, each column represents a reaction's catalysts
2. 'stacked' mode: Each reaction's catalysts get their own chart, variants shown as different colored lines

You can specify the reactions and layout using parameters:
    "catalyst_count": {
        # Required: specify BioCyc reaction IDs to visualize
        "BioCyc_ID": ["Name1", "Name2", ...],
        # Optional: specify variants to visualize
        # If not specified, all variants will be used
        "variant": [1, 2, ...],
        # Optional: specify generations to visualize
        # If not specified, all generations will be used
        "generation": [1, 2, ...],
        # Optional: specify layout mode ('grid' or 'stacked')
        # Default: 'stacked'
        "layout": "stacked"  # or "grid"
        }

This script uses SQL to efficiently calculate catalyst counts directly in the database,
reducing memory usage and improving performance.
"""

import altair as alt
import os
from typing import Any
import pickle

import polars as pl
from duckdb import DuckDBPyConnection
import pandas as pd

from ecoli.library.parquet_emitter import open_arbitrary_sim_data, field_metadata
from ecoli.analysis.utils import create_base_to_extended_mapping


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
    """Visualize catalyst counts over time for specified BioCyc reactions across generations."""

    # Get parameters
    biocyc_ids = params.get("BioCyc_ID", [])
    if not biocyc_ids:
        print(
            "[ERROR] No BioCyc_ID found in params. Please specify reaction IDs to visualize."
        )
        return None

    if isinstance(biocyc_ids, str):
        biocyc_ids = [biocyc_ids]

    # Get layout mode (default to 'stacked')
    layout_mode = params.get("layout", "stacked").lower()
    if layout_mode not in ["grid", "stacked"]:
        print(f"[WARNING] Invalid layout mode '{layout_mode}'. Using 'stacked' mode.")
        layout_mode = "stacked"

    print(
        f"[INFO] Visualizing catalyst counts for {len(biocyc_ids)} reactions: {biocyc_ids}"
    )
    print(f"[INFO] Using layout mode: {layout_mode}")

    # Load sim_data to get reaction_to_catalyst mapping
    try:
        with open_arbitrary_sim_data(sim_data_dict) as f:
            sim_data = pickle.load(f)
        reaction_to_catalyst = sim_data.process.metabolism.reaction_catalysts
        print(
            f"[INFO] Loaded reaction to catalyst mapping with {len(reaction_to_catalyst)} reactions"
        )
    except Exception as e:
        print(f"[ERROR] Error loading sim_data: {e}")
        return None

    # Create base to extended reaction mapping
    base_to_extended_mapping = create_base_to_extended_mapping(sim_data_dict)
    if not base_to_extended_mapping:
        print("[ERROR] Could not create base to extended reaction mapping")
        return None

    # Load catalyst IDs from config
    try:
        catalyst_ids = field_metadata(
            conn, config_sql, "listeners__fba_results__catalyst_counts"
        )
        print(f"[INFO] Total catalysts in sim_data: {len(catalyst_ids)}")
    except Exception as e:
        print(f"[ERROR] Error loading catalyst IDs: {e}")
        return None

    # Build catalyst calculation SQL for efficient processing
    catalyst_calculation_sql, valid_biocyc_ids = build_catalyst_calculation_sql(
        biocyc_ids,
        base_to_extended_mapping,
        reaction_to_catalyst,
        catalyst_ids,
        history_sql,
    )

    if not catalyst_calculation_sql or not valid_biocyc_ids:
        print("[ERROR] Could not build catalyst calculation SQL")
        return None

    print(f"[INFO] Processing {len(valid_biocyc_ids)} valid BioCyc IDs")

    # Execute the optimized SQL query
    try:
        df = conn.sql(catalyst_calculation_sql).pl()
        print(f"[INFO] Loaded data with {df.height} time steps")
    except Exception as e:
        print(f"[ERROR] Error executing catalyst calculation SQL: {e}")
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

    # Calculate average catalyst counts based on layout mode
    if layout_mode == "grid":
        # For grid mode: calculate averages by variant, generation, and reaction
        avg_data = []
        for biocyc_id in valid_biocyc_ids:
            catalyst_col = f"{biocyc_id}_catalyst_count"
            if catalyst_col in df.columns:
                variant_gen_avgs = df.group_by(["variant", "generation"]).agg(
                    pl.col(catalyst_col).mean().alias("avg_catalyst_count")
                )

                for row in variant_gen_avgs.iter_rows(named=True):
                    avg_data.append(
                        {
                            "biocyc_id": biocyc_id,
                            "variant": row["variant"],
                            "generation": row["generation"],
                            "avg_catalyst_count": row["avg_catalyst_count"],
                        }
                    )

        avg_df = pl.DataFrame(avg_data)
    else:
        # For stacked mode: calculate averages by variant and reaction
        avg_catalyst_counts = {}
        for biocyc_id in valid_biocyc_ids:
            catalyst_col = f"{biocyc_id}_catalyst_count"
            if catalyst_col in df.columns:
                variant_avgs = df.group_by("variant").agg(
                    pl.col(catalyst_col).mean().alias("avg_catalyst_count")
                )
                avg_catalyst_counts[biocyc_id] = variant_avgs

    # Create visualization based on layout mode
    if layout_mode == "grid":
        combined_plot = create_grid_visualization(
            df, avg_df, valid_biocyc_ids, unique_variants, unique_generations
        )
        output_filename = "catalyst_count_grid_analysis.html"
        title_suffix = (
            f"{len(unique_variants)} Variants × {len(valid_biocyc_ids)} Reactions"
        )
    else:
        combined_plot = create_stacked_visualization(
            df, avg_catalyst_counts, valid_biocyc_ids
        )
        output_filename = "catalyst_count_stacked_analysis.html"
        title_suffix = f"Multi-Variant Analysis ({len(valid_biocyc_ids)} reactions)"

    if combined_plot is None:
        print("[ERROR] Failed to create visualization")
        return None

    # Add overall title
    combined_plot = combined_plot.properties(
        title=alt.TitleParams(
            text=f"Catalyst Count Analysis: {title_suffix}",
            fontSize=16,
            anchor="start",
        )
    ).resolve_scale(color="shared")

    # Save the plot
    output_path = os.path.join(outdir, output_filename)
    combined_plot.save(output_path)
    print(f"[INFO] Saved visualization to: {output_path}")

    return combined_plot


def build_catalyst_calculation_sql(
    biocyc_ids,
    base_to_extended_mapping,
    reaction_to_catalyst,
    catalyst_ids,
    history_sql,
):
    """Build SQL query to efficiently calculate catalyst counts for specified BioCyc reactions."""

    # Find catalysts for each BioCyc ID and build SQL columns
    biocyc_to_catalysts = {}
    valid_biocyc_ids = []
    catalyst_calculations = []

    for biocyc_id in biocyc_ids:
        catalysts = set()

        # Get extended reaction IDs for this BioCyc ID
        extended_ids = base_to_extended_mapping.get(biocyc_id, [])

        if not extended_ids:
            print(
                f"[WARNING] No extended reaction IDs found for BioCyc ID: {biocyc_id}"
            )
            continue

        # Find catalysts for all extended reactions
        for ext_id in extended_ids:
            if ext_id in reaction_to_catalyst:
                reaction_catalysts = reaction_to_catalyst[ext_id]
                catalysts.update(reaction_catalysts)

        if catalysts:
            # Convert catalyst IDs to indices in the catalyst_ids array
            catalyst_indices = []
            for cat_id in catalysts:
                try:
                    idx = catalyst_ids.index(cat_id)
                    catalyst_indices.append(idx)
                except ValueError:
                    print(
                        f"[WARNING] Catalyst {cat_id} not found in catalyst_ids array"
                    )

            if catalyst_indices:
                biocyc_to_catalysts[biocyc_id] = {
                    "catalyst_ids": list(catalysts),
                    "catalyst_indices": catalyst_indices,
                }
                valid_biocyc_ids.append(biocyc_id)

                # Build SQL calculation for this BioCyc ID
                # Convert 0-based indices to 1-based for DuckDB SQL
                sql_indices = [str(idx + 1) for idx in catalyst_indices]
                catalyst_sum = " + ".join([f"catalysts[{idx}]" for idx in sql_indices])
                # Use quotes around column name to handle special characters like hyphens
                catalyst_calculations.append(
                    f'({catalyst_sum}) AS "{biocyc_id}_catalyst_count"'
                )

                print(
                    f"[INFO] Found {len(catalyst_indices)} catalysts for {biocyc_id}: {list(catalysts)}"
                )
            else:
                print(f"[WARNING] No valid catalyst indices found for {biocyc_id}")
        else:
            print(f"[WARNING] No catalysts found for BioCyc ID: {biocyc_id}")

    if not valid_biocyc_ids or not catalyst_calculations:
        print("[ERROR] No valid BioCyc IDs with catalysts found")
        return None, []

    # Build the complete SQL query
    catalyst_calculations_str = ",\n    ".join(catalyst_calculations)

    sql = f"""
    WITH renamed AS (
        SELECT 
            time / 60.0 AS time_min,
            generation,
            variant,
            listeners__fba_results__catalyst_counts AS catalysts
        FROM ({history_sql})
    )
    SELECT 
        time_min,
        generation,
        variant,
        {catalyst_calculations_str}
    FROM renamed
    ORDER BY variant, generation, time_min
    """

    print(f"[INFO] Built SQL with {len(catalyst_calculations)} catalyst calculations")
    return sql, valid_biocyc_ids


def create_grid_visualization(
    df, avg_df, valid_biocyc_ids, unique_variants, unique_generations
):
    """Create grid layout visualization (rows = variants, columns = reactions)."""

    def create_subplot_chart(variant, biocyc_id):
        """Create a single subplot for a specific variant-reaction combination."""
        catalyst_col = f"{biocyc_id}_catalyst_count"

        # Check if the column exists in dataframe
        if catalyst_col not in df.columns:
            print(f"[WARNING] Column {catalyst_col} not found in dataframe")
            return None

        # Filter data for this variant and reaction
        subplot_data = (
            df.filter(pl.col("variant") == variant)
            .select(["time_min", "generation", catalyst_col])
            .filter(pl.col(catalyst_col).is_not_null())
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
                    f"{catalyst_col}:Q",
                    title="Total Catalyst Count"
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
                tooltip=["time_min:Q", f"{catalyst_col}:Q", "generation:N"],
            )
        )

        # For grid mode, we don't show average lines to keep the visualization clean
        # Combine all elements
        combined = line_chart.resolve_scale(color="shared")

        # Add title only for top row
        if variant == unique_variants[0]:
            combined = combined.properties(title=f"{biocyc_id}")

        combined = combined.properties(width=400, height=300)

        return combined

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
    return combined_plot


def create_stacked_visualization(df, avg_catalyst_counts, valid_biocyc_ids):
    """Create stacked layout visualization (one chart per reaction, variants as colored lines)."""

    def create_catalyst_count_chart(biocyc_id, catalyst_col, variant_avgs):
        """Create a line chart for a single reaction's catalyst counts with average lines for each variant."""

        # Check if the column exists in dataframe
        if catalyst_col not in df.columns:
            print(f"[WARNING] Column {catalyst_col} not found in dataframe")
            return None

        # Select only the columns we need to minimize data transfer
        data = df.select(["time_min", "generation", "variant", catalyst_col])

        # Remove any null values
        data = data.filter(pl.col(catalyst_col).is_not_null())

        if data.height == 0:
            print(f"[WARNING] No valid data for reaction {biocyc_id}")
            return None

        # Main catalyst count line chart (different variants as different colored lines)
        catalyst_chart = (
            alt.Chart(data)
            .mark_line(strokeWidth=2)
            .encode(
                x=alt.X("time_min:Q", title="Time (min)"),
                y=alt.Y(f"{catalyst_col}:Q", title="Total Catalyst Count"),
                color=alt.Color("variant:N", legend=alt.Legend(title="Variant")),
                tooltip=[
                    "time_min:Q",
                    f"{catalyst_col}:Q",
                    "variant:N",
                    "generation:N",
                ],
            )
        )

        # Create average lines for each variant
        avg_line_data = []
        for row in variant_avgs.iter_rows(named=True):
            variant_name = row["variant"]
            avg_value = row["avg_catalyst_count"]
            avg_line_data.append(
                {
                    "variant": variant_name,
                    "avg_catalyst_count": avg_value,
                    "label": f"{variant_name} Avg: {avg_value:.2f}",
                }
            )

        if avg_line_data:
            avg_line_df = pd.DataFrame(avg_line_data)

            avg_lines = (
                alt.Chart(avg_line_df)
                .mark_rule(strokeDash=[5, 5], strokeWidth=2)
                .encode(
                    y=alt.Y("avg_catalyst_count:Q"),
                    color=alt.Color(
                        "variant:N", legend=None
                    ),  # Use same color scale as main chart
                    tooltip=["label:N"],
                )
            )
        else:
            avg_lines = alt.Chart().mark_point()  # Empty chart

        # Combine catalyst count line and average lines
        combined_chart = (
            (catalyst_chart + avg_lines)
            .properties(
                title=f"Catalyst Count vs Time: {biocyc_id}", width=600, height=300
            )
            .resolve_scale(y="shared", color="shared")
        )

        return combined_chart

    # Create charts for each reaction
    charts = []

    for biocyc_id in valid_biocyc_ids:
        catalyst_col = f"{biocyc_id}_catalyst_count"
        variant_avgs = avg_catalyst_counts.get(biocyc_id)
        if variant_avgs is not None:
            chart = create_catalyst_count_chart(biocyc_id, catalyst_col, variant_avgs)
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

    return combined_plot
