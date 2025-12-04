"""
Base reaction flux analysis script with per-generation common-category extraction and burst detection.

This script contains the original helper functions for loading FBA data, mapping
extended -> base reactions, computing base reaction net fluxes, categorizing base
reactions by flux behavior (always_positive, always_negative, oscillating,
always_zero), plotting utilities, and saving results.

Added functionality:
- For each generation present in the history SQL, compute base reaction categories
  using the exact same categorization logic.
- Compute the intersection (common base reactions) across all generations for
  each category.
- Save those common reaction names into four separate CSV files in outdir:
    - common_base_reactions_always_zero.csv
    - common_base_reactions_always_positive.csv
    - common_base_reactions_always_negative.csv
    - common_base_reactions_oscillating.csv
- Compute "burst" base reactions for categories always_positive, always_negative, and oscillating.
  A base reaction is considered a burst (for a category) if, for every generation,
  when that reaction belongs to that category in that generation, its zero_ratio
  (fraction of timepoints with essentially zero flux) is > burst_threshold.
  The default burst_threshold is 0.1 (configurable via params["burst_threshold"]).
- Save three CSVs (one-column each) listing burst base reaction names:
    - burst_base_reaction_always_positive.csv
    - burst_base_reaction_always_negative.csv
    - burst_base_reaction_oscillating.csv

Modified plotting:
- Removed simple scatter plots
- Combined positive and negative scatter plots into unified unidirectional plot

Important constraints preserved:
- The data loading function `load_fba_data` is left intact (not modified).
- The category determination logic (in `categorize_base_reactions_by_flux_behavior`)
  is not altered.
- `plot(...)` remains present and acts as the main entry point.

Usage:
    Call plot(...) with the same parameters expected by the original script.
"""

import os
from typing import Any
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.stats import gaussian_kde
from duckdb import DuckDBPyConnection

# --- BEGIN: Existing helper imports and functions (kept exactly as requested) ---
# Note: These functions are identical to the code you provided. I have not changed
# the internals of load_fba_data, or the categorization logic.
from ecoli.library.parquet_emitter import field_metadata
from ecoli.analysis.utils import create_base_to_extended_mapping


def load_fba_data(
    conn: DuckDBPyConnection, history_sql: str, config_sql: str, sim_data_dict: dict
) -> tuple[pd.DataFrame, dict]:
    """
    Load FBA flux data using DuckDB connection and SQL queries.

    Parameters:
    - conn: DuckDB connection
    - history_sql: SQL query for historical data
    - config_sql: SQL query for configuration data
    - sim_data_dict: Dictionary with sim_data information

    Returns:
    - df: DataFrame with flux data (time points x extended reactions)
    - metadata: Dictionary with experiment metadata
    """
    print("[INFO] Loading FBA flux data via SQL...")

    try:
        # Create base to extended reaction mapping
        base_to_extended_mapping = create_base_to_extended_mapping(sim_data_dict)
        if not base_to_extended_mapping:
            raise Exception("Could not create base to extended reaction mapping")

        # Load the reaction IDs from the config - this is the array that maps to flux matrix columns
        reaction_ids = field_metadata(
            conn, config_sql, "listeners__fba_results__reaction_fluxes"
        )
        print(f"[INFO] Total reactions in sim_data: {len(reaction_ids)}")

        # Required columns for the query
        required_columns = [
            "time",
            "generation",
            "listeners__fba_results__reaction_fluxes",
        ]

        # Build SQL query
        sql = f"""
        SELECT {", ".join(required_columns)}
        FROM ({history_sql})
        ORDER BY generation, time
        """

        # Execute query
        df_pl = conn.sql(sql).pl()

        if df_pl.is_empty():
            raise Exception("No data found")
        print(f"[INFO] Loaded data with {df_pl.height} time steps")

        # Extract flux matrix and convert to pandas DataFrame
        flux_matrix = df_pl["listeners__fba_results__reaction_fluxes"].to_numpy()
        flux_matrix = np.array([np.array(row) for row in flux_matrix])

        # Create DataFrame with extended reactions
        df = pd.DataFrame(flux_matrix, columns=reaction_ids)

        # Add time information
        time_data = df_pl.select(["time"]).to_pandas()
        df = pd.concat([time_data, df], axis=1)

        # Drop initial time point where time == 0
        df = df[df["time"] != 0].reset_index(drop=True)

        # Create metadata dictionary
        metadata = {
            "n_extended_reactions": len(reaction_ids),
            "n_timepoints": len(df),
            "extended_reaction_names": reaction_ids,
            "base_to_extended_mapping": base_to_extended_mapping,
        }

        print("[INFO] Successfully loaded data:")
        print(f"  - Time points: {len(df)}")
        print(f"  - Extended reactions: {len(reaction_ids)}")
        print(f"  - Base reaction mapping entries: {len(base_to_extended_mapping)}")

        return df, metadata

    except Exception as e:
        print(f"[ERROR] Failed to load data: {str(e)}")
        raise


def map_extended_to_base_reactions(extended_reactions, base_to_extended_mapping):
    """
    Map extended reactions to base reactions and identify forward/reverse relationships.

    Parameters:
    - extended_reactions: List of extended reaction names
    - base_to_extended_mapping: Dict mapping base reaction ID to list of extended reaction names

    Returns:
    - base_reaction_mapping: Dict with base reaction info including forward/reverse extended reactions
    - extended_to_base_map: Dict mapping each extended reaction to its base reaction
    """
    # Create reverse mapping from extended to base
    extended_to_base_map = {}
    for base_rxn, extended_list in base_to_extended_mapping.items():
        for extended_rxn in extended_list:
            extended_to_base_map[extended_rxn] = base_rxn

    base_reaction_mapping = {}

    for extended_reaction in extended_reactions:
        # Get base reaction name from mapping
        base_reaction = extended_to_base_map.get(extended_reaction)

        if base_reaction is None:
            print(
                f"[WARNING] No base reaction found for extended reaction: {extended_reaction}"
            )
            continue

        # Initialize base reaction entry if not exists
        if base_reaction not in base_reaction_mapping:
            base_reaction_mapping[base_reaction] = {
                "forward_extended": [],
                "reverse_extended": [],
                "all_extended": [],
            }

        # Determine if this is a forward or reverse extended reaction
        if extended_reaction.endswith(" (reverse)"):
            base_reaction_mapping[base_reaction]["reverse_extended"].append(
                extended_reaction
            )
        else:
            base_reaction_mapping[base_reaction]["forward_extended"].append(
                extended_reaction
            )

        base_reaction_mapping[base_reaction]["all_extended"].append(extended_reaction)

    print("[INFO] Base reaction mapping results:")
    print(f"  - Total base reactions: {len(base_reaction_mapping)}")
    print(f"  - Extended reactions mapped: {len(extended_to_base_map)}")

    # Print statistics about forward/reverse distributions
    forward_only = sum(
        1
        for info in base_reaction_mapping.values()
        if len(info["forward_extended"]) > 0 and len(info["reverse_extended"]) == 0
    )
    reverse_only = sum(
        1
        for info in base_reaction_mapping.values()
        if len(info["forward_extended"]) == 0 and len(info["reverse_extended"]) > 0
    )
    both_directions = sum(
        1
        for info in base_reaction_mapping.values()
        if len(info["forward_extended"]) > 0 and len(info["reverse_extended"]) > 0
    )

    print(f"  - Base reactions with forward extended only: {forward_only}")
    print(f"  - Base reactions with reverse extended only: {reverse_only}")
    print(
        f"  - Base reactions with both forward and reverse extended: {both_directions}"
    )

    return base_reaction_mapping, extended_to_base_map


def compute_base_reaction_fluxes(flux_df, base_reaction_mapping):
    """
    Compute base reaction fluxes by summing forward extended and subtracting reverse extended fluxes.

    Parameters:
    - flux_df: DataFrame with extended reaction flux values
    - base_reaction_mapping: Dict with base reaction info

    Returns:
    - base_flux_df: DataFrame with base reaction net flux values
    - base_reaction_details: Dict with detailed info about each base reaction
    """
    base_flux_data = {}
    base_reaction_details = {}

    for base_reaction, info in base_reaction_mapping.items():
        forward_extended = info["forward_extended"]
        reverse_extended = info["reverse_extended"]

        # Sum forward extended fluxes
        forward_flux = pd.Series(0.0, index=flux_df.index)
        if forward_extended:
            for ext_reaction in forward_extended:
                if ext_reaction in flux_df.columns:
                    forward_flux += flux_df[ext_reaction]

        # Sum reverse extended fluxes
        reverse_flux = pd.Series(0.0, index=flux_df.index)
        if reverse_extended:
            for ext_reaction in reverse_extended:
                if ext_reaction in flux_df.columns:
                    reverse_flux += flux_df[ext_reaction]

        # Net flux = forward - reverse
        net_flux = forward_flux - reverse_flux
        base_flux_data[base_reaction] = net_flux

        # Store details for analysis
        base_reaction_details[base_reaction] = {
            "forward_extended": forward_extended,
            "reverse_extended": reverse_extended,
            "n_forward_extended": len(forward_extended),
            "n_reverse_extended": len(reverse_extended),
            "total_extended": len(info["all_extended"]),
        }

    base_flux_df = pd.DataFrame(base_flux_data)

    print("[INFO] Base reaction flux computation results:")
    print(f"  - Base reactions computed: {len(base_flux_df.columns)}")
    print(f"  - Time points: {len(base_flux_df)}")

    return base_flux_df, base_reaction_details


def categorize_base_reactions_by_flux_behavior(base_flux_df, eps=1e-30):
    """
    Categorize base reactions based on their flux behavior across time steps.

    Parameters:
    - base_flux_df: DataFrame with base reaction net flux values
    - eps: Small tolerance for zero comparison

    Returns:
    - always_positive: List of base reactions that are always >= 0 and have max > 0
    - always_negative: List of base reactions that are always <= 0 and have max abs > 0
    - oscillating: List of base reactions that change sign
    - always_zero: List of base reactions that are always zero
    - base_reaction_categories: Dictionary with detailed categorization info
    """
    always_positive = []
    always_negative = []
    oscillating = []
    always_zero = []
    base_reaction_categories = {}

    for base_reaction in base_flux_df.columns:
        flux_values = base_flux_df[base_reaction].values

        # Check for positive, negative, and zero values
        has_positive = np.any(flux_values > eps)
        has_negative = np.any(flux_values < -eps)
        has_zero = np.any(np.abs(flux_values) <= eps)

        min_flux = flux_values.min()
        max_flux = flux_values.max()
        max_abs_flux = np.max(np.abs(flux_values))

        # Categorize based on behavior
        if max_abs_flux <= eps:  # All values are essentially zero
            always_zero.append(base_reaction)
            category = "always_zero"
        elif not has_negative and has_positive:  # All values >= -eps and has some > eps
            always_positive.append(base_reaction)
            category = "always_positive"
        elif not has_positive and has_negative:  # All values <= eps and has some < -eps
            always_negative.append(base_reaction)
            category = "always_negative"
        elif has_positive and has_negative:  # Has both positive and negative values
            oscillating.append(base_reaction)
            category = "oscillating"
        else:
            # This case should be covered by always_zero, but keep as safety net
            always_zero.append(base_reaction)
            category = "always_zero"

        base_reaction_categories[base_reaction] = {
            "category": category,
            "min_flux": min_flux,
            "max_flux": max_flux,
            "max_abs_flux": max_abs_flux,
            "has_positive": has_positive,
            "has_negative": has_negative,
            "has_zero": has_zero,
        }

    print("\n[INFO] Base reaction categorization by flux behavior:")
    print(f"  - Always positive (>= 0, max > 0): {len(always_positive)}")
    print(f"  - Always negative (<= 0, max abs > 0): {len(always_negative)}")
    print(f"  - Oscillating (changes sign): {len(oscillating)}")
    print(f"  - Always zero (max abs ≈ 0): {len(always_zero)}")

    return (
        always_positive,
        always_negative,
        oscillating,
        always_zero,
        base_reaction_categories,
    )


def print_base_reaction_category_summaries(
    positive_df, negative_df, oscillating_df, always_zero_df
):
    """Print summary information for each base reaction category."""

    if len(positive_df) > 0:
        print(
            "\n[INFO] Always Positive Base Reactions (max flux > 0) - Top 5 most active (lowest zero ratio):"
        )
        top_positive = positive_df.sort_values("zero_ratio").head(5)
        for _, row in top_positive.iterrows():
            print(
                f"  {row['base_reaction']}: zero_ratio={row['zero_ratio']:.4f}, max_flux={row['max_flux']:.2e}, ext_reactions={row['total_extended']}"
            )

    if len(negative_df) > 0:
        print(
            "\n[INFO] Always Negative Base Reactions (max abs flux > 0) - Top 5 most active (lowest zero ratio):"
        )
        top_negative = negative_df.sort_values("zero_ratio").head(5)
        for _, row in top_negative.iterrows():
            print(
                f"  {row['base_reaction']}: zero_ratio={row['zero_ratio']:.4f}, max_abs_flux={row['max_abs_flux']:.2e}, ext_reactions={row['total_extended']}"
            )

    if len(oscillating_df) > 0:
        print(
            "\n[INFO] Oscillating Base Reactions - Top 5 most active (lowest zero ratio):"
        )
        top_oscillating = oscillating_df.sort_values("zero_ratio").head(5)
        for _, row in top_oscillating.iterrows():
            print(
                f"  {row['base_reaction']}: zero_ratio={row['zero_ratio']:.4f}, max_abs_flux={row['max_abs_flux']:.2e}, ext_reactions={row['total_extended']}"
            )

    if len(always_zero_df) > 0:
        print("\n[INFO] Always Zero Base Reactions - First 5 examples:")
        first_zero = always_zero_df.head(5)
        for _, row in first_zero.iterrows():
            print(
                f"  {row['base_reaction']}: max_abs_flux={row['max_abs_flux']:.2e}, ext_reactions={row['total_extended']}"
            )


def create_unified_directional_flux_plot(
    positive_df, negative_df, epsilon_log, base_reaction_details, outdir
):
    """Create unified heat scatter plot combining always positive and always negative base reactions."""

    # Check if we have data to plot
    if len(positive_df) == 0 and len(negative_df) == 0:
        print(
            "[WARNING] No positive or negative base reactions found. Skipping unified plot."
        )
        return

    # Combine the dataframes
    combined_data = []

    # Add positive reactions
    for idx, row in positive_df.iterrows():
        combined_data.append(
            {
                "base_reaction": row["base_reaction"],
                "category": "Always Positive",
                "zero_ratio": row["zero_ratio"],
                "max_abs_flux": row["max_abs_flux"],
                "log_max_abs_flux": row["log_max_abs_flux"],
                "min_flux": row["min_flux"],
                "max_flux": row["max_flux"],
                "total_extended": row["total_extended"],
            }
        )

    # Add negative reactions
    for idx, row in negative_df.iterrows():
        combined_data.append(
            {
                "base_reaction": row["base_reaction"],
                "category": "Always Negative",
                "zero_ratio": row["zero_ratio"],
                "max_abs_flux": row["max_abs_flux"],
                "log_max_abs_flux": row["log_max_abs_flux"],
                "min_flux": row["min_flux"],
                "max_flux": row["max_flux"],
                "total_extended": row["total_extended"],
            }
        )

    if not combined_data:
        print("[WARNING] No combined data available for unified plot.")
        return

    combined_df = pd.DataFrame(combined_data)

    # Prepare data for plotting
    x = combined_df["log_max_abs_flux"]
    y = combined_df["zero_ratio"]
    categories = combined_df["category"]

    # Calculate point density using gaussian_kde for heat scatter plot
    if len(combined_df) > 1:  # Need at least 2 points for KDE
        xy = np.vstack([x, y])
        density = gaussian_kde(xy)(xy)
    else:
        density = np.array([1.0])  # Single point gets density of 1

    # Create hover text with base reaction information
    hover_text = []
    for idx, row in combined_df.iterrows():
        base_reaction = row["base_reaction"]
        details = base_reaction_details.get(base_reaction, {})

        # Create extended reaction info
        forward_ext = details.get("forward_extended", [])
        reverse_ext = details.get("reverse_extended", [])

        ext_info = f"Forward: {len(forward_ext)} extended, Reverse: {len(reverse_ext)} extended"
        if len(forward_ext) <= 3:
            ext_info += (
                f"<br>Forward: {', '.join(forward_ext) if forward_ext else 'None'}"
            )
        if len(reverse_ext) <= 3:
            ext_info += (
                f"<br>Reverse: {', '.join(reverse_ext) if reverse_ext else 'None'}"
            )

        hover_text.append(
            f"<b>Base Reaction:</b> {base_reaction}<br>"
            + f"<b>Extended Reactions:</b> {ext_info}<br>"
            + f"<b>Category:</b> {row['category']}<br>"
            + f"<b>Zero Ratio:</b> {row['zero_ratio']:.4f}<br>"
            + f"<b>Max |Net Flux|:</b> {row['max_abs_flux']:.2e}<br>"
            + f"<b>Min Net Flux:</b> {row['min_flux']:.2e}<br>"
            + f"<b>Max Net Flux:</b> {row['max_flux']:.2e}<br>"
            + f"<b>Log |Max Net Flux|:</b> {row['log_max_abs_flux']:.2f}<br>"
            + f"<b>Point Density:</b> {density[idx]:.6f}"
        )

    # Create subplot with marginal histograms
    fig_heat = make_subplots(
        rows=2,
        cols=2,
        column_widths=[0.9, 0.1],
        row_heights=[0.1, 0.9],
        specs=[
            [{"secondary_y": False}, {"secondary_y": False}],
            [{"secondary_y": False}, {"secondary_y": False}],
        ],
        vertical_spacing=0.05,
        horizontal_spacing=0.05,
        subplot_titles=("", "", "", ""),
    )

    # Define colors for categories
    color_map = {
        "Always Positive": "rgba(70, 130, 180, 0.8)",  # Steel blue
        "Always Negative": "rgba(220, 20, 60, 0.8)",  # Crimson
    }

    # Add scatter traces for each category
    for category in ["Always Positive", "Always Negative"]:
        mask = categories == category
        if not mask.any():
            continue

        category_x = x[mask]
        category_y = y[mask]
        category_density = density[mask]
        category_hover = [hover_text[i] for i in range(len(hover_text)) if mask.iloc[i]]

        fig_heat.add_trace(
            go.Scatter(
                x=category_x,
                y=category_y,
                mode="markers",
                marker=dict(
                    size=8,
                    color=category_density,
                    colorscale="Plasma",
                    opacity=0.8,
                    line=dict(
                        width=0.5, color=color_map[category].replace("0.8", "1.0")
                    ),
                ),
                text=category_hover,
                hovertemplate="%{text}<extra></extra>",
                name=category,
                showlegend=True,
            ),
            row=2,
            col=1,
        )

    # Add colorbar for density
    fig_heat.data[0].marker.colorbar = dict(
        title=dict(text="Point Density", font=dict(size=14)),
        tickfont=dict(size=12),
        thickness=15,
        len=0.7,
        x=1.02,  # Position colorbar to the right
    )

    # Top density curve (x-axis distribution, row=1, col=1)
    if len(x) > 1:
        # Create smooth density curve for x-axis
        x_range = np.linspace(x.min(), x.max(), 250)
        x_density = gaussian_kde(x)
        x_density_values = x_density(x_range)
    else:
        # Single point case
        x_range = np.array([x.iloc[0]])
        x_density_values = np.array([1.0])

    fig_heat.add_trace(
        go.Scatter(
            x=x_range,
            y=x_density_values,
            mode="lines",
            line=dict(color="steelblue", width=3),
            fill="tozeroy",
            fillcolor="rgba(70, 130, 180, 0.3)",
            name="X Density",
            showlegend=False,
        ),
        row=1,
        col=1,
    )

    # Right density curve (y-axis distribution, row=2, col=2)
    if len(y) > 1:
        # Create smooth density curve for y-axis
        y_range = np.linspace(y.min(), y.max(), 250)
        y_density = gaussian_kde(y)
        y_density_values = y_density(y_range)
    else:
        # Single point case
        y_range = np.array([y.iloc[0]])
        y_density_values = np.array([1.0])

    fig_heat.add_trace(
        go.Scatter(
            x=y_density_values,
            y=y_range,
            mode="lines",
            line=dict(color="lightcoral", width=3),
            fill="tozerox",
            fillcolor="rgba(240, 128, 128, 0.3)",
            name="Y Density",
            showlegend=False,
        ),
        row=2,
        col=2,
    )

    # Update layout for heat scatter with histograms
    fig_heat.update_layout(
        title=dict(
            text="<b>Unified Directional Heat Scatter Plot: Always Positive and Always Negative Base Reaction Net Flux</b>",
            font=dict(size=18),
            x=0.5,
            xanchor="center",
        ),
        plot_bgcolor="white",
        paper_bgcolor="white",
        font=dict(family="Arial", size=12),
        width=1000,
        height=800,
        margin=dict(l=80, r=120, t=100, b=80),
    )

    # Update axes for main plot
    fig_heat.update_xaxes(
        title=dict(text="log₁₀(ε + |Max Net Flux|)", font=dict(size=14)),
        tickfont=dict(size=12),
        gridcolor="rgba(128,128,128,0.2)",
        gridwidth=1,
        row=2,
        col=1,
    )
    fig_heat.update_yaxes(
        title=dict(text="Zero Flux Ratio", font=dict(size=14)),
        tickfont=dict(size=12),
        gridcolor="rgba(128,128,128,0.2)",
        gridwidth=1,
        row=2,
        col=1,
    )

    # Update axes for histograms (remove tick labels and titles)
    fig_heat.update_xaxes(showticklabels=False, title="", row=1, col=1)
    fig_heat.update_yaxes(showticklabels=False, title="", row=1, col=1)
    fig_heat.update_xaxes(showticklabels=False, title="", row=2, col=2)
    fig_heat.update_yaxes(showticklabels=False, title="", row=2, col=2)

    # Hide the top-right subplot
    fig_heat.update_xaxes(visible=False, row=1, col=2)
    fig_heat.update_yaxes(visible=False, row=1, col=2)

    # Add statistics annotation
    stats_text = (
        f"Base Reactions (Always Positive): {len(positive_df):,}<br>"
        + f"Base Reactions (Always Negative): {len(negative_df):,}<br>"
        + f"Total Base Reactions: {len(combined_df):,}<br>"
        + f"ε = {epsilon_log:.0e}<br>"
        + f"|Max Net Flux| Range: {combined_df['max_abs_flux'].min():.2e} to {combined_df['max_abs_flux'].max():.2e}<br>"
        + f"Zero Ratio Range: {combined_df['zero_ratio'].min():.4f} to {combined_df['zero_ratio'].max():.4f}<br>"
        + f"Extended Reactions Range: {combined_df['total_extended'].min()} to {combined_df['total_extended'].max()}"
    )

    if len(combined_df) > 1:
        stats_text += f"<br>Density Range: {density.min():.2e} - {density.max():.2e}"

    fig_heat.add_annotation(
        x=0.02,
        y=0.48,
        xref="paper",
        yref="paper",
        text=stats_text,
        showarrow=False,
        font=dict(size=11, color="black"),
        bgcolor="rgba(255,255,255,0.9)",
        bordercolor="rgba(128,128,128,0.5)",
        borderwidth=1,
        borderpad=10,
        xanchor="left",
        yanchor="top",
    )

    # Save plot
    filename = os.path.join(
        outdir,
        "unified_directional_heat_scatter_base_reactions.html",
    )

    fig_heat.write_html(filename)

    print("\n[INFO] Unified directional base reaction flux plot saved:")
    print(f"  - {filename}")


def save_base_reaction_results(
    comprehensive_df,
    oscillating_df,
    always_zero_df,
    base_reaction_details,
    base_reaction_mapping,
    active_base_flux_df,
    metadata,
    outdir,
):
    """Save all base reaction results to CSV files with experiment metadata in outdir."""

    # Create output directory if it doesn't exist
    os.makedirs(outdir, exist_ok=True)

    # Create filename prefix
    prefix = "base_reaction_analysis"

    # Save comprehensive base reaction metrics
    comprehensive_filename = os.path.join(outdir, f"{prefix}_metrics.csv")
    comprehensive_df.to_csv(comprehensive_filename, index=False)
    print(
        f"\n[INFO] Comprehensive base reaction metrics saved to '{comprehensive_filename}'"
    )

    # Save oscillating base reactions specifically
    if len(oscillating_df) > 0:
        oscillating_filename = os.path.join(outdir, f"{prefix}_oscillating.csv")
        oscillating_df.to_csv(oscillating_filename, index=False)
        print(f"[INFO] Oscillating base reactions saved to '{oscillating_filename}'")
    else:
        print("[INFO] No oscillating base reactions found.")

    # Save always zero base reactions specifically
    if len(always_zero_df) > 0:
        zero_filename = os.path.join(outdir, f"{prefix}_always_zero.csv")
        always_zero_df.to_csv(zero_filename, index=False)
        print(f"[INFO] Always zero base reactions saved to '{zero_filename}'")
    else:
        print("[INFO] No always zero base reactions found.")

    # Save base reaction mapping details
    mapping_data = []
    for base_reaction, info in base_reaction_mapping.items():
        mapping_data.append(
            {
                "base_reaction": base_reaction,
                "forward_extended": "; ".join(info["forward_extended"]),
                "reverse_extended": "; ".join(info["reverse_extended"]),
                "n_forward_extended": len(info["forward_extended"]),
                "n_reverse_extended": len(info["reverse_extended"]),
                "total_extended": len(info["all_extended"]),
            }
        )

    mapping_df = pd.DataFrame(mapping_data)
    mapping_filename = os.path.join(outdir, f"{prefix}_extended_mapping.csv")
    mapping_df.to_csv(mapping_filename, index=False)
    print(
        f"[INFO] Base reaction to extended reaction mapping saved to '{mapping_filename}'"
    )

    # Save filtered active base reaction flux data
    flux_filename = os.path.join(outdir, f"{prefix}_filtered_flux.csv")
    active_base_flux_df.to_csv(flux_filename, index=False, encoding="utf-8-sig")
    print(f"[INFO] Filtered active base reactions saved to '{flux_filename}'")

    # Save metadata
    metadata_filename = os.path.join(outdir, f"{prefix}_metadata.csv")
    metadata_for_csv = {
        k: v for k, v in metadata.items() if not isinstance(v, (dict, list, np.ndarray))
    }  # Only save simple types
    metadata_for_csv["n_base_reactions"] = len(comprehensive_df)
    metadata_df = pd.DataFrame([metadata_for_csv])
    metadata_df.to_csv(metadata_filename, index=False)
    print(f"[INFO] Experiment metadata saved to '{metadata_filename}'")

    # Print detailed summary statistics by category
    print("\n[INFO] Detailed Summary Statistics by Category for Base Reactions:")

    for category in [
        "always_positive",
        "always_negative",
        "oscillating",
        "always_zero",
    ]:
        cat_df = comprehensive_df[comprehensive_df["category"] == category]
        if len(cat_df) > 0:
            print(
                f"\n  {category.replace('_', ' ').title()} Base Reactions ({len(cat_df)}):"
            )
            print(
                f"    Zero ratio range: {cat_df['zero_ratio'].min():.4f} - {cat_df['zero_ratio'].max():.4f}"
            )
            print(
                f"    |Max net flux| range: {cat_df['max_abs_flux'].min():.2e} - {cat_df['max_abs_flux'].max():.2e}"
            )
            print(
                f"    Min net flux range: {cat_df['min_flux'].min():.2e} - {cat_df['min_flux'].max():.2e}"
            )
            print(
                f"    Max net flux range: {cat_df['max_flux'].min():.2e} - {cat_df['max_flux'].max():.2e}"
            )
            print(
                f"    Extended reactions per base: {cat_df['total_extended'].min()} - {cat_df['total_extended'].max()}"
            )

            # Count reactions with different flux behaviors
            has_zero_count = cat_df["has_zero"].sum()
            print(f"    Base reactions with zero flux points: {has_zero_count}")

            # Extended reaction statistics
            total_forward_ext = cat_df["n_forward_extended"].sum()
            total_reverse_ext = cat_df["n_reverse_extended"].sum()
            print(f"    Total forward extended reactions: {total_forward_ext}")
            print(f"    Total reverse extended reactions: {total_reverse_ext}")

    print(f"\nTotal base reactions: {len(comprehensive_df)}")
    print(f"Total extended reactions mapped: {metadata['n_extended_reactions']}")


# --- END: Existing helper functions ---

# --- BEGIN: New helpers & updated plot() that compute common base reaction names across generations and burst detection ---


def _get_distinct_generations(conn: DuckDBPyConnection, history_sql: str) -> list:
    """
    Return a sorted list of distinct generation values from the history SQL.

    We treat history_sql as a subquery (it may already be a SELECT ...).
    """
    gen_query = f"SELECT DISTINCT generation FROM ({history_sql}) ORDER BY generation"
    gen_pl = conn.sql(gen_query).pl()
    if gen_pl.is_empty():
        return []
    gen_df = gen_pl.to_pandas()
    # Expect a column named 'generation'
    generations = gen_df["generation"].tolist()
    return generations


def _load_generation_flux_df(
    conn: DuckDBPyConnection, history_sql: str, reaction_ids: list, generation: int
) -> pd.DataFrame:
    """
    Load flux dataframe for a specific generation.

    Returns DataFrame where columns are: 'time' + reaction_ids (extended reaction columns).
    Drops time==0 rows to match the main loader behavior.
    """
    # Build SQL for this generation
    required_columns = [
        "time",
        "generation",
        "listeners__fba_results__reaction_fluxes",
    ]
    sql = f"""
    SELECT {", ".join(required_columns)}
    FROM ({history_sql})
    WHERE generation = {generation}
    ORDER BY time
    """
    df_pl = conn.sql(sql).pl()
    if df_pl.is_empty():
        return pd.DataFrame()  # empty

    flux_matrix = df_pl["listeners__fba_results__reaction_fluxes"].to_numpy()
    flux_matrix = np.array([np.array(row) for row in flux_matrix])
    df_ext = pd.DataFrame(flux_matrix, columns=reaction_ids)

    time_data = df_pl.select(["time"]).to_pandas()
    df_full = pd.concat([time_data, df_ext], axis=1)
    # Drop initial time point where time == 0 to remain consistent
    df_full = df_full[df_full["time"] != 0].reset_index(drop=True)
    return df_full


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
    """
    Preprocesses FBA flux data by mapping extended reactions to base reactions,
    computes net fluxes for base reactions (forward extended - reverse extended),
    categorizes base reactions based on flux behavior, creates unified visualizations,
    and additionally computes base reactions that are common across all generations
    for each category (always_zero, always_positive, always_negative, oscillating).

    Also computes "burst" base reactions for categories always_positive,
    always_negative, and oscillating: a base reaction is considered a burst if
    in every generation the reaction (when in that category) has zero_ratio >
    burst_threshold.

    Modified plotting:
    - Removed simple scatter plots
    - Combined positive and negative scatter plots into unified unidirectional plot

    Saves:
      - Four CSVs with common reactions per category (one-column: base_reaction_name)
      - Three CSVs with burst reactions per category (one-column: base_reaction_name)

    Returns the same tuple as the earlier design plus printing results.
    """

    # Get parameters with defaults
    zero_threshold = params.get("zero_threshold", 0.999)
    eps = params.get("eps", 1e-30)
    epsilon_log = params.get("epsilon_log", 1e-30)
    burst_threshold = params.get("burst_threshold", 0.1)

    print("[INFO] Starting base reaction flux analysis...")
    print(
        f"[INFO] Parameters: zero_threshold={zero_threshold}, eps={eps}, epsilon_log={epsilon_log}, burst_threshold={burst_threshold}"
    )
    print(f"[INFO] Output directory: {outdir}")

    try:
        # Load data (this uses the original loader - we DO NOT modify it)
        df, metadata = load_fba_data(conn, history_sql, config_sql, sim_data_dict)

        # Separate flux data (drop time columns)
        extended_flux_df = df.drop(columns=["time"])

        print(
            f"[INFO] Original data: {len(extended_flux_df.columns)} extended reactions, {len(extended_flux_df)} time points"
        )

        # Map extended reactions to base reactions using the full set
        base_reaction_mapping, extended_to_base_map = map_extended_to_base_reactions(
            extended_flux_df.columns.tolist(), metadata["base_to_extended_mapping"]
        )

        # Compute base reaction fluxes for the entire dataset
        base_flux_df, base_reaction_details = compute_base_reaction_fluxes(
            extended_flux_df, base_reaction_mapping
        )

        # Categorize base reactions by flux behavior for the entire dataset (keeps behavior logic unchanged)
        (
            always_positive,
            always_negative,
            oscillating,
            always_zero,
            base_reaction_categories,
        ) = categorize_base_reactions_by_flux_behavior(base_flux_df, eps)

        # Compute zero-flux ratio per base reaction (using absolute values)
        zero_counts = (np.abs(base_flux_df) <= eps).sum(axis=0)
        zero_ratio = zero_counts / base_flux_df.shape[0]

        # Create comprehensive dataframe with all metrics
        comprehensive_data = []
        for base_reaction in base_flux_df.columns:
            cat_info = base_reaction_categories[base_reaction]
            details = base_reaction_details[base_reaction]
            comprehensive_data.append(
                {
                    "base_reaction": base_reaction,
                    "category": cat_info["category"],
                    "zero_ratio": zero_ratio[base_reaction],
                    "min_flux": cat_info["min_flux"],
                    "max_flux": cat_info["max_flux"],
                    "max_abs_flux": cat_info["max_abs_flux"],
                    "log_max_abs_flux": np.log10(
                        epsilon_log + cat_info["max_abs_flux"]
                    ),
                    "has_positive": cat_info["has_positive"],
                    "has_negative": cat_info["has_negative"],
                    "has_zero": cat_info["has_zero"],
                    "n_forward_extended": details["n_forward_extended"],
                    "n_reverse_extended": details["n_reverse_extended"],
                    "total_extended": details["total_extended"],
                }
            )

        comprehensive_df = pd.DataFrame(comprehensive_data)

        # Filter out base reactions with high zero_ratio
        active_base_reactions = comprehensive_df[
            comprehensive_df["zero_ratio"] < zero_threshold
        ]["base_reaction"].tolist()
        active_base_flux_df = base_flux_df[active_base_reactions]
        print(
            f"\n[INFO] Base reactions remaining after filtering (zero_ratio < {zero_threshold}): {len(active_base_reactions)}"
        )

        # Create separate datasets for visualization
        positive_df = comprehensive_df[
            (comprehensive_df["category"] == "always_positive")
            & (comprehensive_df["max_flux"] > 0)
        ].copy()

        negative_df = comprehensive_df[
            (comprehensive_df["category"] == "always_negative")
            & (comprehensive_df["max_abs_flux"] > 0)
        ].copy()

        oscillating_df = comprehensive_df[
            comprehensive_df["category"] == "oscillating"
        ].copy()
        always_zero_df = comprehensive_df[
            comprehensive_df["category"] == "always_zero"
        ].copy()

        # Print filtering results
        print("\n[INFO] Filtered datasets for visualization:")
        print(f"  - Always positive with max flux > 0: {len(positive_df)}")
        print(f"  - Always negative with max abs flux > 0: {len(negative_df)}")
        print(f"  - Oscillating: {len(oscillating_df)}")
        print(f"  - Always zero: {len(always_zero_df)}")

        # Print top/bottom reactions for each category
        print_base_reaction_category_summaries(
            positive_df, negative_df, oscillating_df, always_zero_df
        )

        # Create unified directional visualization (combines positive and negative)
        create_unified_directional_flux_plot(
            positive_df, negative_df, epsilon_log, base_reaction_details, outdir
        )

        # Save the standard outputs (comprehensive and per-category CSVs, mapping, fluxes, metadata)
        save_base_reaction_results(
            comprehensive_df,
            oscillating_df,
            always_zero_df,
            base_reaction_details,
            base_reaction_mapping,
            active_base_flux_df,
            metadata,
            outdir,
        )

        #
        # NEW: Per-generation category extraction, intersections across generations,
        # and burst detection; save results into separate single-column CSV files.
        #
        print(
            "\n[INFO] Computing per-generation categories, intersections across generations, and burst detection..."
        )

        # 1) get list of distinct generations from the history_sql
        generations = _get_distinct_generations(conn, history_sql)
        print(f"[INFO] Found generations: {generations}")

        # Prepare a dict to collect sets per generation for each category
        per_gen_category_sets = {}
        # categories we care about
        categories = [
            "always_zero",
            "always_positive",
            "always_negative",
            "oscillating",
        ]
        for g in generations:
            per_gen_category_sets[g] = {cat: set() for cat in categories}

        # Also prepare dicts to collect per-generation sets of reactions that satisfy zero_ratio > burst_threshold
        per_gen_burst_candidate_sets = {}
        for g in generations:
            per_gen_burst_candidate_sets[g] = {
                "always_positive": set(),
                "always_negative": set(),
                "oscillating": set(),
            }

        # Make sure outdir exists
        os.makedirs(outdir, exist_ok=True)

        # Helper to write a set of base names to a single-column CSV with header 'base_reaction_name'
        def _write_one_column_csv(base_names_set: set, filepath: str):
            if not base_names_set:
                # create empty dataframe with correct column
                pd.DataFrame(columns=["base_reaction_name"]).to_csv(
                    filepath, index=False
                )
            else:
                df_out = pd.DataFrame(
                    sorted(list(base_names_set)), columns=["base_reaction_name"]
                )
                df_out.to_csv(filepath, index=False)

        # If no generation information found, still create the common and burst CSVs empty
        if len(generations) == 0:
            print(
                "[WARNING] No generations found in history_sql. Creating empty CSV files for common categories and bursts."
            )
            # Common (four categories)
            _write_one_column_csv(
                set(), os.path.join(outdir, "common_base_reactions_always_zero.csv")
            )
            _write_one_column_csv(
                set(), os.path.join(outdir, "common_base_reactions_always_positive.csv")
            )
            _write_one_column_csv(
                set(), os.path.join(outdir, "common_base_reactions_always_negative.csv")
            )
            _write_one_column_csv(
                set(), os.path.join(outdir, "common_base_reactions_oscillating.csv")
            )
            # Burst files (three categories)
            _write_one_column_csv(
                set(), os.path.join(outdir, "burst_base_reaction_always_positive.csv")
            )
            _write_one_column_csv(
                set(), os.path.join(outdir, "burst_base_reaction_always_negative.csv")
            )
            _write_one_column_csv(
                set(), os.path.join(outdir, "burst_base_reaction_oscillating.csv")
            )
            print(f"[INFO] Empty CSVs written to: {outdir}")
        else:
            # For each generation, load its flux data and compute categories and zero ratios
            reaction_ids = metadata["extended_reaction_names"]
            processed_generations = []
            for g in generations:
                print(f"\n[INFO] Processing generation: {g}")
                gen_df = _load_generation_flux_df(conn, history_sql, reaction_ids, g)
                if gen_df.empty:
                    print(
                        f"[WARNING] Generation {g} had no data after filtering time==0. Skipping."
                    )
                    continue

                processed_generations.append(g)

                # Get extended flux data (drop time column)
                gen_extended_flux_df = gen_df.drop(columns=["time"])

                # Map extended to base reactions for this generation (mapping function is unchanged)
                gen_base_reaction_mapping, gen_extended_to_base_map = (
                    map_extended_to_base_reactions(
                        gen_extended_flux_df.columns.tolist(),
                        metadata["base_to_extended_mapping"],
                    )
                )

                # Compute base fluxes for this generation
                gen_base_flux_df, gen_base_reaction_details = (
                    compute_base_reaction_fluxes(
                        gen_extended_flux_df, gen_base_reaction_mapping
                    )
                )

                # Compute zero_ratio per base reaction in this generation (abs <= eps)
                # number of timepoints for the generation:
                n_timepoints_gen = (
                    gen_base_flux_df.shape[0] if gen_base_flux_df.shape[0] > 0 else 1
                )
                gen_zero_counts = (np.abs(gen_base_flux_df) <= eps).sum(axis=0)
                gen_zero_ratio = gen_zero_counts / n_timepoints_gen  # pandas Series

                # Categorize base reactions for this generation using SAME logic (eps retained)
                (
                    gen_always_positive,
                    gen_always_negative,
                    gen_oscillating,
                    gen_always_zero,
                    gen_base_reaction_categories,
                ) = categorize_base_reactions_by_flux_behavior(gen_base_flux_df, eps)

                # Collect sets of category membership for this generation
                per_gen_category_sets[g]["always_zero"] = set(gen_always_zero)
                per_gen_category_sets[g]["always_positive"] = set(gen_always_positive)
                per_gen_category_sets[g]["always_negative"] = set(gen_always_negative)
                per_gen_category_sets[g]["oscillating"] = set(gen_oscillating)

                # For burst detection: within this generation, a reaction is a burst candidate
                # if it belongs to the category and its zero_ratio > burst_threshold.
                # We record such candidates per generation for each category of interest.
                for rxn in gen_always_positive:
                    # default to 0 if missing
                    zr = float(gen_zero_ratio.get(rxn, 1.0))
                    if zr > burst_threshold:
                        per_gen_burst_candidate_sets[g]["always_positive"].add(rxn)
                for rxn in gen_always_negative:
                    zr = float(gen_zero_ratio.get(rxn, 1.0))
                    if zr > burst_threshold:
                        per_gen_burst_candidate_sets[g]["always_negative"].add(rxn)
                for rxn in gen_oscillating:
                    zr = float(gen_zero_ratio.get(rxn, 1.0))
                    if zr > burst_threshold:
                        per_gen_burst_candidate_sets[g]["oscillating"].add(rxn)

                print(
                    f"[INFO] Generation {g} category counts: "
                    f"always_zero={len(gen_always_zero)}, always_positive={len(gen_always_positive)}, "
                    f"always_negative={len(gen_always_negative)}, oscillating={len(gen_oscillating)}"
                )
                print(
                    f"[INFO] Generation {g} burst candidate counts (zr > {burst_threshold}): "
                    f"always_positive={len(per_gen_burst_candidate_sets[g]['always_positive'])}, "
                    f"always_negative={len(per_gen_burst_candidate_sets[g]['always_negative'])}, "
                    f"oscillating={len(per_gen_burst_candidate_sets[g]['oscillating'])}"
                )

            # Use only processed_generations (those with data)
            if len(processed_generations) == 0:
                print(
                    "[WARNING] No generations had usable data. Writing empty CSVs for common and burst outputs."
                )
                # Common (four categories)
                _write_one_column_csv(
                    set(), os.path.join(outdir, "common_base_reactions_always_zero.csv")
                )
                _write_one_column_csv(
                    set(),
                    os.path.join(outdir, "common_base_reactions_always_positive.csv"),
                )
                _write_one_column_csv(
                    set(),
                    os.path.join(outdir, "common_base_reactions_always_negative.csv"),
                )
                _write_one_column_csv(
                    set(), os.path.join(outdir, "common_base_reactions_oscillating.csv")
                )
                # Burst files (three categories)
                _write_one_column_csv(
                    set(),
                    os.path.join(outdir, "burst_base_reaction_always_positive.csv"),
                )
                _write_one_column_csv(
                    set(),
                    os.path.join(outdir, "burst_base_reaction_always_negative.csv"),
                )
                _write_one_column_csv(
                    set(), os.path.join(outdir, "burst_base_reaction_oscillating.csv")
                )
                print(f"[INFO] Empty CSVs written to: {outdir}")
            else:
                # Compute intersection across processed generations for each category (common sets)
                common_by_category = {}
                for cat in categories:
                    sets_list = [
                        per_gen_category_sets[g][cat] for g in processed_generations
                    ]
                    if not sets_list:
                        common = set()
                    else:
                        common = sets_list[0].copy()
                        for s in sets_list[1:]:
                            common &= s
                    common_by_category[cat] = common
                    print(
                        f"[INFO] Common across processed generations for '{cat}': {len(common)}"
                    )

                # Save the common sets (one-column CSVs)
                _write_one_column_csv(
                    common_by_category.get("always_zero", set()),
                    os.path.join(outdir, "common_base_reactions_always_zero.csv"),
                )
                _write_one_column_csv(
                    common_by_category.get("always_positive", set()),
                    os.path.join(outdir, "common_base_reactions_always_positive.csv"),
                )
                _write_one_column_csv(
                    common_by_category.get("always_negative", set()),
                    os.path.join(outdir, "common_base_reactions_always_negative.csv"),
                )
                _write_one_column_csv(
                    common_by_category.get("oscillating", set()),
                    os.path.join(outdir, "common_base_reactions_oscillating.csv"),
                )
                print(
                    f"[INFO] Common base reaction CSV files (one per category) saved into: {outdir}"
                )

                # Compute burst intersection across processed generations for each burst category
                burst_categories = ["always_positive", "always_negative", "oscillating"]
                burst_common_by_category = {}
                for bcat in burst_categories:
                    sets_list = [
                        per_gen_burst_candidate_sets[g][bcat]
                        for g in processed_generations
                    ]
                    if not sets_list:
                        burst_common = set()
                    else:
                        burst_common = sets_list[0].copy()
                        for s in sets_list[1:]:
                            burst_common &= s
                    burst_common_by_category[bcat] = burst_common
                    print(
                        f"[INFO] Burst-common across processed generations for '{bcat}': {len(burst_common)}"
                    )

                # Save burst CSVs (one-column each)
                _write_one_column_csv(
                    burst_common_by_category.get("always_positive", set()),
                    os.path.join(outdir, "burst_base_reaction_always_positive.csv"),
                )
                _write_one_column_csv(
                    burst_common_by_category.get("always_negative", set()),
                    os.path.join(outdir, "burst_base_reaction_always_negative.csv"),
                )
                _write_one_column_csv(
                    burst_common_by_category.get("oscillating", set()),
                    os.path.join(outdir, "burst_base_reaction_oscillating.csv"),
                )
                print(f"[INFO] Burst base reaction CSV files saved into: {outdir}")

        print(
            "\n[INFO] Base reaction flux preprocessing, visualization, common-category extraction, and burst detection complete."
        )
        print(f"[INFO] All files saved to directory: {outdir}")

        # Return same things as original function signature (plus printing above)
        return (
            comprehensive_df,
            active_base_flux_df,
            oscillating_df,
            always_zero_df,
            base_reaction_details,
            base_reaction_mapping,
            metadata,
        )

    except Exception as e:
        print(f"[ERROR] Analysis failed: {str(e)}")
        import traceback

        traceback.print_exc()
        return None
