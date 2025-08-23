"""
This script preprocesses FBA flux data by mapping extended reactions to base reactions,
computes net fluxes for base reactions (forward extended - reverse extended),
categorizes base reactions based on flux behavior (always positive, always negative, or oscillating),
and creates separate visualizations for each category.

Modified to work with DuckDB connection and SQL queries instead of direct file loading.
All outputs are saved to outdir.
"""

import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.stats import gaussian_kde
import os
from typing import Any
from duckdb import DuckDBPyConnection

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


def create_base_reaction_flux_plots(
    flux_data, direction, filename_suffix, epsilon_log, base_reaction_details, outdir
):
    """Create heat scatter and simple scatter plots for base reactions in a given flux direction."""

    if len(flux_data) == 0:
        return

    # Prepare data for plotting
    x = flux_data["log_max_abs_flux"]
    y = flux_data["zero_ratio"]

    # Calculate point density using gaussian_kde for heat scatter plot
    if len(flux_data) > 1:  # Need at least 2 points for KDE
        xy = np.vstack([x, y])
        density = gaussian_kde(xy)(xy)
    else:
        density = np.array([1.0])  # Single point gets density of 1

    # Create hover text with base reaction information
    hover_text = []
    simple_hover_text = []

    for idx, row in flux_data.iterrows():
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

        # For heat scatter (with density)
        hover_text.append(
            f"<b>Base Reaction:</b> {base_reaction}<br>"
            + f"<b>Extended Reactions:</b> {ext_info}<br>"
            + f"<b>Category:</b> {row['category']}<br>"
            + f"<b>Zero Ratio:</b> {row['zero_ratio']:.4f}<br>"
            + f"<b>Max |Net Flux|:</b> {row['max_abs_flux']:.2e}<br>"
            + f"<b>Min Net Flux:</b> {row['min_flux']:.2e}<br>"
            + f"<b>Max Net Flux:</b> {row['max_flux']:.2e}<br>"
            + f"<b>Log |Max Net Flux|:</b> {row['log_max_abs_flux']:.2f}<br>"
            + f"<b>Point Density:</b> {density[list(flux_data.index).index(idx)]:.6f}"
        )

        # For simple scatter (without density)
        simple_hover_text.append(
            f"<b>Base Reaction:</b> {base_reaction}<br>"
            + f"<b>Extended Reactions:</b> {ext_info}<br>"
            + f"<b>Category:</b> {row['category']}<br>"
            + f"<b>Zero Ratio:</b> {row['zero_ratio']:.4f}<br>"
            + f"<b>Max |Net Flux|:</b> {row['max_abs_flux']:.2e}<br>"
            + f"<b>Min Net Flux:</b> {row['min_flux']:.2e}<br>"
            + f"<b>Max Net Flux:</b> {row['max_flux']:.2e}<br>"
            + f"<b>Log |Max Net Flux|:</b> {row['log_max_abs_flux']:.2f}"
        )

    # 1. HEAT SCATTER PLOT WITH MARGINAL HISTOGRAMS
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

    # Main scatter plot (bottom left, row=2, col=1)
    fig_heat.add_trace(
        go.Scatter(
            x=x,
            y=y,
            mode="markers",
            marker=dict(
                size=8,
                color=density,
                colorscale="Plasma",
                opacity=0.8,
                colorbar=dict(
                    title=dict(text="Point Density", font=dict(size=14)),
                    tickfont=dict(size=12),
                    thickness=15,
                    len=0.7,
                    x=1.02,  # Position colorbar to the right
                ),
                line=dict(width=0.5, color="white"),
            ),
            text=hover_text,
            hovertemplate="%{text}<extra></extra>",
            name="Base Reactions",
            showlegend=False,
        ),
        row=2,
        col=1,
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
            text=f"<b>Heat Scatter Plot with Marginal Density Curves: {direction} Base Reaction Net Flux</b>",
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
        f"Base Reactions ({direction}): {len(flux_data):,}<br>"
        + f"ε = {epsilon_log:.0e}<br>"
        + f"|Max Net Flux| Range: {flux_data['max_abs_flux'].min():.2e} to {flux_data['max_abs_flux'].max():.2e}<br>"
        + f"Zero Ratio Range: {flux_data['zero_ratio'].min():.4f} to {flux_data['zero_ratio'].max():.4f}<br>"
        + f"Extended Reactions Range: {flux_data['total_extended'].min()} to {flux_data['total_extended'].max()}"
    )

    if len(flux_data) > 1:
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

    # 2. SIMPLE SCATTER PLOT (original version without histograms)
    fig_simple = go.Figure()

    fig_simple.add_trace(
        go.Scatter(
            x=x,
            y=y,
            mode="markers",
            marker=dict(
                size=6,
                color="steelblue",
                opacity=0.7,
                line=dict(width=0.5, color="white"),
            ),
            text=simple_hover_text,
            hovertemplate="%{text}<extra></extra>",
            name="Base Reactions",
        )
    )

    fig_simple.update_layout(
        title=dict(
            text=f"<b>Simple Scatter Plot: {direction} Base Reaction Net Flux - Zero Ratio vs |Max Net Flux|</b>",
            font=dict(size=18),
            x=0.5,
            xanchor="center",
        ),
        xaxis=dict(
            title=dict(text="log₁₀(ε + |Max Net Flux|)", font=dict(size=14)),
            tickfont=dict(size=12),
            gridcolor="rgba(128,128,128,0.2)",
            gridwidth=1,
        ),
        yaxis=dict(
            title=dict(text="Zero Flux Ratio", font=dict(size=14)),
            tickfont=dict(size=12),
            gridcolor="rgba(128,128,128,0.2)",
            gridwidth=1,
        ),
        plot_bgcolor="white",
        paper_bgcolor="white",
        font=dict(family="Arial", size=12),
        width=900,
        height=700,
        margin=dict(l=80, r=80, t=100, b=80),
        showlegend=False,
    )

    # Add statistics for simple plot
    simple_stats_text = (
        f"Base Reactions ({direction}): {len(flux_data):,}<br>"
        + f"ε = {epsilon_log:.0e}<br>"
        + f"|Max Net Flux| Range: {flux_data['max_abs_flux'].min():.2e} to {flux_data['max_abs_flux'].max():.2e}<br>"
        + f"Zero Ratio Range: {flux_data['zero_ratio'].min():.4f} to {flux_data['zero_ratio'].max():.4f}<br>"
        + f"Extended Reactions Range: {flux_data['total_extended'].min()} to {flux_data['total_extended'].max()}"
    )

    fig_simple.add_annotation(
        x=0.02,
        y=0.48,
        xref="paper",
        yref="paper",
        text=simple_stats_text,
        showarrow=False,
        font=dict(size=11, color="black"),
        bgcolor="rgba(255,255,255,0.9)",
        bordercolor="rgba(128,128,128,0.5)",
        borderwidth=1,
        borderpad=10,
        xanchor="left",
        yanchor="top",
    )

    # Save plots
    heat_filename = os.path.join(
        outdir,
        f"heat_scatter_with_density_curves_base_reactions_{filename_suffix}.html",
    )
    simple_filename = os.path.join(
        outdir, f"simple_scatter_plot_base_reactions_{filename_suffix}.html"
    )

    fig_heat.write_html(heat_filename)
    fig_simple.write_html(simple_filename)

    print(f"\n[INFO] {direction} base reaction flux plots saved:")
    print(f"  - {heat_filename}")
    print(f"  - {simple_filename}")


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
    categorizes base reactions based on flux behavior, and creates separate
    visualizations for each category with marginal histograms.

    Parameters from params dict:
    - zero_threshold: Fraction of zeros above which a base reaction is considered inactive (default: 0.999)
    - eps: Small tolerance for zero comparison (default: 1e-30)
    - epsilon_log: Small value to add to max flux for log transformation (default: 1e-30)
    """

    # Get parameters with defaults
    zero_threshold = params.get("zero_threshold", 0.999)
    eps = params.get("eps", 1e-30)
    epsilon_log = params.get("epsilon_log", 1e-30)

    print("[INFO] Starting base reaction flux analysis...")
    print(
        f"[INFO] Parameters: zero_threshold={zero_threshold}, eps={eps}, epsilon_log={epsilon_log}"
    )
    print(f"[INFO] Output directory: {outdir}")

    try:
        # Load data
        df, metadata = load_fba_data(conn, history_sql, config_sql, sim_data_dict)

        # Separate flux data (drop time columns)
        extended_flux_df = df.drop(columns=["time"])

        print(
            f"[INFO] Original data: {len(extended_flux_df.columns)} extended reactions, {len(extended_flux_df)} time points"
        )

        # Map extended reactions to base reactions
        base_reaction_mapping, extended_to_base_map = map_extended_to_base_reactions(
            extended_flux_df.columns.tolist(), metadata["base_to_extended_mapping"]
        )

        # Compute base reaction fluxes
        base_flux_df, base_reaction_details = compute_base_reaction_fluxes(
            extended_flux_df, base_reaction_mapping
        )

        # Categorize base reactions by flux behavior
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

        # Create visualizations with marginal histograms
        if len(positive_df) > 0:
            create_base_reaction_flux_plots(
                positive_df,
                "Always Positive",
                "always_positive",
                epsilon_log,
                base_reaction_details,
                outdir,
            )
        else:
            print(
                "\n[WARNING] No always positive base reactions found. Skipping positive plots."
            )

        if len(negative_df) > 0:
            create_base_reaction_flux_plots(
                negative_df,
                "Always Negative",
                "always_negative",
                epsilon_log,
                base_reaction_details,
                outdir,
            )
        else:
            print(
                "\n[WARNING] No always negative base reactions found. Skipping negative plots."
            )

        # Save results
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

        print("\n[INFO] Base reaction flux preprocessing and visualization complete.")
        print(f"[INFO] All files saved to directory: {outdir}")

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
