"""
This script preprocesses FBA flux data by mapping extended reactions to base reactions,
computes net fluxes for base reactions (forward extended - reverse extended),
identifies oscillating base reactions (those that take both positive and negative values),
computes dynamic metrics for oscillating reactions (positive/negative ratios, oscillation times/frequency),
and creates a semi-log heat scatter plot (with marginal density curves) focused on oscillating reactions.

Modified to work with DuckDB connection and SQL queries instead of direct file loading.
All outputs are saved to outdir.

Key changes from the original:
1. Removed unified directional combined heat/scatter plotting (no Always Positive / Always Negative / Always Zero plots).
2. Removed per-category statistics for always positive / always negative / always zero; focus is only on oscillating reactions.
3. Added a semi-log heat scatter plot for oscillating reactions:
   - x axis: oscillation_frequency = oscillation_times / total_timepoints
     where oscillation_times is the number of sign changes in the sequence after removing near-zero points.
   - y axis: log10(positive_ratio / negative_ratio), where ratios are computed using epsilon thresholds.
   - Both axes have marginal density curves (KDE).
   - All comparisons to zero use eps: positive if > eps, negative if < -eps, zero if abs <= eps.

eps can be set use params.
"""

import os
from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd
from duckdb import DuckDBPyConnection
from scipy.stats import gaussian_kde
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from ecoli.library.parquet_emitter import field_metadata
from ecoli.analysis.utils import create_base_to_extended_mapping


def load_fba_data(
    conn: DuckDBPyConnection, history_sql: str, config_sql: str, sim_data_dict: dict
) -> Tuple[pd.DataFrame, dict]:
    """
    Load FBA flux data using DuckDB connection and SQL queries.

    Returns:
    - df: DataFrame with flux data (time points x extended reactions, with 'time' column included)
    - metadata: Dictionary with experiment metadata
    """
    print("[INFO] Loading FBA flux data via SQL...")

    try:
        base_to_extended_mapping = create_base_to_extended_mapping(sim_data_dict)
        if not base_to_extended_mapping:
            raise Exception("Could not create base to extended reaction mapping")

        reaction_ids = field_metadata(
            conn, config_sql, "listeners__fba_results__reaction_fluxes"
        )
        print(f"[INFO] Total reactions in sim_data: {len(reaction_ids)}")

        required_columns = [
            "time",
            "generation",
            "listeners__fba_results__reaction_fluxes",
        ]

        sql = f"""
        SELECT {", ".join(required_columns)}
        FROM ({history_sql})
        ORDER BY generation, time
        """

        df_pl = conn.sql(sql).pl()

        if df_pl.is_empty():
            raise Exception("No data found")
        print(f"[INFO] Loaded data with {df_pl.height} time steps")

        flux_matrix = df_pl["listeners__fba_results__reaction_fluxes"].to_numpy()
        flux_matrix = np.array([np.array(row) for row in flux_matrix])

        df = pd.DataFrame(flux_matrix, columns=reaction_ids)

        time_data = df_pl.select(["time"]).to_pandas()
        df = pd.concat([time_data, df], axis=1)

        # Drop initial time point where time == 0
        df = df[df["time"] != 0].reset_index(drop=True)

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
        base_reaction = extended_to_base_map.get(extended_reaction)

        if base_reaction is None:
            print(
                f"[WARNING] No base reaction found for extended reaction: {extended_reaction}"
            )
            continue

        if base_reaction not in base_reaction_mapping:
            base_reaction_mapping[base_reaction] = {
                "forward_extended": [],
                "reverse_extended": [],
                "all_extended": [],
            }

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

    # Print concise distribution summary (only informative)
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


def compute_base_reaction_fluxes(flux_df: pd.DataFrame, base_reaction_mapping: dict):
    """
    Compute base reaction net fluxes (forward - reverse).
    Returns:
    - base_flux_df: DataFrame with base reaction net flux values (columns = base reactions)
    - base_reaction_details: dict with details per base reaction
    """
    base_flux_data = {}
    base_reaction_details = {}

    for base_reaction, info in base_reaction_mapping.items():
        forward_extended = info["forward_extended"]
        reverse_extended = info["reverse_extended"]

        forward_flux = pd.Series(0.0, index=flux_df.index)
        if forward_extended:
            for ext_reaction in forward_extended:
                if ext_reaction in flux_df.columns:
                    forward_flux = forward_flux.add(
                        flux_df[ext_reaction], fill_value=0.0
                    )

        reverse_flux = pd.Series(0.0, index=flux_df.index)
        if reverse_extended:
            for ext_reaction in reverse_extended:
                if ext_reaction in flux_df.columns:
                    reverse_flux = reverse_flux.add(
                        flux_df[ext_reaction], fill_value=0.0
                    )

        net_flux = forward_flux - reverse_flux
        base_flux_data[base_reaction] = net_flux

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


def categorize_base_reactions_by_flux_behavior(
    base_flux_df: pd.DataFrame, eps: float = 1e-30
):
    """
    Categorize base reactions by behavior using epsilon threshold.
    Only returns oscillating list in informative print; full categories still returned (for possible downstream needs).
    """
    always_positive = []
    always_negative = []
    oscillating = []
    always_zero = []
    base_reaction_categories = {}

    for base_reaction in base_flux_df.columns:
        flux_values = base_flux_df[base_reaction].values

        has_positive = np.any(flux_values > eps)
        has_negative = np.any(flux_values < -eps)
        has_zero = np.any(np.abs(flux_values) <= eps)

        min_flux = flux_values.min()
        max_flux = flux_values.max()
        max_abs_flux = np.max(np.abs(flux_values))

        if max_abs_flux <= eps:
            always_zero.append(base_reaction)
            category = "always_zero"
        elif not has_negative and has_positive:
            always_positive.append(base_reaction)
            category = "always_positive"
        elif not has_positive and has_negative:
            always_negative.append(base_reaction)
            category = "always_negative"
        elif has_positive and has_negative:
            oscillating.append(base_reaction)
            category = "oscillating"
        else:
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
    print(f"  - Oscillating (changes sign): {len(oscillating)}")

    return (
        always_positive,
        always_negative,
        oscillating,
        always_zero,
        base_reaction_categories,
    )


def compute_oscillating_dynamics_metrics(
    base_flux_df: pd.DataFrame, oscillating_reactions: list, eps: float = 1e-30
) -> pd.DataFrame:
    """
    Compute dynamic metrics for oscillating reactions:
    - positive_ratio, negative_ratio, zero_ratio (all relative to total timepoints)
    - oscillation_times: number of sign changes in the non-zero filtered sequence
    - oscillation_frequency = oscillation_times / total_timepoints
    - log_pos_neg_ratio = log10(positive_ratio / negative_ratio), handled safely with eps
    """
    oscillating_data = []

    for base_reaction in oscillating_reactions:
        flux_values = base_flux_df[base_reaction].values
        n_timepoints = len(flux_values)

        positive_count = np.sum(flux_values > eps)
        negative_count = np.sum(flux_values < -eps)
        zero_count = np.sum(np.abs(flux_values) <= eps)

        positive_ratio = positive_count / n_timepoints
        negative_ratio = negative_count / n_timepoints
        zero_ratio = zero_count / n_timepoints

        # Remove near-zero values for oscillation counting
        non_zero_mask = np.abs(flux_values) > eps
        non_zero_flux = flux_values[non_zero_mask]

        if len(non_zero_flux) <= 1:
            oscillation_times = 0
            oscillation_frequency = 0.0
        else:
            # Count sign changes in consecutive non-zero sequence
            signs = np.sign(non_zero_flux)
            sign_changes = np.sum(signs[1:] != signs[:-1])
            oscillation_times = int(sign_changes)
            oscillation_frequency = oscillation_times / n_timepoints

        # Compute log ratio safely
        if negative_ratio > 0:
            log_pos_neg_ratio = np.log10(positive_ratio / negative_ratio)
        elif positive_ratio > 0:
            log_pos_neg_ratio = np.log10(positive_ratio / eps)
        else:
            # Both zero (shouldn't be oscillating) -> fallback 0
            log_pos_neg_ratio = 0.0

        oscillating_data.append(
            {
                "base_reaction": base_reaction,
                "positive_ratio": positive_ratio,
                "negative_ratio": negative_ratio,
                "zero_ratio": zero_ratio,
                "positive_count": int(positive_count),
                "negative_count": int(negative_count),
                "zero_count": int(zero_count),
                "oscillation_times": oscillation_times,
                "oscillation_frequency": oscillation_frequency,
                "log_pos_neg_ratio": log_pos_neg_ratio,
                "n_timepoints": n_timepoints,
                "non_zero_points": int(len(non_zero_flux)),
                "min_flux": float(np.min(flux_values)),
                "max_flux": float(np.max(flux_values)),
                "max_abs_flux": float(np.max(np.abs(flux_values))),
            }
        )

    oscillating_metrics = pd.DataFrame(oscillating_data)

    if len(oscillating_metrics) > 0:
        print(
            f"\n[INFO] Oscillating dynamics metrics computed for {len(oscillating_metrics)} reactions:"
        )
        print(
            f"  - Oscillation frequency range: {oscillating_metrics['oscillation_frequency'].min():.4f} to {oscillating_metrics['oscillation_frequency'].max():.4f}"
        )
        print(
            f"  - Log(pos/neg ratio) range: {oscillating_metrics['log_pos_neg_ratio'].min():.4f} to {oscillating_metrics['log_pos_neg_ratio'].max():.4f}"
        )
        print(
            f"  - Positive ratio range: {oscillating_metrics['positive_ratio'].min():.4f} to {oscillating_metrics['positive_ratio'].max():.4f}"
        )
        print(
            f"  - Negative ratio range: {oscillating_metrics['negative_ratio'].min():.4f} to {oscillating_metrics['negative_ratio'].max():.4f}"
        )
    else:
        print("\n[WARN] No oscillating metrics computed (empty list).")

    return oscillating_metrics


def create_oscillating_dynamics_plot(
    oscillating_metrics: pd.DataFrame,
    base_reaction_details: Dict[str, dict],
    outdir: str,
    eps: float = 1e-30,
):
    """
    Create a semi-log heat scatter plot for oscillating reactions with marginal density curves.
    - X-axis: oscillation_frequency
    - Y-axis: log10(positive_ratio / negative_ratio)
    Saves HTML file to outdir.
    """
    if oscillating_metrics is None or len(oscillating_metrics) == 0:
        print("[WARNING] No oscillating reactions found for dynamics plot.")
        return

    # Prepare data
    x = oscillating_metrics["oscillation_frequency"]
    y = oscillating_metrics["log_pos_neg_ratio"]

    # Compute density for scatter coloring
    if len(oscillating_metrics) > 1:
        xy = np.vstack([x, y])
        try:
            density = gaussian_kde(xy)(xy)
        except Exception:
            # If KDE fails due to singular matrix etc., fallback to 1D product KDE
            density = np.ones(len(x))
    else:
        density = np.array([1.0])

    # Hover text
    hover_text = []
    for idx, row in oscillating_metrics.iterrows():
        base_reaction = row["base_reaction"]
        details = base_reaction_details.get(base_reaction, {})
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
            + f"<b>Oscillation Frequency:</b> {row['oscillation_frequency']:.6f}<br>"
            + f"<b>Oscillation Times:</b> {row['oscillation_times']}<br>"
            + f"<b>Log(Pos/Neg Ratio):</b> {row['log_pos_neg_ratio']:.4f}<br>"
            + f"<b>Positive Ratio:</b> {row['positive_ratio']:.4f}<br>"
            + f"<b>Negative Ratio:</b> {row['negative_ratio']:.4f}<br>"
            + f"<b>Zero Ratio:</b> {row['zero_ratio']:.4f}<br>"
            + f"<b>Max |Net Flux|:</b> {row['max_abs_flux']:.2e}<br>"
            + f"<b>Min Net Flux:</b> {row['min_flux']:.2e}<br>"
            + f"<b>Max Net Flux:</b> {row['max_flux']:.2e}<br>"
            + f"<b>Non-zero Points:</b> {row['non_zero_points']}/{row['n_timepoints']}<br>"
            + f"<b>Point Density:</b> {density[idx]:.6e}"
        )

    # Build subplots: top density (x), right density (y), main scatter bottom-left
    fig = make_subplots(
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
    )

    # Main scatter (row=2, col=1)
    fig.add_trace(
        go.Scatter(
            x=x,
            y=y,
            mode="markers",
            marker=dict(
                size=8,
                color=density,
                colorscale="Viridis",
                opacity=0.85,
                colorbar=dict(
                    title=dict(text="Point Density", font=dict(size=12)),
                    tickfont=dict(size=11),
                    thickness=15,
                    len=0.7,
                    x=1.02,
                ),
                line=dict(width=0.5, color="white"),
            ),
            text=hover_text,
            hovertemplate="%{text}<extra></extra>",
            name="Oscillating Reactions",
            showlegend=False,
        ),
        row=2,
        col=1,
    )

    # Top density (x distribution)
    if len(x) > 1:
        x_range = np.linspace(x.min(), x.max(), 250)
        try:
            x_density = gaussian_kde(x)
            x_density_values = x_density(x_range)
        except Exception:
            x_density_values = np.ones_like(x_range)
    else:
        x_range = np.array([x.iloc[0]])
        x_density_values = np.array([1.0])

    fig.add_trace(
        go.Scatter(
            x=x_range,
            y=x_density_values,
            mode="lines",
            line=dict(color="darkorange", width=3),
            fill="tozeroy",
            fillcolor="rgba(255, 140, 0, 0.25)",
            showlegend=False,
        ),
        row=1,
        col=1,
    )

    # Right density (y distribution)
    if len(y) > 1:
        y_range = np.linspace(y.min(), y.max(), 250)
        try:
            y_density = gaussian_kde(y)
            y_density_values = y_density(y_range)
        except Exception:
            y_density_values = np.ones_like(y_range)
    else:
        y_range = np.array([y.iloc[0]])
        y_density_values = np.array([1.0])

    fig.add_trace(
        go.Scatter(
            x=y_density_values,
            y=y_range,
            mode="lines",
            line=dict(color="darkgreen", width=3),
            fill="tozerox",
            fillcolor="rgba(0, 100, 0, 0.25)",
            showlegend=False,
        ),
        row=2,
        col=2,
    )

    # Layout and axes
    fig.update_layout(
        title=dict(
            text="<b>Oscillating Base Reactions: Frequency vs Positive/Negative Bias (semi-log)</b>",
            font=dict(size=18),
            x=0.5,
        ),
        plot_bgcolor="white",
        paper_bgcolor="white",
        font=dict(family="Arial", size=12),
        width=1000,
        height=800,
        margin=dict(l=80, r=120, t=120, b=80),
    )

    fig.update_xaxes(
        title=dict(
            text="Oscillation Frequency (oscillations per time step)",
            font=dict(size=14),
        ),
        row=2,
        col=1,
        gridcolor="rgba(128,128,128,0.2)",
        gridwidth=1,
    )
    fig.update_yaxes(
        title=dict(text="log₁₀(Positive Ratio / Negative Ratio)", font=dict(size=14)),
        row=2,
        col=1,
        gridcolor="rgba(128,128,128,0.2)",
        gridwidth=1,
    )

    # Add hline at y=0 (equal pos/neg)
    fig.add_hline(
        y=0,
        line=dict(color="red", width=1, dash="dash"),
        annotation=dict(text="Equal Pos/Neg Bias", font=dict(size=10, color="red")),
        row=2,
        col=1,
    )

    # Hide top-right subplot (unused)
    fig.update_xaxes(visible=False, row=1, col=2)
    fig.update_yaxes(visible=False, row=1, col=2)

    # Remove ticks for density subplots
    fig.update_xaxes(showticklabels=False, row=1, col=1)
    fig.update_yaxes(showticklabels=False, row=1, col=1)
    fig.update_xaxes(showticklabels=False, row=2, col=2)
    fig.update_yaxes(showticklabels=False, row=2, col=2)

    # Stats annotation
    stats_text = (
        f"Oscillating Base Reactions: {len(oscillating_metrics):,}<br>"
        + f"ε = {eps:.0e}<br>"
        + f"Oscillation Frequency Range: {oscillating_metrics['oscillation_frequency'].min():.4f} to {oscillating_metrics['oscillation_frequency'].max():.4f}<br>"
        + f"Log(Pos/Neg Ratio) Range: {oscillating_metrics['log_pos_neg_ratio'].min():.4f} to {oscillating_metrics['log_pos_neg_ratio'].max():.4f}<br>"
        + f"Oscillation Times Range: {oscillating_metrics['oscillation_times'].min()} to {oscillating_metrics['oscillation_times'].max()}<br>"
        + f"Max |Net Flux| Range: {oscillating_metrics['max_abs_flux'].min():.2e} to {oscillating_metrics['max_abs_flux'].max():.2e}"
    )
    if len(oscillating_metrics) > 1:
        stats_text += f"<br>Density Range: {density.min():.2e} - {density.max():.2e}"

    fig.add_annotation(
        x=0.42,
        y=0.78,
        xref="paper",
        yref="paper",
        text=stats_text,
        showarrow=False,
        font=dict(size=11, color="black"),
        bgcolor="rgba(255,255,255,0.95)",
        bordercolor="rgba(128,128,128,0.5)",
        borderwidth=1,
        borderpad=8,
        xanchor="left",
        yanchor="top",
    )

    # Save plot
    filename = os.path.join(outdir, "oscillating_reactions_dynamics_analysis.html")
    os.makedirs(outdir, exist_ok=True)
    fig.write_html(filename)
    print("\n[INFO] Oscillating reactions dynamics plot saved:")
    print(f"  - {filename}")


def print_oscillating_reaction_summaries(oscillating_metrics: pd.DataFrame):
    """Print top summaries for oscillating reactions."""
    if oscillating_metrics is None or len(oscillating_metrics) == 0:
        print("[INFO] No oscillating reactions to summarize.")
        return

    print(
        "\n[INFO] Oscillating Base Reactions - Top 10 most dynamic (highest oscillation frequency):"
    )
    top_dynamic = oscillating_metrics.sort_values(
        "oscillation_frequency", ascending=False
    ).head(10)
    for _, row in top_dynamic.iterrows():
        print(
            f"  {row['base_reaction']}: freq={row['oscillation_frequency']:.6f}, "
            f"times={row['oscillation_times']}, pos_ratio={row['positive_ratio']:.3f}, "
            f"neg_ratio={row['negative_ratio']:.3f}, log_ratio={row['log_pos_neg_ratio']:.3f}"
        )

    print(
        "\n[INFO] Oscillating Base Reactions - Top 10 most positive-biased (highest log pos/neg ratio):"
    )
    top_positive_biased = oscillating_metrics.sort_values(
        "log_pos_neg_ratio", ascending=False
    ).head(10)
    for _, row in top_positive_biased.iterrows():
        print(
            f"  {row['base_reaction']}: log_ratio={row['log_pos_neg_ratio']:.3f}, "
            f"pos_ratio={row['positive_ratio']:.3f}, neg_ratio={row['negative_ratio']:.3f}, "
            f"freq={row['oscillation_frequency']:.6f}"
        )

    print(
        "\n[INFO] Oscillating Base Reactions - Top 10 most negative-biased (lowest log pos/neg ratio):"
    )
    top_negative_biased = oscillating_metrics.sort_values(
        "log_pos_neg_ratio", ascending=True
    ).head(10)
    for _, row in top_negative_biased.iterrows():
        print(
            f"  {row['base_reaction']}: log_ratio={row['log_pos_neg_ratio']:.3f}, "
            f"pos_ratio={row['positive_ratio']:.3f}, neg_ratio={row['negative_ratio']:.3f}, "
            f"freq={row['oscillation_frequency']:.6f}"
        )


def save_oscillating_results(
    oscillating_metrics: pd.DataFrame,
    base_reaction_details: dict,
    base_reaction_mapping: dict,
    metadata: dict,
    outdir: str,
):
    """Save oscillating reaction results to CSV files with experiment metadata in outdir."""
    os.makedirs(outdir, exist_ok=True)
    prefix = "oscillating_reaction_analysis"

    # Save metrics
    metrics_filename = os.path.join(outdir, f"{prefix}_dynamics_metrics.csv")
    oscillating_metrics.to_csv(metrics_filename, index=False)
    print(
        f"\n[INFO] Oscillating reaction dynamics metrics saved to '{metrics_filename}'"
    )

    # Save mapping for oscillating reactions
    oscillating_reactions = (
        set(oscillating_metrics["base_reaction"].tolist())
        if len(oscillating_metrics) > 0
        else set()
    )
    mapping_data = []
    for base_reaction, info in base_reaction_mapping.items():
        if base_reaction in oscillating_reactions:
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

    if mapping_data:
        mapping_df = pd.DataFrame(mapping_data)
        mapping_filename = os.path.join(outdir, f"{prefix}_extended_mapping.csv")
        mapping_df.to_csv(mapping_filename, index=False)
        print(
            f"[INFO] Oscillating reaction extended mapping saved to '{mapping_filename}'"
        )
    else:
        print("[INFO] No oscillating mapping data to save.")

    # Save metadata (only simple types)
    metadata_filename = os.path.join(outdir, f"{prefix}_metadata.csv")
    metadata_for_csv = {
        k: v for k, v in metadata.items() if not isinstance(v, (dict, list, np.ndarray))
    }
    metadata_for_csv["n_oscillating_reactions"] = (
        len(oscillating_metrics) if oscillating_metrics is not None else 0
    )
    metadata_df = pd.DataFrame([metadata_for_csv])
    metadata_df.to_csv(metadata_filename, index=False)
    print(f"[INFO] Oscillating analysis metadata saved to '{metadata_filename}'")


def plot(
    params: dict[str, Any],
    conn: DuckDBPyConnection,
    history_sql: str,
    config_sql: str,
    success_sql: str,  # kept for compatibility though not used in this focused analysis
    sim_data_dict: dict[str, dict[int, str]],
    validation_data_paths: list[str],
    outdir: str,
    variant_metadata: dict[str, dict[int, Any]],
    variant_names: dict[str, str],
):
    """
    Main entry point: loads data via DuckDB, computes base net fluxes, finds oscillating reactions,
    computes dynamic metrics for oscillating reactions, produces the semi-log heat scatter plot,
    and saves results to outdir.

    params keys:
      - eps: tolerance for zero comparisons (default 1e-30)
    """
    eps = params.get("eps", 1e-30)

    print("[INFO] Starting oscillating base reaction analysis...")
    print(f"[INFO] Parameters: eps={eps}")
    print(f"[INFO] Output directory: {outdir}")

    try:
        # Load data
        df, metadata = load_fba_data(conn, history_sql, config_sql, sim_data_dict)

        # Extract extended fluxes (drop 'time')
        if "time" in df.columns:
            extended_flux_df = df.drop(columns=["time"])
        else:
            extended_flux_df = df.copy()

        print(
            f"[INFO] Loaded extended flux matrix: {extended_flux_df.shape[1]} extended reactions, {extended_flux_df.shape[0]} time points"
        )

        # Map extended -> base reactions
        base_reaction_mapping, extended_to_base_map = map_extended_to_base_reactions(
            extended_flux_df.columns.tolist(), metadata["base_to_extended_mapping"]
        )

        # Compute base net fluxes
        base_flux_df, base_reaction_details = compute_base_reaction_fluxes(
            extended_flux_df, base_reaction_mapping
        )

        # Categorize to find oscillating
        _, _, oscillating, _, base_reaction_categories = (
            categorize_base_reactions_by_flux_behavior(base_flux_df, eps)
        )

        if len(oscillating) == 0:
            print("[INFO] No oscillating base reactions detected. Exiting.")
            return None

        # Compute oscillating dynamics metrics
        oscillating_metrics = compute_oscillating_dynamics_metrics(
            base_flux_df, oscillating, eps
        )

        # Print oscillating summaries
        print_oscillating_reaction_summaries(oscillating_metrics)

        # Create plot
        create_oscillating_dynamics_plot(
            oscillating_metrics, base_reaction_details, outdir, eps
        )

        # Save oscillating results
        save_oscillating_results(
            oscillating_metrics,
            base_reaction_details,
            base_reaction_mapping,
            metadata,
            outdir,
        )

        print("\n[INFO] Oscillating base reaction analysis complete.")
        print(f"[INFO] All files saved to directory: {outdir}")

        return {
            "oscillating_metrics": oscillating_metrics,
            "base_reaction_details": base_reaction_details,
            "base_reaction_mapping": base_reaction_mapping,
            "metadata": metadata,
        }

    except Exception as e:
        print(f"[ERROR] Analysis failed: {str(e)}")
        import traceback

        traceback.print_exc()
        return None
