"""
Small Molecule Total vs Free Counts Plotly!

For each tracked small molecule, this plots the time-averaged total count
against the time-averaged free count. Points below the y=x line have
small molecules sequestered in equilibrium complexes, two component system
(TCS) complexes, and/or DNA-bound transcription factors (TFs).

Free counts are read straight from the standard ``bulk`` table and total counts
come directly from the SmallMoleculeCounts listener's totalSmallMoleculeCounts.
The sequestration breakdown is recomputed from stoich maps using the
compute_avg_sequestration() function.

This file produces two plots:
  1. small_molecule_total_vs_free_counts.html -- active small molecules
     highlighted by sequestration type.
  2. small_molecule_total_vs_free_counts_highlighted.html -- all points in one
     color with user-specified small molecules highlighted in red.

NOTE: to keep the plots readable, this file only plots small molecules that are
"active" in the simulation (i.e. are trackable via bulk molecules and have
nonzero counts at some point during the sim).
"""

import os
import pickle
from typing import Any
import numpy as np
import plotly.graph_objects as go
import polars as pl
from duckdb import DuckDBPyConnection

from ecoli.library.parquet_emitter import (
    field_metadata,
    open_arbitrary_sim_data,
)
from ecoli.library.small_molecule_sequestration import compute_avg_sequestration
from ecoli.library.small_molecule_analysis import (
    CATEGORY_COLORS,
    categorize_expr,
    count_cells,
    read_avg_free_from_bulk,
    read_avg_listener_list,
    resolve_highlights,
)

# Default highlight list if "highlight_metabolites" is not given in params:
DEFAULT_HIGHLIGHT_METABOLITES = ["ATP", "LEU[c]", "TRP"]


def _hover_text(rows):
    """Build the per-point hover strings (shared by both figures)."""
    return [
        f"<b>{row['metabolite']}</b><br>"
        + f"Avg. total: {row['avg_total']:.1f}<br>"
        + f"Avg. free: {row['avg_free']:.1f}<br>"
        + f"Avg. bound (total): {row['avg_bound_total']:.1f}<br>"
        + f"  in equilibrium complexes: {row['avg_eq_bound']:.1f}<br>"
        + f"  in two component system complexes: {row['avg_tcs_bound']:.1f}<br>"
        + f"  in DNA-bound TFs: {row['avg_bound_tf']:.1f}<br>"
        + f"Fraction free: {row['fraction_free']:.3f}"
        for row in rows
    ]


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

    # Load the small molecule IDs from listener metadata:
    sm_ids = field_metadata(
        conn, config_sql, "listeners__small_molecule_counts__totalSmallMoleculeCounts"
    )

    # Optionally skip the first N generations for all averages computed:
    skip_n_gens = params.get("skip_n_gens", 0)
    if skip_n_gens:
        history_sql = f"SELECT * FROM ({history_sql}) WHERE generation >= {skip_n_gens}"

    # Read in total counts from the listener and free counts straight from
    # the standard bulk table:
    avg_total = read_avg_listener_list(
        conn, history_sql, "listeners__small_molecule_counts__totalSmallMoleculeCounts"
    )

    with open_arbitrary_sim_data(sim_data_dict) as f:
        sim_data = pickle.load(f)

    avg_free = read_avg_free_from_bulk(conn, history_sql, sim_data, sm_ids)

    # Recompute the time-averaged sequestration breakdown:
    seq = compute_avg_sequestration(conn, history_sql, sim_data, sm_ids)
    # Small molecules in equilibrium complexes:
    avg_eq = seq["avg_eq"]
    # Combine both TCS sources (Pi in phospho-proteins + ligands in TCS subunits)
    avg_tcs = seq["avg_tcs_pi"] + seq["avg_tcs_complex"]
    # Small molecules bound to DNA-bound TFs:
    avg_bound_tf = seq["avg_bound_tf"]

    # Plot every metabolite ever present in nonzero counts during the sim:
    n_zero = int((avg_total <= 0).sum())
    n_cells = count_cells(conn, history_sql)
    print(
        f"{n_zero} of {len(sm_ids)} metabolites had zero counts the "
        f"entire simulation (not plotted)."
    )

    # Build per-metabolite DataFrame
    df = (
        pl.DataFrame(
            {
                "metabolite": sm_ids,
                "avg_total": avg_total.tolist(),
                "avg_free": avg_free.tolist(),
                "avg_eq_bound": avg_eq.tolist(),
                "avg_tcs_bound": avg_tcs.tolist(),
                "avg_bound_tf": avg_bound_tf.tolist(),
            }
        )
        .with_columns(
            [
                (pl.col("avg_total") - pl.col("avg_free")).alias("avg_bound_total"),
                (pl.col("avg_free") / pl.col("avg_total").clip(lower_bound=1e-9)).alias(
                    "fraction_free"
                ),
            ]
        )
        .filter(pl.col("avg_total") > 0)
    )

    df = df.with_columns(categorize_expr())

    cat_counts = dict(df.group_by("sequestration_type").len().iter_rows())

    # Plotly figure:
    min_val = float(df["avg_total"].min())
    max_val = float(df["avg_total"].max())
    axis_range = [min_val * 0.8, max_val * 1.25]

    fig = go.Figure()

    # Add y=x reference line
    fig.add_trace(
        go.Scatter(
            x=axis_range,
            y=axis_range,
            mode="lines",
            line=dict(color="black", width=1, dash="dash"),
            opacity=0.5,
            showlegend=False,
            hoverinfo="skip",
        )
    )

    # Add scatter points for each sequestration type:
    for seq_type, color in CATEGORY_COLORS:
        if seq_type not in cat_counts:
            continue

        df_subset = df.filter(pl.col("sequestration_type") == seq_type)
        label = f"{seq_type} ({cat_counts[seq_type]})"

        marker_size = 8 if seq_type == "No sequestration" else 12

        hover_text = [
            f"<b>{row['metabolite']}</b><br>"
            + f"Avg. total: {row['avg_total']:.1f}<br>"
            + f"Avg. free: {row['avg_free']:.1f}<br>"
            + f"Avg. bound (total): {row['avg_bound_total']:.1f}<br>"
            + f"  in eq complexes: {row['avg_eq_bound']:.1f}<br>"
            + f"  in TCS: {row['avg_tcs_bound']:.1f}<br>"
            + f"  in DNA-bound TFs: {row['avg_bound_tf']:.1f}<br>"
            + f"Fraction free: {row['fraction_free']:.3f}"
            for row in df_subset.to_dicts()
        ]

        fig.add_trace(
            go.Scatter(
                x=df_subset["avg_total"].to_list(),
                y=df_subset["avg_free"].to_list(),
                mode="markers",
                name=label,
                marker=dict(
                    color=color,
                    size=marker_size,
                    opacity=0.7,
                    line=dict(width=0.5, color="white"),
                ),
                hovertext=hover_text,
                hoverinfo="text",
            )
        )

    # Update layout:
    fig.update_layout(
        title=dict(
            text=(
                f"Average Total vs Free Metabolite Counts<br>"
                f"averaged over {n_cells} cells"
                + (f" (first {skip_n_gens} gens skipped)" if skip_n_gens else "")
                + f"<br>n={len(df)} metabolites plotted ({n_zero} had zero counts "
                f"over the entire sim)"
            ),
            font=dict(size=14),
        ),
        xaxis=dict(
            title="Log10(Average Total Count)",
            type="log",
            range=[np.log10(axis_range[0]), np.log10(axis_range[1])],
            showgrid=True,
            gridcolor="lightgray",
        ),
        yaxis=dict(
            title="Log10(Average Free Count)",
            type="log",
            range=[np.log10(axis_range[0]), np.log10(axis_range[1])],
            showgrid=True,
            gridcolor="lightgray",
        ),
        plot_bgcolor="white",
        width=700,
        height=700,
        hovermode="closest",
        legend=dict(
            title=dict(text="Sequestration (count)"),
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01,
        ),
    )

    # Save the figure
    fig.write_html(os.path.join(outdir, "small_molecule_total_vs_free_counts.html"))

    # Plot 2: Highlight a user-specified list of metabolites in the plot:
    highlight_list = params.get("highlight_metabolites", DEFAULT_HIGHLIGHT_METABOLITES)
    highlight_set, messages = resolve_highlights(highlight_list, list(sm_ids))
    print("--- NOTES ABOUT COMPARTMENT TAGS FOR SPECIFIED METABOLITES ---")
    for msg in messages:
        print(msg)
    if not messages:
        print(
            "(nothing to note with this list of metabolites specified for highlighting)"
        )

    df_h = df.with_columns(
        pl.col("metabolite").is_in(list(highlight_set)).alias("highlighted")
    )
    df_hi = df_h.filter(pl.col("highlighted"))
    df_other = df_h.filter(~pl.col("highlighted"))
    n_hi = len(df_hi)

    fig_h = go.Figure()
    fig_h.add_trace(
        go.Scatter(
            x=axis_range,
            y=axis_range,
            mode="lines",
            line=dict(color="black", width=1, dash="dash"),
            opacity=0.5,
            showlegend=False,
            hoverinfo="skip",
        )
    )
    # Plot all points in light gray first:
    fig_h.add_trace(
        go.Scatter(
            x=df_other["avg_total"].to_list(),
            y=df_other["avg_free"].to_list(),
            mode="markers",
            name=f"Other ({len(df_other)})",
            marker=dict(
                color="lightslategray",
                size=6,
                opacity=0.45,
                line=dict(width=0.3, color="white"),
            ),
            hovertext=_hover_text(df_other.to_dicts()),
            hoverinfo="text",
        )
    )
    # Plot highlighted points on top in red:
    fig_h.add_trace(
        go.Scatter(
            x=df_hi["avg_total"].to_list(),
            y=df_hi["avg_free"].to_list(),
            mode="markers",
            name=f"Highlighted ({n_hi})",
            marker=dict(
                color="red", size=12, opacity=0.95, line=dict(width=0.6, color="black")
            ),
            hovertext=_hover_text(df_hi.to_dicts()),
            hoverinfo="text",
        )
    )
    fig_h.update_layout(
        title=dict(
            text=(
                f"Total vs Free Metabolite Counts (n={len(df)},"
                f" time-averaged over {n_cells} cells)"
            ),
            font=dict(size=14),
        ),
        xaxis=dict(
            title="Log10(Average Total Count)",
            type="log",
            range=[np.log10(axis_range[0]), np.log10(axis_range[1])],
            showgrid=True,
            gridcolor="lightgray",
        ),
        yaxis=dict(
            title="Log10(Average Free Count)",
            type="log",
            range=[np.log10(axis_range[0]), np.log10(axis_range[1])],
            showgrid=True,
            gridcolor="lightgray",
        ),
        plot_bgcolor="white",
        width=700,
        height=700,
        hovermode="closest",
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
    )
    fig_h.write_html(
        os.path.join(outdir, "small_molecule_total_vs_free_counts_highlighted.html")
    )
