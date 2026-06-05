"""
Metabolite Total vs Free Counts (y=x scatter) plotly!

For each tracked metabolite, plots the time-averaged total count (x-axis)
against the time-averaged free count (y-axis). Points on the y=x line have
all their counts in the free pool. Points below the line have metabolites
sequestered in equilibrium complexes or TCS phosphorylated molecules (both of
which can also be found in bound TFs for those that are actively modeled).

Color coding:
  - Blue: no sequestration in complexes (sits on y=x)
  - Orange: bound in equilibrium complexes only
  - Green: bound in TCS phosphorylation only (Pi[c])
  - Purple: bound in both complex types
  - Pink: bound in actively modeled bound transcription factors

"""

import os
from typing import Any
import numpy as np
import plotly.graph_objects as go
import polars as pl
from duckdb import DuckDBPyConnection

from ecoli.library.parquet_emitter import (
    field_metadata,
    read_stacked_columns,
    ndlist_to_ndarray,
)

# Minimum average total count to include a metabolite (filters out metabolites
# that are never present in the simulation (usually condition dependent)).
MIN_AVG_TOTAL_COUNT = 1.0


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

    # Load metabolite IDs from listener metadata:
    metabolite_ids = field_metadata(
        conn, config_sql, "listeners__metabolite_counts__totalMetaboliteCounts"
    )

    # Load time-series data and compute time averages
    query = [
        "listeners__metabolite_counts__totalMetaboliteCounts AS total",
        "listeners__metabolite_counts__freeMetaboliteCounts AS free",
        "listeners__metabolite_counts__metabolitesInEquilibriumComplexes AS eq_bound",
        "listeners__metabolite_counts__metabolitesInTCSPhosphorylation AS tcs_bound",
        "listeners__metabolite_counts__metabolitesInTCSComplexes AS tcs_complex_bound",
        "listeners__metabolite_counts__metabolitesInBoundTFs AS bound_tf",
    ]
    raw = pl.DataFrame(
        read_stacked_columns(history_sql, query, order_results=True, conn=conn)
    )

    # Convert list columns to numpy arrays and compute time averages
    total_arr = ndlist_to_ndarray(raw["total"].to_list())  # (T, n_mets)
    free_arr = ndlist_to_ndarray(raw["free"].to_list())
    eq_arr = ndlist_to_ndarray(raw["eq_bound"].to_list())
    tcs_arr = ndlist_to_ndarray(raw["tcs_bound"].to_list())
    tcs_complex_arr = ndlist_to_ndarray(raw["tcs_complex_bound"].to_list())
    bound_tf_arr = ndlist_to_ndarray(raw["bound_tf"].to_list())

    avg_total = total_arr.mean(axis=0)
    avg_free = free_arr.mean(axis=0)
    avg_eq = eq_arr.mean(axis=0)
    # Combine both TCS sources (Pi in phospho-proteins + ligand in TCS complexes)
    avg_tcs = (tcs_arr + tcs_complex_arr).mean(axis=0)
    # Ligand (eq complexes) and Pi (from TCS complexes) in DNA-bound TFs
    avg_bound_tf = bound_tf_arr.mean(axis=0)

    # Build per-metabolite DataFrame
    df = (
        pl.DataFrame(
            {
                "metabolite": metabolite_ids,
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
        .filter(pl.col("avg_total") >= MIN_AVG_TOTAL_COUNT)
    )

    # Classify sequestration type (priority order; DNA-bound TF pulled out
    # first since those metabolites are also technically in an eq complex).
    df = df.with_columns(
        pl.when(pl.col("avg_bound_tf") > 0)
        .then(pl.lit("In DNA-bound TF"))
        .when((pl.col("avg_eq_bound") > 0) & (pl.col("avg_tcs_bound") > 0))
        .then(pl.lit("In eq + TCS complex"))
        .when(pl.col("avg_eq_bound") > 0)
        .then(pl.lit("In eq complex"))
        .when(pl.col("avg_tcs_bound") > 0)
        .then(pl.lit("In TCS"))
        .otherwise(pl.lit("No sequestration"))
        .alias("sequestration_type")
    )

    # Append the per-category count to each label, e.g. "In eq complex (8)",
    # so the legend shows how many metabolites fall in each category.
    cat_counts = dict(df.group_by("sequestration_type").len().iter_rows())
    df = df.with_columns(
        pl.col("sequestration_type")
        .map_elements(lambda c: f"{c} ({cat_counts.get(c, 0)})", return_dtype=pl.Utf8)
        .alias("sequestration_label")
    )

    # Build Plotly figure (log-log so values spanning many orders of
    # magnitude are not crammed into a corner)
    min_val = float(df["avg_total"].min())
    max_val = float(df["avg_total"].max())
    axis_range = [min_val * 0.8, max_val * 1.25]

    # Fixed base-category -> color mapping
    base_colors = {
        "No sequestration": "lightsteelblue",
        "In eq complex": "darkorange",
        "In TCS": "mediumseagreen",
        "In eq + TCS complex": "mediumpurple",
        "In DNA-bound TF": "crimson",
    }

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

    # Add scatter points for each sequestration type
    for seq_type in [
        "No sequestration",
        "In eq complex",
        "In TCS",
        "In eq + TCS complex",
        "In DNA-bound TF",
    ]:
        if seq_type not in cat_counts:
            continue

        df_subset = df.filter(pl.col("sequestration_type") == seq_type)
        label = f"{seq_type} ({cat_counts[seq_type]})"

        # Determine marker size (matching Altair's 40 vs 80)
        marker_size = 8 if seq_type == "No sequestration" else 12

        # Build hover text
        hover_text = [
            f"<b>{row['metabolite']}</b><br>"
            + f"Avg total: {row['avg_total']:.1f}<br>"
            + f"Avg free: {row['avg_free']:.1f}<br>"
            + f"Avg bound (total): {row['avg_bound_total']:.1f}<br>"
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
                    color=base_colors[seq_type],
                    size=marker_size,
                    opacity=0.7,
                    line=dict(width=0.5, color="white"),
                ),
                hovertext=hover_text,
                hoverinfo="text",
            )
        )

    # Update layout
    fig.update_layout(
        title=dict(
            text=(
                f"Average Total vs Free Metabolite Counts  "
                f"(n={len(df)} metabolites, avg total ≥ {MIN_AVG_TOTAL_COUNT})  |  "
                "Points below y=x are sequestered"
            ),
            font=dict(size=14),
        ),
        xaxis=dict(
            title="Average Total Count (log scale)",
            type="log",
            range=[np.log10(axis_range[0]), np.log10(axis_range[1])],
            showgrid=True,
            gridcolor="lightgray",
        ),
        yaxis=dict(
            title="Average Free Count (log scale)",
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
    fig.write_html(os.path.join(outdir, "metabolite_total_vs_free_counts.html"))
