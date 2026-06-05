"""
Metabolite Total vs Free Counts (y=x scatter)

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

import altair as alt
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

    # Build hover tooltip
    tooltip = [
        alt.Tooltip("metabolite:N", title="Metabolite"),
        alt.Tooltip("avg_total:Q", title="Avg total", format=".1f"),
        alt.Tooltip("avg_free:Q", title="Avg free", format=".1f"),
        alt.Tooltip("avg_bound_total:Q", title="Avg bound (total)", format=".1f"),
        alt.Tooltip("avg_eq_bound:Q", title="  in eq complexes", format=".1f"),
        alt.Tooltip("avg_tcs_bound:Q", title="  in TCS", format=".1f"),
        alt.Tooltip("avg_bound_tf:Q", title="  in DNA-bound TFs", format=".1f"),
        alt.Tooltip("fraction_free:Q", title="Fraction free", format=".3f"),
    ]

    # Build scatter plot (log-log so values spanning many orders of
    # magnitude are not crammed into a corner)
    min_val = float(df["avg_total"].min())
    max_val = float(df["avg_total"].max())
    log_scale_x = alt.Scale(type="log", domain=[min_val * 0.8, max_val * 1.25])
    log_scale_y = alt.Scale(type="log", domain=[min_val * 0.8, max_val * 1.25])

    # y=x reference line
    ref_line = (
        alt.Chart(pl.DataFrame({"x": [min_val * 0.8, max_val * 1.25]}))
        .mark_line(strokeDash=[4, 2], color="black", opacity=0.5)
        .encode(
            x=alt.X("x:Q", scale=log_scale_x),
            y=alt.Y("x:Q", scale=log_scale_y),
        )
    )

    # Fixed base-category -> color, built into a dynamic domain/range using the
    # labels (which include per-category counts). Only categories present are
    # shown in the legend.
    base_colors = [
        ("No sequestration", "lightsteelblue"),
        ("In eq complex", "darkorange"),
        ("In TCS", "mediumseagreen"),
        ("In eq + TCS complex", "mediumpurple"),
        ("In DNA-bound TF", "crimson"),
    ]
    domain_labels = []
    range_colors = []
    for base, color in base_colors:
        if base in cat_counts:
            domain_labels.append(f"{base} ({cat_counts[base]})")
            range_colors.append(color)
    color_scale = alt.Scale(domain=domain_labels, range=range_colors)

    scatter = (
        alt.Chart(df)
        .mark_circle(opacity=0.7)
        .encode(
            x=alt.X(
                "avg_total:Q",
                scale=log_scale_x,
                title="Average Total Count (log scale)",
            ),
            y=alt.Y(
                "avg_free:Q", scale=log_scale_y, title="Average Free Count (log scale)"
            ),
            color=alt.Color(
                "sequestration_label:N",
                scale=color_scale,
                title="Sequestration (count)",
            ),
            size=alt.condition(
                alt.datum.sequestration_type == "No sequestration",
                alt.value(40),
                alt.value(80),
            ),
            tooltip=tooltip,
        )
    )

    chart = (
        (ref_line + scatter)
        .properties(
            title=(
                f"Average Total vs Free Metabolite Counts  "
                f"(n={len(df)} metabolites, avg total ≥ {MIN_AVG_TOTAL_COUNT})  |  "
                "Points below y=x are sequestered"
            ),
            width=600,
            height=600,
        )
        .configure_view(fill="white")
        .interactive()
    )

    chart.save(os.path.join(outdir, "metabolite_total_vs_free_counts.html"))
