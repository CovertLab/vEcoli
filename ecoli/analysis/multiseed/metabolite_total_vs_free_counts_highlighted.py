"""
Metabolite Total vs Free Counts (y=x scatter) with highlighting capabilities!

Same as metabolite_total_vs_free_counts.py, but instead of coloring points
by sequestration type, ALL points are plotted in one color and a
user-specified list of metabolites is highlighted in red so they are easy
to locate on the plot.

Set HIGHLIGHT_METABOLITES at the top of this file to the metabolite IDs you
want to highlight (with or without compartment tags, e.g. 'ATP' or
'ATP[c]'). Matching is done on the bare ID so either form works.

Points on the y=x line have all their counts in the free pool. Points below
the line have metabolites sequestered in equilibrium, TCS complexes, or bound
transcription factors (which can be either TCS complexes or eq complexes).
"""

import os
import re
from typing import Any

import altair as alt
import polars as pl
from duckdb import DuckDBPyConnection

from ecoli.library.parquet_emitter import (
    field_metadata,
    read_stacked_columns,
    ndlist_to_ndarray,
)

# Metabolites to highlight in red (with or without compartment tags)
HIGHLIGHT_METABOLITES = ["ATP[c]", "Pi[c]", "AMP[c]"]

# Minimum average total count to include a metabolite (filters out
# metabolites that are never present (condition dependent).
MIN_AVG_TOTAL_COUNT = 1.0


def _bare(mol_id: str) -> str:
    """Strip a compartment tag like [c] from a molecule ID."""
    return re.split(r"\[.\]", mol_id)[0]


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
    # Load metabolite IDs from listener metadata
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

    total_arr = ndlist_to_ndarray(raw["total"].to_list())
    free_arr = ndlist_to_ndarray(raw["free"].to_list())
    eq_arr = ndlist_to_ndarray(raw["eq_bound"].to_list())
    tcs_arr = ndlist_to_ndarray(raw["tcs_bound"].to_list())
    tcs_complex_arr = ndlist_to_ndarray(raw["tcs_complex_bound"].to_list())
    bound_tf_arr = ndlist_to_ndarray(raw["bound_tf"].to_list())

    avg_total = total_arr.mean(axis=0)
    avg_free = free_arr.mean(axis=0)
    avg_eq = eq_arr.mean(axis=0)
    avg_tcs = (tcs_arr + tcs_complex_arr).mean(axis=0)
    avg_bound_tf = bound_tf_arr.mean(axis=0)

    # Determine which metabolites to highlight (match on bare ID)
    highlight_bare = {_bare(m) for m in HIGHLIGHT_METABOLITES}
    is_highlighted = [_bare(m) in highlight_bare for m in metabolite_ids]

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
                "highlighted": is_highlighted,
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

    # Group label with per-group count in parentheses for the legend:
    n_highlighted = int(df["highlighted"].sum())
    n_other = len(df) - n_highlighted
    df = df.with_columns(
        pl.when(pl.col("highlighted"))
        .then(pl.lit(f"Highlighted ({n_highlighted})"))
        .otherwise(pl.lit(f"Other ({n_other})"))
        .alias("highlight_group")
    )

    # Tooltip
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

    # Build scatter plot (log-log)
    min_val = float(df["avg_total"].min())
    max_val = float(df["avg_total"].max())
    log_scale = alt.Scale(type="log", domain=[min_val * 0.8, max_val * 1.25])

    ref_line = (
        alt.Chart(pl.DataFrame({"x": [min_val * 0.8, max_val * 1.25]}))
        .mark_line(strokeDash=[4, 2], color="black", opacity=0.5)
        .encode(
            x=alt.X("x:Q", scale=log_scale),
            y=alt.Y("x:Q", scale=log_scale),
        )
    )

    # All points one neutral color; highlighted ones red and larger.
    # Plot non-highlighted first, then highlighted on top.
    color_scale = alt.Scale(
        domain=[f"Other ({n_other})", f"Highlighted ({n_highlighted})"],
        range=["lightslategray", "red"],
    )

    base = (
        alt.Chart(df)
        .mark_circle()
        .encode(
            x=alt.X(
                "avg_total:Q", scale=log_scale, title="Average Total Count (log scale)"
            ),
            y=alt.Y(
                "avg_free:Q", scale=log_scale, title="Average Free Count (log scale)"
            ),
            color=alt.Color("highlight_group:N", scale=color_scale, title="Group"),
            size=alt.condition(alt.datum.highlighted, alt.value(110), alt.value(35)),
            opacity=alt.condition(
                alt.datum.highlighted, alt.value(1.0), alt.value(0.45)
            ),
            order=alt.Order("highlighted:Q", sort="ascending"),
            tooltip=tooltip,
        )
    )

    n_highlighted = int(df["highlighted"].sum())
    chart = (
        (ref_line + base)
        .properties(
            title=(
                f"Average Total vs Free Metabolite Counts  "
                f"(n={len(df)} metabolites)  |  "
                f"{n_highlighted} highlighted in red  |  "
                "Points below y=x are sequestered"
            ),
            width=600,
            height=600,
        )
        .configure_view(fill="white")
        .interactive()
    )

    chart.save(os.path.join(outdir, "metabolite_total_vs_free_counts_highlighted.html"))
