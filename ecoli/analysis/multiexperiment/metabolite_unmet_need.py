"""
Plot unmet homeostatic need for metabolites: top-5 bar chart (mean |unmet need|)
and time-series lines for metabolites of interest.

Unmet need = (target_homeostatic_dmdt - estimated_homeostatic_dmdt) / homeostatic_count,
then replace inf, then divide by counts_to_molar (per row).
Uses Altair instead of Plotly.
"""

from __future__ import annotations

import os
from typing import Any, TYPE_CHECKING

import altair as alt
import polars as pl

from ecoli.library.parquet_emitter import (
    field_metadata,
    read_stacked_columns,
)

if TYPE_CHECKING:
    from duckdb import DuckDBPyConnection

DEFAULT_TOP_N = 8
PASTEL = [
    "#8dd3c7",
    "#EECE9D",
    "#bebada",
    "#fb8072",
    "#80b1d3",
    "#fdb462",
    "#b3de69",
    "#fccde5",
]


def plot(
    params: dict[str, Any],
    conn: "DuckDBPyConnection",
    history_sql: str,
    config_sql: str,
    success_sql: str,
    sim_data_dict: dict[str, dict[int, str]],
    validation_data_paths: list[str],
    outdir: str,
    variant_metadata: dict[str, dict[int, Any]],
    variant_names: dict[str, str],
) -> None:
    """Plot top-5 unmet need bar chart and metabolite-of-interest time series (Altair)."""
    top_n = params.get("top_n", DEFAULT_TOP_N)
    metabolites_of_interest = params.get("metabolites_of_interest")  # None => use top_n

    try:
        homeostatic_ids = field_metadata(
            conn, config_sql, "listeners__fba_results__estimated_homeostatic_dmdt"
        )
    except Exception:
        print(
            "metabolite_unmet_need: listeners__fba_results__estimated_homeostatic_dmdt "
            "not in config (e.g. non-metabolism_redux); skipping."
        )
        return

    bulk_ids = field_metadata(conn, config_sql, "bulk")
    try:
        homeostatic_bulk_idx_1based = [
            bulk_ids.index(met_id) + 1 for met_id in homeostatic_ids
        ]
    except ValueError as e:
        print(
            f"metabolite_unmet_need: homeostatic metabolite not in bulk: {e}; skipping."
        )
        return

    n_met = len(homeostatic_ids)

    query_cols = [
        "time",
        "generation",
        "lineage_seed",
        "agent_id",
        "listeners__fba_results__estimated_homeostatic_dmdt AS estimated_dmdt",
        "listeners__fba_results__target_homeostatic_dmdt AS target_dmdt",
        f"list_select(bulk, {homeostatic_bulk_idx_1based}) AS homeostatic_counts",
        "listeners__enzyme_kinetics__counts_to_molar AS counts_to_molar",
    ]

    raw = pl.DataFrame(
        read_stacked_columns(
            history_sql,
            query_cols,
            conn=conn,
            order_results=True,
            success_sql=success_sql,
            remove_first=True,
        )
    )

    if raw.is_empty():
        print("metabolite_unmet_need: no rows returned; skipping.")
        return

    # Compute unmet need per metabolite: (target - estimated) / homeostatic_count, then / counts_to_molar
    for i in range(n_met):
        est = pl.col("estimated_dmdt").list.get(i)
        tgt = pl.col("target_dmdt").list.get(i)
        cnt = pl.col("homeostatic_counts").list.get(i)
        denom = pl.when(cnt == 0).then(None).otherwise(cnt)
        ratio = (tgt - est) / denom / pl.col("counts_to_molar")
        raw = raw.with_columns(
            pl.when(ratio.is_infinite()).then(None).otherwise(ratio).alias(f"unmet_{i}")
        )

    # Cell-relative time
    t_min = raw.group_by(["generation", "lineage_seed", "agent_id"]).agg(
        pl.col("time").min().alias("t_min")
    )
    raw = raw.join(t_min, on=["generation", "lineage_seed", "agent_id"])
    raw = raw.with_columns(
        ((pl.col("time") - pl.col("t_min")) / 60.0).alias("Time_min")
    )

    # Long format: one row per (row, metabolite)
    id_vars = ["Time_min", "generation", "lineage_seed", "agent_id"]
    value_vars = [f"unmet_{i}" for i in range(n_met)]
    long = raw.select(id_vars + value_vars).melt(
        id_vars=id_vars,
        value_vars=value_vars,
        variable_name="met_key",
        value_name="unmet_need",
    )
    long = long.with_columns(
        pl.col("met_key").str.replace("unmet_", "").cast(pl.Int32).alias("met_idx")
    )
    met_df = pl.DataFrame({"met_idx": range(n_met), "metabolite": homeostatic_ids})
    long = long.join(met_df, on="met_idx")

    # Aggregate: mean per (Time_min, metabolite)
    agg = (
        long.group_by("Time_min", "metabolite")
        .agg(pl.col("unmet_need").mean().alias("unmet_need"))
        .sort("Time_min", "metabolite")
    )

    # Top N by mean absolute unmet need over time
    met_score = (
        agg.group_by("metabolite")
        .agg(pl.col("unmet_need").abs().mean().alias("mean_abs_unmet"))
        .sort("mean_abs_unmet", descending=True)
    )
    top_mets = met_score.head(top_n)["metabolite"].to_list()
    top_bar = met_score.filter(pl.col("metabolite").is_in(top_mets))

    # Metabolites for line plot: user list or top N
    line_mets = (
        metabolites_of_interest if metabolites_of_interest is not None else top_mets
    )
    line_mets = [m for m in line_mets if m in homeostatic_ids]
    if not line_mets:
        line_mets = top_mets

    agg_line = agg.filter(pl.col("metabolite").is_in(line_mets))

    df_bar = top_bar.to_pandas()
    df_line = agg_line.to_pandas()

    # Consistent colors
    all_mets = list(dict.fromkeys(top_mets + line_mets))
    color_domain = all_mets
    color_range = [PASTEL[i % len(PASTEL)] for i in range(len(color_domain))]

    # Top: bar chart — mean_abs_unmet is always positive, use symlog to handle near-zero
    bar_base = alt.Chart(df_bar).encode(
        x=alt.X("metabolite:N", title="Metabolite", sort="-y"),
        color=alt.Color(
            "metabolite:N",
            scale=alt.Scale(domain=color_domain, range=color_range),
            legend=None,
        ),
        tooltip=["metabolite:N", "mean_abs_unmet:Q"],
    )

    bars = bar_base.mark_bar(cornerRadiusEnd=8, size=28).encode(
        y=alt.Y(
            "mean_abs_unmet:Q",
            title="Unmet need (mean |L1 diff|)",
            scale=alt.Scale(
                type="symlog"
            ),  # ✅ safe here since mean_abs_unmet > 0 always
        ),
    )

    bar_labels = bar_base.mark_text(
        align="center",
        baseline="bottom",
        dy=-4,  # small offset above bar top
        fontSize=12,
        fontWeight="bold",
    ).encode(
        y=alt.Y("mean_abs_unmet:Q", scale=alt.Scale(type="symlog")),
        text=alt.Text(
            "mean_abs_unmet:Q", format=".2e"
        ),  # ✅ scientific notation e.g. 5.1e+03
    )

    bar_chart = (bars + bar_labels).properties(height=220, width=600)

    # Bottom: line chart — unmet_need can be negative, so NO log scale
    line_chart = (
        alt.Chart(df_line)
        .mark_line(strokeWidth=2)
        .encode(
            x=alt.X("Time_min:Q", title="Time (min)"),
            y=alt.Y(
                "unmet_need:Q",
                title="L1 |Target - Estimate|",
            ),
            color=alt.Color(
                "metabolite:N",
                scale=alt.Scale(domain=color_domain, range=color_range),
                legend=alt.Legend(title="Metabolite"),
            ),
            tooltip=["Time_min:Q", "metabolite:N", "unmet_need:Q"],
        )
        .properties(height=380, width=600)
    )

    title = "Unmet homeostatic need summary + culprit metabolite timeseries"

    combined = alt.vconcat(bar_chart, line_chart).properties(title=title)

    combined.save(os.path.join(outdir, "WC_metabolite_unmet_need.html"))
    print(f"Saved metabolite unmet need to {outdir}")
    return
