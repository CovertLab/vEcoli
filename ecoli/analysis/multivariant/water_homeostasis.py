"""
Plot WATER[c]'s homeostatic tracking over time for multivariant simulation.

For each variant: (1) WATER[c]'s actual bulk count over time, and (2) the
mismatch between the FBA's achieved and target dm/dt for WATER[c]
(``estimated_homeostatic_dmdt - target_homeostatic_dmdt``). Post-fix, the
mismatch should sit at ~0 throughout (the hard constraint in
``NetworkFlowModel.solve()`` forces an exact match); pre-fix, it grows
without bound as the deficit runs away.

One pair of line subplots per variant, averaged across all cells in that
variant (aligned by time since birth).
"""

from __future__ import annotations

import os
from typing import Any, TYPE_CHECKING

import altair as alt
import polars as pl

from ecoli.analysis.multivariant.utils import create_variant_label
from ecoli.library.parquet_emitter import field_metadata, read_stacked_columns

if TYPE_CHECKING:
    from duckdb import DuckDBPyConnection

alt.data_transformers.enable("vegafusion")

WATER_ID = "WATER[c]"
DEFAULT_SUBPLOT_WIDTH = 600


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
    """One pair of line subplots (count, dm/dt mismatch) per variant."""
    experiment_id = next(iter(variant_metadata.keys()), None)
    per_variant_params: dict[int, Any] = (
        variant_metadata[experiment_id] if experiment_id else {}
    )

    subplot_width = int(params.get("subplot_width", DEFAULT_SUBPLOT_WIDTH))

    homeostatic_ids = field_metadata(
        conn, config_sql, "listeners__fba_results__homeostatic_metabolite_counts"
    )
    try:
        water_idx = homeostatic_ids.index(WATER_ID) + 1
    except ValueError as e:
        print(f"water_homeostasis: {WATER_ID} not in homeostatic metabolites: {e}")
        return

    query_cols = [
        "time",
        "generation",
        "lineage_seed",
        "agent_id",
        f"listeners__fba_results__homeostatic_metabolite_counts[{water_idx}] AS water_count",
        f"listeners__fba_results__target_homeostatic_dmdt[{water_idx}] AS target_dmdt",
        f"listeners__fba_results__estimated_homeostatic_dmdt[{water_idx}] AS est_dmdt",
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
        print("water_homeostasis: no rows returned; skipping.")
        return

    raw = raw.with_columns(
        (pl.col("est_dmdt") - pl.col("target_dmdt")).alias("mismatch")
    )

    # Relative time per (variant, generation, lineage_seed, agent_id)
    t_min = raw.group_by(["variant", "generation", "lineage_seed", "agent_id"]).agg(
        pl.col("time").min().alias("t_min")
    )
    raw = raw.join(t_min, on=["variant", "generation", "lineage_seed", "agent_id"])
    raw = raw.with_columns(
        ((pl.col("time") - pl.col("t_min")) / 60.0).alias("Time_min")
    )

    agg = (
        raw.group_by("variant", "Time_min")
        .agg(
            pl.col("water_count").mean().alias("water_count"),
            pl.col("mismatch").mean().alias("mismatch"),
        )
        .sort("variant", "Time_min")
    )

    variants = agg["variant"].unique().sort()
    w = subplot_width

    subplot_charts = []
    for variant_val in variants:
        sub = agg.filter(pl.col("variant") == variant_val)
        if sub.is_empty():
            continue
        label = create_variant_label(variant_val, per_variant_params)
        df = sub.to_pandas()

        count_chart = (
            alt.Chart(df)
            .mark_line(strokeWidth=2, color="#80b1d3")
            .encode(
                x=alt.X("Time_min:Q", title="Time (min)"),
                y=alt.Y("water_count:Q", title="WATER[c] count"),
                tooltip=["Time_min:Q", "water_count:Q"],
            )
            .properties(height=250, width=w, title=f"WATER[c] count - {label}")
        )

        mismatch_line = (
            alt.Chart(df)
            .mark_line(strokeWidth=2, color="#fb8072")
            .encode(
                x=alt.X("Time_min:Q", title="Time (min)"),
                y=alt.Y("mismatch:Q", title="Achieved - target dm/dt"),
                tooltip=["Time_min:Q", "mismatch:Q"],
            )
        )
        zero_rule = (
            alt.Chart(pl.DataFrame({"y": [0]}).to_pandas())
            .mark_rule(strokeDash=[4, 4], color="gray")
            .encode(y="y:Q")
        )
        mismatch_chart = (mismatch_line + zero_rule).properties(
            height=250, width=w, title=f"WATER[c] dm/dt mismatch - {label}"
        )

        subplot_charts.append(count_chart)
        subplot_charts.append(mismatch_chart)

    if not subplot_charts:
        print("water_homeostasis: no per-variant data after aggregation; skipping.")
        return

    combined = alt.vconcat(*subplot_charts).properties(
        title="WATER[c] homeostatic tracking by variant"
    )

    out_path = os.path.join(outdir, "water_homeostasis.html")
    combined.save(out_path)
    print(f"Saved water homeostasis tracking (multivariant) to {out_path}")
