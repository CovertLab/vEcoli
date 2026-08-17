"""
Plot active/inactive molecule counts over time for the two-component-system
transcription factors touched by this session's water-mass-action rate-law
fix (ArcA and PhoP), for multivariant simulation.

One line plot per variant per TF, averaged across all cells in that variant
(aligned by time since birth), with a line for the active and inactive pools.
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

# (display name, active/phosphorylated form, inactive/unphosphorylated form)
TFS = [
    ("ArcA", "PHOSPHO-ARCA[c]", "ARCA-MONOMER[c]"),
    ("PhoP", "PHOSPHO-PHOP[c]", "PHOP-MONOMER[c]"),
]
DEFAULT_SUBPLOT_WIDTH = 600


def _tf_subplot_charts(
    tf_name: str,
    active_id: str,
    inactive_id: str,
    conn: "DuckDBPyConnection",
    history_sql: str,
    config_sql: str,
    success_sql: str,
    per_variant_params: dict[int, Any],
    subplot_width: int,
) -> list[alt.Chart]:
    """One line subplot per variant for a single TF's active/inactive counts."""
    bulk_ids = field_metadata(conn, config_sql, "bulk")
    try:
        active_idx = bulk_ids.index(active_id) + 1
        inactive_idx = bulk_ids.index(inactive_id) + 1
    except ValueError as e:
        print(f"arca_activity: {tf_name} molecule not in bulk: {e}; skipping.")
        return []

    query_cols = [
        "time",
        "generation",
        "lineage_seed",
        "agent_id",
        f"bulk[{active_idx}] AS active_count",
        f"bulk[{inactive_idx}] AS inactive_count",
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
        print(f"arca_activity: no rows returned for {tf_name}; skipping.")
        return []

    # Relative time per (variant, generation, lineage_seed, agent_id)
    t_min = raw.group_by(["variant", "generation", "lineage_seed", "agent_id"]).agg(
        pl.col("time").min().alias("t_min")
    )
    raw = raw.join(t_min, on=["variant", "generation", "lineage_seed", "agent_id"])
    raw = raw.with_columns(
        ((pl.col("time") - pl.col("t_min")) / 60.0).alias("Time_min")
    )

    active_label = f"Active ({active_id})"
    inactive_label = f"Inactive ({inactive_id})"

    long = raw.select(["variant", "Time_min", "active_count", "inactive_count"]).melt(
        id_vars=["variant", "Time_min"],
        value_vars=["active_count", "inactive_count"],
        variable_name="state",
        value_name="count",
    )
    long = long.with_columns(
        pl.col("state")
        .replace(
            {
                "active_count": active_label,
                "inactive_count": inactive_label,
            }
        )
        .alias("state")
    )

    agg = (
        long.group_by("variant", "Time_min", "state")
        .agg(pl.col("count").mean().alias("count"))
        .sort("variant", "Time_min", "state")
    )

    variants = agg["variant"].unique().sort()

    color_domain = [active_label, inactive_label]
    color_range = ["#fb8072", "#80b1d3"]

    subplot_charts = []
    for variant_val in variants:
        sub = agg.filter(pl.col("variant") == variant_val)
        if sub.is_empty():
            continue
        variant_label = create_variant_label(variant_val, per_variant_params)
        title = f"{tf_name} - {variant_label}"
        df = sub.to_pandas()

        line_chart = (
            alt.Chart(df)
            .mark_line(strokeWidth=2)
            .encode(
                x=alt.X("Time_min:Q", title="Time (min)"),
                y=alt.Y("count:Q", title="Molecule count"),
                color=alt.Color(
                    "state:N",
                    scale=alt.Scale(domain=color_domain, range=color_range),
                    legend=alt.Legend(title=f"{tf_name} state"),
                ),
                tooltip=["Time_min:Q", "state:N", "count:Q"],
            )
            .properties(height=300, width=subplot_width, title=title)
        )

        subplot_charts.append(line_chart)

    return subplot_charts


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
    """One line subplot per variant per TF, aggregated across all cells in that variant."""
    experiment_id = next(iter(variant_metadata.keys()), None)
    per_variant_params: dict[int, Any] = (
        variant_metadata[experiment_id] if experiment_id else {}
    )

    subplot_width = int(params.get("subplot_width", DEFAULT_SUBPLOT_WIDTH))

    all_charts = []
    for tf_name, active_id, inactive_id in TFS:
        all_charts.extend(
            _tf_subplot_charts(
                tf_name,
                active_id,
                inactive_id,
                conn,
                history_sql,
                config_sql,
                success_sql,
                per_variant_params,
                subplot_width,
            )
        )

    if not all_charts:
        print("arca_activity: no per-variant data after aggregation; skipping.")
        return

    combined = alt.vconcat(*all_charts).properties(
        title="Two-component-system TF active/inactive counts by variant"
    )

    out_path = os.path.join(outdir, "arca_activity.html")
    combined.save(out_path)
    print(f"Saved TCS activity (multivariant) to {out_path}")
