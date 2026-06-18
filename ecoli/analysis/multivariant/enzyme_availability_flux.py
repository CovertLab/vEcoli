"""
Plot enzyme availability (bulk counts) vs unmet homeostatic need over
continuous simulation time for multivariant simulation.

One dual-y-axis Altair subplot per variant, stacked vertically. Within each
variant, lines represent the mean over all cells on a continuous time axis
(relative to the first timestep of each lineage seed); generation is used as
the detail encoding to draw separate line segments at each cell division
without resetting the time axis.

DISCLAIMER: This analysis is only meant for metabolism-redux and
metabolism-redux-classic. metabolism.py lacks necessary listeners due to differences
in problem formulation
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING, Any

import altair as alt
import plotly.express as px
import polars as pl

from ecoli.analysis.multivariant.utils import create_variant_label
from ecoli.library.parquet_emitter import (
    field_metadata,
    read_stacked_columns,
)

if TYPE_CHECKING:
    from duckdb import DuckDBPyConnection

alt.data_transformers.enable("vegafusion")

DEFAULT_ENZYME_NAME = "UDP-NACMURALA-GLU-LIG-MONOMER[c]"
DEFAULT_MET = "CPD-12261[p]"
PASTEL_COLOR = px.colors.qualitative.Pastel


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
    """Plot mean enzyme counts and unmet need vs time, one subplot per variant."""
    experiment_id = next(iter(variant_metadata.keys()), None)
    per_variant_params: dict[int, Any] = (
        variant_metadata[experiment_id] if experiment_id else {}
    )

    enzyme_name = params.get("enzyme_name", DEFAULT_ENZYME_NAME)
    met = params.get("met", DEFAULT_MET)

    enzyme_label = f"{enzyme_name} (counts)"
    need_label = f"{met} unmet need (normalized dmdt diff)"

    bulk_ids = field_metadata(conn, config_sql, "bulk")
    homeostatic_metabolite = field_metadata(
        conn, config_sql, "listeners__fba_results__estimated_homeostatic_dmdt"
    )

    assert enzyme_name in bulk_ids, (
        f"{enzyme_name} not found in bulk metadata; cannot proceed."
    )
    assert met in homeostatic_metabolite, (
        f"{met} not found in estimated_homeostatic_dmdt metadata; cannot proceed."
    )

    enzyme_idx = bulk_ids.index(enzyme_name)
    met_idx = homeostatic_metabolite.index(met)

    query = [
        f"list_select(bulk, [{enzyme_idx + 1}])[1] AS enzyme_count",
        f"listeners__fba_results__estimated_homeostatic_dmdt[{met_idx + 1}] AS met_estimated_dmdt",
        f"listeners__fba_results__target_homeostatic_dmdt[{met_idx + 1}] AS met_target_dmdt",
        f"listeners__fba_results__homeostatic_metabolite_counts[{met_idx + 1}] AS met_counts",
    ]

    raw = pl.DataFrame(
        read_stacked_columns(
            history_sql,
            query,
            order_results=True,
            conn=conn,
            remove_first=True,
        )
    )

    # Continuous time per lineage_seed: subtract the global minimum time for
    # each seed so the x-axis starts at 0 and runs unbroken across generations.
    min_t = raw.group_by(["lineage_seed"]).agg(pl.col("time").min().alias("t_min"))
    raw = raw.join(min_t, on=["lineage_seed"])
    raw = raw.with_columns(
        ((pl.col("time") - pl.col("t_min")) / 60).alias("Time (min)")
    )

    raw = raw.with_columns(
        (
            (pl.col("met_target_dmdt") - pl.col("met_estimated_dmdt"))
            / pl.col("met_counts")
        ).alias("unmet_need")
    )

    df_plot = pl.DataFrame(
        {
            "Time (min)": raw["Time (min)"],
            "variant": raw["variant"],
            "generation": raw["generation"],
            "lineage_seed": raw["lineage_seed"],
            "enzyme count": raw["enzyme_count"],
            "unmet need": raw["unmet_need"],
        }
    )

    color_scale = alt.Scale(
        domain=[enzyme_label, need_label],
        range=[PASTEL_COLOR[0], PASTEL_COLOR[1]],
    )

    line_enz = (
        alt.Chart()
        .mark_line(strokeWidth=1.5)
        .transform_calculate(trace=f"'{enzyme_label}'")
        .encode(
            x=alt.X("Time (min):Q", title="Time (min)"),
            y=alt.Y(
                "mean(enzyme count):Q",
                title=enzyme_label,
                axis=alt.Axis(orient="left"),
            ),
            color=alt.Color(
                "trace:N",
                scale=color_scale,
                legend=alt.Legend(title="Trace"),
            ),
            detail=alt.Detail("generation:N"),
        )
    )

    line_need = (
        alt.Chart()
        .mark_line(strokeWidth=1.5)
        .transform_calculate(trace=f"'{need_label}'")
        .encode(
            x=alt.X("Time (min):Q", title="Time (min)"),
            y=alt.Y(
                "mean(unmet need):Q",
                title=need_label,
                axis=alt.Axis(orient="right"),
            ),
            color=alt.Color(
                "trace:N",
                scale=color_scale,
                legend=alt.Legend(title="Trace"),
            ),
            detail=alt.Detail("generation:N"),
        )
    )

    variants = df_plot["variant"].unique().sort()
    plots = []
    for variant_val in variants:
        variant_name = create_variant_label(variant_val, per_variant_params)
        variant_data = df_plot.filter(pl.col("variant") == variant_val).to_pandas()

        subplot = (
            alt.layer(line_enz, line_need, data=variant_data)
            .resolve_scale(y="independent")
            .properties(width=600, height=300, title=variant_name)
        )
        plots.append(subplot)

    final = (
        alt.vconcat(*plots)
        .resolve_scale(x="independent", y="independent")
        .properties(title="Enzyme availability and unmet need by variant")
    )

    safe_met = met.replace("[", "_").replace("]", "_")
    out_path = os.path.join(outdir, f"WC_{safe_met}_enzyme_availability.html")
    final.save(out_path)
    print(f"Saved multivariant enzyme availability vs unmet need to {out_path}")
