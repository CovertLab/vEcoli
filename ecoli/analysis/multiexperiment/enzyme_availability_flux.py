"""
Plot enzyme availability (bulk/monomer counts) vs unmet homeostatic need
(normalized |target_dmdt - estimated_dmdt|) over cell-cycle time for
multiexperiment data.

Uses Altair with dual y-axis. Enzyme counts come from listeners__monomer_counts
(or bulk for complexes); unmet need from listeners__fba_results
estimated_homeostatic_dmdt vs target_homeostatic_dmdt for a chosen metabolite.
"""

from __future__ import annotations
import plotly.express as px

import os
from typing import TYPE_CHECKING, Any

import altair as alt
import polars as pl

from ecoli.library.parquet_emitter import (
    field_metadata,
    read_stacked_columns,
)

if TYPE_CHECKING:
    from duckdb import DuckDBPyConnection

alt.data_transformers.enable("vegafusion")

# Defaults matching the notebook snippet
DEFAULT_ENZYME_NAME = "UDP-NACMURALA-GLU-LIG-MONOMER[c]"
DEFAULT_MET = "CPD-12261[p]"
TIME_BIN_MIN = 0.5  # minutes, for aggregating over cells
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
    """Plot mean enzyme counts and mean unmet need vs cell-cycle time (Altair, dual y-axis)."""

    # ---- Parse config parameters ----
    enzyme_name = params.get("enzyme_name", DEFAULT_ENZYME_NAME)
    met = params.get("met", DEFAULT_MET)
    group_by = params.get("group_by", "gen_seed")  # or "generation" or "none"

    enzyme_label = f"{enzyme_name} (counts)"
    need_label = f"{met} unmet need (normalized dmdt diff)"

    # --- Get enzyme and metabolite indexes from metadata ---
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

    # ---- Get enzyme counts and dmdt estimates from listeners ----
    query = [
        f"list_select(bulk, [{enzyme_idx + 1}])[1] AS enzyme_count",  # ! DuckDB is 1-indexed
        f"listeners__fba_results__estimated_homeostatic_dmdt[{met_idx + 1}] AS met_estimated_dmdt",  # ! DuckDB is 1-indexed
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

    # --- Reformat Dataframe ---
    if group_by == "gen_seed" or group_by == "generation":
        min_t = raw.group_by(["generation", "lineage_seed"]).agg(
            pl.col("time").min().alias("t_min")
        )
        raw = raw.join(min_t, on=["generation", "lineage_seed"])
        raw = raw.with_columns(
            ((pl.col("time") - pl.col("t_min")) / 60).alias("Time (min)")
        )
    else:
        raw = raw.with_columns(
            ((pl.col("time") - pl.col("time").min()) / 60).alias("Time (min)")
        )

    # Unmet need = L1 |target - estimate|/count
    raw = raw.with_columns(
        (
            (pl.col("met_target_dmdt") - pl.col("met_estimated_dmdt"))
            / pl.col("met_counts")
        ).alias("unmet_need")
    )

    raw = raw.with_columns(
        gen_seed=(
            pl.lit("gen=")
            + pl.col("generation").cast(pl.Utf8)
            + pl.lit(", seed=")
            + pl.col("lineage_seed").cast(pl.Utf8)
        )
    )

    new_columns = {
        "Time (min)": raw["Time (min)"],
        "group_by": raw[group_by],
        "experiment_id": raw["experiment_id"],
        "generation": raw["generation"],
        "enzyme count": raw["enzyme_count"],
        "unmet need": raw["unmet_need"],
    }

    df_plot = pl.DataFrame(new_columns)

    # Dual-axis plot with Altair (two layers, resolve_scale(y='independent'))
    base = alt.Chart()

    # Shared color scale across both layers
    color_scale = alt.Scale(
        domain=[f"{enzyme_name} (counts)", f"{met} unmet need (normalized dmdt diff)"],
        range=[PASTEL_COLOR[0], PASTEL_COLOR[1]],
    )

    line_enz = (
        base.mark_line(strokeWidth=1.5)
        .transform_calculate(trace=f"'{enzyme_name} (counts)'")
        .encode(
            x=alt.X("Time (min):Q", title="Time (min)"),
            y=alt.Y(
                "mean(enzyme count):Q", title=enzyme_label, axis=alt.Axis(orient="left")
            ),
            color=alt.Color(
                "trace:N",
                scale=color_scale,  # ✅ shared scale
                legend=alt.Legend(title="Trace"),
            ),
            detail=alt.Detail("generation:N"),
        )
    )

    line_need = (
        base.mark_line(strokeWidth=1.5)
        .transform_calculate(trace=f"'{met} unmet need (normalized dmdt diff)'")
        .encode(
            x=alt.X("Time (min):Q", title="Time (min)"),
            y=alt.Y(
                "mean(unmet need):Q",
                title=need_label,
                axis=alt.Axis(orient="right"),
            ),
            color=alt.Color(
                "trace:N",
                scale=color_scale,  # ✅ same shared scale
                legend=alt.Legend(title="Trace"),
            ),
            detail=alt.Detail("generation:N"),
        )
    )

    chart = (
        alt.layer(line_enz, line_need, data=df_plot)
        .resolve_scale(y="independent")
        .properties(width=600, height=300)
        .facet(facet=alt.Facet("experiment_id:N"), columns=2)
        .resolve_scale(y="independent")
        .properties(title="Mean enzyme counts and unmet need across experiments")
    )

    safe_met = met.replace("[", "_").replace("]", "_")
    out_path = os.path.join(outdir, f"WC_{safe_met}_enzyme_availability.html")
    chart.save(out_path)

    print(f"Saved enzyme availability vs unmet need to {out_path}")
    return
