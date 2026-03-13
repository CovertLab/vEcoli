"""
Plot absolute / normalized cell mass (dry mass) over time across multiple
experiments. time is divided by generation.

The absolute / normalized dry mass plots will have one line with std and mean shading per
seed and generation.
"""

from typing import Any, TYPE_CHECKING
import os

from ecoli.library.parquet_emitter import (
    read_stacked_columns,
)
import altair as alt
import polars as pl
import pandas as pd

if TYPE_CHECKING:
    from duckdb import DuckDBPyConnection


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
):
    """Plot cell mass (dry mass) over time per experiment, one subplot per experiment."""
    query = [
        "listeners__mass__dry_mass AS dry_mass",
        "listeners__mass__dry_mass_fold_change AS dry_mass_fold_change",
        "time/60 AS time_min",
    ]

    raw = pl.DataFrame(
        read_stacked_columns(history_sql, query, order_results=True, conn=conn)
    )

    if raw.shape[0] > 5000:
        print(
            f"Warning: raw data has {raw.shape[0]} rows, which may lead to slow plotting. Consider downsampling."
        )
        alt.data_transformers.enable("vegafusion")

    # Use max simulation length across all experiments so x-axis is consistent
    max_time_min = float(raw["time_min"].max())
    x_scale = alt.Scale(domain=[0, max_time_min], nice=False)

    # --- Create line and spread plots for absolute dry mass ---
    line_abs = (
        alt.Chart(raw)
        .mark_line(strokeWidth=0.5)
        .transform_calculate(
            gen_seed="'gen=' + datum.generation + ', seed=' + datum.lineage_seed"
        )
        .encode(
            x=alt.X("time_min:Q", title="Time (min)", scale=x_scale),
            y=alt.Y("mean(dry_mass):Q", title="Average Dry Mass (fg)"),
            color=alt.Color(
                "gen_seed:N", legend=alt.Legend(title="Generation and Seed")
            ),
        )
        .properties(width=600, height=300)
    ).interactive()

    spread_abs = (
        alt.Chart(raw)
        .mark_area(opacity=0.3)
        .transform_calculate(
            gen_seed="'gen=' + datum.generation + ', seed=' + datum.lineage_seed"
        )
        .encode(
            x=alt.X("time_min:Q", title="Time (min)", scale=x_scale),
            y=alt.Y("ci0(dry_mass):Q"),
            y2=alt.Y2("ci1(dry_mass):Q"),
            color=alt.Color(
                "gen_seed:N", legend=alt.Legend(title="Generation and Seed")
            ),
        )
        .properties(width=600, height=300)
    ).interactive()

    line_rel = (
        alt.Chart(raw)
        .mark_line(strokeWidth=0.5)
        .transform_calculate(
            gen_seed="'gen=' + datum.generation + ', seed=' + datum.lineage_seed"
        )
        .encode(
            x=alt.X("time_min:Q", title="Time (min)", scale=x_scale),
            y=alt.Y("mean(dry_mass_fold_change):Q", title="Normalized Dry Mass"),
            color=alt.Color(
                "gen_seed:N", legend=alt.Legend(title="Generation and Seed")
            ),
        )
        .properties(width=600, height=300)
    ).interactive()

    spread_rel = (
        alt.Chart(raw)
        .mark_area(opacity=0.3)
        .transform_calculate(
            gen_seed="'gen=' + datum.generation + ', seed=' + datum.lineage_seed"
        )
        .encode(
            x=alt.X("time_min:Q", title="Time (min)", scale=x_scale),
            y=alt.Y("ci0(dry_mass_fold_change):Q"),
            y2=alt.Y2("ci1(dry_mass_fold_change):Q"),
            color=alt.Color(
                "gen_seed:N", legend=alt.Legend(title="Generation and Seed")
            ),
        )
        .properties(width=600, height=300)
    ).interactive()

    reference_line = (
        alt.Chart(pd.DataFrame({"y": [2]}))
        .mark_rule(color="red", strokeDash=[5, 5], strokeWidth=1)
        .encode(y="y:Q")
    ).interactive()

    absolute_mass_plot = (spread_abs + line_abs).properties(
        title="Average Absolute Dry Mass with Spread Over Time"
    )

    relative_mass_plot = (spread_rel + line_rel + reference_line).properties(
        title="Average Normalized Dry Mass with Spread Over Time"
    )

    figure = absolute_mass_plot | relative_mass_plot

    out_path = os.path.join(outdir, "multiexperiment_cell_mass.html")
    figure.save(out_path)
    print(f"Saved multi-experiment cell mass visualization to: {out_path}")

    return
