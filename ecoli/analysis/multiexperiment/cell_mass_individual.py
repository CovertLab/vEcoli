"""
Plot absolute / normalized cell mass (dry mass) over time across multiple
experiments. time is divided by generation.

This script will have one pair of absolute / normalized dry mass plots per experiment
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

    # --- Create line and spread plots for relative dry mass ---
    experiments = raw.select("experiment_id").unique().to_series().to_list()
    plots = []

    for exp_id in experiments:
        exp_df = raw.filter(pl.col("experiment_id") == exp_id).to_pandas()
        display_name = exp_id if len(exp_id) <= 40 else exp_id[:37] + "..."

        base = alt.Chart(exp_df).add_selection(alt.selection_interval(bind="scales"))

        tooltip_fields = ["time_min:Q", "generation:N"]
        base_encode = {
            "x": alt.X("time_min:Q", title="Time (min)", scale=alt.Scale(nice=False)),
            "color": alt.Color(
                "generation:N",
                legend=alt.Legend(title="Generation"),
                scale=alt.Scale(scheme="category10"),
            ),
        }

        mass_plot = (
            base.mark_line(strokeWidth=2.5)
            .encode(
                x=base_encode["x"],
                color=base_encode["color"],
                tooltip=tooltip_fields + ["dry_mass:Q"],
                detail="lineage_seed:N",
                y=alt.Y(
                    "dry_mass:Q",
                    title="Dry Mass (fg)",
                    scale=alt.Scale(nice=False),
                ),
            )
            .properties(
                width=400,
                height=200,
                title=f"{display_name} - Absolute Dry Mass",
            )
        )

        norm_mass_plot = (
            base.mark_line(strokeWidth=2.5)
            .encode(
                x=base_encode["x"],
                color=base_encode["color"],
                tooltip=tooltip_fields + ["dry_mass_fold_change:Q"],
                detail="lineage_seed:N",
                y=alt.Y(
                    "dry_mass_fold_change:Q",
                    title="Normalized Dry Mass",
                    scale=alt.Scale(nice=False),
                ),
            )
            .properties(
                width=400,
                height=200,
                title=f"{display_name} - Normalized Dry Mass",
            )
        )

        reference_line = (
            alt.Chart(pd.DataFrame({"y": [2]}))
            .mark_rule(color="red", strokeDash=[5, 5], strokeWidth=1)
            .encode(y="y:Q")
        )
        norm_mass_plot = norm_mass_plot + reference_line

        variant_combined = (
            alt.hconcat(mass_plot, norm_mass_plot)
            .resolve_scale(x="shared")
            .properties(title=f"{display_name} Cell Mass")
        )
        plots.append(variant_combined)

    final_plot = plots[0] if len(plots) == 1 else alt.vconcat(*plots)
    final_plot = final_plot.resolve_scale(x="independent", y="independent").properties(
        title="Multi-Experiment Cell Mass Analysis"
    )

    out_path = os.path.join(outdir, "multiexperiment_cell_mass_individual.html")
    final_plot.save(out_path)

    print(f"Saved multi-experiment cell mass visualization to: {out_path}")

    return
