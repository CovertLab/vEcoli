"""
Plot weighted objective function terms over time across multiple experiments.

Uses the same objective terms and weights as single/weighted_objective_terms.
Plots mean objective term (weighted and unweighted) per experiment over time,
with one subplot per experiment (or one line per experiment by term).

Ported from single/weighted_objective_terms.py.
"""

from typing import Any, TYPE_CHECKING
import os

from ecoli.library.parquet_emitter import read_stacked_columns
import altair as alt
import polars as pl
import numpy as np

if TYPE_CHECKING:
    from duckdb import DuckDBPyConnection

COLORS_256 = [  # From colorbrewer2.org, qualitative 8-class set 1
    [228, 26, 28],
    [55, 126, 184],
    [77, 175, 74],
    [152, 78, 163],
    [255, 127, 0],
    [255, 255, 51],
    [166, 86, 40],
    [247, 129, 191],
]

COLORS = ["#%02x%02x%02x" % (color[0], color[1], color[2]) for color in COLORS_256]

OBJECTIVE_QUERY = {
    "secretion": "listeners__fba_results__secretion_term",
    "efficiency": "listeners__fba_results__efficiency_term",
    "kinetics": "listeners__fba_results__kinetics_term",  # 0.00001
    "diversity": "listeners__fba_results__diversity_term",  # 0.001 Heena's addition to minimize number of reactions with no flow
    "homeostatic": "listeners__fba_results__homeostatic_term",
}


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
    group_by = params.get("group_by", "gen_seed")

    """Plot mean weighted/unweighted objective terms over time, faceted by experiment."""

    query = [f"{listener} AS {term}_term" for term, listener in OBJECTIVE_QUERY.items()]

    objective_data = pl.DataFrame(
        read_stacked_columns(history_sql, query, order_results=True, conn=conn)
    )

    objective_data = objective_data.with_columns(
        gen_seed=(
            pl.lit("gen=")
            + pl.col("generation").cast(pl.Utf8)
            + pl.lit(", seed=")
            + pl.col("lineage_seed").cast(pl.Utf8)
        )
    )

    if group_by == "gen_seed" or group_by == "generation":
        min_t = objective_data.group_by(["generation", "lineage_seed"]).agg(
            pl.col("time").min().alias("t_min")
        )
        objective_data = objective_data.join(min_t, on=["generation", "lineage_seed"])
        objective_data = objective_data.with_columns(
            ((pl.col("time") - pl.col("t_min")) / 60).alias("Time (min)")
        )
    else:
        objective_data = objective_data.with_columns(
            ((pl.col("time") - pl.col("time").min()) / 60).alias("Time (min)")
        )

    new_columns = {
        "Time (min)": (objective_data["time"] - objective_data["time"].min()) / 60,
        "group_by": objective_data[group_by],
        "experiment_id": objective_data["experiment_id"],
        "generation": objective_data["generation"],
        **{
            f"{k} weighted": objective_data[f"{k}_term"]
            for k, v in OBJECTIVE_QUERY.items()
        },
        # **{
        #     f"{k} unweighted": objective_data[f'{k}_term'] / OBJECTIVE_WEIGHTS[k]
        #     for k, v in OBJECTIVE_WEIGHTS.items()
        # },
    }

    df = pl.DataFrame(new_columns)

    # Long form for plotting
    melted = df.melt(
        id_vars=["Time (min)", "group_by", "generation", "experiment_id"],
        variable_name="Term",
        value_name="Objective Term",
    )

    line = (
        alt.Chart()
        .mark_line(strokeWidth=0.5)
        .encode(
            x=alt.X("Time (min):Q", title="Time (min)"),
            y=alt.Y("mean(Objective Term):Q", title="Mean Objective Terms"),
            color=alt.Color("Term:N", legend=alt.Legend(title="Objective Terms")),
            detail=alt.Detail(
                "generation:N"
            ),  # separate path per generation so no line across division
        )
    )

    spread = (
        alt.Chart()
        .mark_area(opacity=0.3)
        .encode(
            x=alt.X("Time (min):Q", title="Time (min)"),
            y=alt.Y("ci0(Objective Term):Q"),
            y2=alt.Y2("ci1(Objective Term):Q"),
            color=alt.Color("Term:N", legend=alt.Legend(title="Objective Terms")),
            detail=alt.Detail(
                "generation:N"
            ),  # separate path per generation so no line across division
        )
    )

    # --- Save Plot 1: Combined plot with mean and spread ---
    n_experiments = len(np.unique(objective_data["experiment_id"]))

    figure_combined = (
        alt.layer(spread, line, data=melted)
        .facet(column=alt.Facet("group_by:N"))
        .resolve_scale(
            x="independent"
        )  # tight x-axis per facet, 0 to max for that group
        .properties(
            title=f"Objective Terms across {n_experiments} experiments grouped by {group_by}"
        )
    )
    out_path = os.path.join(outdir, "multiexperiment_weighted_objective_terms.html")
    figure_combined.save(out_path)

    # --- Save Plot 2: Individual Objective Term per Experiment (Mean across gen and seed) ---
    figure_individual = (
        alt.layer(spread, line, data=melted)
        .facet(facet=alt.Facet("experiment_id:N"), columns=5)
        .resolve_scale(
            x="independent"
        )  # tight x-axis per facet, 0 to max for that group
        .properties(
            title="Mean Objective Term of each experiment across the entire sim (gen and seed)"
        )
    )
    out_path = os.path.join(
        outdir, "multiexperiment_weighted_objective_terms_individual.html"
    )
    figure_individual.save(out_path)

    print(f"Saved multi-experiment weighted objective terms to: {out_path}")
    return
