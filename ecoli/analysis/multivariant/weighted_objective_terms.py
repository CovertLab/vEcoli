"""
Plot weighted objective function terms over time for multivariant simulation.

One subplot per variant, stacked vertically. Within each variant the mean
(± CI spread) of each objective term is shown over continuous simulation time,
with lines broken at cell division (detail by generation).
"""

from typing import Any, TYPE_CHECKING
import os

from ecoli.analysis.multivariant.utils import create_variant_label
from ecoli.library.parquet_emitter import read_stacked_columns
import altair as alt
import polars as pl

if TYPE_CHECKING:
    from duckdb import DuckDBPyConnection

OBJECTIVE_QUERY = {
    "secretion": "listeners__fba_results__secretion_term",
    "efficiency": "listeners__fba_results__efficiency_term",
    "kinetics": "listeners__fba_results__kinetics_term",
    "diversity": "listeners__fba_results__diversity_term",
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
    """Plot mean weighted objective terms over time, one subplot per variant."""
    experiment_id = next(iter(variant_metadata.keys()), None)
    per_variant_params: dict[int, Any] = (
        variant_metadata[experiment_id] if experiment_id else {}
    )

    query = [f"{listener} AS {term}_term" for term, listener in OBJECTIVE_QUERY.items()]

    objective_data = pl.DataFrame(
        read_stacked_columns(history_sql, query, order_results=True, conn=conn)
    )

    # Relative time per lineage_seed so time is continuous across generations
    min_t = objective_data.group_by(["lineage_seed"]).agg(
        pl.col("time").min().alias("t_min")
    )
    objective_data = objective_data.join(min_t, on=["lineage_seed"])
    objective_data = objective_data.with_columns(
        ((pl.col("time") - pl.col("t_min")) / 60).alias("Time (min)")
    )

    new_columns = {
        "Time (min)": objective_data["Time (min)"],
        "variant": objective_data["variant"],
        "generation": objective_data["generation"],
        "lineage_seed": objective_data["lineage_seed"],
        **{f"{k} weighted": objective_data[f"{k}_term"] for k in OBJECTIVE_QUERY},
    }
    df = pl.DataFrame(new_columns)

    melted = df.melt(
        id_vars=["Time (min)", "variant", "generation", "lineage_seed"],
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
            # Break line at cell division
            detail=alt.Detail("generation:N"),
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
            detail=alt.Detail("generation:N"),
        )
    )

    variants = melted["variant"].unique().sort()
    plots = []
    for variant_val in variants:
        variant_name = create_variant_label(variant_val, per_variant_params)
        variant_melted = melted.filter(pl.col("variant") == variant_val).to_pandas()

        subplot = alt.layer(spread, line, data=variant_melted).properties(
            width=600, height=250, title=variant_name
        )
        plots.append(subplot)

    final = (
        alt.vconcat(*plots)
        .resolve_scale(x="independent", y="shared")
        .properties(title="Weighted Objective Terms by Variant")
    )

    out_path = os.path.join(outdir, "weighted_objective_terms.html")
    final.save(out_path)
    print(f"Saved multivariant weighted objective terms to: {out_path}")
