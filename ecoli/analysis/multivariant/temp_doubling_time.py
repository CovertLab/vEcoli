"""
Doubling time histograms for any number of variants.
"""

import altair as alt

# noinspection PyUnresolvedReferences
from duckdb import DuckDBPyConnection
import polars as pl
from typing import Any

from ecoli.library.parquet_emitter import read_stacked_columns


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
    """
    Line plot of doubling time vs generation for each lineage seed. Only works for lineage
    simulations with ``single_daughters`` set to True.
    """
    doubling_time_sql = read_stacked_columns(
        history_sql,
        ["time"],
        order_results=False,
    )
    data = conn.sql(f"""
        WITH initial_aggregation AS (
            SELECT (max(time) - min(time)) / 3600 AS "Doubling Time (hr)", experiment_id, variant, lineage_seed, generation, agent_id
            FROM ({doubling_time_sql})
            GROUP BY experiment_id, variant, lineage_seed, generation, agent_id
        ),
        max_generation AS (
            SELECT MAX(generation) AS max_gen, experiment_id, variant, lineage_seed
            FROM ({history_sql})
            GROUP BY experiment_id, variant, lineage_seed
        )
        SELECT "Doubling Time (hr)", lineage_seed AS Seed, generation AS Generation, max_gen, experiment_id, variant, agent_id
        FROM initial_aggregation JOIN max_generation USING (experiment_id, variant, lineage_seed)
    """).pl()
    fail_daughter_one = [2, 3, 6, 10, 13]
    # fail_daughter_one = [9]
    doubling_times = data.filter(
        (
            (pl.col("Generation") != pl.col("max_gen"))
            | (
                (pl.col("Generation") == 24)
                & ~(
                    pl.col("agent_id").str.ends_with("1")
                    & pl.col("Seed").is_in(fail_daughter_one)
                )
            )
            | (
                pl.col("agent_id").str.ends_with("1")
                & ~pl.col("Seed").is_in(fail_daughter_one)
            )
        )
        & (pl.col("Doubling Time (hr)") != 3.0)
    )
    death_times = data.filter(
        (
            (pl.col("Generation") == pl.col("max_gen"))
            & (
                (pl.col("Generation") != 24)
                | (
                    pl.col("agent_id").str.ends_with("1")
                    & pl.col("Seed").is_in(fail_daughter_one)
                )
            )
            & (
                ~pl.col("agent_id").str.ends_with("1")
                | pl.col("Seed").is_in(fail_daughter_one)
            )
        )
        | (pl.col("Doubling Time (hr)") == 3.0)
    )

    selection = alt.selection_point(fields=["Seed"], bind="legend")
    chart = (
        alt.Chart(doubling_times)
        .mark_line()
        .encode(
            x="Generation",
            y="Doubling Time (hr)",
            color=alt.Color("Seed", type="nominal"),
            tooltip=["Doubling Time (hr)", "Seed"],
            opacity=alt.when(selection).then(alt.value(1)).otherwise(alt.value(0.2)),
        )
        .add_params(selection)
        .interactive()
    )
    death_points = (
        alt.Chart(death_times)
        .mark_point(shape="cross")
        .encode(
            x="Generation",
            y="Doubling Time (hr)",
            color=alt.Color("Seed", type="nominal"),
            opacity=alt.when(selection).then(alt.value(1)).otherwise(alt.value(0.2)),
            tooltip=["Doubling Time (hr)", "Seed"],
        )
    )
    exp_avg = alt.Chart().mark_rule(strokeDash=[2, 2]).encode(y=alt.datum(1 / 0.47))
    # exp_avg = alt.Chart().mark_rule(strokeDash=[2, 2]).encode(y=alt.datum(67/60))
    sim_avg_df = doubling_times.group_by("experiment_id", "variant", "Generation").agg(
        pl.mean("Doubling Time (hr)")
    )
    sim_avg = (
        alt.Chart(sim_avg_df)
        .mark_line(strokeDash=[2, 2])
        .encode(x="Generation", y="Doubling Time (hr)", tooltip=["Doubling Time (hr)"])
    )
    chart = chart + exp_avg + sim_avg + death_points
    chart.save(f"{outdir}/doubling_time.html")
