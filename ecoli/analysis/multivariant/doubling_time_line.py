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
    doubling_times = conn.sql(f"""
        SELECT (max(time) - min(time)) / 3600 AS 'Doubling Time (hr)', experiment_id, variant, lineage_seed, generation, agent_id
        FROM ({doubling_time_sql})
        GROUP BY experiment_id, variant, lineage_seed, generation, agent_id
    """).pl()
    successful_sims = conn.sql(success_sql).pl()
    death_times = doubling_times.join(
        successful_sims,
        how="anti",
        on=["experiment_id", "variant", "lineage_seed", "agent_id"],
    )
    doubling_times = doubling_times.join(
        successful_sims,
        how="semi",
        on=["experiment_id", "variant", "lineage_seed", "agent_id"],
    )

    selection = alt.selection_point(fields=["lineage_seed"], bind="legend")
    chart = (
        alt.Chart(doubling_times)
        .mark_line()
        .encode(
            x="generation",
            y="Doubling Time (hr)",
            color=alt.Color("lineage_seed", type="nominal"),
            tooltip=["Doubling Time (hr)", "lineage_seed"],
            opacity=alt.when(selection).then(alt.value(1)).otherwise(alt.value(0.2)),
        )
        .add_params(selection)
        .interactive()
    )
    death_points = (
        alt.Chart(death_times)
        .mark_point(shape="cross")
        .encode(
            x="generation",
            y="Doubling Time (hr)",
            color=alt.Color("lineage_seed", type="nominal"),
            opacity=alt.when(selection).then(alt.value(1)).otherwise(alt.value(0.2)),
            tooltip=["Doubling Time (hr)", "lineage_seed"],
        )
    )
    exp_avg = alt.Chart().mark_rule(strokeDash=[2, 2]).encode(y=alt.datum(1 / 0.47))
    sim_avg_df = doubling_times.group_by("experiment_id", "variant", "generation").agg(
        pl.mean("Doubling Time (hr)")
    )
    sim_avg = (
        alt.Chart(sim_avg_df)
        .mark_line(strokeDash=[2, 2])
        .encode(x="generation", y="Doubling Time (hr)", tooltip=["Doubling Time (hr)"])
    )
    chart = chart + exp_avg + sim_avg + death_points
    chart.save(f"{outdir}/doubling_time.html")
