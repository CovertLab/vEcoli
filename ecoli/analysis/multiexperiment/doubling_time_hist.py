"""
Histogram of doubling times across multiple experiments, with average marked.
"""

from typing import Any, cast

import altair as alt
import polars as pl
from duckdb import DuckDBPyConnection

from ecoli.library.parquet_emitter import read_stacked_columns, skip_n_gens


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
    Configure number of initial generations to skip using ``skip_n_gens`` in params.
    """
    doubling_time_sql: str = cast(
        str,
        read_stacked_columns(
            history_sql,
            ["time"],
            order_results=False,
            success_sql=success_sql,
        ),
    )

    """
    params["skip_n_gens"]: int specifies how many initial generations to skip when calculating doubling times
    
    params["plot_by"]: str specifies how to group the data when plotting doubling times; default is "experiment_id"
        - "experiment_id": distribution of doubling times per experiment, grouping variant, generation, and lineage_seed together
        - "variant": distribution of doubling times per variant
        - "generation": distribution of generational doubling time considering all experiments and variants
        - "lineage_seed": distribution of lineage doubling time considering all experiments and variants
        - "gen_seed": distribution of doubling time of gen_seed combo, grouping experiment_id and variant
    """
    skip_n_gens_val = params.get("skip_n_gens", 0)
    plot_by = params.get("plot_by", "gen_seed")
    doubling_time_sql = skip_n_gens(doubling_time_sql, skip_n_gens_val)

    doubling_times = conn.sql(f"""
        SELECT (max(time) - min(time)) / 60 AS "Doubling Time (min)", experiment_id, variant, lineage_seed, generation, agent_id
        FROM ({doubling_time_sql})
        GROUP BY experiment_id, variant, lineage_seed, generation, agent_id
    """).pl()

    avg_doubling_time = doubling_times["Doubling Time (min)"].mean()
    if avg_doubling_time is None:
        raise ValueError("No doubling times found in the data.")
    avg_doubling_time_rounded = round(cast(float, avg_doubling_time), 2)

    hist = (
        alt.Chart(doubling_times)
        .transform_calculate(
            gen_seed="'gen=' + datum.generation + ', seed=' + datum.lineage_seed"
        )
        .mark_bar()
        .encode(
            x=alt.X(
                "Doubling Time (min):Q",
                bin=alt.Bin(maxbins=40, step=0.1),
                axis=alt.Axis(title="Doubling Time (min)", labelFlush=False),
            ),
            y=alt.Y("count()", axis=alt.Axis(title="Frequency")),
            color=alt.Color(
                f"{plot_by}:N",
                legend=alt.Legend(title=f"{plot_by}"),
                scale=alt.Scale(scheme="category20"),
            ),
            tooltip=[
                alt.Tooltip("Doubling Time (min)", bin=alt.Bin(maxbins=40, step=0.1)),
                "count()",
                # "experiment_id",
            ],
        )
    )

    avg_df = pl.DataFrame({"avg": [avg_doubling_time]})

    rule = (
        alt.Chart(avg_df)
        .mark_rule(color="red", strokeDash=[5, 5], size=2)
        .encode(
            x=alt.X("avg:Q"),
            tooltip=[
                alt.Tooltip("avg", title=f"Average: {avg_doubling_time_rounded} min"),
            ],
        )
    )

    text = (
        alt.Chart(avg_df)
        .mark_text(align="left", baseline="middle", dx=7, dy=-20, color="red")
        .encode(
            x=alt.X("avg:Q"),
            text=alt.value(f"{avg_doubling_time_rounded} min"),
            tooltip=[
                alt.Tooltip("avg", title=f"Average: {avg_doubling_time_rounded} min"),
            ],
        )
    )

    chart = hist + rule + text
    chart = chart.properties(title="Distribution of Doubling Times (Multi-Experiment)")
    chart = chart.interactive()

    out_path = f"{outdir}/doubling_time_histogram.html"
    chart.save(out_path)
    print(f"Saved doubling time histogram to: {out_path}")

    return
