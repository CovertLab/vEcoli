import altair as alt

# noinspection PyUnresolvedReferences
from duckdb import DuckDBPyConnection
import polars as pl
from typing import Any, cast

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
    Histogram of doubling times with average marked.

    Configure number of initial generations to skip using ``skip_n_gens`` key
    in params.
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
    # Skip first 8 generations to avoid initialization bias
    doubling_time_sql = skip_n_gens(doubling_time_sql, params["skip_n_gens"])
    doubling_times = conn.sql(f"""
        SELECT (max(time) - min(time)) / 60 AS 'Doubling Time (min)', experiment_id, variant, lineage_seed, generation, agent_id
        FROM ({doubling_time_sql})
        GROUP BY experiment_id, variant, lineage_seed, generation, agent_id
    """).pl()
    # Calculate the average doubling time
    avg_doubling_time = doubling_times["Doubling Time (min)"].mean()
    if avg_doubling_time is None:
        raise ValueError("No doubling times found in the data.")
    # Round for display
    avg_doubling_time_rounded = round(cast(float, avg_doubling_time), 2)

    # Create the base histogram - Define the main X-axis here
    hist = (
        alt.Chart(doubling_times)
        .mark_bar()
        .encode(
            x=alt.X(
                "Doubling Time (min)",
                bin=alt.Bin(maxbins=40),
                axis=alt.Axis(title="Doubling Time (min)", labelFlush=False),
            ),  # Explicit title
            y=alt.Y("count()", axis=alt.Axis(title="Frequency")),  # Nicer Y-axis title
            tooltip=[
                alt.Tooltip("Doubling Time (min)", bin=alt.Bin(maxbins=40)),
                "count()",
            ],
        )
    )

    # Create a DataFrame for the rule and text
    avg_df = pl.DataFrame({"avg": [avg_doubling_time]})

    # Create the vertical rule for the average - Disable its X-axis labels/title
    rule = (
        alt.Chart(avg_df)
        .mark_rule(color="red", strokeDash=[5, 5], size=2)
        .encode(
            x=alt.X("avg:Q"),
            tooltip=[
                alt.Tooltip("avg", title=f"Average: {avg_doubling_time_rounded} min")
            ],
        )
    )

    # Create the text label for the average line - Disable its X-axis labels/title
    text = (
        alt.Chart(avg_df)
        .mark_text(align="left", baseline="middle", dx=7, dy=-20, color="red")
        .encode(
            x=alt.X("avg:Q"),
            text=alt.value(f"{avg_doubling_time_rounded} min"),
            tooltip=[
                alt.Tooltip("avg", title=f"Average: {avg_doubling_time_rounded} min")
            ],
        )
    )

    # Combine the histogram, the rule, and the text label
    # The final chart will use the axis definitions from the 'hist' layer
    chart = hist + rule + text

    # Add overall chart title
    chart = chart.properties(title="Distribution of Doubling Times")

    # Optional: Add interactivity if desired
    chart = chart.interactive()

    # Save the chart
    chart.save(f"{outdir}/doubling_time_histogram.html")
