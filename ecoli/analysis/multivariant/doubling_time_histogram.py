"""
Plot histograms of doubling times and compares the distributions across
different variants. optionally plots percentage of simulations that reach
the simulation-specified number of generations without failing.
"""

import altair as alt


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
    DOUBLING_TIME_LIMIT_MIN = 180
    IGNORE_FIRST_N_GENS = 4
    PLOT_COMPLETION_RATES = True
    time_sql = read_stacked_columns(history_sql, ["time"], order_results=False)
    full_df = conn.sql(f"""
          SELECT
              experiment_id,
              variant,
              lineage_seed,
              generation,
              agent_id,
              min(time) AS start_time,
              max(time) AS end_time,
              (max(time) - min(time)) / 60 AS doubling_time_min
          FROM ({time_sql})
          GROUP BY experiment_id, variant, lineage_seed, generation, agent_id
      """).pl()

    successful_sims = conn.sql(success_sql).pl()
    full_df = full_df.join(
        successful_sims,
        how="semi",
        on=["experiment_id", "variant", "lineage_seed", "generation", "agent_id"],
    )

    # Filter out timeout cells and early generations
    full_df = full_df.filter(
        (full_df["doubling_time_min"] <= DOUBLING_TIME_LIMIT_MIN)
        & (full_df["generation"] >= IGNORE_FIRST_N_GENS)
    )

    # Plot 1: Histogram of doubling times
    hist = (
        alt.Chart(full_df)
        .mark_bar(opacity=0.6)
        .encode(
            x=alt.X(
                "doubling_time_min:Q",
                bin=alt.Bin(maxbins=36),
                title="Doubling Time (min)",
            ),
            y=alt.Y("count()", title="Frequency"),
            color=alt.Color("variant:N", title="Variant"),
            tooltip=["variant:N", "count()"],
        )
        .properties(title="Distribution of Doubling Times")
    )

    # Plot 2: Completion rates by variant (optional)
    if PLOT_COMPLETION_RATES:
        gen0_counts = (
            full_df.filter(pl.col("generation") == 0).group_by("variant").count()
        )
        last_gen = full_df["generation"].max()
        last_gen_counts = (
            full_df.filter(pl.col("generation") == last_gen).group_by("variant").count()
        )

        rates = gen0_counts.join(
            last_gen_counts, on="variant", suffix="_last", how="left"
        )
        rates = rates.with_columns(
            ((rates["count_last"] / rates["count"]).alias("completion_rate"))
        )

        rate_plot = (
            alt.Chart(rates)
            .mark_bar()
            .encode(
                x=alt.X("variant:N", title="Variant"),
                y=alt.Y("completion_rate:Q", title="Completion Rate"),
                color="variant:N",
                tooltip=["variant:N", "completion_rate:Q"],
            )
            .properties(title="Completion Rates by Variant")
        )

        combined = alt.vconcat(hist, rate_plot).resolve_scale(color="independent")
    else:
        combined = hist

    combined.save(f"{outdir}/doubling_time_histogram.html")
