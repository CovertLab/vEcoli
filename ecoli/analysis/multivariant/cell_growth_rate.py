"""
Plot cell growth rate (1/hour) over time for multivariant simulation in vEcoli, and:
0. you can specify variants and generations to analyze, like:
        \"multivariant\": {
            ......
            \"cell_growth_rate\": {
                \"variant\": [0, 1, ...],
                \"generation\": [1, 2, ....]
                }
            ......
            }
1. each variant has its own plot;
2. at each subplot, time is divided by generation id;

It can also be used at multigeneration analysis.
"""

import os
from typing import Any
import altair as alt
import pickle
import polars as pl
import numpy as np
from duckdb import DuckDBPyConnection
import pandas as pd

from ecoli.library.parquet_emitter import open_arbitrary_sim_data


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
    """Visualize cell growth rate metrics for multivariant E. coli simulation."""

    # Load simulation data to get reference doubling time
    with open_arbitrary_sim_data(sim_data_dict) as f:
        sim_data = pickle.load(f)

    # Reference line for expected growth rate
    sim_doubling_time = sim_data.doubling_time.asNumber()
    ref_growth_rate = float(np.log(2)) / sim_doubling_time * 60  # Convert to 1/hour

    # Required columns for analysis
    required_columns = [
        "time",
        "variant",
        "generation",
        "listeners__mass__instantaneous_growth_rate",
    ]

    # Build SQL query
    all_columns = ", ".join(required_columns)
    sql = f"""
    SELECT {all_columns}
    FROM ({history_sql})
    ORDER BY variant, generation, time
    """

    df = conn.sql(sql).pl()

    # Configuration parameters for filtering
    target_variants = params.get("variant", None)  # List of variant IDs or None for all
    target_generations = params.get(
        "generation", None
    )  # List of generation IDs or None for all

    # Filter by specified variants and generations
    if target_variants is not None:
        print(f"[INFO] Target variants: {target_variants}")
        df = df.filter(pl.col("variant").is_in(target_variants))
    if target_generations is not None:
        print(f"[INFO] Target generations: {target_generations}")
        df = df.filter(pl.col("generation").is_in(target_generations))

    # Time processing
    df = df.with_columns((pl.col("time") / 60).alias("time_min"))

    # Calculate cell doubling time from instantaneous growth rate
    if "listeners__mass__instantaneous_growth_rate" in df.columns:
        # Convert from 1/sec to 1/hour
        growth_rate = pl.col("listeners__mass__instantaneous_growth_rate") * 3600

        # Sanitize doubling time values
        growth_rate_valid_condition = (
            (growth_rate > 0)
            & growth_rate.is_finite()
            & (growth_rate < 2 * ref_growth_rate)
        )

        df = df.with_columns(
            pl.when(growth_rate_valid_condition)
            .then(growth_rate)
            .otherwise(None)
            .alias("growth_rate_per_hour")
        )

    # Specify variants for subplot creation
    unique_variants = df.select("variant").unique().sort("variant")["variant"].to_list()

    if not unique_variants:
        # Create fallback chart if no data
        fallback_df = pd.DataFrame(
            {"message": ["No data available"], "x": [0], "y": [0]}
        )
        fallback_chart = (
            alt.Chart(fallback_df)
            .mark_text(size=20, color="red")
            .encode(x="x:Q", y="y:Q", text="message:N")
            .properties(width=600, height=400, title="No Data Available")
        )
        out_path = os.path.join(outdir, "cell_growth_rate_report.html")
        fallback_chart.save(out_path)
        print(f"[ERROR] No data available. Saved fallback to: {out_path}")
        return fallback_chart

    # Create subplot for each variant
    charts = []

    for variant_id in unique_variants:
        variant_df = df.filter(pl.col("variant") == variant_id)

        if variant_df.height == 0:
            continue

        # Get variant name for title
        variant_name = variant_names.get(str(variant_id), f"Variant {variant_id}")

        # Create line chart for growth rate over time
        base_chart = alt.Chart(variant_df)

        # Growth rate line
        growth_line = base_chart.mark_line(point=True, strokeWidth=2).encode(
            x=alt.X("time_min:Q", title="Time (min)"),
            y=alt.Y("growth_rate_per_hour:Q", title="Growth Rate (1/hour)"),
            color=alt.Color(
                "generation:N",
                legend=alt.Legend(title="Generation"),
                scale=alt.Scale(scheme="category10"),
            ),
            tooltip=["time_min:Q", "growth_rate_per_hour:Q", "generation:N"],
        )

        # Reference growth rate line
        ref_line = (
            alt.Chart(pd.DataFrame({"ref_rate": [ref_growth_rate]}))
            .mark_rule(color="red", strokeDash=[5, 5], strokeWidth=2)
            .encode(
                y="ref_rate:Q",
                tooltip=alt.value(f"Reference rate: {ref_growth_rate:.3f} /hour"),
            )
        )

        # Combine lines
        variant_chart = (
            (growth_line + ref_line)
            .properties(
                title=f"{variant_name} - Cell Growth Rate", width=500, height=300
            )
            .resolve_scale(color="independent", y="shared")
        )

        charts.append(variant_chart)

    # Arrange charts in a grid layout
    if len(charts) == 1:
        combined_chart = charts[0]
    elif len(charts) == 2:
        combined_chart = alt.hconcat(*charts)
    else:
        # For more than 2 charts, arrange in rows of 2
        rows = []
        for i in range(0, len(charts), 2):
            if i + 1 < len(charts):
                rows.append(alt.hconcat(charts[i], charts[i + 1]))
            else:
                rows.append(charts[i])
        combined_chart = alt.vconcat(*rows)

    # Add overall title for multiple plkots
    if len(charts) > 1:
        final_chart = combined_chart.resolve_scale(
            x="shared", y="independent", color="independent"
        ).properties(title="Cell Growth Rate Analysis - Multivariant Comparison")
    else:
        final_chart = combined_chart

    # Save the visualization
    out_path = os.path.join(outdir, "multivariant_cell_growth_rate_report.html")
    final_chart.save(out_path)
    print(f"Saved cell growth rate visualization to: {out_path}")

    return final_chart
