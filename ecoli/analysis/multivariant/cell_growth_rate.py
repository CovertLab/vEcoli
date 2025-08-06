"""
Plot cell growth rate (1/hour) over time for multivariant simulation in vEcoli, and:
0. you can specify variants and generations to analyze, like:
        \"multivariant\": {
            ......
            \"cell_growth_rate\": {
                # Optional: specify variants and generations to visualize
                # If not specified, all will be used
                \"variant\": [0, 1, ...],
                \"generation\": [1, 2, ....],
                \"show_reference\": true/false
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


# ------------------------------------- #
def calculate_average_growth_rates(df, variant_names, group_by_generation=False):
    """
    Calculate average cell growth rate for each variant, optionally grouped by generation.

    Args:
        df: Polars DataFrame with processed growth rate data
        variant_names: Dictionary mapping variant IDs to names
        group_by_generation: If True, group by both variant and generation;
                           if False, group by variant only

    Returns:
        Polars DataFrame with average growth rates
    """

    # Determine grouping columns
    group_cols = ["variant"]
    sort_cols = ["variant"]

    if group_by_generation:
        group_cols.append("generation")
        sort_cols.append("generation")

    # Calculate average growth rate
    avg_growth_df = (
        df.filter(pl.col("growth_rate_per_hour").is_not_null())
        .group_by(group_cols)
        .agg(
            [
                pl.col("growth_rate_per_hour").mean().alias("avg_growth_rate"),
                pl.col("growth_rate_per_hour").std().alias("std_growth_rate"),
                pl.col("growth_rate_per_hour").count().alias("data_points"),
                pl.col("growth_rate_per_hour").min().alias("min_growth_rate"),
                pl.col("growth_rate_per_hour").max().alias("max_growth_rate"),
            ]
        )
        .sort(sort_cols)
    )

    # Add variant names
    avg_growth_df = avg_growth_df.with_columns(
        pl.col("variant")
        .map_elements(
            lambda x: variant_names.get(str(x), f"Variant {x}"), return_dtype=pl.Utf8
        )
        .alias("variant_name")
    )

    return avg_growth_df


def create_growth_rate_comparison_chart(
    avg_by_variant, avg_by_variant_gen, ref_growth_rate, show_reference=True
):
    """Create charts comparing average growth rates across variants and generations."""

    # Chart 1: Average by variant only
    avg_variant_df = avg_by_variant.to_pandas()

    variant_bars = (
        alt.Chart(avg_variant_df)
        .mark_bar()
        .encode(
            x=alt.X("variant_name:N", title="Variant"),
            y=alt.Y("avg_growth_rate:Q", title="Average Growth Rate (1/hour)"),
            color=alt.Color(
                "variant_name:N", legend=None, scale=alt.Scale(scheme="category10")
            ),
            tooltip=[
                "variant_name:N",
                "avg_growth_rate:Q",
                "std_growth_rate:Q",
                "data_points:Q",
            ],
        )
    )

    variant_error_bars = (
        alt.Chart(avg_variant_df)
        .mark_errorbar()
        .encode(x="variant_name:N", y="avg_growth_rate:Q", yError="std_growth_rate:Q")
    )

    variant_chart_layers = [variant_bars, variant_error_bars]
    if show_reference:
        variant_ref_line = (
            alt.Chart(pd.DataFrame({"ref_rate": [ref_growth_rate]}))
            .mark_rule(strokeDash=[5, 5], strokeWidth=2, color="red")
            .encode(
                y="ref_rate:Q",
                tooltip=alt.value(f"Reference: {ref_growth_rate:.3f} /hour"),
            )
        )
        variant_chart_layers.append(variant_ref_line)

    variant_chart = (
        alt.layer(*variant_chart_layers)
        .properties(title="Average Growth Rate by Variant", width=350, height=300)
        .resolve_scale(color="independent")
    )

    # Chart 2: Average by variant and generation
    avg_gen_df = avg_by_variant_gen.to_pandas()

    # Create a combined label for x-axis
    avg_gen_df["variant_gen_label"] = (
        avg_gen_df["variant_name"] + " G" + avg_gen_df["generation"].astype(str)
    )

    gen_bars = (
        alt.Chart(avg_gen_df)
        .mark_bar()
        .encode(
            x=alt.X(
                "variant_gen_label:N",
                title="Variant - Generation",
                sort=alt.Sort(["variant", "generation"]),
            ),
            y=alt.Y("avg_growth_rate:Q", title="Average Growth Rate (1/hour)"),
            color=alt.Color(
                "variant_name:N",
                legend=alt.Legend(title="Variant"),
                scale=alt.Scale(scheme="category10"),
            ),
            tooltip=[
                "variant_name:N",
                "generation:O",
                "avg_growth_rate:Q",
                "std_growth_rate:Q",
                "data_points:Q",
            ],
        )
    )

    gen_error_bars = (
        alt.Chart(avg_gen_df)
        .mark_errorbar()
        .encode(
            x="variant_gen_label:N", y="avg_growth_rate:Q", yError="std_growth_rate:Q"
        )
    )

    # Conditional reference line for generation chart
    generation_chart_layers = [gen_bars, gen_error_bars]
    if show_reference:
        gen_ref_line = (
            alt.Chart(
                pd.DataFrame(
                    {
                        "ref_rate": [ref_growth_rate],
                        "legend_label": ["Reference Simulation Growth Rate"],
                    }
                )
            )
            .mark_rule(strokeDash=[5, 5], strokeWidth=2)
            .encode(
                y="ref_rate:Q",
                color=alt.Color(
                    "legend_label:N",
                    scale=alt.Scale(range=["red"]),
                    legend=alt.Legend(title="Reference"),
                ),
                tooltip=alt.value(f"Reference: {ref_growth_rate:.3f} /hour"),
            )
        )
        generation_chart_layers.append(gen_ref_line)

    generation_chart = (
        alt.layer(*generation_chart_layers)
        .properties(
            title="Average Growth Rate by Variant and Generation", width=500, height=300
        )
        .resolve_scale(color="independent")
    )

    # Combine both charts horizontally
    combined_chart = (
        alt.hconcat(variant_chart, generation_chart)
        .resolve_scale(color="shared", y="shared")
        .properties(title="Cell Growth Rate Analysis - Comprehensive Comparison")
    )

    return combined_chart


# ------------------------------------- #


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
    show_reference = params.get(
        "show_reference", True
    )  # Whether to show reference line
    print(f"[INFO] Show reference line: {show_reference}")

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
        # Create chart layers list
        chart_layers = [growth_line]

        # Conditionally add reference growth rate line
        if show_reference:
            ref_line = (
                alt.Chart(
                    pd.DataFrame(
                        {
                            "ref_rate": [ref_growth_rate],
                            "legend_label": ["Reference Simulation Growth Rate"],
                        }
                    )
                )
                .mark_rule(strokeDash=[5, 5], strokeWidth=2)
                .encode(
                    y="ref_rate:Q",
                    color=alt.Color(
                        "legend_label:N",
                        scale=alt.Scale(range=["red"]),
                        legend=alt.Legend(title="Reference"),
                    ),
                    tooltip=alt.value(f"Reference rate: {ref_growth_rate:.3f} /hour"),
                )
            )
            chart_layers.append(ref_line)

        # Combine layers
        variant_chart = (
            alt.layer(*chart_layers)
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

    # Calculate averages
    avg_by_variant = calculate_average_growth_rates(
        df, variant_names, group_by_generation=False
    )
    avg_by_variant_gen = calculate_average_growth_rates(
        df, variant_names, group_by_generation=True
    )

    # Optional: Save results to CSV
    avg_by_variant.write_csv(
        os.path.join(outdir, "average_growth_rates_by_variant.csv")
    )
    avg_by_variant_gen.write_csv(
        os.path.join(outdir, "average_growth_rates_by_variant_generation.csv")
    )

    # Create and save comprehensive comparison chart
    comprehensive_chart = create_growth_rate_comparison_chart(
        avg_by_variant, avg_by_variant_gen, ref_growth_rate, show_reference
    )
    comparison_path = os.path.join(outdir, "average_growth_rate_comparison.html")
    comprehensive_chart.save(comparison_path)
    print(f"Saved comprehensive growth rate comparison to: {comparison_path}")

    return final_chart
