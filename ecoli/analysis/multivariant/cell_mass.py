"""
Plot absolue / normalized cell mass over time for multivariant simulation in vEcoli, and:
1. each variant has its own plot;
2. at each subplot, time is divided by generation id;

It can also be used at multigeneration analysis.
"""

import os
from typing import Any
import altair as alt
import polars as pl
import pandas as pd
from duckdb import DuckDBPyConnection


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
    # Load data with required columns
    required_columns = [
        "time",
        "variant",
        "generation",
        "agent_id",
        "listeners__mass__dry_mass",
    ]

    sql = f"""
    SELECT {", ".join(required_columns)}
    FROM ({history_sql})
    WHERE agent_id = 0
    ORDER BY variant, generation, time
    """

    df = conn.sql(sql).pl()

    # Process time and mass data
    df = df.with_columns(
        [
            (pl.col("time") / 60).alias("time_min"),
        ]
    )

    # Calculate initial mass for each generation for normalization
    generation_stats = df.group_by(["variant", "generation"]).agg(
        [pl.col("listeners__mass__dry_mass").min().alias("initial_mass")]
    )

    # Join back to main dataframe and calculate normalized mass
    df = df.join(generation_stats, on=["variant", "generation"], how="left")

    # Calculate normalized mass
    df = df.with_columns(
        [
            (pl.col("listeners__mass__dry_mass") / pl.col("initial_mass")).alias(
                "dry_mass_normalized"
            )
        ]
    )

    # Get variants and create plots
    variants = df.select("variant").unique().to_series().to_list()

    # ----------------------------------------#
    plots = []

    for variant in variants:
        variant_df = df.filter(pl.col("variant") == variant).to_pandas()
        variant_name = variant_names.get(variant, f"Variant {variant}")

        # Create base chart with line plots only
        base = alt.Chart(variant_df).add_selection(
            alt.selection_interval(bind="scales")
        )

        # Base encoding
        base_encode = {
            "x": alt.X("time_min:Q", title="Time (min)", scale=alt.Scale(nice=False)),
            "color": alt.Color(
                "generation:N",
                legend=alt.Legend(title="Generation"),
                scale=alt.Scale(scheme="category10"),
            ),
            "tooltip": ["time_min:Q", "generation:N"],
        }

        # Absolute dry mass plot
        mass_plot = (
            base.mark_line(strokeWidth=2.5)
            .encode(
                y=alt.Y(
                    "listeners__mass__dry_mass:Q",
                    title="Dry Mass (fg)",
                    scale=alt.Scale(nice=False),
                ),
                tooltip=base_encode["tooltip"] + ["listeners__mass__dry_mass:Q"],
                **{k: v for k, v in base_encode.items() if k != "tooltip"},
            )
            .properties(
                width=400, height=200, title=f"{variant_name} - Absolute Dry Mass"
            )
        )

        # Normalized dry mass plot
        norm_mass_plot = (
            base.mark_line(strokeWidth=2.5)
            .encode(
                y=alt.Y(
                    "dry_mass_normalized:Q",
                    title="Normalized Dry Mass",
                    scale=alt.Scale(nice=False),
                ),
                tooltip=base_encode["tooltip"] + ["dry_mass_normalized:Q"],
                **{k: v for k, v in base_encode.items() if k != "tooltip"},
            )
            .properties(
                width=400, height=200, title=f"{variant_name} - Normalized Dry Mass"
            )
        )

        # Add reference line at y=2 (doubling mass)
        reference_line = (
            alt.Chart(pd.DataFrame({"y": [2]}))
            .mark_rule(color="red", strokeDash=[5, 5], strokeWidth=1)
            .encode(y="y:Q")
        )

        norm_mass_plot = norm_mass_plot + reference_line

        # Combine plots for this variant
        variant_combined = (
            alt.hconcat(mass_plot, norm_mass_plot)
            .resolve_scale(x="shared")
            .properties(title=f"{variant_name} Cell Mass Analysis")
        )

        plots.append(variant_combined)

    # Create combined plot
    final_plot = plots[0] if len(plots) == 1 else alt.vconcat(*plots)
    final_plot = final_plot.resolve_scale(x="independent", y="independent").properties(
        title="Multi-Variant Cell Mass Analysis"
    )

    # Save plot
    out_path = os.path.join(outdir, "multivariant_cell_mass_report.html")
    final_plot.save(out_path)
    print(f"Saved multi-variant cell mass visualization to: {out_path}")

    return final_plot
