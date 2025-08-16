"""
Visualize specific protein counts over time across generations

You can specify the protein to visualize using the 'protein_id' parameter in params:
    "protein_count": {
        "protein_id": ["Name1", "Name2", ...],
        }
"""

import altair as alt
import os
from typing import Any
import numpy as np
import pickle

import polars as pl
from duckdb import DuckDBPyConnection

from ecoli.library.parquet_emitter import open_arbitrary_sim_data, read_stacked_columns


def plot(
    params: dict[str, Any],
    conn: DuckDBPyConnection,
    history_sql: str,
    config_sql: str,
    success_sql: str,
    sim_data_paths: dict[str, dict[int, str]],
    validation_data_paths: list[str],
    outdir: str,
    variant_metadata: dict[str, dict[int, Any]],
    variant_names: dict[str, str],
):
    """Visualize protein counts over time for specified protein ID across generations."""

    # Load sim_data
    with open_arbitrary_sim_data(sim_data_paths) as f:
        sim_data = pickle.load(f)

    # Get protein ID from parameters
    protein_id = params.get("protein_id")

    if not protein_id:
        print("[ERROR] 'protein_id' parameter is required")
        return None

    # Get monomer IDs from simulation data
    sim_monomer_ids = sim_data.process.translation.monomer_data["id"]

    # Find the index of the target protein
    try:
        protein_idx = np.where(sim_monomer_ids == protein_id)[0]
        if len(protein_idx) == 0:
            print(f"[ERROR] Protein ID '{protein_id}' not found in simulation data")
            return None
        protein_idx = protein_idx[0]
        print(f"[INFO] Found protein '{protein_id}' at index {protein_idx}")
    except Exception as e:
        print(f"[ERROR] Error finding protein ID '{protein_id}': {e}")
        return None

    # Step 1: Get protein counts data using the direct column approach
    subquery = read_stacked_columns(
        history_sql, ["listeners__monomer_counts"], order_results=False
    )

    sql = f"""
    WITH unnested_counts AS (
        SELECT unnest(listeners__monomer_counts) AS counts,
            generate_subscripts(listeners__monomer_counts, 1) AS idx,
            generation, time
        FROM ({subquery})
    )
    SELECT time, generation,
            counts as protein_count
    FROM unnested_counts
    WHERE idx = {protein_idx + 1}  -- SQL uses 1-based indexing
    ORDER BY generation, time
    """

    df = conn.sql(sql).pl()

    # Step 2: Process the data
    # Convert time to minutes
    df = df.with_columns((pl.col("time") / 60).alias("time_min"))

    # Calculate statistics per generation
    generation_stats = df.group_by("generation").agg(
        [
            pl.col("protein_count").mean().alias("mean_count"),
            pl.col("protein_count").std().alias("std_count"),
            pl.col("protein_count").min().alias("min_count"),
            pl.col("protein_count").max().alias("max_count"),
            pl.col("protein_count").count().alias("n_points"),
        ]
    )

    # Convert to pandas for Altair
    plot_df_pd = df.to_pandas()
    stats_df_pd = generation_stats.to_pandas()

    print(f"Data shape: {plot_df_pd.shape}")
    print(f"Generations: {sorted(plot_df_pd['generation'].unique())}")
    print(
        f"Time range: {plot_df_pd['time_min'].min():.1f} - {plot_df_pd['time_min'].max():.1f} min"
    )

    # ------------------------ #

    # Create line chart for protein counts over time
    def create_protein_count_chart():
        base = alt.Chart(plot_df_pd)

        line = (
            base.mark_line(point=True, strokeWidth=2)
            .encode(
                x=alt.X("time_min:Q", title="Time (min)"),
                y=alt.Y("protein_count:Q", title="Protein Count"),
                color=alt.Color(
                    "generation:N",
                    legend=alt.Legend(title="Generation"),
                    scale=alt.Scale(scheme="category10"),
                ),
                tooltip=["time_min:Q", "generation:N", "protein_count:Q", "agent_id:N"],
            )
            .properties(
                title=f"Protein Count of {protein_id} Over Time Across Generations",
                width=800,
                height=400,
            )
        )
        return line

    # Create line chart showing mean ± std per generation
    def create_generation_summary_chart():
        # Calculate time-averaged values per generation
        gen_summary = df.group_by("generation").agg(
            [
                pl.col("protein_count").mean().alias("mean_count"),
                pl.col("protein_count").std().alias("std_count"),
                pl.col("time_min").mean().alias("mean_time"),
            ]
        )

        gen_summary_pd = gen_summary.to_pandas()

        # Calculate error bars
        gen_summary_pd["upper"] = (
            gen_summary_pd["mean_count"] + gen_summary_pd["std_count"]
        )
        gen_summary_pd["lower"] = (
            gen_summary_pd["mean_count"] - gen_summary_pd["std_count"]
        )

        base = alt.Chart(gen_summary_pd)

        # Error bars
        error_bars = base.mark_errorbar(extent="stdev").encode(
            x=alt.X("generation:N", title="Generation"),
            y=alt.Y("mean_count:Q", title="Mean Protein Count"),
            yError=alt.YError("std_count:Q"),
        )

        # Points
        points = base.mark_point(size=100, filled=True).encode(
            x=alt.X("generation:N", title="Generation"),
            y=alt.Y("mean_count:Q", title="Mean Protein Count"),
            tooltip=["generation:N", "mean_count:Q", "std_count:Q"],
        )

        chart = (error_bars + points).properties(
            title=f"Mean Protein Count of {protein_id} by Generation",
            width=600,
            height=300,
        )

        return chart

    # Create histogram of protein counts by generation
    def create_protein_count_histogram():
        hist = (
            alt.Chart(plot_df_pd)
            .mark_bar(opacity=0.7)
            .encode(
                x=alt.X(
                    "protein_count:Q", bin=alt.Bin(maxbins=30), title="Protein Count"
                ),
                y=alt.Y("count():Q", title="Frequency"),
                color=alt.Color("generation:N", scale=alt.Scale(scheme="category10")),
            )
            .properties(
                title=f"Distribution of {protein_id} Counts by Generation",
                width=600,
                height=300,
            )
        )
        return hist

    # Create box plot by generation
    def create_boxplot_by_generation():
        boxplot = (
            alt.Chart(plot_df_pd)
            .mark_boxplot(extent="min-max")
            .encode(
                x=alt.X("generation:N", title="Generation"),
                y=alt.Y("protein_count:Q", title="Protein Count"),
                color=alt.Color("generation:N", scale=alt.Scale(scheme="category10")),
            )
            .properties(
                title=f"Protein Count Distribution of {protein_id} by Generation",
                width=600,
                height=300,
            )
        )
        return boxplot

    # ------------------------ #

    # Generate all charts
    main_chart = create_protein_count_chart()
    generation_summary = create_generation_summary_chart()
    histogram = create_protein_count_histogram()
    boxplot = create_boxplot_by_generation()

    # Combine charts in a comprehensive layout
    combined_chart = alt.vconcat(
        # Main time series chart
        main_chart,
        # Generation summary and boxplot
        alt.hconcat(generation_summary, boxplot),
        # Histogram
        alt.hconcat(histogram),
        title=f"Comprehensive Protein Count Analysis: {protein_id}",
    ).resolve_scale(color="independent")

    # Save the visualization
    out_path = os.path.join(outdir, "protein_count.html")
    combined_chart.save(out_path)
    print(f"Saved protein count visualization to: {out_path}")

    # Save summary statistics
    stats_path = os.path.join(outdir, "protein_count_stats.csv")
    stats_df_pd.to_csv(stats_path, index=False)
    print(f"Saved summary statistics to: {stats_path}")

    # Print summary statistics
    print(f"\nSummary Statistics for {protein_id}:")
    print("=" * 50)
    for _, row in stats_df_pd.iterrows():
        gen = int(row["generation"])
        print(
            f"Generation {gen}: Mean={row['mean_count']:.2f}, "
            f"Std={row['std_count']:.2f}, "
            f"Range=[{row['min_count']:.0f}, {row['max_count']:.0f}], "
            f"N={row['n_points']}"
        )

    return combined_chart
