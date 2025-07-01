"""
The multigeneration analysis method `replication`
1. Record the DNA polymerase position vs time
2. Record # of pairs of replication forks
3. Record the factors of critical initial mass and dry mass
4. Record # of oriC
"""

import altair as alt
import os
from typing import Any
import pickle

from duckdb import DuckDBPyConnection
import polars as pl

from ecoli.library.parquet_emitter import (
    open_arbitrary_sim_data,
    read_stacked_columns,
)

CRITICAL_N = [1, 2, 4, 8]

# ----------------------------------------- #


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
    """Create comprehensive replication visualization plots for E. coli simulation data."""
    # Load sim_data to get genome length
    with open_arbitrary_sim_data(sim_data_dict) as f:
        sim_data = pickle.load(f)
    genome_length = len(sim_data.process.replication.genome_sequence)

    # Define data columns with proper listener names and aliases
    data_columns = [
        'time / 3600 AS "Time (hr)"',
        "listeners__replication_data__fork_coordinates AS fork_coordinates",
        "listeners__replication_data__number_of_oric AS number_of_oric",
        "listeners__mass__cell_mass AS cell_mass",
        "listeners__mass__dry_mass AS dry_mass",
        "listeners__replication_data__critical_initiation_mass AS critical_initiation_mass",
        "listeners__replication_data__critical_mass_per_oric AS critical_mass_per_oric",
    ]

    # Load data
    plot_data = read_stacked_columns(history_sql, data_columns, conn=conn)

    # Convert to DataFrame
    df = pl.DataFrame(plot_data)

    # Process fork coordinates and calculate pairs of forks using Polars
    if "fork_coordinates" in df.columns:
        df = df.with_columns(
            pairs_of_forks=pl.col("fork_coordinates")
            .list.eval(~pl.element().is_nan())
            .list.sum()
            / 2
        )

    # Calculate critical mass equivalents
    if "cell_mass" in df.columns and "critical_initiation_mass" in df.columns:
        df = df.with_columns(
            critical_mass_equivalents=(
                pl.col("cell_mass") / pl.col("critical_initiation_mass")
            )
        )

    # ----------------------------------------- #
    # Create visualization functions
    def create_fork_positions_plot():
        """Create DNA polymerase positions scatter plot."""
        if "fork_coordinates" not in df.columns:
            return None

        # Explode fork coordinates and filter out NaN values
        fork_df = (
            df.select(["Time (hr)", "fork_coordinates"])
            .explode("fork_coordinates")
            .filter(~pl.col("fork_coordinates").is_nan())
            .rename({"fork_coordinates": "Position"})
        )

        if fork_df.height == 0:
            return None
        return (
            alt.Chart(fork_df)
            .mark_circle(size=5, opacity=0.7)
            .encode(
                x=alt.X("Time (hr):Q", title="Time (hr)"),
                y=alt.Y(
                    "Position:Q",
                    scale=alt.Scale(domain=[-genome_length / 2, genome_length / 2]),
                    axis=alt.Axis(
                        values=[-genome_length / 2, 0, genome_length / 2],
                        labelExpr="datum.value == 0 ? 'oriC' : (datum.value < 0 ? '-terC' : '+terC')",
                    ),
                    title="DNA polymerase position (nt)",
                ),
            )
            .properties(title="DNA Polymerase Positions", width=600, height=120)
        )

    def create_pairs_of_forks_plot():
        """Create pairs of replication forks line plot."""
        if "pairs_of_forks" not in df.columns:
            return None

        return (
            alt.Chart(df)
            .mark_line(strokeWidth=2)
            .encode(
                x=alt.X("Time (hr):Q", title="Time (hr)"),
                y=alt.Y(
                    "pairs_of_forks:Q",
                    scale=alt.Scale(domain=[0, 6]),
                    title="Pairs of forks",
                ),
            )
            .properties(title="Pairs of Replication Forks", width=600, height=100)
        )

    def create_critical_mass_plot():
        """Create critical mass equivalents plot with reference lines."""
        if "critical_mass_equivalents" not in df.columns:
            return None

        # Main line plot
        base_plot = (
            alt.Chart(df)
            .mark_line(strokeWidth=2)
            .encode(
                x=alt.X("Time (hr):Q", title="Time (hr)"),
                y=alt.Y(
                    "critical_mass_equivalents:Q",
                    title="Factors of critical initiation mass",
                ),
            )
        )

        # Reference lines for critical N values
        reference_data = pl.DataFrame(
            {"y": CRITICAL_N, "label": [f"N={n}" for n in CRITICAL_N]}
        )

        reference_lines = (
            alt.Chart(reference_data)
            .mark_rule(strokeDash=[5, 5], color="gray", opacity=0.7)
            .encode(y="y:Q")
        )

        # Text labels for reference lines
        reference_labels = (
            alt.Chart(reference_data)
            .mark_text(align="left", dx=5, fontSize=10, color="gray")
            .encode(y="y:Q", text="label:N")
            .transform_calculate(x="0")
            .encode(x=alt.X("x:Q"))
        )

        return (base_plot + reference_lines + reference_labels).properties(
            title="Factors of Critical Initiation Mass", width=600, height=100
        )

    def create_mass_plot(column_name: str, title: str, y_title: str):
        """Create a generic mass plot."""
        if column_name not in df.columns:
            return None

        return (
            alt.Chart(df)
            .mark_line(strokeWidth=2)
            .encode(
                x=alt.X("Time (hr):Q", title="Time (hr)"),
                y=alt.Y(f"{column_name}:Q", title=y_title),
            )
            .properties(title=title, width=600, height=100)
        )

    # ----------------------------------------- #
    # Generate all plots
    plots = []

    # 1. Fork positions
    fork_plot = create_fork_positions_plot()
    if fork_plot:
        plots.append(fork_plot)

    # 2. Pairs of forks
    pairs_plot = create_pairs_of_forks_plot()
    if pairs_plot:
        plots.append(pairs_plot)

    # 3. Critical mass equivalents
    critical_plot = create_critical_mass_plot()
    if critical_plot:
        plots.append(critical_plot)

    # 4. Dry mass
    dry_mass_plot = create_mass_plot("dry_mass", "Dry Mass", "Dry mass (fg)")
    if dry_mass_plot:
        plots.append(dry_mass_plot)

    # 5. Number of oriC
    oric_plot = create_mass_plot("number_of_oric", "Number of oriC", "Number of oriC")
    if oric_plot:
        plots.append(oric_plot)

    # 6. Critical mass per oriC
    mass_per_oric_plot = create_mass_plot(
        "critical_mass_per_oric", "Critical Mass per oriC", "Critical mass per oriC"
    )
    if mass_per_oric_plot:
        plots.append(mass_per_oric_plot)

    # Combine plots or create fallback
    if plots:
        combined_plot = alt.vconcat(*plots).resolve_scale(x="shared")
        print(f"Created visualization with {len(plots)} subplots")
    else:
        # Fallback plot if no data available
        fallback_data = pl.DataFrame(
            {"x": [0], "y": [0], "text": ["No data available for plotting"]}
        )
        combined_plot = (
            alt.Chart(fallback_data)
            .mark_text(fontSize=20, color="red")
            .encode(x=alt.X("x:Q", axis=None), y=alt.Y("y:Q", axis=None), text="text:N")
            .properties(width=600, height=400, title="Replication Data Visualization")
        )
        print("No plottable data found - created fallback message")

    # Save the plot
    output_path = os.path.join(outdir, "replication_report.html")
    combined_plot.save(output_path)
    print(f"Saved visualization to: {output_path}")

    return combined_plot
