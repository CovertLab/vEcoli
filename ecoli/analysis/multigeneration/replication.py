import altair as alt
import os
from typing import Any
import pickle

from duckdb import DuckDBPyConnection
import polars as pl
import numpy as np

from ecoli.library.parquet_emitter import (
    open_arbitrary_sim_data,
    read_stacked_columns,
)

CRITICAL_N = [1, 2, 4, 8]


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
    # Load sim_data to get genome length
    with open_arbitrary_sim_data(sim_data_dict) as f:
        sim_data = pickle.load(f)
    genome_length = len(sim_data.process.replication.genome_sequence)

    # Load all the required data
    data_columns = [
        "time",
        "listenersreplication_datanumberOfOric",
        "listenersreplication_datacriticalInitiationMass",
        "listenersreplication_datacriticalMassPerOriC",
        "listenersreplication_datafork_coordinates",
        "listenersmassdrymass",
        "listenersmasscellmass",
    ]

    plot_data = read_stacked_columns(
        history_sql,
        data_columns,
        conn=conn,
    )

    # Convert to DataFrame and add time in hours
    df = pl.DataFrame(plot_data).with_columns(
        **{"Time (hr)": pl.col("time") / 3600}  # Convert seconds to hours
    )

    # Calculate pairs of forks from fork coordinates
    # Fork coordinates is a 2D array, count non-NaN values and divide by 2
    fork_coords = df["listenersreplication_datafork_coordinates"].to_numpy()
    pairs_of_forks = []
    for coord_array in fork_coords:
        if coord_array is not None and len(coord_array) > 0:
            pairs_of_forks.append(np.sum(~np.isnan(coord_array)) / 2)
        else:
            pairs_of_forks.append(0)

    df = df.with_columns(pl.Series("pairs_of_forks", pairs_of_forks))

    # Calculate critical mass equivalents
    df = df.with_columns(
        (
            pl.col("listenersmasscellmass")
            / pl.col("listenersreplication_datacriticalInitiationMass")
        ).alias("critical_mass_equivalents")
    )

    # Create individual plots

    # 1. Fork positions plot - this is complex due to the 2D nature, we'll create a simplified version
    fork_positions_data = []
    for i, (time_val, coords) in enumerate(zip(df["Time (hr)"], fork_coords)):
        if coords is not None and len(coords) > 0:
            for coord in coords:
                if not np.isnan(coord):
                    fork_positions_data.append(
                        {"Time (hr)": time_val, "Position": coord}
                    )

    if fork_positions_data:
        fork_df = pl.DataFrame(fork_positions_data)
        fork_plot = (
            alt.Chart(fork_df)
            .mark_circle(size=5)
            .encode(
                x=alt.X("Time (hr):Q"),
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
            .properties(title="DNA Polymerase Positions", width=600, height=100)
        )
    else:
        # Create empty plot if no fork data
        fork_plot = (
            alt.Chart(pl.DataFrame({"x": [0], "y": [0]}))
            .mark_text(text="No fork data available")
            .encode(x="x:Q", y="y:Q")
            .properties(width=600, height=100)
        )

    # 2. Pairs of forks plot
    pairs_plot = df.plot.line(
        x="Time (hr)",
        y=alt.Y(
            "pairs_of_forks", scale=alt.Scale(domain=[0, 6]), title="Pairs of forks"
        ),
    ).properties(title="Pairs of Replication Forks", width=600, height=100)

    # 3. Critical mass equivalents plot with reference lines
    base_critical_plot = df.plot.line(
        x="Time (hr)",
        y=alt.Y(
            "critical_mass_equivalents", title="Factors of critical initiation mass"
        ),
    )

    # Add reference lines for critical N values
    reference_lines = (
        alt.Chart(
            pl.DataFrame({"y": CRITICAL_N, "label": [f"N={n}" for n in CRITICAL_N]})
        )
        .mark_rule(strokeDash=[5, 5], color="black")
        .encode(y="y:Q")
    )

    critical_plot = (base_critical_plot + reference_lines).properties(
        title="Factors of Critical Initiation Mass", width=600, height=100
    )

    # 4. Dry mass plot
    dry_mass_plot = df.plot.line(
        x="Time (hr)",
        y=alt.Y("listenersmassdryMass", title="Dry mass (fg)"),
    ).properties(title="Dry Mass", width=600, height=100)

    # 5. Number of oriC plot
    oric_plot = df.plot.line(
        x="Time (hr)",
        y=alt.Y("listenersreplication_datanumberOfOric", title="Number of oriC"),
    ).properties(title="Number of oriC", width=600, height=100)

    # 6. Critical mass per oriC plot
    mass_per_oric_plot = df.plot.line(
        x="Time (hr)",
        y=alt.Y(
            "listenersreplication_datacriticalMassPerOriC",
            title="Critical mass per oriC",
        ),
    ).properties(title="Critical Mass per oriC", width=600, height=100)

    # Combine all plots vertically
    combined_plot = alt.vconcat(
        fork_plot,
        pairs_plot,
        critical_plot,
        dry_mass_plot,
        oric_plot,
        mass_per_oric_plot,
    ).resolve_scale(x="shared")

    # Save the plot
    combined_plot.save(os.path.join(outdir, "replication.html"))
