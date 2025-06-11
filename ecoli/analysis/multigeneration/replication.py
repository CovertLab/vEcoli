"""plots replication fork position, fork counts, cell mass, number of oric.
Option to plot mass per oric and ratio of cell mass to critical initiation mass"""

import os
from typing import Any
import altair as alt


from duckdb import DuckDBPyConnection
import pickle
import polars as pl

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
    with open_arbitrary_sim_data(sim_data_dict) as f:
        sim_data = pickle.load(f)
    genome_length = len(sim_data.process.replication.genome_sequence)

    columns = {
        "Time": "time",
        "Dry": "listeners__mass__dry_mass",
        "Cell_mass": "listeners__mass__cell_mass",
        "Fork_coord": "listeners__replication_data__fork_coordinates",
        # "Crit_mass":"listeners__replication_data__critical_initiation_mass",        #this is not in replication listener yet?
        "Oric": "listeners__replication_data__number_of_oric",
        # "Mass_per_oric":"listeners__replication_data__critical_mass_per_oric",           #this is not in replication listener yet?
    }

    replication_data = read_stacked_columns(
        history_sql, list(columns.values()), conn=conn
    )
    replication_data = pl.from_arrow(replication_data)
    replication_data = replication_data.with_columns(
        (pl.col("time") - pl.col("time").min()).alias("time_min") / 60 / 60
    )

    # Get fork positions
    fork_coord = replication_data.select(
        ["time_min", "listeners__replication_data__fork_coordinates"]
    ).explode("listeners__replication_data__fork_coordinates")
    fork_coord = fork_coord.rename(
        {
            "time_min": "Time (hr)",
            "listeners__replication_data__fork_coordinates": "Fork coordinate",
        }
    )
    # Get fork counts
    fork_coord_exploded = replication_data.select(
        ["time_min", "listeners__replication_data__fork_coordinates"]
    ).explode("listeners__replication_data__fork_coordinates")

    valid_fork_coords = fork_coord_exploded.filter(
        ~pl.col("listeners__replication_data__fork_coordinates").is_null()
        & ~pl.col("listeners__replication_data__fork_coordinates").is_nan()
    )
    fork_counts = valid_fork_coords.group_by("time_min").agg(
        pl.count("listeners__replication_data__fork_coordinates").alias("Fork pairs")
    )

    replication_data = replication_data.join(fork_counts, on="time_min", how="left")

    """ 
    # critical initiation mass

   replication_data = replication_data.with_columns((pl.col("listeners__mass__cell_mass") / pl.col("listeners__replication_data__critical_initiation_mass")).alias("Critical mass equivalents"))

    """
    # plot fork coordinates
    fork_chart = (
        alt.Chart(fork_coord)
        .mark_point()
        .encode(
            x="Time (hr):Q",
            y=alt.Y(
                "Fork coordinate:Q",
                scale=alt.Scale(domain=[-genome_length / 2, genome_length / 2]),
            ),
        )
        .properties(title="DNA polymerase position (nt)")
    )

    # plot fork counts
    fork_count_chart = (
        alt.Chart(replication_data)
        .mark_line()
        .encode(
            x="time_min:Q",
            y=alt.Y("Fork pairs:Q", scale=alt.Scale(domain=[0, 6])),
        )
        .properties(title="Pairs of replication forks")
    )

    """
    # plot critical mass fraction
    crit_mass_chart = alt.Chart(replication_data).mark_line().encode(
        x="time_min:Q",
        y="Critical mass equivalents:Q",
    ).properties(title="Factors of critical initiation mass")
    for n in CRITICAL_N:
        crit_mass_chart += alt.Chart(pl.DataFrame({"y": [n]})).mark_rule().encode(y="y:Q")
    """
    # plot dry mass
    dry_mass_chart = (
        alt.Chart(replication_data)
        .mark_line()
        .encode(
            x="time_min:Q",
            y=alt.Y("listeners__mass__dry_mass:Q", title="Dry mass (fg)"),
        )
        .properties(title="Dry Mass")
    )

    # plot number of oriC
    oric_chart = (
        alt.Chart(replication_data)
        .mark_line()
        .encode(
            x="time_min:Q",
            y="listeners__replication_data__number_of_oric:Q",
        )
        .properties(title="Number of oriC")
    )

    """
    # plot mass per oriC
    mass_per_oric_chart = alt.Chart(replication_data).mark_line().encode(
        x="time_min:Q",
        y="listeners__replication_data__critical_mass_per_oric:Q",
    ).properties(title="Critical mass per oriC")

    
    final_chart = alt.vconcat(
        fork_chart, fork_count_chart, crit_mass_chart,
        dry_mass_chart, oric_chart, mass_per_oric_chart
    ).resolve_scale(y='independent')
    """
    final_chart = alt.vconcat(
        fork_chart, fork_count_chart, dry_mass_chart, oric_chart
    ).resolve_scale(y="independent")
    final_chart.save(os.path.join(outdir, "replication.html"))
