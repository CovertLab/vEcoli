"""
Plot mass components (dry mass, protein mass, rRNA mass, mRNA mass, dna mass) over time for multiple generations
"""

import os
from typing import Any

import altair as alt
import polars as pl
from duckdb import DuckDBPyConnection

from ecoli.library.parquet_emitter import read_stacked_columns


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
    mass_columns = {
        "Dry mass": "listeners__mass__dry_mass",
        "Protein mass": "listeners__mass__protein_mass",
        "rRNA mass": "listeners__mass__rRna_mass",
        "mRNA mass": "listeners__mass__mRna_mass",
        "DNA mass": "listeners__mass__dna_mass",
    }

    columns_to_read = ["time"] + list(mass_columns.values())
    df = pl.DataFrame(read_stacked_columns(history_sql, columns_to_read, conn=conn))

    rename_map = {v: k for k, v in mass_columns.items()}
    df = df.rename(rename_map)

    df = df.with_columns([(pl.col("time") / 3600).alias("Time (hr)")])

    melted = df.melt(
        id_vars="Time (hr)",
        value_vars=list(mass_columns.keys()),
        variable_name="Mass Component",
        value_name="Mass (fg)",
    )

    chart = (
        alt.Chart(melted)
        .mark_line()
        .encode(
            x=alt.X("Time (hr):Q", title="Time (hr)"),
            y=alt.Y("Mass (fg):Q", title="Mass (fg)"),
            color=alt.Color("Mass Component:N", title="Mass Component"),
            tooltip=["Time (hr)", "Mass Component", "Mass (fg)"],
        )
        .properties(title="Cell Mass Fractions")
        .interactive()
    )

    chart.save(os.path.join(outdir, "mass_fraction_summary.html"))
