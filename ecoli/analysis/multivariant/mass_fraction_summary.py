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

    read_columns = ["time", "variant", "lineage_seed"] + list(mass_columns.values())
    df = pl.DataFrame(read_stacked_columns(history_sql, read_columns, conn=conn))

    df = df.with_columns([(pl.col("time") / 3600).alias("Time (hr)")]).rename(
        {v: k for k, v in mass_columns.items()}
    )

    melted_df = df.melt(
        id_vars=["Time (hr)", "variant", "lineage_seed"],
        value_vars=list(mass_columns.keys()),
        variable_name="Mass Component",
        value_name="Mass (fg)",
    )
    melted_df = melted_df.with_columns(
        [
            pl.col("variant").alias("Variant Name"),
            pl.col("lineage_seed").cast(str).alias("Seed"),
        ]
    )

    chart = (
        alt.Chart(melted_df)
        .mark_line()
        .encode(
            x=alt.X("Time (hr):Q", title="Time (hr)"),
            y=alt.Y("Mass (fg):Q"),
            color=alt.Color("Variant Name:N", title="Variant"),
            tooltip=["Time (hr)", "Submass", "Mass (fg)", "Variant Name"],
        )
        .facet(facet=alt.Facet("Submass:N", title=None), columns=2)
        .properties(title=" Cell Mass Components by Variant")
        .interactive()
    )

    # Save as HTML
    chart.save(os.path.join(outdir, "mass_fraction_summary.html"))
