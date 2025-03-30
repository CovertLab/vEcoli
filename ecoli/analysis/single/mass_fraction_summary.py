import os
from typing import Any, cast

from duckdb import DuckDBPyConnection
import polars as pl
import altair as alt

from ecoli.library.parquet_emitter import num_cells, read_stacked_columns

COLORS_256 = [  # From colorbrewer2.org, qualitative 8-class set 1
    [228, 26, 28],
    [55, 126, 184],
    [77, 175, 74],
    [152, 78, 163],
    [255, 127, 0],
    [255, 255, 51],
    [166, 86, 40],
    [247, 129, 191],
]

COLORS = ["#%02x%02x%02x" % (color[0], color[1], color[2]) for color in COLORS_256]


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
    assert num_cells(conn, config_sql) == 1, (
        "Mass fraction summary plot requires single-cell data."
    )

    mass_columns = {
        "Protein": "listeners__mass__protein_mass",
        "tRNA": "listeners__mass__tRna_mass",
        "rRNA": "listeners__mass__rRna_mass",
        "mRNA": "listeners__mass__mRna_mass",
        "DNA": "listeners__mass__dna_mass",
        "Small Mol": "listeners__mass__smallMolecule_mass",
        "Dry": "listeners__mass__dry_mass",
    }
    mass_data = pl.DataFrame(
        read_stacked_columns(history_sql, list(mass_columns.values()), conn=conn)
    )

    fractions = {
        k: cast(float, (mass_data[v] / mass_data["listeners__mass__dry_mass"]).mean())
        for k, v in mass_columns.items()
    }
    new_columns = {
        "Time (min)": (mass_data["time"] - mass_data["time"].min()) / 60,
        **{
            f"{k} ({fractions[k]:.3f})": mass_data[v] / mass_data[v][0]
            for k, v in mass_columns.items()
        },
    }
    mass_fold_change_df = pl.DataFrame(new_columns)

    # Altair requires long form data (also no periods in column names)
    melted_df = mass_fold_change_df.melt(
        id_vars="Time (min)",
        variable_name="Submass",
        value_name="Mass (normalized by t = 0 min)",
    )

    chart = (
        alt.Chart(melted_df)
        .mark_line()
        .encode(
            x=alt.X("Time (min):Q", title="Time (min)"),
            y=alt.Y("Mass (normalized by t = 0 min):Q"),
            color=alt.Color("Submass:N", scale=alt.Scale(range=COLORS)),
        )
        .properties(
            title="Biomass components (average fraction of total dry mass in parentheses)"
        )
    )
    chart.save(os.path.join(outdir, "mass_fraction_summary.html"))
