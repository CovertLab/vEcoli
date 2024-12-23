import os
from typing import Any, cast

from duckdb import DuckDBPyConnection
import polars as pl
import hvplot.polars

from ecoli.library.parquet_emitter import num_cells, read_stacked_columns

hvplot.extension("matplotlib")

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
    sim_data_paths: dict[str, dict[int, str]],
    validation_data_paths: list[str],
    outdir: str,
    variant_metadata: dict[str, dict[int, Any]],
    variant_names: dict[str, str],
):
    assert (
        num_cells(conn, config_sql) == 1
    ), "Mass fraction summary plot requires single-cell data."

    flux_columns = {
        "reaction flux": "listeners__FBA_results__reaction_fluxes",
        # "external exchange fluxes": "listeners__FBA_results__external_exchange_fluxes",
        "target flux": "listeners__enzyme_kinetics__target_fluxes",
    }
    flux_data = read_stacked_columns(
        history_sql, list(flux_columns.values()), conn=conn
    )
    flux_data = pl.DataFrame(flux_data)

    import ipdb; ipdb.set_trace()

    flux_data.write_csv(os.path.join(outdir, "flux_data.csv"))
    # fractions = {
    #     k: (flux_data[v] / flux_data["listeners__mass__dry_mass"]).mean()
    #     for k, v in flux_columns.items()
    # }
    # new_columns = {
    #     "Time (min)": (flux_data["time"] - flux_data["time"].min()) / 60,
    #     **{
    #         f"{k} ({cast(float, fractions[k]):.3f})": flux_data[v] / flux_data[v][0]
    #         for k, v in flux_columns.items()
    #     },
    # }
    # mass_fold_change = pl.DataFrame(new_columns)
    # plot_namespace = mass_fold_change.plot
    # # hvplot.output(backend='matplotlib')
    # plotted_data = plot_namespace.line(
    #     x="Time (min)",
    #     ylabel="Mass (normalized by t = 0 min)",
    #     title="Biomass components (average fraction of total dry mass in parentheses)",
    #     color=COLORS,
    # )
    # hvplot.save(plotted_data, os.path.join(outdir, "mass_fraction_summary.html"))
    # hvplot.save(plotted_data, 'mass_fraction_summary.png', dpi=300)
