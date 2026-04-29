"""
Central Carbon Metabolism Flux comparison against Toya 2010 for multivariant simulation.

Scatter + boxplot overlay: X-axis = Toya 2010 flux; Y-axis = simulated flux.
Fluxes are aggregated across all seeds and generations within each variant.
One subplot per variant, faceted in a grid.
"""

from typing import Any, TYPE_CHECKING
import os
import pickle
import numpy as np

from wholecell.utils import units, toya
from fsspec import open as fsspec_open
from ecoli.library.parquet_emitter import (
    field_metadata,
    ndlist_to_ndarray,
    open_arbitrary_sim_data,
    read_stacked_columns,
)
import altair as alt
import polars as pl

if TYPE_CHECKING:
    from duckdb import DuckDBPyConnection

COUNTS_UNITS = units.mmol
VOLUME_UNITS = units.L
MASS_UNITS = units.g
TIME_UNITS = units.s
CONC_UNITS = COUNTS_UNITS / VOLUME_UNITS
FLUX_UNITS = COUNTS_UNITS / VOLUME_UNITS / TIME_UNITS
TIMESTEP = 1 * TIME_UNITS


def plot(
    params: dict[str, Any],
    conn: "DuckDBPyConnection",
    history_sql: str,
    config_sql: str,
    success_sql: str,
    sim_data_dict: dict[str, dict[int, str]],
    validation_data_paths: list[str],
    outdir: str,
    variant_metadata: dict[str, dict[int, Any]],
    variant_names: dict[str, str],
):
    REDUXCLASSIC = params.get("is_reduxclassic", True)

    with open_arbitrary_sim_data(sim_data_dict) as f:
        sim_data = pickle.load(f)
    with fsspec_open(validation_data_paths[0], "rb") as f:
        validation_data = pickle.load(f)

    cell_density = sim_data.constants.cell_density

    query = [
        "listeners__mass__cell_mass AS cell_mass",
        "listeners__mass__dry_mass AS dry_mass",
        "listeners__fba_results__base_reaction_fluxes AS base_reaction_fluxes",
        "listeners__enzyme_kinetics__counts_to_molar AS counts_to_molar",
    ]

    raw = pl.DataFrame(
        read_stacked_columns(
            history_sql, query, order_results=True, conn=conn, remove_first=REDUXCLASSIC
        )
    )

    reaction_ids = np.array(
        field_metadata(conn, config_sql, "listeners__fba_results__base_reaction_fluxes")
    )
    toya_reactions = validation_data.reactionFlux.toya2010fluxes["reactionID"]
    common_reactions = [
        r for r in toya_reactions if r in {r: i for i, r in enumerate(reaction_ids)}
    ]

    flux_matrix = ndlist_to_ndarray(raw["base_reaction_fluxes"])

    if REDUXCLASSIC:
        counts_to_molar = raw["counts_to_molar"].to_numpy()[:, np.newaxis]
        sim_reaction_fluxes = CONC_UNITS / TIMESTEP * counts_to_molar * flux_matrix
    else:
        sim_reaction_fluxes = CONC_UNITS / TIMESTEP * flux_matrix

    unique_variants = sorted(raw["variant"].unique().to_list())
    variant_col = raw["variant"].to_numpy()

    all_dfs = []
    for variant_val in unique_variants:
        variant_label = variant_names.get(variant_val, f"Variant {variant_val}")
        mask = variant_col == variant_val

        cell_masses_ref = units.fg * raw.filter(pl.Series(mask))["cell_mass"]
        dry_masses_ref = units.fg * raw.filter(pl.Series(mask))["dry_mass"]

        toya_fluxes = toya.adjust_toya_data(
            validation_data.reactionFlux.toya2010fluxes["reactionFlux"],
            cell_masses_ref,
            dry_masses_ref,
            cell_density,
        )
        toya_stdevs = toya.adjust_toya_data(
            validation_data.reactionFlux.toya2010fluxes["reactionFluxStdev"],
            cell_masses_ref,
            dry_masses_ref,
            cell_density,
        )

        sim_flux_means, sim_flux_stdevs = toya.process_simulated_fluxes(
            toya_reactions, reaction_ids, sim_reaction_fluxes[mask, :]
        )
        toya_flux_means = toya.process_toya_data(
            common_reactions, toya_reactions, toya_fluxes
        )
        toya_flux_stdevs = toya.process_toya_data(
            common_reactions, toya_reactions, toya_stdevs
        )

        sim_means_num = sim_flux_means.asNumber(FLUX_UNITS)
        sim_stdevs_num = sim_flux_stdevs.asNumber(FLUX_UNITS)
        toya_means_num = toya_flux_means.asNumber(FLUX_UNITS)
        toya_stdevs_num = toya_flux_stdevs.asNumber(FLUX_UNITS)

        ss_res = np.sum((sim_means_num - toya_means_num) ** 2)
        ss_tot = np.sum((toya_means_num - np.mean(toya_means_num)) ** 2)
        r_squared = float(1 - ss_res / ss_tot)
        pearson_r = float(np.corrcoef(sim_means_num, toya_means_num)[0, 1])
        pearson_r2 = pearson_r**2

        all_dfs.append(
            pl.DataFrame(
                {
                    "reaction": list(toya_reactions),
                    "toya_mean": toya_means_num,
                    "toya_stdev": toya_stdevs_num,
                    "sim_mean": sim_means_num,
                    "sim_stdev": sim_stdevs_num,
                    "toya_lo": toya_means_num - toya_stdevs_num,
                    "toya_hi": toya_means_num + toya_stdevs_num,
                    "sim_lo": sim_means_num - sim_stdevs_num,
                    "sim_hi": sim_means_num + sim_stdevs_num,
                    "variant_label": [variant_label] * len(toya_reactions),
                    "pearson_r2": [pearson_r2] * len(toya_reactions),
                    "r_squared": [r_squared] * len(toya_reactions),
                }
            )
        )

    df_all = pl.concat(all_dfs)
    flux_unit_str = FLUX_UNITS.strUnit()

    points = (
        alt.Chart()
        .mark_point(color="steelblue", size=20, filled=True, opacity=0.4)
        .encode(
            x=alt.X("toya_mean:Q", title=f"Toya 2010 Reaction Flux {flux_unit_str}"),
            y=alt.Y("sim_mean:Q", title=f"Mean WCM Reaction Flux {flux_unit_str}"),
            detail=alt.Detail("reaction:N"),
            tooltip=["variant_label:N", "toya_mean:Q", "sim_mean:Q"],
        )
    )

    boxes = (
        alt.Chart()
        .mark_boxplot(
            color="steelblue",
            opacity=0.6,
            size=10,
            outliers=False,
        )
        .encode(
            x=alt.X("mean(toya_mean):Q", title=""),
            y=alt.Y("mean(sim_mean):Q", title=""),
            detail=alt.Detail("reaction:N"),
        )
    )

    x_errorbars = (
        alt.Chart()
        .mark_rule(color="black", strokeWidth=1)
        .encode(
            x=alt.X("mean(toya_lo):Q"),
            x2="mean(toya_hi):Q",
            y=alt.Y("mean(sim_mean):Q"),
            detail=alt.Detail("reaction:N"),
        )
    )

    # groupby uses variant_label (fixes the bug where source hardcoded "generation")
    annotation = (
        alt.Chart()
        .mark_text(
            lineBreak="\n", align="left", baseline="top", dx=5, dy=5, fontSize=11
        )
        .encode(
            x=alt.value(0),
            y=alt.value(0),
        )
        .transform_aggregate(
            pearson_r2="mean(pearson_r2)",
            r_squared="mean(r_squared)",
            groupby=["variant_label"],
        )
        .transform_calculate(
            multiline_label="'Pearson R\u00b2 = ' + format(datum.pearson_r2, '.2f') + '\\n' + "
            "'R\u00b2 to y=x is ' + format(datum.r_squared, '.2f')"
        )
        .encode(text="multiline_label:N")
    )

    final_chart = (
        alt.layer(boxes, points, x_errorbars, annotation, data=df_all)
        .properties(width=500, height=450)
        .facet(facet=alt.Facet("variant_label:N", title="Variant"), columns=5)
        .resolve_scale(y="independent")
        .configure_view(strokeWidth=0, fill=None)
        .properties(title="Central Carbon Metabolism Flux by Variant")
    )

    final_chart.save(os.path.join(outdir, "centralCarbonMetabolismBoxScatter.html"))
