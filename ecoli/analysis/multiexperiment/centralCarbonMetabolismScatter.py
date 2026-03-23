"""
Central Carbon Metabolism Flux comparison against Toya 2010 across multiple
experiments.

Mirrors the single/centralCarbonMetabolismScatter.py scatter layout:
  - X-axis: Toya 2010 experimentally measured flux (fixed; one value per reaction).
  - Y-axis: simulated flux.

For each reaction (at its Toya x position) the chart shows a boxplot whose
statistics are computed from ALL simulated flux values pooled across every
experiment and every timestep within that generation.  The Toya ±stdev is
drawn as a horizontal error bar at the box median.

The chart is faceted by generation (one row per generation).

Note: mark_boxplot in Vega-Lite does not reliably group by exact quantitative
x values, so box statistics are computed explicitly in Polars and drawn with
primitive marks (bar + rule + tick).
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
    plot_by = params.get("plot_by", "generation")
    REDUXCLASSIC = params.get("is_reduxclassic", True)

    with open_arbitrary_sim_data(sim_data_dict) as f:
        sim_data = pickle.load(f)
    with fsspec_open(validation_data_paths[0], "rb") as f:
        validation_data = pickle.load(f)

    cell_density = sim_data.constants.cell_density

    # ---------------------------------------
    # --- Read mass and FBA flux columns ---
    # ---------------------------------------
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

    reaction_id_to_index = {r: i for i, r in enumerate(reaction_ids)}
    common_reactions = [r for r in toya_reactions if r in reaction_id_to_index]

    flux_matrix = ndlist_to_ndarray(raw["base_reaction_fluxes"])

    if REDUXCLASSIC:
        # convert counts to mmol/L/s; must be numpy (n_timesteps,1) to broadcast with flux_matrix
        counts_to_molar = raw["counts_to_molar"].to_numpy()[:, np.newaxis]
        sim_reaction_fluxes = (
            CONC_UNITS / TIMESTEP * counts_to_molar * flux_matrix
        )  # mmol/L/s
    else:
        sim_reaction_fluxes = CONC_UNITS / TIMESTEP * flux_matrix  # mmol/L/s

    # -----------------------------------------------
    # --- Format Simulated DataFrame with toya.py ---
    # -----------------------------------------------
    raw = raw.with_columns(
        pl.concat_str(
            pl.lit("experiment_id="),
            pl.col("experiment_id"),
            pl.lit(", variant="),
            pl.col("variant").cast(pl.Utf8),
            pl.lit(", seed="),
            pl.col("lineage_seed").cast(pl.Utf8),
            pl.lit(", generation="),
            pl.col("generation").cast(pl.Utf8),
        ).alias("sim_meta")
    )

    sim_meta = np.unique(raw["sim_meta"])

    # -----------------------------------------------
    # --- Build one unified DataFrame across all sims ---
    # -----------------------------------------------
    all_dfs = []
    for sim in sim_meta:
        mask = raw["sim_meta"] == sim
        sim_row = raw.filter(mask).row(0, named=True)

        experiment_id = sim_row["experiment_id"]
        variant = sim_row["variant"]
        lineage_seed = sim_row["lineage_seed"]
        generation = sim_row["generation"]

        cell_masses_ref = units.fg * raw.filter(mask)["cell_mass"]
        dry_masses_ref = units.fg * raw.filter(mask)["dry_mass"]

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
                    "sim_meta": [sim] * len(toya_reactions),
                    "pearson_r2": [pearson_r2] * len(toya_reactions),
                    "r_squared": [r_squared] * len(toya_reactions),
                    "experiment_id": [experiment_id] * len(toya_reactions),
                    "variant": [variant] * len(toya_reactions),
                    "lineage_seed": [lineage_seed] * len(toya_reactions),
                    "generation": [generation] * len(toya_reactions),
                }
            )
        )

    df_all = pl.concat(all_dfs)
    flux_unit_str = FLUX_UNITS.strUnit()

    # -----------------------------------------------
    # --- Build chart facet by sim_meta -------------
    # -----------------------------------------------
    base = alt.Chart().encode(
        x=alt.X("mean(toya_mean):Q", title=f"Toya 2010 Reaction Flux {flux_unit_str}"),
        y=alt.Y("mean(sim_mean):Q", title=f"Mean WCM Reaction Flux {flux_unit_str}"),
        detail=alt.Detail("reaction:N"),
        tooltip=[
            "reaction:N",
            "mean(toya_mean):Q",
            "mean(sim_mean):Q",
        ],
    )

    points = base.mark_point(color="steelblue", size=50, filled=True)

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

    y_errorbars = (
        alt.Chart()
        .mark_rule(color="black", strokeWidth=1)
        .encode(
            y=alt.Y("mean(sim_lo):Q"),
            y2="mean(sim_hi):Q",
            x=alt.X("mean(toya_mean):Q"),
            detail=alt.Detail("reaction:N"),
        )
    )

    # Annotation layer for per-facet stats
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
            groupby=[plot_by],
        )
        .transform_calculate(
            multiline_label="'Pearson R² = ' + format(datum.pearson_r2, '.2f') + '\\n' + "
            "'R² to y=x is ' + format(datum.r_squared, '.2f')"
        )
        .encode(text="multiline_label:N")
    )

    final_chart = (
        alt.layer(points, x_errorbars, y_errorbars, annotation, data=df_all)
        .properties(width=300, height=300)
        .facet(facet=alt.Facet(f"{plot_by}:N", title=f"{plot_by}"), columns=5)
        .resolve_scale(y="independent")
        .configure_view(strokeWidth=0, fill=None)
        .properties(title="Central Carbon Metabolism Flux")
    )

    final_chart.save(os.path.join(outdir, "centralCarbonMetabolismScatter.html"))
