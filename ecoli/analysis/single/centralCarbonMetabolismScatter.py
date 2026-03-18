"""
Central Carbon Metabolism Flux comparison against Toya 2010 measured fluxes.

Scatter plot of mean simulated reaction fluxes (y-axis) vs. Toya 2010
experimentally measured fluxes (x-axis), with error bars on both axes
and a Pearson R correlation in the title.

Ported from wcEcoli
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

# --- Import and Define Units ---
COUNTS_UNITS = units.mmol
VOLUME_UNITS = units.L
MASS_UNITS = units.g
TIME_UNITS = units.s
CONC_UNITS = COUNTS_UNITS / VOLUME_UNITS
FLUX_UNITS = COUNTS_UNITS / VOLUME_UNITS / TIME_UNITS

TIMESTEP = 1 * TIME_UNITS  # time step of the simulation


def plot(
    params: dict[str, Any],
    conn: "DuckDBPyConnection",
    history_sql: str,
    config_sql: str,
    success_sql: str,
    sim_data_paths: dict[str, dict[int, str]],
    validation_data_paths: list[str],
    outdir: str,
    variant_metadata: dict[str, dict[int, Any]],
    variant_names: dict[str, str],
):
    """
    Args:
        params: Unused; reserved for future parameters.
    """
    # -------------------------------------------
    # --- Load Simulation and Validation Data ---
    # -------------------------------------------
    with open_arbitrary_sim_data(sim_data_paths) as f:
        sim_data = pickle.load(f)
    with fsspec_open(validation_data_paths[0], "rb") as f:
        validation_data = pickle.load(f)

    cell_density = sim_data.constants.cell_density

    # --------------------------------------
    # --- Read mass and FBA flux columns ---
    # --------------------------------------
    query = [
        "listeners__mass__cell_mass AS cell_mass",
        "listeners__mass__dry_mass AS dry_mass",
        "listeners__fba_results__base_reaction_fluxes AS base_reaction_fluxes",
    ]

    raw = pl.DataFrame(
        read_stacked_columns(history_sql, query, order_results=True, conn=conn)
    )

    cell_masses = units.fg * raw["cell_mass"]  # (n_timesteps,)
    dry_masses = units.fg * raw["dry_mass"]  # (n_timesteps,)
    flux_matrix = ndlist_to_ndarray(
        raw["base_reaction_fluxes"]
    )  # (n_timesteps, n_rxns)

    sim_reaction_fluxes = CONC_UNITS / TIMESTEP * flux_matrix  # mmol/L/s

    # Retrieve reaction IDs from config metadata
    reaction_ids = np.array(
        field_metadata(conn, config_sql, "listeners__fba_results__base_reaction_fluxes")
    )

    # -----------------------------
    # --- Load Toya 2010 Fluxes ---
    # -----------------------------
    toya_reactions = validation_data.reactionFlux.toya2010fluxes["reactionID"]

    toya_fluxes = toya.adjust_toya_data(
        validation_data.reactionFlux.toya2010fluxes["reactionFlux"],
        cell_masses,
        dry_masses,
        cell_density,
    )  # outputs in mmol/L/s
    toya_stdevs = toya.adjust_toya_data(
        validation_data.reactionFlux.toya2010fluxes["reactionFluxStdev"],
        cell_masses,
        dry_masses,
        cell_density,
    )  # outputs in mmol/L/s

    # ------------------------------------------------------------------
    # Align simulated and Toya fluxes to matching reaction IDs
    # ------------------------------------------------------------------
    sim_flux_means, sim_flux_stdevs = toya.process_simulated_fluxes(
        toya_reactions, reaction_ids, sim_reaction_fluxes
    )
    toya_flux_means = toya.process_toya_data(
        toya_reactions, toya_reactions, toya_fluxes
    )
    toya_flux_stdevs = toya.process_toya_data(
        toya_reactions, toya_reactions, toya_stdevs
    )

    sim_means_num = sim_flux_means.asNumber(FLUX_UNITS)
    sim_stdevs_num = sim_flux_stdevs.asNumber(FLUX_UNITS)
    toya_means_num = toya_flux_means.asNumber(FLUX_UNITS)
    toya_stdevs_num = toya_flux_stdevs.asNumber(FLUX_UNITS)

    # ------------------------------------------------------------------
    # Altair scatter with error bars
    # ------------------------------------------------------------------
    pearson_r = float(np.corrcoef(sim_means_num, toya_means_num)[0, 1])
    flux_unit_str = FLUX_UNITS.strUnit()

    df = pl.DataFrame(
        {
            "reaction": list(toya_reactions),
            "toya_mean": toya_means_num,
            "toya_stdev": toya_stdevs_num,
            "sim_mean": sim_means_num,
            "sim_stdev": sim_stdevs_num,
            # Pre-compute error bar bounds so Altair can encode them directly
            "toya_lo": toya_means_num - toya_stdevs_num,
            "toya_hi": toya_means_num + toya_stdevs_num,
            "sim_lo": sim_means_num - sim_stdevs_num,
            "sim_hi": sim_means_num + sim_stdevs_num,
        }
    )

    base = alt.Chart(df).encode(
        x=alt.X("toya_mean:Q", title=f"Toya 2010 Reaction Flux {flux_unit_str}"),
        y=alt.Y("sim_mean:Q", title=f"Mean WCM Reaction Flux {flux_unit_str}"),
        tooltip=["reaction:N", "toya_mean:Q", "sim_mean:Q"],
    )

    points = base.mark_point(color="steelblue", size=50, filled=True)

    x_errorbars = base.mark_errorbar().encode(
        x=alt.X("toya_lo:Q", title=f"Toya 2010 Reaction Flux {flux_unit_str}"),
        x2="toya_hi:Q",
        color=alt.value("black"),
    )

    y_errorbars = base.mark_errorbar().encode(
        y=alt.Y("sim_lo:Q", title=f"Mean WCM Reaction Flux {flux_unit_str}"),
        y2="sim_hi:Q",
        color=alt.value("black"),
    )

    chart = (
        (points + x_errorbars + y_errorbars)
        .properties(
            title=f"Central Carbon Metabolism Flux, Pearson R = {pearson_r:.2f}",
            width=500,
            height=450,
        )
        .configure_view(strokeWidth=0, fill=None)
    )

    chart.save(os.path.join(outdir, "centralCarbonMetabolismScatter.html"))
    chart.save(os.path.join(outdir, "centralCarbonMetabolismScatter.svg"))
