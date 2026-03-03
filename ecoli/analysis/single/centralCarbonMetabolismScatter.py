"""
Central Carbon Metabolism Flux comparison against Toya 2010 measured fluxes.

Scatter plot of mean simulated reaction fluxes (y-axis) vs. Toya 2010
experimentally measured fluxes (x-axis), with error bars on both axes
and a Pearson R correlation in the title.

Ported from wcEcoli
"""

from typing import Any, cast, TYPE_CHECKING
import os
import pickle
import numpy as np
import plotly.graph_objects as go
from wholecell.utils import units, toya
from ecoli.library.parquet_emitter import (
    field_metadata,
    ndlist_to_ndarray,
    open_arbitrary_sim_data,
    read_stacked_columns,
)

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
    with open(validation_data_paths[0], "rb") as f:
        validation_data = pickle.load(f)

    cell_density = sim_data.constants.cell_density

    # --------------------------------------------------------------
    # --- Read mass and FBA flux columns from parquet via DuckDB ---
    # --------------------------------------------------------------
    subquery = cast(
        str,
        read_stacked_columns(
            history_sql,
            [
                "listeners__mass__cell_mass",
                "listeners__mass__dry_mass",
                "listeners__fba_results__base_reaction_fluxes",
            ],
            order_results=False,
        ),
    )

    raw = conn.sql(f"""
            SELECT
                list(listeners__mass__cell_mass ORDER BY time) AS cell_mass,
                list(listeners__mass__dry_mass ORDER BY time) AS dry_mass,
                list(listeners__fba_results__base_reaction_fluxes ORDER BY time)
                    AS base_reaction_fluxes,
            FROM ({subquery})
            GROUP BY experiment_id, variant, lineage_seed, generation, agent_id
        """).pl()

    cell_masses = units.fg * ndlist_to_ndarray(raw["cell_mass"])[0]  # (n_timesteps,)
    dry_masses = units.fg * ndlist_to_ndarray(raw["dry_mass"])[0]  # (n_timesteps,)
    flux_matrix = ndlist_to_ndarray(raw["base_reaction_fluxes"])[
        0
    ]  # (n_timesteps, n_rxns)

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
    # Plotly scatter with error bars
    # ------------------------------------------------------------------

    pearson_r = float(np.corrcoef(sim_means_num, toya_means_num)[0, 1])

    scatter = go.Scatter(
        x=toya_means_num,
        y=sim_means_num,
        mode="markers",
        error_x=dict(type="data", array=toya_stdevs_num, visible=True, color="black"),
        error_y=dict(type="data", array=sim_stdevs_num, visible=True, color="black"),
        marker=dict(color="steelblue", size=7),
        text=list(toya_reactions),
        hovertemplate=(
            "<b>%{text}</b><br>"
            f"Toya 2010: %{{x:.4f}} {FLUX_UNITS.strUnit()}<br>"
            f"Simulation: %{{y:.4f}} {FLUX_UNITS.strUnit()}<extra></extra>"
        ),
    )

    fig = go.Figure(data=[scatter])
    fig.update_layout(
        title=f"Central Carbon Metabolism Flux, Pearson R = {pearson_r:.2f}",
        xaxis_title=f"Toya 2010 Reaction Flux {FLUX_UNITS.strUnit()}",
        yaxis_title=f"Mean WCM Reaction Flux {FLUX_UNITS.strUnit()}",
        width=700,
        height=600,
        template="plotly_white",
        paper_bgcolor="rgba(255, 0, 0, 0)",
        plot_bgcolor="rgba(255, 0, 0, 0)",
    )

    fig.write_html(os.path.join(outdir, "centralCarbonMetabolismScatter.html"))
