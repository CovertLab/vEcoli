"""
Central Carbon Metabolism Flux comparison against Toya 2010 across multiple
experiments.

Scatter + distribution (violin and strip) of simulated reaction fluxes (y-axis)
per Toya 2010 reaction (fixed set on x/facets). Each experiment has ~m
datapoints per reaction; we show the distribution across experiments via
violin + strip plots faceted by reaction and experiment_id.

Ported from single/centralCarbonMetabolismScatter.py.
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
    """
    Build long dataframe: experiment_id, reaction, sim_flux (and optional toya_mean).
    Then facet by reaction (fixed set) and experiment_id; in each cell show
    violin + strip of sim_flux.
    """
    with open_arbitrary_sim_data(sim_data_dict) as f:
        sim_data = pickle.load(f)
    with fsspec_open(validation_data_paths[0], "rb") as f:
        validation_data = pickle.load(f)

    cell_density = sim_data.constants.cell_density

    query = [
        "listeners__mass__cell_mass AS cell_mass",
        "listeners__mass__dry_mass AS dry_mass",
        "listeners__fba_results__base_reaction_fluxes AS base_reaction_fluxes",
    ]

    raw = pl.DataFrame(
        read_stacked_columns(
            history_sql, query, order_results=True, conn=conn, remove_first=True
        )
    )

    reaction_ids = np.array(
        field_metadata(conn, config_sql, "listeners__fba_results__base_reaction_fluxes")
    )
    toya_reactions = validation_data.reactionFlux.toya2010fluxes["reactionID"]

    reaction_id_to_index = {r: i for i, r in enumerate(reaction_ids)}
    common_reactions = [r for r in toya_reactions if r in reaction_id_to_index]

    flux_matrix = ndlist_to_ndarray(raw["base_reaction_fluxes"])
    sim_reaction_fluxes = CONC_UNITS / TIMESTEP * flux_matrix

    # Reference Toya fluxes (fixed x): use first experiment's masses for adjustment
    first_exp = raw["experiment_id"][0]
    first_mask = raw["experiment_id"] == first_exp
    cell_masses = units.fg * raw.filter(first_mask)["cell_mass"]
    dry_masses = units.fg * raw.filter(first_mask)["dry_mass"]
    toya_fluxes = toya.adjust_toya_data(
        validation_data.reactionFlux.toya2010fluxes["reactionFlux"],
        cell_masses,
        dry_masses,
        cell_density,
    )
    toya_flux_means = toya.process_toya_data(
        common_reactions, toya_reactions, toya_fluxes
    )
    toya_means_num = toya_flux_means.asNumber(FLUX_UNITS)

    # Long data: one row per (row_index, reaction) -> experiment_id, reaction, sim_flux
    n_rows = flux_matrix.shape[0]
    rows_list = []
    for i in range(n_rows):
        exp_id = raw["experiment_id"][i]
        for j, r in enumerate(common_reactions):
            idx = reaction_id_to_index[r]
            val = sim_reaction_fluxes[i, idx].asNumber(FLUX_UNITS)
            rows_list.append(
                {"experiment_id": exp_id, "reaction": r, "sim_flux": float(val)}
            )

    df = pl.DataFrame(rows_list)

    # Optional: add Toya reference per reaction for tooltip/reference line
    toya_ref = pl.DataFrame(
        {
            "reaction": common_reactions,
            "toya_mean": toya_means_num.tolist(),
        }
    )
    df = df.join(toya_ref, on="reaction")

    # Jitter for strip so points don't overlap on one x when overlaid on violin
    np.random.seed(42)
    jitter = pl.Series("x_jitter", np.random.uniform(-0.02, 0.02, len(df)))
    df = df.with_columns(jitter)

    flux_min = float(df["sim_flux"].min())
    flux_max = float(df["sim_flux"].max())
    extent = [flux_min, flux_max]
    flux_unit_str = FLUX_UNITS.strUnit()

    base = alt.Chart(df)

    # Density per (experiment_id, reaction) so each facet cell gets the right distribution
    violin = (
        base.transform_density(
            "sim_flux",
            groupby=["experiment_id", "reaction"],
            as_=["sim_flux", "density"],
            extent=extent,
        )
        .mark_area(orient="horizontal", opacity=0.4)
        .encode(
            x=alt.X("density:Q", stack="center", axis=None),
            y=alt.Y("sim_flux:Q", title=f"Simulated flux {flux_unit_str}"),
            color=alt.Color("experiment_id:N", legend=alt.Legend(title="Experiment")),
        )
        .properties(width=100, height=280)
    )

    strip = (
        base.mark_circle(size=14, opacity=0.6, color="black")
        .encode(
            x=alt.X(
                "x_jitter:Q",
                title=None,
                axis=None,
                scale=alt.Scale(domain=[-0.05, 0.05]),
            ),
            y=alt.Y("sim_flux:Q", title=f"Simulated flux {flux_unit_str}"),
        )
        .properties(width=100, height=280)
    )

    # Layer violin (distribution) + strip (points) per cell; facet by reaction (columns) and experiment (rows)
    figure = (
        (violin + strip)
        .resolve_scale(y="shared", x="independent")
        .properties(
            title="Central Carbon Metabolism Flux by experiment (violin + strip)"
        )
    )

    out_path = os.path.join(
        outdir, "multiexperiment_centralCarbonMetabolismScatter.html"
    )
    figure.save(out_path)
    print(f"Saved multi-experiment central carbon metabolism scatter to: {out_path}")
    return
