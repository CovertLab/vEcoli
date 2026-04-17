from typing import Any, TYPE_CHECKING
import numpy as np

from wholecell.utils import units
from ecoli.library.parquet_emitter import (
    field_metadata,
    named_idx,
    read_stacked_columns,
)
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
    # plot_by = params.get("plot_by", "sim_meta")
    REDUXCLASSIC = params.get("is_reduxclassic", True)

    # with open_arbitrary_sim_data(sim_data_dict) as f:
    #     sim_data = pickle.load(f)

    # ---------------------------------------
    # --- Read mass and FBA flux columns ---
    # ---------------------------------------

    fba_reaction_ids = np.array(
        field_metadata(conn, config_sql, "listeners__fba_results__solution_fluxes")
    )
    kinetic_reactions = np.array(
        field_metadata(
            conn, config_sql, "listeners__fba_results__target_kinetic_fluxes"
        )
    )
    kinetic_reactions_idx = list(
        np.where(np.isin(fba_reaction_ids, kinetic_reactions))[0]
    )

    query = [
        named_idx(
            "listeners__fba_results__estimated_fluxes",
            list(kinetic_reactions),
            [kinetic_reactions_idx],
        ),
        "listeners__fba_results__target_kinetic_fluxes AS target_kinetic_fluxes",
        "listeners__enzyme_kinetics__counts_to_molar AS counts_to_molar",
    ]

    raw = pl.DataFrame(
        read_stacked_columns(
            history_sql,
            query,
            order_results=True,
            conn=conn,
            success_sql=success_sql,
            remove_first=REDUXCLASSIC,
        )
    )

    return raw
