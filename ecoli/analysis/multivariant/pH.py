import os
import numpy as np
import matplotlib.pyplot as plt

from typing import Any, TYPE_CHECKING
from ecoli.library.parquet_emitter import (
    read_stacked_columns,
    field_metadata,
    named_idx,
)
from vivarium.library.units import units
from scipy.constants import N_A

if TYPE_CHECKING:
    from duckdb import DuckDBPyConnection

N_AVOGADRO = N_A / units.mol
VOLUME_UNITS = units.fL


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
    All options have default values (do not need to be explicitly provided).

    Args:
        params: Dictionary of parameters given under analysis
            name in configuration JSON. Config options look like this:

            .. code-block:: json

                {
                }
    """
    # Retrieve reaction IDs from config metadata
    bulk_ids = field_metadata(conn, config_sql, "bulk")

    proton_idx = np.nonzero(
        np.array(["PROTON[c]", "PROTON[p]", "PROTON[e]"])[:, np.newaxis] == bulk_ids
    )[1]

    df = read_stacked_columns(
        history_sql,
        [
            named_idx(
                "bulk",
                ["PROTON[c]", "PROTON[p]", "PROTON[e]"],
                [list(proton_idx)],
            ),
            "listeners__mass__volume",
        ],
        remove_first=True,
        conn=conn,
    )

    # Get total protons in moles, cell volume in fL
    total_protons = (
        df[["PROTON[c]", "PROTON[p]", "PROTON[e]"]].to_numpy().sum(axis=1) / N_AVOGADRO
    )
    cellVolume = df["listeners__mass__volume"].to_numpy() * VOLUME_UNITS

    # Get pH
    pH = -np.log10((total_protons / cellVolume).to("M").magnitude)

    fig, ax = plt.subplots()

    ax.plot(df["time"], pH)

    # Labels
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("pH")

    fig.savefig(os.path.join(outdir, "pH.png"))
    fig.savefig(os.path.join(outdir, "pH.svg"))
