"""Build a Causality-viewer `seriesOut.zip` from the variant sim_data pickle
and the analysis framework's already-narrowed history parquet.

Wraps the standalone `buildCausalityNetwork.py` CLI in the standard
`plot(params, conn, history_sql, ...)` interface so seriesOut can be
generated as part of a normal analysis run.

Requires single-cell input (one experiment × variant × seed × generation ×
agent). Writes `seriesOut/seriesOut.zip` into `outdir`.
"""

import os
from typing import Any

from duckdb import DuckDBPyConnection

from ecoli.analysis.causality_network import read_dynamics
from ecoli.analysis.causality_network.build_network import BuildNetwork
from ecoli.library.parquet_emitter import num_cells
from wholecell.utils import filepath as fp


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
        "build_causality_network requires single-cell data (one experiment × "
        "variant × seed × generation × agent). Filter the run before invoking."
    )

    exp_id = next(iter(sim_data_paths))
    sim_data_path = next(iter(sim_data_paths[exp_id].values()))

    series_out = os.path.join(outdir, "seriesOut")
    fp.makedirs(series_out)

    check_sanity = bool(params.get("check_sanity", False))
    network = BuildNetwork(sim_data_path, series_out, check_sanity)
    node_list, edge_list = network.build_nodes_and_edges()

    read_dynamics.convert_dynamics(
        series_out,
        network.sim_data,
        node_list,
        edge_list,
        exp_id,
        conn,
        history_sql,
        config_sql,
    )
