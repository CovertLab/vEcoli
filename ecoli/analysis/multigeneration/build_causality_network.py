"""Multigeneration variant of `build_causality_network`.

Accepts multi-cell input as long as it's a **single lineage** — i.e. one
(experiment, variant, lineage_seed) with single-daughter descendants. Cells
in a single-daughter lineage don't coexist in time, so ordering the parquet
rows by (generation, time) yields a monotone timeseries that convert_dynamics
can consume as-is. The stricter single-cell guard in
`ecoli/analysis/single/build_causality_network.py` still applies to that
version; here we only require the lineage to be unbranched.
"""

import os
from typing import Any

from duckdb import DuckDBPyConnection

from ecoli.analysis.causality_network import read_dynamics
from ecoli.analysis.causality_network.build_network import BuildNetwork
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
    # Guard against branched lineages (multi-daughter / multi-seed) — in
    # those, cells coexist and the flat concatenation is not a valid
    # timeseries. Count unique (variant, lineage_seed) tuples.
    lineage_count = conn.sql(
        f"SELECT COUNT(DISTINCT (variant, lineage_seed)) "
        f"FROM ({config_sql})"
    ).fetchone()[0]
    assert lineage_count == 1, (
        f"build_causality_network (multigeneration) requires a single "
        f"lineage (one variant × lineage_seed). Got {lineage_count} distinct "
        f"lineages — filter the run before invoking."
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
