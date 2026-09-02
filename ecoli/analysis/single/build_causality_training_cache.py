"""Build a self-contained, training-ready Zarr cache of one cell's causality
network directly from the variant sim_data pickle and the analysis
framework's already-narrowed history parquet.

Sibling to ``build_causality_network.py`` (same ``plot(params, conn,
history_sql, ...)`` interface, same node/edge/dynamics extraction), not a
replacement -- this writes a ``.zarr`` store instead of ``seriesOut.zip``,
for the ``causality_gnn`` training pipeline to consume directly (via
``manifest.py``/``materialize.py``) without a JSON-parsing round trip. If
you also want the interactive Causality viewer for this cell, run
``build_causality_network`` as well; the two don't conflict or duplicate
each other's output.

Requires single-cell input (one experiment × variant × seed × generation ×
agent), same as ``build_causality_network``. Writes
``causality_training_cache/<cell_key>.zarr`` into ``outdir``, where
``cell_key`` is ``v{variant}_s{lineage_seed}_g{generation}_a{agent_id}`` --
the same convention ``causality_gnn.manifest`` expects when discovering
these caches from a Hive-partitioned output tree.
"""

import os
from typing import Any

from duckdb import DuckDBPyConnection

from ecoli.analysis.causality_network.build_network import BuildNetwork
from ecoli.analysis.causality_network.write_training_cache import write_cell_zarr_store
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
        "build_causality_training_cache requires single-cell data (one experiment × "
        "variant × seed × generation × agent). Filter the run before invoking."
    )

    exp_id = next(iter(sim_data_paths))
    sim_data_path = next(iter(sim_data_paths[exp_id].values()))

    # variant/lineage_seed/generation/agent_id identify this cell uniquely within
    # a multi-generation/multi-lineage experiment -- read off the (single-cell,
    # asserted above) config rows the same way num_cells groups by them.
    cell_id_row = conn.sql(
        f"SELECT DISTINCT variant, lineage_seed, generation, agent_id "
        f"FROM ({config_sql})"
    ).fetchone()
    variant, lineage_seed, generation, agent_id = cell_id_row
    cell_key = f"v{variant}_s{lineage_seed}_g{generation}_a{agent_id}"

    cache_dir = os.path.join(outdir, "causality_training_cache")
    fp.makedirs(cache_dir)
    store_path = os.path.join(cache_dir, f"{cell_key}.zarr")

    check_sanity = bool(params.get("check_sanity", False))
    shard = bool(params.get("shard", False))

    network = BuildNetwork(sim_data_path, cache_dir, check_sanity)
    node_list, edge_list = network.build_nodes_and_edges()

    write_cell_zarr_store(
        store_path,
        cell_key,
        node_list,
        edge_list,
        network.sim_data,
        conn,
        history_sql,
        config_sql,
        cell_meta={
            "variant": str(variant),
            "lineage_seed": str(lineage_seed),
            "generation": str(generation),
            "agent_id": str(agent_id),
        },
        shard=shard,
    )
