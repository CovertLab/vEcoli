"""
Write a causality network's nodes/edges/dynamics directly into a
self-contained, training-ready Zarr store -- skipping the JSON-per-node /
zip serialization ``convert_dynamics`` uses entirely.

This is a SIBLING to ``read_dynamics.convert_dynamics``, not a replacement:
the existing seriesOut.zip path (and everything downstream of it -- the
Causality viewer, the Marimo notebooks) is completely untouched. This
module exists because a separate training pipeline (the ``causality_gnn``
package) re-parses seriesOut.zip once per cell to build the same tensors
this module builds directly here, and at the scale of many cells that
JSON-parse round trip is pure overhead for a consumer that was never going
to render an interactive graph in the first place.

Design choice worth being explicit about: each cell gets its OWN
self-contained ``.zarr`` store (its own copy of the small graph/topology
arrays, plus its own features/time arrays) rather than many cells
appending into one shared store. This is deliberate: many single-cell
analysis jobs finishing around the same time is the realistic execution
pattern here (this runs as part of each cell's own analysis step), and a
self-contained per-cell store needs zero coordination between writers --
each one only ever touches its own directory. The tradeoff is a small
amount of duplicated topology data per cell (the graph arrays are a small
fraction of one cell's total store size); pooling many of these
self-contained per-cell stores into one canonical multi-cell training
store is a separate, later, much cheaper (array-copy, no JSON) step --
see the ``causality_gnn`` package's ``manifest.py``/``materialize.py``.

Tensor construction below intentionally DUPLICATES (rather than imports)
``causality_gnn/graph.py``'s ``build_graph_tensors`` logic, applied to the
live ``node_list``/``edge_list``/``Node`` objects available at generation
time instead of to JSON parsed back out of a zip. This keeps vEcoli's own
environment free of any dependency on the separate training package (the
same zero-cross-dependency principle already applied in the other
direction: the training package has zero dependency on vEcoli). The two
implementations must be kept in sync if the node/edge schema changes;
that's a real, accepted maintenance cost of the decoupling, not an
oversight.
"""

from __future__ import annotations

import os
from typing import Any, Optional

import numpy as np
import zarr

from ecoli.analysis.causality_network.read_dynamics import (
    build_node_dynamics,
    build_timeseries_and_indexes,
)

# One chunk per timestep row, same as causality_gnn/materialize.py -- this is
# what makes single-random-timestep reads during training touch exactly one
# chunk. Only used when shard=False; see write_cell_zarr_store's docstring.
DEFAULT_SHARD_ROWS = 128


def _build_tensors(node_list: list[dict], edge_list: list[dict], sim_data, indexes, volume, timeseries):
    """Mirrors causality_gnn/graph.py's build_graph_tensors -- see this
    module's docstring for why this is a deliberate duplication, not an
    import. Keep in sync with that file if the node/edge dict schema
    changes.
    """
    node_ids = [n["ID"] for n in node_list]
    node_types = [n["type"] for n in node_list]
    node_classes = [n["class"] for n in node_list]
    node_id_to_idx = {nid: i for i, nid in enumerate(node_ids)}

    type_vocab = sorted(set(node_types))
    type_to_idx = {t: i for i, t in enumerate(type_vocab)}
    class_vocab = sorted(set(node_classes))
    class_to_idx = {c: i for i, c in enumerate(class_vocab)}

    node_type_idx = np.array([type_to_idx[t] for t in node_types], dtype=np.int16)
    node_class_idx = np.array([class_to_idx[c] for c in node_classes], dtype=np.int8)

    time = np.asarray(timeseries["time"])
    T = len(time)
    N = len(node_ids)
    raw_features = np.zeros((T, N), dtype=np.float32)
    feature_mask = np.zeros(N, dtype=bool)
    feature_names: list[Optional[str]] = [None] * N

    for node_dict in node_list:
        node_id = node_dict["ID"]
        idx = node_id_to_idx[node_id]
        node = build_node_dynamics(node_dict, sim_data, indexes, volume, timeseries)
        if not node.dynamics:
            continue
        # "Primary" series = whichever key the node's reader function set
        # first (Python dicts preserve insertion order) -- same convention
        # causality_gnn/graph.py uses via primary_series_index=0.
        key = next(iter(node.dynamics))
        arr = np.asarray(node.dynamics[key], dtype=np.float64)
        if arr.shape[0] != T:
            continue
        raw_features[:, idx] = arr.astype(np.float32)
        feature_mask[idx] = True
        feature_names[idx] = key

    edge_proc_vocab = sorted(set(e["process"] for e in edge_list))
    proc_to_idx = {p: i for i, p in enumerate(edge_proc_vocab)}

    E = len(edge_list)
    src_idx = np.zeros(E, dtype=np.int32)
    dst_idx = np.zeros(E, dtype=np.int32)
    edge_proc_idx = np.zeros(E, dtype=np.int8)
    edge_sign = np.zeros(E, dtype=np.float32)
    edge_logmag = np.zeros(E, dtype=np.float32)
    edge_has_stoich = np.zeros(E, dtype=np.float32)

    dropped = 0
    keep = np.ones(E, dtype=bool)
    for i, e in enumerate(edge_list):
        s = node_id_to_idx.get(e["src_node_id"])
        d = node_id_to_idx.get(e["dst_node_id"])
        if s is None or d is None:
            dropped += 1
            keep[i] = False
            continue
        src_idx[i] = s
        dst_idx[i] = d
        edge_proc_idx[i] = proc_to_idx[e["process"]]
        st = e["stoichiometry"]
        if isinstance(st, str):
            pass  # catalyst / regulatory edge: sign=0, logmag=0, has_stoich=0
        else:
            st = float(st)
            edge_sign[i] = float(np.sign(st))
            edge_logmag[i] = float(np.log1p(abs(st)))
            edge_has_stoich[i] = 1.0

    if dropped:
        src_idx = src_idx[keep]
        dst_idx = dst_idx[keep]
        edge_proc_idx = edge_proc_idx[keep]
        edge_sign = edge_sign[keep]
        edge_logmag = edge_logmag[keep]
        edge_has_stoich = edge_has_stoich[keep]

    return dict(
        node_ids=node_ids, node_types=node_types, node_classes=node_classes,
        type_vocab=type_vocab, class_vocab=class_vocab,
        node_type_idx=node_type_idx, node_class_idx=node_class_idx,
        raw_features=raw_features, feature_mask=feature_mask,
        feature_names=feature_names, time=time,
        edge_proc_vocab=edge_proc_vocab, src_idx=src_idx, dst_idx=dst_idx,
        edge_proc_idx=edge_proc_idx, edge_sign=edge_sign, edge_logmag=edge_logmag,
        edge_has_stoich=edge_has_stoich, n_dropped_edges=dropped,
    )


def _write_graph_group(store: zarr.Group, t: dict) -> None:
    g = store.create_group("graph")
    g.create_array("node_type_idx", shape=t["node_type_idx"].shape, dtype="int16")[:] = t["node_type_idx"]
    g.create_array("node_class_idx", shape=t["node_class_idx"].shape, dtype="int8")[:] = t["node_class_idx"]
    g.create_array("feature_mask", shape=t["feature_mask"].shape, dtype="bool")[:] = t["feature_mask"]
    g.create_array("src_idx", shape=t["src_idx"].shape, dtype="int32")[:] = t["src_idx"]
    g.create_array("dst_idx", shape=t["dst_idx"].shape, dtype="int32")[:] = t["dst_idx"]
    g.create_array("edge_proc_idx", shape=t["edge_proc_idx"].shape, dtype="int8")[:] = t["edge_proc_idx"]
    g.create_array("edge_sign", shape=t["edge_sign"].shape, dtype="float32")[:] = t["edge_sign"]
    g.create_array("edge_logmag", shape=t["edge_logmag"].shape, dtype="float32")[:] = t["edge_logmag"]
    g.create_array("edge_has_stoich", shape=t["edge_has_stoich"].shape, dtype="float32")[:] = t["edge_has_stoich"]
    store.attrs.update({
        "node_ids": t["node_ids"],
        "node_types": t["node_types"],
        "node_classes": t["node_classes"],
        "type_vocab": t["type_vocab"],
        "class_vocab": t["class_vocab"],
        "edge_proc_vocab": t["edge_proc_vocab"],
        "feature_names": t["feature_names"],
        "n_nodes": len(t["node_ids"]),
        "n_edges": int(t["src_idx"].shape[0]),
    })


def write_cell_zarr_store(
    store_path: str,
    cell_key: str,
    node_list: list[dict],
    edge_list: list[dict],
    sim_data,
    conn,
    history_sql: str,
    config_sql: str,
    cell_meta: Optional[dict[str, Any]] = None,
    shard: bool = False,
    shard_rows: int = DEFAULT_SHARD_ROWS,
) -> None:
    """Build this cell's graph + dynamics tensors and write them into a new,
    self-contained Zarr store at ``store_path`` (created fresh -- this does
    NOT append to an existing multi-cell store; see this module's docstring
    for why each cell gets its own store).

    ``shard``: when False (the default), the per-timestep arrays are
    chunked one row per chunk -- one physical file per timestep, which
    maximizes single-random-row read speed but means a T-timestep cell
    produces roughly T separate small files on disk. At the scale of many
    thousands of cells that can add up to a very large number of small
    files (a real filesystem/backup concern on some storage backends, not
    a training-speed one -- random-row reads stay fast either way). When
    True, ``shard_rows`` consecutive chunks are bundled into one physical
    file (Zarr's sharding codec) -- far fewer files on disk, while a
    reader can still fetch a single row without reading the whole shard.
    Left off by default because it hasn't been the bottleneck for any
    workload actually run against this yet; flip it on if total file count
    becomes a real problem at your actual cell count.
    """
    timeseries, indexes, volume = build_timeseries_and_indexes(
        sim_data, conn, history_sql, config_sql
    )
    t = _build_tensors(node_list, edge_list, sim_data, indexes, volume, timeseries)

    if os.path.exists(store_path):
        raise FileExistsError(
            f"{store_path} already exists -- write_cell_zarr_store always creates a "
            f"fresh, self-contained per-cell store and never appends to an existing one."
        )
    store = zarr.open_group(store_path, mode="a")
    _write_graph_group(store, t)

    T, N = t["raw_features"].shape
    chunks = (1, N)
    shards = (min(shard_rows, T), N) if shard else None

    cells_group = store.create_group("cells")
    cell_group = cells_group.create_group(cell_key)
    feat_arr = cell_group.create_array(
        "features", shape=(T, N), chunks=chunks, shards=shards, dtype="float32"
    )
    feat_arr[:] = t["raw_features"]
    time_arr = cell_group.create_array("time", shape=(T,), chunks=(T,), dtype="float64")
    time_arr[:] = t["time"]

    # Per-cell metadata lives on the CELL's OWN group attrs, not a shared
    # mutable registry dict at the store's top level -- this store only
    # ever holds one cell so it wouldn't matter here, but keeping the same
    # convention as the (multi-writer) pooled store means a later
    # zarr-to-zarr ingestion step can treat both the same way: discover
    # cells by listing `cells/`'s children and reading each one's own
    # attrs, never by trusting one global list something else might have
    # raced on.
    cell_group.attrs.update({"T": int(T), **(cell_meta or {})})

    print(
        f"Wrote training cache for {cell_key!r} -> {store_path} "
        f"(T={T}, N={N}, E={t['src_idx'].shape[0]}, {t['n_dropped_edges']} edges dropped, "
        f"sharded={shard})"
    )
