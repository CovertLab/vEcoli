"""
Turn a loaded CausalityBundle into plain tensors: node type/class indices,
a (T, N) raw scalar feature matrix (one primary series per node), and
directed edge index / edge-attribute tensors (process-type embedding index,
stoichiometry sign, log-magnitude, and a has-stoichiometry flag).

This is pure graph/array construction -- no model, no training.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .bundle import CausalityBundle


@dataclass
class GraphTensors:
    node_ids: list[str]
    node_types: list[str]
    node_classes: list[str]
    type_vocab: list[str]
    class_vocab: list[str]
    node_type_idx: np.ndarray  # (N,) int
    node_class_idx: np.ndarray  # (N,) int

    # (T, N) raw (untransformed) scalar feature per node, 0 where unavailable
    raw_features: np.ndarray
    feature_mask: np.ndarray  # (N,) bool -- True if node has real dynamics
    feature_names: list[str | None]  # which series-type each node's scalar is
    time: np.ndarray  # (T,)

    edge_proc_vocab: list[str]
    src_idx: np.ndarray  # (E,) int
    dst_idx: np.ndarray  # (E,) int
    edge_proc_idx: np.ndarray  # (E,) int
    edge_sign: np.ndarray  # (E,) float32
    edge_logmag: np.ndarray  # (E,) float32
    edge_has_stoich: np.ndarray  # (E,) float32
    n_dropped_edges: int

    @property
    def n_nodes(self) -> int:
        return len(self.node_ids)

    @property
    def n_edges(self) -> int:
        return int(self.src_idx.shape[0])

    @property
    def n_node_types(self) -> int:
        return len(self.type_vocab)

    @property
    def n_node_classes(self) -> int:
        return len(self.class_vocab)

    @property
    def n_edge_types(self) -> int:
        return len(self.edge_proc_vocab)


def build_graph_tensors(
    bundle: CausalityBundle, primary_series_index: int = 0
) -> GraphTensors:
    """Build graph + feature tensors from a loaded ``CausalityBundle``.

    ``primary_series_index`` picks which series-dict entry to use as each
    node's single scalar feature when a node has more than one (e.g. a Gene
    node has both "transcription probability" and "gene copy number" --
    index 0 takes whichever the causality-network builder listed first,
    matching the real per-node-type reader functions' own convention).
    """
    node_ids = [n["ID"] for n in bundle.nodes]
    node_types = [n["type"] for n in bundle.nodes]
    node_classes = [n["class"] for n in bundle.nodes]
    node_id_to_idx = {nid: i for i, nid in enumerate(node_ids)}

    type_vocab = sorted(set(node_types))
    type_to_idx = {t: i for i, t in enumerate(type_vocab)}
    class_vocab = sorted(set(node_classes))
    class_to_idx = {c: i for i, c in enumerate(class_vocab)}

    node_type_idx = np.array([type_to_idx[t] for t in node_types], dtype=np.int64)
    node_class_idx = np.array([class_to_idx[c] for c in node_classes], dtype=np.int64)

    T = len(bundle.time)
    N = len(node_ids)
    raw_features = np.zeros((T, N), dtype=np.float64)
    feature_mask = np.zeros(N, dtype=bool)
    feature_names: list[str | None] = [None] * N

    for node_id, series in bundle.dynamics.items():
        if node_id == "time" or not series:
            continue
        idx = node_id_to_idx.get(node_id)
        if idx is None:
            continue
        keys = list(series.keys())
        if primary_series_index >= len(keys):
            continue
        key = keys[primary_series_index]
        arr = np.asarray(series[key], dtype=np.float64)
        if arr.shape[0] != T:
            continue
        raw_features[:, idx] = arr
        feature_mask[idx] = True
        feature_names[idx] = key

    edge_proc_vocab = sorted(set(e["process"] for e in bundle.edges))
    proc_to_idx = {p: i for i, p in enumerate(edge_proc_vocab)}

    E = len(bundle.edges)
    src_idx = np.zeros(E, dtype=np.int64)
    dst_idx = np.zeros(E, dtype=np.int64)
    edge_proc_idx = np.zeros(E, dtype=np.int64)
    edge_sign = np.zeros(E, dtype=np.float32)
    edge_logmag = np.zeros(E, dtype=np.float32)
    edge_has_stoich = np.zeros(E, dtype=np.float32)

    dropped = 0
    keep = np.ones(E, dtype=bool)
    for i, e in enumerate(bundle.edges):
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

    return GraphTensors(
        node_ids=node_ids,
        node_types=node_types,
        node_classes=node_classes,
        type_vocab=type_vocab,
        class_vocab=class_vocab,
        node_type_idx=node_type_idx,
        node_class_idx=node_class_idx,
        raw_features=raw_features,
        feature_mask=feature_mask,
        feature_names=feature_names,
        time=np.asarray(bundle.time),
        edge_proc_vocab=edge_proc_vocab,
        src_idx=src_idx,
        dst_idx=dst_idx,
        edge_proc_idx=edge_proc_idx,
        edge_sign=edge_sign,
        edge_logmag=edge_logmag,
        edge_has_stoich=edge_has_stoich,
        n_dropped_edges=dropped,
    )
