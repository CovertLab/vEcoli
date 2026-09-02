"""
Reader for a Causality ``seriesOut.zip`` bundle -- the exact artifact
produced by ``ecoli.analysis.causality_network.build_causality_network``
(and its ``single``/``multigeneration`` analysis-script wrappers), and
consumed by ``ecoli.analysis.causality_network.viewer.CausalityBundle`` /
the Marimo causality notebooks in vEcoli.

This module deliberately duplicates (rather than imports) the small amount
of parsing logic needed, so that this whole package has **zero dependency**
on the ``ecoli`` / ``wholecell`` / ``vEcoli`` packages and can run on any
machine with just numpy + torch + (orjson or the stdlib ``json`` module).

Expected zip layout (unchanged if you'd rather point ``bundle_path`` at an
already-unzipped directory with the same files)::

    nodes.json      # list[dict]: ID, type, class, name, synonyms, ...
    edges.json      # list[dict]: src_node_id, dst_node_id, stoichiometry, process
    series.json     # dict: node_id -> [{"index", "units", "type", "filename"}, ...]
    series/<hash>.json  # list[dict]: {"units", "type", "id", "dynamics": [...]}
"""

from __future__ import annotations

import json
import os
import zipfile
from dataclasses import dataclass, field
from typing import Any

import numpy as np

try:
    import orjson

    def _loads(b: bytes) -> Any:
        return orjson.loads(b)
except ImportError:  # pragma: no cover - orjson is an optional speedup only

    def _loads(b: bytes) -> Any:
        return json.loads(b)


NODES_FILE = "nodes.json"
EDGES_FILE = "edges.json"
SERIES_INDEX_FILE = "series.json"
TIME_NODE_ID = "time"


@dataclass
class CausalityBundle:
    """Eagerly loads nodes/edges/series-index and per-node dynamics.

    Unlike the viewer's lazy, cached reader (built for an interactive UI),
    this loads every node's dynamics up front, since a training pipeline
    needs all of them anyway.
    """

    bundle_path: str
    nodes: list[dict[str, Any]] = field(default_factory=list, init=False)
    edges: list[dict[str, Any]] = field(default_factory=list, init=False)
    # node_id -> {series_type: np.ndarray}
    dynamics: dict[str, dict[str, np.ndarray]] = field(default_factory=dict, init=False)
    # node_id -> {series_type: units string}
    units: dict[str, dict[str, str]] = field(default_factory=dict, init=False)
    time: np.ndarray = field(default=None, init=False)

    def __post_init__(self) -> None:
        if os.path.isdir(self.bundle_path):
            self._load_from_dir(self.bundle_path)
        else:
            self._load_from_zip(self.bundle_path)

        self.nodes_by_id = {n["ID"]: n for n in self.nodes}

        time_series = self.dynamics.get(TIME_NODE_ID, {})
        self.time = np.asarray(time_series.get("time", []))

    # ------------------------------------------------------------------
    def _load_from_zip(self, path: str) -> None:
        with zipfile.ZipFile(path, "r") as zf:
            self.nodes = _loads(zf.read(NODES_FILE))
            self.edges = _loads(zf.read(EDGES_FILE))
            series_index = _loads(zf.read(SERIES_INDEX_FILE))
            for node_id, entries in series_index.items():
                if not entries:
                    continue
                filename = entries[0]["filename"]
                raw = _loads(zf.read(f"series/{filename}"))
                self.dynamics[node_id] = {
                    rec["type"]: np.asarray(rec["dynamics"]) for rec in raw
                }
                self.units[node_id] = {rec["type"]: rec.get("units", "") for rec in raw}

    def _load_from_dir(self, path: str) -> None:
        def read_json(name):
            with open(os.path.join(path, name), "rb") as f:
                return _loads(f.read())

        self.nodes = read_json(NODES_FILE)
        self.edges = read_json(EDGES_FILE)
        series_index = read_json(SERIES_INDEX_FILE)
        for node_id, entries in series_index.items():
            if not entries:
                continue
            filename = entries[0]["filename"]
            with open(os.path.join(path, "series", filename), "rb") as f:
                raw = _loads(f.read())
            self.dynamics[node_id] = {
                rec["type"]: np.asarray(rec["dynamics"]) for rec in raw
            }
            self.units[node_id] = {rec["type"]: rec.get("units", "") for rec in raw}

    # ------------------------------------------------------------------
    def summary(self) -> str:
        n_with_dynamics = sum(1 for k in self.dynamics if k != TIME_NODE_ID)
        return (
            f"CausalityBundle({self.bundle_path!r}): "
            f"{len(self.nodes)} nodes, {len(self.edges)} edges, "
            f"{n_with_dynamics} nodes with dynamics, "
            f"{len(self.time)} timesteps"
        )
