"""
Helper for reading a Causality ``seriesOut.zip`` bundle from Python.

Consumed by the Marimo notebooks under ``notebooks/marimo/causality/``.
The bundle stays on disk; per-node dynamics files are read lazily and cached.
"""

from __future__ import annotations

import zipfile
from collections import defaultdict
from functools import lru_cache
from typing import Any, Iterable

import numpy as np
import orjson


NODES_FILE = "nodes.json"
EDGES_FILE = "edges.json"
SERIES_INDEX_FILE = "series.json"
TIME_NODE_ID = "time"


class CausalityBundle:
    """Random-access view over a Causality ``seriesOut.zip`` bundle.

    Loads the small manifests (``nodes.json``, ``edges.json``, ``series.json``)
    eagerly and keeps the zip open for lazy per-node dynamics reads.
    """

    def __init__(self, zip_path: str):
        self.zip_path = zip_path
        self._zip = zipfile.ZipFile(zip_path, "r")

        self.nodes: list[dict[str, Any]] = orjson.loads(self._zip.read(NODES_FILE))
        self.edges: list[dict[str, Any]] = orjson.loads(self._zip.read(EDGES_FILE))
        self.series_index: dict[str, list[dict[str, Any]]] = orjson.loads(
            self._zip.read(SERIES_INDEX_FILE)
        )

        self.nodes_by_id: dict[str, dict[str, Any]] = {n["ID"]: n for n in self.nodes}

        self._name_index: dict[str, list[str]] = defaultdict(list)
        for n in self.nodes:
            for key in _searchable_keys(n):
                self._name_index[key.lower()].append(n["ID"])

        self.outgoing: dict[str, list[str]] = defaultdict(list)
        self.incoming: dict[str, list[str]] = defaultdict(list)
        for e in self.edges:
            self.outgoing[e["src_node_id"]].append(e["dst_node_id"])
            self.incoming[e["dst_node_id"]].append(e["src_node_id"])

        self.node_types: list[str] = sorted({n["type"] for n in self.nodes})
        self.node_classes: list[str] = sorted({n["class"] for n in self.nodes})

        time_series = self.get_dynamics(TIME_NODE_ID)
        self.time: np.ndarray = np.asarray(time_series.get("time", []))

    # ------------- lookup helpers -------------

    def get_node(self, node_id: str) -> dict[str, Any] | None:
        return self.nodes_by_id.get(node_id)

    def search(self, query: str, limit: int = 25) -> list[dict[str, Any]]:
        """Case-insensitive substring match on name, ID, and synonyms."""
        if not query:
            return []
        q = query.lower()
        seen: set[str] = set()
        hits: list[dict[str, Any]] = []
        for key, node_ids in self._name_index.items():
            if q not in key:
                continue
            for nid in node_ids:
                if nid in seen:
                    continue
                seen.add(nid)
                hits.append(self.nodes_by_id[nid])
                if len(hits) >= limit:
                    return hits
        return hits

    def get_dynamics(self, node_id: str) -> dict[str, np.ndarray]:
        """Return ``{series_type: values}`` for the node, loading lazily."""
        return self._load_dynamics(node_id)

    @lru_cache(maxsize=256)
    def _load_dynamics(self, node_id: str) -> dict[str, np.ndarray]:
        entries = self.series_index.get(node_id)
        if not entries:
            return {}
        filename = entries[0]["filename"]
        raw = orjson.loads(self._zip.read(f"series/{filename}"))
        return {rec["type"]: np.asarray(rec["dynamics"]) for rec in raw}

    def get_series_meta(self, node_id: str) -> list[dict[str, Any]]:
        """Return the series.json metadata rows for ``node_id`` (units, type)."""
        return self.series_index.get(node_id, [])

    # ------------- graph traversal -------------

    def neighbors(
        self,
        node_id: str,
        direction: str = "both",
        depth: int = 1,
    ) -> set[str]:
        """BFS from ``node_id`` following the requested edge direction(s).

        ``direction`` is one of ``"upstream"``, ``"downstream"``, or ``"both"``.
        Returned set does not include the starting node.
        """
        if direction not in {"upstream", "downstream", "both"}:
            raise ValueError(f"Unknown direction {direction!r}")

        frontier: set[str] = {node_id}
        visited: set[str] = {node_id}
        for _ in range(max(depth, 0)):
            next_frontier: set[str] = set()
            for n in frontier:
                if direction in ("downstream", "both"):
                    next_frontier.update(self.outgoing.get(n, ()))
                if direction in ("upstream", "both"):
                    next_frontier.update(self.incoming.get(n, ()))
            next_frontier.difference_update(visited)
            if not next_frontier:
                break
            visited.update(next_frontier)
            frontier = next_frontier
        visited.discard(node_id)
        return visited

    def upstream(self, node_id: str, depth: int = 1) -> set[str]:
        return self.neighbors(node_id, "upstream", depth)

    def downstream(self, node_id: str, depth: int = 1) -> set[str]:
        return self.neighbors(node_id, "downstream", depth)

    def edges_between(self, node_ids: Iterable[str]) -> list[dict[str, Any]]:
        """Return every edge whose src and dst both lie in ``node_ids``."""
        keep = set(node_ids)
        return [
            e for e in self.edges
            if e["src_node_id"] in keep and e["dst_node_id"] in keep
        ]

    # ------------- context management -------------

    def close(self) -> None:
        self._zip.close()

    def __enter__(self) -> "CausalityBundle":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()


def _searchable_keys(node: dict[str, Any]) -> Iterable[str]:
    """Yield strings we want to make searchable for ``node``."""
    for key in ("ID", "name"):
        val = node.get(key)
        if isinstance(val, str) and val:
            yield val
    for syn in node.get("synonyms", []) or []:
        if isinstance(syn, str) and syn:
            yield syn
