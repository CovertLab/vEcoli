"""
causality_gnn: a standalone (no vEcoli/ecoli/wholecell dependency) pipeline
that trains a graph neural network surrogate directly from the JSON/zip
output of vEcoli's causality-network analysis module -- the same
`seriesOut.zip` bundle consumed by the Causality viewer and the Marimo
causality notebooks.

Only dependencies: numpy, torch, orjson (falls back to the stdlib `json`
module if orjson isn't installed).

Typical usage (see train.py for the CLI):

    from causality_gnn.bundle import CausalityBundle
    from causality_gnn.graph import build_graph_tensors
    from causality_gnn.dataset import build_training_arrays
    from causality_gnn.model import BaselineCausalityGNN

    bundle = CausalityBundle("seriesOut.zip")
    graph = build_graph_tensors(bundle)
    data = build_training_arrays(graph, train_frac=0.8)
    model = BaselineCausalityGNN(graph.n_node_types, graph.n_node_classes, graph.n_edge_types)
"""

__all__ = ["bundle", "graph", "dataset", "model", "train"]
