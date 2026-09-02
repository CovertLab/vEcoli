"""
CLI entry point: train the baseline causality-network GNN directly from a
`seriesOut.zip` bundle (or an unzipped directory with the same layout).

This script has NO dependency on ecoli/wholecell/vEcoli -- it is meant to
run on a separate machine (ideally with a GPU) from wherever the bundle
was generated.

Usage:
    python -m causality_gnn.train --bundle /path/to/seriesOut.zip \\
        --epochs 15 --batch-size 8 --hidden-dim 16 --n-layers 2 \\
        --train-frac 0.8 --out-dir ./run1

Writes to --out-dir:
    history.json         per-epoch train/val MSE + persistence baselines
    model.pt              trained model state_dict
    example_predictions.json   a few true-vs-predicted node trajectories
"""

from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
import torch

from .bundle import CausalityBundle
from .dataset import build_training_arrays, iter_batches, persistence_baseline
from .graph import build_graph_tensors
from .model import BaselineCausalityGNN


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--bundle", required=True, help="Path to seriesOut.zip or unzipped dir"
    )
    p.add_argument("--out-dir", default="./causality_gnn_run")
    p.add_argument("--epochs", type=int, default=15)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--val-batch-size", type=int, default=8)
    p.add_argument("--train-frac", type=float, default=0.8)
    p.add_argument("--hidden-dim", type=int, default=16)
    p.add_argument("--n-layers", type=int, default=2)
    p.add_argument("--lr", type=float, default=2e-3)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cpu")
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Build the graph/dataset/model and run ONE forward pass "
        "(no backward, no optimizer step, no epochs) to sanity-check "
        "shapes, then exit without training.",
    )
    return p.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    device = torch.device(args.device)

    print(f"Loading bundle from {args.bundle} ...")
    bundle = CausalityBundle(args.bundle)
    print("  " + bundle.summary())

    print("Building graph tensors...")
    graph = build_graph_tensors(bundle)
    print(
        f"  N={graph.n_nodes} nodes, E={graph.n_edges} edges "
        f"({graph.n_dropped_edges} dropped for missing endpoints), "
        f"{graph.feature_mask.sum()} nodes with usable dynamics, "
        f"{graph.n_node_types} node types, {graph.n_edge_types} edge/process types"
    )

    print("Building standardized (t, t+1) training arrays...")
    data = build_training_arrays(graph, train_frac=args.train_frac)
    print(
        f"  T={graph.raw_features.shape[0]}, T_train={data.T_train}, "
        f"{len(data.train_t_idx)} train pairs, {len(data.val_t_idx)} val pairs"
    )
    baseline_train, baseline_val = persistence_baseline(data)
    print(
        f"  persistence baseline: train MSE={baseline_train:.4f}  val MSE={baseline_val:.4f}"
    )

    X = torch.tensor(data.standardized, dtype=torch.float32, device=device)
    delta = torch.tensor(data.delta, dtype=torch.float32, device=device)
    feat_mask = torch.tensor(data.feature_mask, dtype=torch.bool, device=device)
    node_type_idx = torch.tensor(graph.node_type_idx, dtype=torch.long, device=device)
    node_class_idx = torch.tensor(graph.node_class_idx, dtype=torch.long, device=device)
    src = torch.tensor(graph.src_idx, dtype=torch.long, device=device)
    dst = torch.tensor(graph.dst_idx, dtype=torch.long, device=device)
    edge_proc_idx = torch.tensor(graph.edge_proc_idx, dtype=torch.long, device=device)
    edge_scalar = torch.tensor(
        np.stack([graph.edge_sign, graph.edge_logmag, graph.edge_has_stoich], axis=1),
        dtype=torch.float32,
        device=device,
    )

    model = BaselineCausalityGNN(
        n_node_types=graph.n_node_types,
        n_node_classes=graph.n_node_classes,
        n_edge_types=graph.n_edge_types,
        hidden_dim=args.hidden_dim,
        n_layers=args.n_layers,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model has {n_params:,} parameters")

    if args.dry_run:
        print("Dry run: single forward pass (no grad, no training)...")
        model.eval()
        with torch.no_grad():
            x0 = X[:1]
            pred = model(
                x0, node_type_idx, node_class_idx, src, dst, edge_proc_idx, edge_scalar
            )
        print(f"  forward pass OK, output shape {tuple(pred.shape)}")
        return

    os.makedirs(args.out_dir, exist_ok=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    history = {"train_loss": [], "val_loss": []}

    t_start = time.time()
    for epoch in range(args.epochs):
        model.train()
        losses = []
        for batch_t in iter_batches(
            data.train_t_idx, args.batch_size, shuffle=True, rng=rng
        ):
            x_t = X[batch_t]
            y_delta = delta[batch_t]
            optimizer.zero_grad()
            pred = model(
                x_t, node_type_idx, node_class_idx, src, dst, edge_proc_idx, edge_scalar
            )
            loss = (pred - y_delta)[:, feat_mask].pow(2).mean()
            loss.backward()
            optimizer.step()
            losses.append(loss.item())

        model.eval()
        val_sq_errs = []
        with torch.no_grad():
            for batch_t in iter_batches(
                data.val_t_idx, args.val_batch_size, shuffle=False, rng=rng
            ):
                x_val = X[batch_t]
                y_val = delta[batch_t]
                pred_val = model(
                    x_val,
                    node_type_idx,
                    node_class_idx,
                    src,
                    dst,
                    edge_proc_idx,
                    edge_scalar,
                )
                val_sq_errs.append((pred_val - y_val)[:, feat_mask].pow(2).flatten())
        val_loss = torch.cat(val_sq_errs).mean().item() if val_sq_errs else float("nan")

        train_loss = float(np.mean(losses))
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        print(
            f"epoch {epoch:3d}  train_mse={train_loss:.4f}  val_mse={val_loss:.4f}  "
            f"({time.time() - t_start:.1f}s elapsed)",
            flush=True,
        )

    with open(os.path.join(args.out_dir, "history.json"), "w") as f:
        json.dump(
            {
                "history": history,
                "persistence_baseline_train": baseline_train,
                "persistence_baseline_val": baseline_val,
                "n_params": n_params,
                "n_nodes": graph.n_nodes,
                "n_edges": graph.n_edges,
                "T_train": data.T_train,
            },
            f,
            indent=2,
        )
    torch.save(model.state_dict(), os.path.join(args.out_dir, "model.pt"))
    print(f"Done. Wrote results to {args.out_dir}/")


if __name__ == "__main__":
    main()
