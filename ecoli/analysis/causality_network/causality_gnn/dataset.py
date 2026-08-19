"""
Turn a GraphTensors' (T, N) raw feature matrix into standardized,
temporally-split (t, t+1) training/validation arrays.

Transform: signed-log1p (``sign(x) * log1p(|x|)``) applied elementwise --
a single, type-agnostic transform that tames the huge dynamic range across
node types (raw counts up to ~1e5+, signed fluxes, [0,1] probabilities)
without needing per-node-type special-casing.

Standardization: per-node z-score using **training-window statistics only**
(no leakage from the validation tail into the train-time mean/std).

Split: temporal, not random -- the last ``1 - train_frac`` fraction of
timesteps is held out as a "predict the future tail of this trajectory"
validation set, since whole-cell state (e.g. cell mass) drifts
systematically over a generation and a random shuffle would leak future
information into training.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .graph import GraphTensors


@dataclass
class TrainingArrays:
    standardized: np.ndarray  # (T, N) float32
    delta: (
        np.ndarray
    )  # (T-1, N) float32, delta[t] = standardized[t+1] - standardized[t]
    feature_mask: np.ndarray  # (N,) bool
    train_t_idx: np.ndarray  # indices t such that (t, t+1) is a training pair
    val_t_idx: np.ndarray  # indices t such that (t, t+1) is a validation pair
    mean: np.ndarray  # (N,) per-node mean (signed-log1p space), from train window
    std: np.ndarray  # (N,) per-node std, from train window
    T_train: int


def signed_log1p(x: np.ndarray) -> np.ndarray:
    return np.sign(x) * np.log1p(np.abs(x))


def build_training_arrays(
    graph: GraphTensors, train_frac: float = 0.8
) -> TrainingArrays:
    T, N = graph.raw_features.shape
    T_train = max(2, int(round(T * train_frac)))

    transformed = signed_log1p(graph.raw_features)

    mean = transformed[:T_train].mean(axis=0)
    std = transformed[:T_train].std(axis=0)
    std = np.where(std < 1e-6, 1.0, std)
    mean[~graph.feature_mask] = 0.0
    std[~graph.feature_mask] = 1.0

    standardized = (transformed - mean) / std
    standardized[:, ~graph.feature_mask] = 0.0
    standardized = standardized.astype(np.float32)

    delta = (standardized[1:] - standardized[:-1]).astype(np.float32)

    train_t_idx = np.arange(0, T_train - 1)
    val_t_idx = np.arange(T_train, T - 1)

    return TrainingArrays(
        standardized=standardized,
        delta=delta,
        feature_mask=graph.feature_mask,
        train_t_idx=train_t_idx,
        val_t_idx=val_t_idx,
        mean=mean,
        std=std,
        T_train=T_train,
    )


def persistence_baseline(data: TrainingArrays) -> tuple[float, float]:
    """MSE of the trivial "predict no change" (delta=0) baseline."""
    train_mse = float(np.mean(data.delta[data.train_t_idx][:, data.feature_mask] ** 2))
    val_mse = (
        float(np.mean(data.delta[data.val_t_idx][:, data.feature_mask] ** 2))
        if len(data.val_t_idx)
        else float("nan")
    )
    return train_mse, val_mse


def iter_batches(
    t_idx: np.ndarray, batch_size: int, shuffle: bool, rng: np.random.Generator
):
    order = rng.permutation(t_idx) if shuffle else t_idx
    for i in range(0, len(order), batch_size):
        yield order[i : i + batch_size]
