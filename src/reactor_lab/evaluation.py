"""Held-out and separated-regime evaluation for reactor surrogates."""

from __future__ import annotations

import numpy as np

from .data import SyntheticDataset
from .surrogate import Surrogate, predict


def regression_metrics(
    simulated: np.ndarray, predicted: np.ndarray
) -> dict[str, float | None]:
    """Compute MAE, RMSE, and R² for one target vector."""
    simulated = np.asarray(simulated)
    predicted = np.asarray(predicted)
    if simulated.shape != predicted.shape or simulated.ndim != 1:
        raise ValueError("simulated and predicted must be one-dimensional arrays of equal shape")
    residual = predicted - simulated
    mae = float(np.mean(np.abs(residual)))
    rmse = float(np.sqrt(np.mean(np.square(residual))))
    denominator = float(np.sum(np.square(simulated - np.mean(simulated))))
    r2 = None if denominator == 0 else 1.0 - float(np.sum(np.square(residual))) / denominator
    return {"mae": mae, "rmse": rmse, "r2": r2}


def evaluate_surrogate(
    model: Surrogate, dataset: SyntheticDataset
) -> dict[str, dict[str, dict[str, float | None]]]:
    """Evaluate each target independently on in-distribution and OOD test data."""
    results = {}
    for split_name in ("test", "ood"):
        split = dataset.get_split(split_name)
        predicted = np.asarray(predict(model, split.features))
        results[split_name] = {
            target_name: regression_metrics(split.targets[:, index], predicted[:, index])
            for index, target_name in enumerate(dataset.metadata["target_names"])
        }
    return results
