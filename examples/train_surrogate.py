"""Generate synthetic data, train a surrogate, and evaluate separated test regimes."""

import json
from pathlib import Path

import numpy as np

from reactor_lab import SamplingConfig, SolverConfig, TrainingConfig
from reactor_lab.data import generate_dataset, save_dataset
from reactor_lab.evaluation import evaluate_surrogate
from reactor_lab.plotting import (
    plot_surrogate_predictions,
    plot_surrogate_residuals,
    plot_training_history,
)
from reactor_lab.surrogate import load_surrogate, predict, save_surrogate, train_surrogate

output_directory = Path("artifacts/stage2")
sampling_config = SamplingConfig(solver=SolverConfig(num_steps=200))
training_config = TrainingConfig()

dataset = generate_dataset(sampling_config)
save_dataset(dataset, output_directory / "dataset.npz")
print(f"Generated {dataset.features.shape[0]} synthetic simulations")

result = train_surrogate(dataset, training_config)
save_surrogate(result.model, output_directory)
(output_directory / "history.json").write_text(
    json.dumps(result.history, indent=2) + "\n", encoding="utf-8"
)

# Evaluate the reloaded artifact so this path also verifies checkpoint completeness.
model = load_surrogate(output_directory)
metrics = evaluate_surrogate(model, dataset)
(output_directory / "metrics.json").write_text(
    json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8"
)

simulated = {}
predicted = {}
for split_name in ("test", "ood"):
    split = dataset.get_split(split_name)
    simulated[split_name] = split.targets
    predicted[split_name] = np.asarray(predict(model, split.features))

plot_surrogate_predictions(
    simulated, predicted, output_directory / "predictions.png"
)
plot_surrogate_residuals(simulated, predicted, output_directory / "residuals.png")
plot_training_history(result.history, output_directory / "training-history.png")

for split_name, split_metrics in metrics.items():
    regime = "in-distribution" if split_name == "test" else "separated regime"
    print(f"{split_name} ({regime}):")
    for target_name, target_metrics in split_metrics.items():
        print(
            f"  {target_name}: MAE={target_metrics['mae']:.6g}, "
            f"RMSE={target_metrics['rmse']:.6g}, R²={target_metrics['r2']:.4f}"
        )
print(f"Saved synthetic dataset, model, metrics, and diagnostic plots to {output_directory}")
