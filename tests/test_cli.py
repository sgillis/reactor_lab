import json

import numpy as np
import pytest

from reactor_lab import load_surrogate, predict
from reactor_lab.cli import main


def test_simulate_command_saves_configuration_summary_and_trajectory(tmp_path) -> None:
    output = tmp_path / "simulation"
    status = main(
        [
            "simulate",
            "--peak-temperature",
            "2100",
            "--residence-time",
            "0.4",
            "--hydrogen-concentration",
            "60",
            "--diameter",
            "0.00004",
            "--initial-oxide-fraction",
            "0.8",
            "--solver-steps",
            "20",
            "--plot",
            "--output-directory",
            str(output),
        ]
    )

    assert status == 0
    assert (output / "trajectories.png").is_file()
    config = json.loads((output / "config.json").read_text(encoding="utf-8"))
    result = json.loads((output / "result.json").read_text(encoding="utf-8"))
    with np.load(output / "trajectory.npz", allow_pickle=False) as trajectory:
        assert trajectory["time"].shape == (21,)
        assert result["final_oxide_fraction"] == pytest.approx(
            float(trajectory["oxide_fraction"][-1])
        )
        assert result["maximum_temperature"] == pytest.approx(
            float(np.max(trajectory["temperature"]))
        )
        assert result["reached_melting"] == bool(trajectory["reached_melting"])
    assert config["reactor"]["peak_temperature"] == 2_100.0
    assert config["solver"]["num_steps"] == 20
    assert result["synthetic"] is True


def test_train_command_saves_reloadable_model_and_evaluation(tmp_path) -> None:
    output = tmp_path / "training"
    status = main(
        [
            "train",
            "--train-samples",
            "8",
            "--epochs",
            "2",
            "--solver-steps",
            "20",
            "--seed",
            "7",
            "--output-directory",
            str(output),
        ]
    )

    assert status == 0
    assert (output / "model.eqx").is_file()
    assert (output / "dataset.npz").is_file()
    assert (output / "predictions.png").is_file()
    run_config = json.loads((output / "config.json").read_text(encoding="utf-8"))
    history = json.loads((output / "history.json").read_text(encoding="utf-8"))
    metrics = json.loads((output / "metrics.json").read_text(encoding="utf-8"))
    assert run_config["training_config"]["epochs"] == 2
    assert len(history["train_loss"]) == len(history["validation_loss"]) == 2
    assert set(metrics) == {"test", "ood"}
    assert metrics["test"]["final_oxide_fraction"]["mae"] >= 0


@pytest.fixture(scope="module")
def demo_output(tmp_path_factory):
    output = tmp_path_factory.mktemp("demo")
    status = main(
        [
            "run-demo",
            "--train-samples",
            "8",
            "--epochs",
            "2",
            "--solver-steps",
            "20",
            "--candidate-count",
            "12",
            "--validation-count",
            "3",
            "--seed",
            "7",
            "--output-directory",
            str(output),
        ]
    )
    assert status == 0
    return output


def test_run_demo_saves_complete_reproducible_workflow(demo_output) -> None:
    expected = {
        "run-config.json",
        "dataset.npz",
        "dataset.json",
        "model.eqx",
        "config.json",
        "history.json",
        "metrics.json",
        "predictions.png",
        "residuals.png",
        "training-history.png",
        "optimization.json",
    }
    assert expected <= {path.name for path in demo_output.iterdir()}
    config = json.loads((demo_output / "run-config.json").read_text(encoding="utf-8"))
    report = json.loads((demo_output / "optimization.json").read_text(encoding="utf-8"))
    assert config["synthetic"] is True
    assert config["sampling_config"]["seed"] == 7
    assert config["training_config"]["seed"] == 7
    assert config["optimization_config"]["seed"] == 7
    assert config["optimization_config"]["candidate_count"] == 12
    assert len(report["validated_candidates"]) <= 3
    if report["recommendation"] is not None:
        assert report["recommendation"]["simulator_feasible"] is True


def test_run_demo_replays_saved_config_reproducibly(demo_output, tmp_path) -> None:
    replay_output = tmp_path / "replayed-demo"
    status = main(
        [
            "run-demo",
            "--config",
            str(demo_output / "run-config.json"),
            "--output-directory",
            str(replay_output),
        ]
    )

    assert status == 0
    for name in ("run-config.json", "config.json", "history.json", "metrics.json"):
        assert json.loads((replay_output / name).read_text(encoding="utf-8")) == json.loads(
            (demo_output / name).read_text(encoding="utf-8")
        )

    with (
        np.load(demo_output / "dataset.npz", allow_pickle=False) as original,
        np.load(replay_output / "dataset.npz", allow_pickle=False) as replayed,
    ):
        for name in ("features", "targets", "split"):
            np.testing.assert_array_equal(replayed[name], original[name])
        features = original["features"].copy()

    np.testing.assert_array_equal(
        predict(load_surrogate(replay_output), features),
        predict(load_surrogate(demo_output), features),
    )

    original_report = json.loads(
        (demo_output / "optimization.json").read_text(encoding="utf-8")
    )
    replayed_report = json.loads(
        (replay_output / "optimization.json").read_text(encoding="utf-8")
    )
    original_report.pop("model_checkpoint")
    replayed_report.pop("model_checkpoint")
    assert replayed_report == original_report


def test_optimize_command_loads_checkpoint_and_reports_validation(
    demo_output, tmp_path
) -> None:
    output = tmp_path / "optimization"
    status = main(
        [
            "optimize",
            "--model-directory",
            str(demo_output),
            "--candidate-count",
            "12",
            "--validation-count",
            "3",
            "--solver-steps",
            "20",
            "--seed",
            "7",
            "--output-directory",
            str(output),
        ]
    )

    assert status == 0
    report = json.loads((output / "optimization.json").read_text(encoding="utf-8"))
    assert report["model_checkpoint"] == str(demo_output)
    assert report["config"]["candidate_count"] == 12
    assert len(report["validated_candidates"]) <= 3


def test_cli_rejects_invalid_workflow_settings(demo_output) -> None:
    with pytest.raises(SystemExit) as error:
        main(
            [
                "optimize",
                "--model-directory",
                str(demo_output),
                "--candidate-count",
                "0",
            ]
        )
    assert error.value.code == 2
