import argparse
import csv
import json
from pathlib import Path
import platform
import statistics
import sys
from time import perf_counter

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from numcompute_stream import (
    DecisionTreeClassifier,
    MixedPreprocessor,
    Pipeline,
    StreamTrainer,
    iter_chunks,
    load_csv,
)


def generate_data(filepath, n_samples=2400, seed=42):
    if type(n_samples) is not int or n_samples < 20:
        raise ValueError("n_samples must be an integer >= 20.")
    rng = np.random.default_rng(seed)
    numeric = rng.normal(size=(n_samples, 3))
    devices = rng.choice(["desktop", "mobile", "tablet"], size=n_samples, p=[0.4, 0.4, 0.2])
    effects = np.where(devices == "mobile", 0.5, np.where(devices == "tablet", -0.5, 0.0))
    boundary = numeric[:, 0] + 0.65 * numeric[:, 1] + effects
    split = n_samples // 2
    boundary[split:] = -numeric[split:, 0] + 0.5 * numeric[split:, 2] + effects[split:] - 0.2
    labels = (boundary > 0).astype(int)
    labels ^= (rng.random(n_samples) < 0.05).astype(int)
    numeric[rng.random(numeric.shape) < 0.04] = np.nan
    with Path(filepath).open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["feature0", "feature1", "feature2", "device", "target"])
        for row, device, label in zip(numeric, devices, labels):
            values = ["" if np.isnan(value) else format(value, ".10g") for value in row]
            writer.writerow(values + [device, int(label)])
    return Path(filepath)


def loop_split(X, encoded_y):
    n_samples, n_features = X.shape
    n_classes = int(np.max(encoded_y)) + 1

    def impurity(counts):
        total = sum(counts)
        return 1.0 - sum((count / total) ** 2 for count in counts)

    parent = [0] * n_classes
    for label in encoded_y:
        parent[int(label)] += 1
    parent_impurity = impurity(parent)
    best_gain = 1e-12
    best_split = None
    for feature in range(n_features):
        for threshold in sorted(set(X[:, feature].tolist()))[:-1]:
            left = [0] * n_classes
            right = [0] * n_classes
            for row, label in zip(X, encoded_y):
                counts = left if row[feature] <= threshold else right
                counts[int(label)] += 1
            weighted = (
                sum(left) * impurity(left) + sum(right) * impurity(right)
            ) / n_samples
            gain = parent_impurity - weighted
            if gain > best_gain + 1e-12:
                best_gain = gain
                best_split = (feature, float(threshold))
    return best_split


def benchmark_vectorisation(repeats=5, n_samples=400):
    rng = np.random.default_rng(42)
    X = rng.normal(size=(n_samples, 3))
    y = (X[:, 0] + 0.5 * X[:, 1] > 0).astype(int)
    model = DecisionTreeClassifier(max_depth=0).fit(X, y)
    encoded_y = np.searchsorted(model.classes_, y)
    expected = loop_split(X, encoded_y)
    actual = model._best_split(X, encoded_y)
    if expected != actual:
        raise AssertionError("Loop and vectorised split searches disagree.")
    loop_times = []
    vectorised_times = []
    for _ in range(repeats):
        started = perf_counter()
        loop_split(X, encoded_y)
        loop_times.append(perf_counter() - started)
        started = perf_counter()
        model._best_split(X, encoded_y)
        vectorised_times.append(perf_counter() - started)
    loop_seconds = statistics.median(loop_times)
    vectorised_seconds = statistics.median(vectorised_times)
    return {
        "operation": "Gini split search",
        "samples": n_samples,
        "features": X.shape[1],
        "repeats": repeats,
        "same_result": True,
        "best_feature": actual[0],
        "best_threshold": actual[1],
        "loop_seconds": loop_seconds,
        "vectorised_seconds": vectorised_seconds,
        "speedup": loop_seconds / vectorised_seconds,
        "loop_trials": loop_times,
        "vectorised_trials": vectorised_times,
    }


def run_stream(X, y, buffer_size, chunk_size=100, common_start=600):
    pipeline = Pipeline([
        ("prepare", MixedPreprocessor(categorical_columns=(3,), scale=True)),
        ("tree", DecisionTreeClassifier(max_depth=5, buffer_size=buffer_size, random_state=42)),
    ])
    trainer = StreamTrainer(pipeline, window_size=500)
    trainer.run(iter_chunks(X, y, chunk_size))
    common_correct = 0.0
    common_samples = 0
    for record in trainer.logs_:
        start = record["samples_seen"] - record["chunk_samples"]
        if start >= common_start and not record["warmup"]:
            common_correct += record["accuracy"] * record["chunk_samples"]
            common_samples += record["chunk_samples"]
    training = [record["training_seconds"] for record in trainer.logs_ if record["retrained"]]
    metrics = trainer.metrics.result()
    return trainer, {
        "buffer_size": buffer_size,
        "accuracy": metrics["accuracy"],
        "common_accuracy": common_correct / common_samples if common_samples else np.nan,
        "common_evaluated_samples": common_samples,
        "evaluated_samples": metrics["samples"],
        "retraining_events": pipeline.n_retrains_,
        "retraining_seconds": sum(training),
        "median_retrain_seconds": statistics.median(training) if training else 0.0,
        "training_seconds": sum(record["training_seconds"] for record in trainer.logs_),
        "prediction_seconds": sum(record["prediction_seconds"] for record in trainer.logs_),
        "peak_state_bytes": max(record["memory_bytes"] for record in trainer.logs_),
        "pending_samples": pipeline.pending_samples_,
    }


def _serializable(value):
    if isinstance(value, np.ndarray):
        return _serializable(value.tolist())
    if isinstance(value, np.generic):
        return _serializable(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if isinstance(value, dict):
        return {key: _serializable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_serializable(item) for item in value]
    return value


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--generate-data", action="store_true")
    parser.add_argument("--rows", type=int, default=2400)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--chunk-size", type=int, default=100)
    parser.add_argument("--buffers", type=int, nargs="+", default=[100, 300, 600])
    parser.add_argument("--data", type=Path, default=ROOT / "demo" / "data" / "stream_data.csv")
    parser.add_argument("--output", type=Path, default=ROOT / "benchmark" / "results_stream.json")
    args = parser.parse_args()
    if args.repeats < 1 or args.chunk_size < 1 or min(args.buffers) < 1:
        raise ValueError("Repeats, chunk size, and buffer sizes must be positive.")
    if not args.data.is_file() or not args.output.is_file():
        raise FileNotFoundError("Create the data and results files before running this script.")
    if args.generate_data:
        generate_data(args.data, args.rows)
    X, y = load_csv(
        args.data, "target", categorical_columns=("device",), target_dtype=int
    )
    common_start = max(
        ((size + args.chunk_size - 1) // args.chunk_size) * args.chunk_size
        for size in args.buffers
    )
    if len(y) <= common_start:
        raise ValueError("The stream must contain samples after the common warm-up.")
    vectorisation = benchmark_vectorisation()
    results = []
    for size in args.buffers:
        trials = []
        retained_logs = None
        for _ in range(args.repeats):
            trainer, summary = run_stream(X, y, size, args.chunk_size, common_start)
            trials.append(summary)
            if retained_logs is None:
                retained_logs = trainer.logs_
        summary = {
            key: statistics.median([trial[key] for trial in trials])
            for key in trials[0]
        }
        summary["buffer_size"] = size
        summary["trials"] = trials
        summary["logs"] = retained_logs
        results.append(summary)
        print(
            f"buffer={size}: common accuracy={summary['common_accuracy']:.4f}, "
            f"retrains={summary['retraining_events']:.0f}, "
            f"train={summary['training_seconds']:.4f}s"
        )
    output = {
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "platform": platform.platform(),
        },
        "configuration": {
            "seed": 42,
            "rows": len(y),
            "chunk_size": args.chunk_size,
            "common_evaluation_start": common_start,
            "repeats": args.repeats,
            "drift_row": len(y) // 2,
        },
        "vectorisation": vectorisation,
        "buffers": results,
    }
    args.output.write_text(
        json.dumps(_serializable(output), indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(f"Split search speedup: {vectorisation['speedup']:.2f}x")
    print(f"Results saved to {args.output}")


if __name__ == "__main__":
    main()
