import csv
from collections import deque
from pathlib import Path
import sys
from time import perf_counter

import numpy as np

from .metrics import ClassificationMetrics, _as_labels


def _memory_bytes(value, seen=None):
    if seen is None:
        seen = set()
    identity = id(value)
    if identity in seen:
        return 0
    seen.add(identity)
    size = sys.getsizeof(value)
    if isinstance(value, np.ndarray):
        return size + (_memory_bytes(value.base, seen) if value.base is not None else 0)
    if isinstance(value, dict):
        return size + sum(
            _memory_bytes(key, seen) + _memory_bytes(item, seen)
            for key, item in value.items()
        )
    if isinstance(value, (list, tuple, set, deque)):
        return size + sum(_memory_bytes(item, seen) for item in value)
    if hasattr(value, "__dict__") and not isinstance(value, type):
        return size + _memory_bytes(vars(value), seen)
    return size


class StreamTrainer:

    def __init__(self, model, metrics=None, window_size=500, stats=None):
        if not callable(getattr(model, "partial_fit", None)):
            raise TypeError("model must implement partial_fit.")
        if not callable(getattr(model, "predict", None)):
            raise TypeError("model must implement predict.")
        self.model = model
        self.metrics = ClassificationMetrics() if metrics is None else metrics
        if not isinstance(self.metrics, ClassificationMetrics):
            raise TypeError("metrics must be a ClassificationMetrics instance.")
        if self.metrics.window_size is not None:
            raise ValueError("Trainer cumulative metrics must not have a rolling window.")
        self.rolling_metrics = ClassificationMetrics(
            average=self.metrics.average,
            pos_label=self.metrics.pos_label,
            zero_division=self.metrics.zero_division,
            window_size=window_size,
        )
        self.stats = stats
        self.logs_ = []
        self.samples_seen_ = 0
        self.last_y_true_ = None
        self.last_y_pred_ = None
        self.last_fit_seconds_ = 0.0

    def _is_fitted(self):
        if hasattr(self.model, "is_fitted_"):
            return bool(self.model.is_fitted_)
        return getattr(self.model, "root_", None) is not None

    def _scores(self, X):
        if not callable(getattr(self.model, "predict_proba", None)):
            return None
        classes = np.asarray(self.model.classes_)
        if len(classes) > 2:
            return None
        probabilities = self.model.predict_proba(X)
        indices = np.flatnonzero(classes == self.metrics.pos_label)
        if len(indices) == 0:
            return np.zeros(len(X))
        return probabilities[:, indices[0]]

    def score_chunk(self, X, y):
        true = _as_labels(y)
        started = perf_counter()
        pred = self.model.predict(X)
        scores = self._scores(X)
        elapsed = perf_counter() - started
        if pred.shape != true.shape:
            raise ValueError("Predictions must match the label shape.")
        self.metrics.update(true, pred, scores)
        self.rolling_metrics.update(true, pred, scores)
        result = self.metrics.result()
        rolling = self.rolling_metrics.result()
        self.last_y_true_ = true.copy()
        self.last_y_pred_ = pred.copy()
        return {
            "accuracy": float(np.mean(true == pred)),
            "cumulative_accuracy": result["accuracy"],
            "rolling_accuracy": rolling["accuracy"],
            "precision": result["precision"],
            "recall": result["recall"],
            "f1": result["f1"],
            "auc": result["auc"],
            "evaluated_samples": result["samples"],
            "prediction_seconds": elapsed,
        }

    def fit_chunk(self, X, y):
        started = perf_counter()
        self.model.partial_fit(X, y)
        self.last_fit_seconds_ = perf_counter() - started
        return self

    def process_chunk(self, X, y):
        X = np.asarray(X)
        y = _as_labels(y)
        if X.ndim != 2 or X.shape[1] == 0 or len(X) != len(y):
            raise ValueError("X and y must have consistent sample and feature shapes.")
        fitted = self._is_fitted()
        previous_retrains = self.model.n_retrains_
        if fitted:
            result = self.score_chunk(X, y)
        else:
            current = self.metrics.result()
            result = {
                "accuracy": np.nan,
                "cumulative_accuracy": current["accuracy"],
                "rolling_accuracy": np.nan,
                "precision": current["precision"],
                "recall": current["recall"],
                "f1": current["f1"],
                "auc": current["auc"],
                "evaluated_samples": current["samples"],
                "prediction_seconds": 0.0,
            }
        if self.stats is not None:
            self.stats.update_stats(X)
        self.fit_chunk(X, y)
        self.samples_seen_ += len(y)
        record = {
            "chunk": len(self.logs_) + 1,
            "chunk_samples": len(y),
            "samples_seen": self.samples_seen_,
            "warmup": not fitted,
            **result,
            "training_seconds": self.last_fit_seconds_,
            "retrained": self.model.n_retrains_ > previous_retrains,
            "n_retrains": self.model.n_retrains_,
            "pending_samples": self.model.pending_samples_,
            "training_samples": self.model.n_samples_fit_,
            "memory_bytes": _memory_bytes((self.model, self.metrics, self.rolling_metrics, self.stats)),
        }
        self.logs_.append(record)
        return record.copy()

    def run(self, chunks):
        for X, y in chunks:
            self.process_chunk(X, y)
        return self.logs_

    def save_logs(self, filepath):
        if not self.logs_:
            raise RuntimeError("There are no logs to save.")
        with Path(filepath).open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(self.logs_[0]))
            writer.writeheader()
            writer.writerows(self.logs_)
        return Path(filepath)
