from collections import deque

import numpy as np

from numcompute.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)


def _as_labels(values):
    values = np.asarray(values)
    if values.ndim != 1 or len(values) == 0:
        raise ValueError("Labels must be non-empty 1D arrays.")
    if values.dtype.kind == "O":
        if not all(isinstance(value, str) for value in values):
            raise ValueError("Labels must be finite numbers or strings.")
        values = values.astype(str)
    if values.dtype.kind in "biuf":
        if not np.all(np.isfinite(values)):
            raise ValueError("Labels must be finite.")
    elif values.dtype.kind not in "US":
        raise ValueError("Labels must be finite numbers or strings.")
    return values


def _label_kind(values):
    return "numeric" if values.dtype.kind in "biuf" else "string"


def _window_size(value):
    if value is not None and (type(value) is not int or value < 1):
        raise ValueError("window_size must be None or a positive integer.")
    return value


def _trim_chunks(chunks, count, limit):
    while limit is not None and count > limit:
        excess = count - limit
        first = chunks[0]
        if len(first[0]) <= excess:
            count -= len(first[0])
            chunks.popleft()
        else:
            chunks[0] = tuple(
                value[excess:].copy() if value is not None else None
                for value in first
            )
            count -= excess
    return count


class ROCAUC:

    def __init__(self, pos_label=1, window_size=None):
        self.pos_label = pos_label
        self.window_size = _window_size(window_size)
        self.reset()

    def reset(self):
        self._chunks = deque()
        self.n_samples_ = 0
        self._kind = None
        return self

    def update(self, y_true, y_score):
        true = _as_labels(y_true)
        scores = np.asarray(y_score, dtype=float)
        if scores.shape != true.shape or not np.all(np.isfinite(scores)):
            raise ValueError("Scores must be finite and match the true labels.")
        kind = _label_kind(true)
        if self._kind is not None and kind != self._kind:
            raise ValueError("Label types must stay consistent across updates.")
        self._kind = kind
        self._chunks.append((true.copy(), scores.copy()))
        self.n_samples_ += len(true)
        self.n_samples_ = _trim_chunks(self._chunks, self.n_samples_, self.window_size)
        return self

    def result(self):
        if not self._chunks:
            return np.nan
        true = np.concatenate([chunk[0] for chunk in self._chunks])
        scores = np.concatenate([chunk[1] for chunk in self._chunks])
        if len(np.unique(true)) != 2:
            return np.nan
        positive = true == self.pos_label
        n_positive = int(positive.sum())
        n_negative = len(true) - n_positive
        if n_positive == 0 or n_negative == 0:
            return np.nan
        order = np.argsort(scores, kind="stable")
        sorted_scores = scores[order]
        starts = np.r_[0, np.flatnonzero(sorted_scores[1:] != sorted_scores[:-1]) + 1]
        ends = np.r_[starts[1:], len(scores)]
        average_ranks = (starts + ends + 1) / 2
        ranks = np.repeat(average_ranks, ends - starts)
        rank_sum = ranks[positive[order]].sum()
        return float(
            (rank_sum - n_positive * (n_positive + 1) / 2)
            / (n_positive * n_negative)
        )


class ClassificationMetrics:

    def __init__(self, average="macro", pos_label=1, zero_division=0, window_size=None):
        if average not in ("binary", "macro", "micro", "weighted", None):
            raise ValueError("Unsupported metric averaging method.")
        if zero_division not in (0, 1):
            raise ValueError("zero_division must be 0 or 1.")
        self.average = average
        self.pos_label = pos_label
        self.zero_division = zero_division
        self.window_size = _window_size(window_size)
        self.reset()

    def reset(self):
        self.labels_ = None
        self.confusion_matrix_ = np.zeros((0, 0), dtype=np.int64)
        self.n_samples_ = 0
        self.total_samples_ = 0
        self._kind = None
        self._chunks = deque()
        self._auc = ROCAUC(pos_label=self.pos_label)
        return self

    def _accumulate(self, true, pred, scores):
        observed = np.unique(np.concatenate((true, pred)))
        labels = observed if self.labels_ is None else np.union1d(self.labels_, observed)
        matrix = np.zeros((len(labels), len(labels)), dtype=np.int64)
        if self.labels_ is not None:
            old_indices = np.searchsorted(labels, self.labels_)
            matrix[np.ix_(old_indices, old_indices)] = self.confusion_matrix_
        indices = np.searchsorted(labels, true) * len(labels) + np.searchsorted(labels, pred)
        matrix += np.bincount(indices, minlength=len(labels) ** 2).reshape(len(labels), -1)
        self.labels_ = labels
        self.confusion_matrix_ = matrix
        self.n_samples_ += len(true)
        if scores is not None:
            self._auc.update(true, scores)

    def update(self, y_true, y_pred, y_score=None):
        true = _as_labels(y_true)
        pred = _as_labels(y_pred)
        if true.shape != pred.shape:
            raise ValueError("True and predicted labels must have the same shape.")
        kind = _label_kind(true)
        if _label_kind(pred) != kind or self._kind is not None and self._kind != kind:
            raise ValueError("Label types must stay consistent.")
        scores = None if y_score is None else np.asarray(y_score, dtype=float)
        if scores is not None:
            if scores.shape != true.shape or not np.all(np.isfinite(scores)):
                raise ValueError("Scores must be finite and match the labels.")
        observed = np.unique(np.concatenate((true, pred)))
        all_labels = observed if self.labels_ is None else np.union1d(self.labels_, observed)
        if self.average == "binary":
            if len(all_labels) > 2:
                raise ValueError("Binary averaging requires at most two classes.")
            if len(all_labels) == 2 and not np.any(all_labels == self.pos_label):
                raise ValueError("pos_label is not one of the binary classes.")

        self._kind = kind
        if self.window_size is None:
            self._accumulate(true, pred, scores)
        else:
            self._chunks.append((true.copy(), pred.copy(), None if scores is None else scores.copy()))
            _trim_chunks(self._chunks, self.n_samples_ + len(true), self.window_size)
            self.labels_ = None
            self.confusion_matrix_ = np.zeros((0, 0), dtype=np.int64)
            self.n_samples_ = 0
            self._auc.reset()
            for chunk in self._chunks:
                self._accumulate(*chunk)
        self.total_samples_ += len(true)
        return self

    def _divide(self, numerator, denominator):
        numerator, denominator = np.broadcast_arrays(
            np.asarray(numerator, dtype=float), np.asarray(denominator, dtype=float)
        )
        return np.divide(
            numerator,
            denominator,
            out=np.full(numerator.shape, float(self.zero_division)),
            where=denominator != 0,
        )

    def _average(self, values, support, accuracy):
        if self.average is None:
            return values.copy()
        if self.average == "binary":
            indices = np.flatnonzero(self.labels_ == self.pos_label)
            return float(values[indices[0]]) if len(indices) else float(self.zero_division)
        if self.average == "macro":
            return float(np.mean(values))
        if self.average == "weighted":
            return float(np.sum(values * support) / support.sum())
        return accuracy

    def result(self):
        labels = np.asarray([]) if self.labels_ is None else self.labels_.copy()
        result = {
            "samples": self.n_samples_,
            "total_samples": self.total_samples_,
            "accuracy": np.nan,
            "error_rate": np.nan,
            "precision": np.nan,
            "recall": np.nan,
            "f1": np.nan,
            "auc": self._auc.result(),
            "labels": labels,
            "confusion_matrix": self.confusion_matrix_.copy(),
        }
        if self.n_samples_ == 0:
            return result
        matrix = self.confusion_matrix_
        tp = np.diag(matrix).astype(float)
        support = matrix.sum(axis=1).astype(float)
        fp = matrix.sum(axis=0) - tp
        fn = support - tp
        accuracy = float(tp.sum() / self.n_samples_)
        result["accuracy"] = accuracy
        result["error_rate"] = 1.0 - accuracy
        values = {
            "precision": self._divide(tp, tp + fp),
            "recall": self._divide(tp, tp + fn),
            "f1": self._divide(2 * tp, 2 * tp + fp + fn),
        }
        for metric, value in values.items():
            result[metric] = self._average(value, support, accuracy)
        return result


class _Metric:

    name = None

    def __init__(self, window_size=None, **options):
        self.metrics = ClassificationMetrics(window_size=window_size, **options)

    def update(self, y_true, y_pred):
        self.metrics.update(y_true, y_pred)
        return self

    def reset(self):
        self.metrics.reset()
        return self

    def result(self):
        return self.metrics.result()[self.name]


class Accuracy(_Metric):
    name = "accuracy"


class ErrorRate(_Metric):
    name = "error_rate"


class Precision(_Metric):
    name = "precision"


class Recall(_Metric):
    name = "recall"


class F1(_Metric):
    name = "f1"


class ConfusionMatrix(_Metric):
    name = "confusion_matrix"


accuracy = accuracy_score
precision = precision_score
recall = recall_score
f1 = f1_score
