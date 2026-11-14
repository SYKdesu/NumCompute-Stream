import numpy as np
import pytest


def test_stream_accuracy_weights_samples_instead_of_chunks():
    from numcompute_stream import Accuracy
    metric = Accuracy().update([1, 1, 1, 1], [1, 1, 1, 1])
    metric.update([0], [1])
    assert metric.result() == 0.8
    metric.reset()
    assert np.isnan(metric.result())


def test_stream_metrics_match_batch_precision_recall_and_f1():
    from numcompute_stream import ClassificationMetrics
    from numcompute.metrics import precision_score, recall_score, f1_score
    true = np.array([0, 1, 2, 0, 1, 2])
    pred = np.array([0, 2, 2, 1, 1, 0])
    metrics = ClassificationMetrics(average="macro")
    metrics.update(true[:2], pred[:2]).update(true[2:], pred[2:])
    result = metrics.result()
    assert result["precision"] == pytest.approx(precision_score(true, pred, average="macro"))
    assert result["recall"] == pytest.approx(recall_score(true, pred, average="macro"))
    assert result["f1"] == pytest.approx(f1_score(true, pred, average="macro"))


def test_stream_confusion_matrix_keeps_old_counts_when_labels_expand():
    from numcompute_stream import ClassificationMetrics
    metrics = ClassificationMetrics().update(["b"], ["b"])
    metrics.update(["a"], ["a"])
    result = metrics.result()
    np.testing.assert_array_equal(result["labels"], ["a", "b"])
    np.testing.assert_array_equal(result["confusion_matrix"], np.eye(2, dtype=int))


def test_stream_rolling_metrics_trim_partial_chunks():
    from numcompute_stream import ClassificationMetrics
    metrics = ClassificationMetrics(window_size=3)
    metrics.update([0, 0, 0], [0, 0, 0])
    metrics.update([1, 1], [0, 1])
    result = metrics.result()
    assert result["samples"] == 3
    assert result["total_samples"] == 5
    assert result["accuracy"] == pytest.approx(2 / 3)


def test_stream_auc_accumulates_scores_and_handles_ties():
    from numcompute_stream import ROCAUC
    auc = ROCAUC()
    assert np.isnan(auc.result())
    auc.update([0], [0.1])
    assert np.isnan(auc.result())
    auc.update([0, 1, 1], [0.4, 0.35, 0.8])
    assert auc.result() == pytest.approx(0.75)
    auc.reset().update([0, 1], [0.5, 0.5])
    assert auc.result() == 0.5


def test_stream_auc_rolling_window():
    from numcompute_stream import ROCAUC
    auc = ROCAUC(window_size=3)
    auc.update([0, 1], [0.1, 0.4])
    auc.update([0, 1], [0.35, 0.8])
    assert auc.result() == 1.0
    assert auc.n_samples_ == 3


def test_stream_binary_metrics_zero_division():
    from numcompute_stream import ClassificationMetrics
    metrics = ClassificationMetrics(average="binary", zero_division=0)
    result = metrics.update([0, 0], [0, 0]).result()
    assert result["precision"] == result["recall"] == result["f1"] == 0


def test_stream_metrics_reject_invalid_update_without_losing_counts():
    from numcompute_stream import ClassificationMetrics
    metrics = ClassificationMetrics().update([0], [0])
    with pytest.raises(ValueError):
        metrics.update([0], [np.nan])
    with pytest.raises(ValueError):
        metrics.update([0], [0], [np.inf])
    assert metrics.result()["samples"] == 1
