import numpy as np
import pytest

from numcompute.metrics import (
    accuracy,
    confusion_matrix,
    f1,
    mean_squared_error,
    precision,
    recall,
)


def test_accuracy_supports_numeric_and_string_labels():
    assert accuracy([1, 0, 1, 1], [1, 1, 1, 0]) == 0.5
    assert accuracy(["cat", "dog"], ["cat", "cat"]) == 0.5


def test_confusion_matrix_uses_true_rows_and_predicted_columns():
    result = confusion_matrix([0, 0, 1, 1], [0, 1, 0, 1])
    np.testing.assert_array_equal(result, [[1, 1], [1, 1]])


def test_confusion_matrix_honors_explicit_label_order():
    result = confusion_matrix(["a", "b", "a"], ["a", "a", "b"], labels=["b", "a"])
    np.testing.assert_array_equal(result, [[0, 1], [1, 1]])


def test_confusion_matrix_normalizes_rows_without_nan():
    result = confusion_matrix([0, 0, 1], [0, 1, 1], labels=[0, 1, 2], normalize="true")
    np.testing.assert_allclose(result, [[0.5, 0.5, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 0.0]])


def test_binary_precision_recall_and_f1():
    y_true = [1, 1, 1, 0]
    y_pred = [1, 1, 0, 1]
    assert precision(y_true, y_pred) == pytest.approx(2 / 3)
    assert recall(y_true, y_pred) == pytest.approx(2 / 3)
    assert f1(y_true, y_pred) == pytest.approx(2 / 3)


def test_multiclass_macro_and_per_class_values():
    y_true = [0, 1, 2, 2]
    y_pred = [0, 2, 2, 1]
    per_class = precision(y_true, y_pred, average=None)
    np.testing.assert_allclose(per_class, [1.0, 0.0, 0.5])
    assert precision(y_true, y_pred, average="macro") == pytest.approx(0.5)


def test_zero_division_is_configurable():
    assert precision([0, 0], [0, 0], pos_label=1, zero_division=1) == 1.0


def test_mean_squared_error_supports_multidimensional_arrays():
    assert mean_squared_error([[1, 2], [3, 4]], [[1, 1], [5, 4]]) == pytest.approx(1.25)


@pytest.mark.parametrize(
    "function",
    [accuracy, precision, recall, f1],
)
def test_classification_metrics_reject_mismatched_lengths(function):
    with pytest.raises(ValueError, match="same shape"):
        function([0, 1], [0])


def test_mse_rejects_non_finite_inputs():
    with pytest.raises(ValueError, match="finite"):
        mean_squared_error([1.0, np.nan], [1.0, 2.0])


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
