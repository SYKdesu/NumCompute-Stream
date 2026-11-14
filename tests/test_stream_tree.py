import numpy as np
import pytest

from numcompute_stream import DecisionTreeClassifier


def test_fit_and_predict():
    tree = DecisionTreeClassifier()
    tree.fit([[1], [2], [8], [9]], [0, 0, 1, 1])

    np.testing.assert_array_equal(
        tree.predict([[1.5], [8.5]]),
        [0, 1],
    )


def test_buffer_threshold_and_overshoot():
    tree = DecisionTreeClassifier(buffer_size=5)

    tree.partial_fit([[1], [2]], [0, 0])
    tree.partial_fit([[8], [9]], [1, 1])

    assert tree.pending_samples_ == 4
    assert tree.n_retrains_ == 0

    with pytest.raises(RuntimeError):
        tree.predict([[1]])

    # Six pending samples exceed the threshold of five.
    tree.partial_fit([[10], [11]], [1, 1])

    assert tree.pending_samples_ == 0
    assert tree.n_retrains_ == 1
    assert len(tree._X_chunks) == 0

    np.testing.assert_array_equal(
        tree.predict([[1], [11]]),
        [0, 1],
    )


def test_old_tree_is_used_between_retrains():
    tree = DecisionTreeClassifier(buffer_size=2)
    tree.partial_fit([[0], [1]], [0, 0])

    old_root = tree.root_
    tree.partial_fit([[0]], [1])

    assert tree.root_ is old_root
    assert tree.pending_samples_ == 1
    assert tree.n_retrains_ == 1
    np.testing.assert_array_equal(tree.predict([[0]]), [0])

def test_fit_resets_stream_state():
    tree = DecisionTreeClassifier(buffer_size=3)
    tree.partial_fit([[0], [1], [2]], [0, 0, 1])
    tree.partial_fit([[3]], [1])
    assert tree.n_retrains_ == 1
    assert tree.pending_samples_ == 1
    tree.fit([[10], [11]], [1, 1])
    assert tree.pending_samples_ == 0
    assert tree.n_retrains_ == 0
    assert tree._X_chunks == []
    assert tree._y_chunks == []
    tree.partial_fit([[12]], [1])
    assert tree.pending_samples_ == 1


def test_fixed_seed_and_streaming_parameter_consistency():
    X = [[0, 0, 0], [1, 1, 1], [8, 8, 8], [9, 9, 9]]
    y = [0, 0, 1, 1]
    first = DecisionTreeClassifier(max_features=1, random_state=7).fit(X, y)
    second = DecisionTreeClassifier(max_features=1, random_state=7).fit(X, y)
    assert first.root_ == second.root_
    expected = int(np.random.default_rng(7).choice(3, size=1)[0])
    assert first.root_["feature"] == expected
    stream = DecisionTreeClassifier(buffer_size=4, max_features=1, random_state=7)
    stream.partial_fit(X[:2], y[:2])
    stream.partial_fit(X[2:], y[2:])
    assert stream.root_ == first.root_


def test_refit_resets_the_random_generator():
    X = [[0, 0, 0], [1, 1, 1], [8, 8, 8], [9, 9, 9]]
    y = [0, 0, 1, 1]
    tree = DecisionTreeClassifier(max_features=1, random_state=7).fit(X, y)
    first_root = tree.root_
    tree.fit(X, y)
    assert tree.root_ == first_root


def test_all_features_matches_default():
    X = [[0, 4], [1, 3], [8, 2], [9, 1]]
    y = [0, 0, 1, 1]
    default = DecisionTreeClassifier().fit(X, y)
    explicit = DecisionTreeClassifier(max_features=2).fit(X, y)
    assert explicit.root_ == default.root_


@pytest.mark.parametrize("method", ["fit", "partial_fit"])
def test_max_features_cannot_exceed_input_columns(method):
    tree = DecisionTreeClassifier(max_features=3)
    with pytest.raises(ValueError, match="max_features"):
        getattr(tree, method)([[0, 1], [2, 3]], [0, 1])
    assert tree.pending_samples_ == 0
    assert tree.root_ is None


@pytest.mark.parametrize("criterion", ["gini", "entropy"])
def test_both_criteria_learn_multiclass_string_labels(criterion):
    X = [[0, 0], [0, 1], [0, 3], [0, 4], [0, 6], [0, 7]]
    y = ["a", "a", "b", "b", "c", "c"]
    tree = DecisionTreeClassifier(criterion=criterion).fit(X, y)
    np.testing.assert_array_equal(tree.predict(X), y)
    assert tree.root_["feature"] == 1


def test_gini_matches_hand_calculation():
    assert DecisionTreeClassifier._gini(np.array([0, 0, 1, 1])) == 0.5
    assert DecisionTreeClassifier._gini(np.array([2, 2, 2])) == 0


def test_entropy_matches_hand_calculation():
    tree = DecisionTreeClassifier(criterion="entropy")
    assert float(tree._impurity([2, 2])) == 1.0
    assert float(tree._impurity([0, 4])) == 0.0


def test_leaf_probabilities_match_class_frequencies():
    tree = DecisionTreeClassifier(max_depth=0).fit([[0], [1], [2]], ["a", "a", "b"])
    np.testing.assert_array_equal(tree.classes_, ["a", "b"])
    np.testing.assert_allclose(tree.predict_proba([[100]]), [[2 / 3, 1 / 3]])


def test_probability_rows_sum_to_one():
    tree = DecisionTreeClassifier().fit([[0], [1], [2], [3]], [0, 0, 1, 1])
    probabilities = tree.predict_proba([[-1], [0.5], [2.5], [9]])
    assert probabilities.shape == (4, 2)
    np.testing.assert_allclose(probabilities.sum(axis=1), 1)
    assert np.all((probabilities >= 0) & (probabilities <= 1))


def test_stream_class_vocabulary_survives_missing_classes():
    tree = DecisionTreeClassifier(buffer_size=2)
    tree.partial_fit([[0], [1]], [0, 1])
    tree.partial_fit([[2], [3]], [1, 1])
    np.testing.assert_array_equal(tree.classes_, [0, 1])
    np.testing.assert_allclose(tree.predict_proba([[0]]), [[0, 1]])


def test_new_class_does_not_change_active_tree_until_retrain():
    tree = DecisionTreeClassifier(buffer_size=2)
    tree.partial_fit([[0], [1]], ["old", "old"])
    old_root = tree.root_
    tree.partial_fit([[2]], ["new"])
    assert tree.root_ is old_root
    np.testing.assert_array_equal(tree.classes_, ["old"])
    assert tree.predict_proba([[0]]).shape == (1, 1)
    tree.partial_fit([[3]], ["new"])
    np.testing.assert_array_equal(tree.classes_, ["new", "old"])
    np.testing.assert_array_equal(tree.predict([[0]]), ["new"])
    np.testing.assert_allclose(tree.predict_proba([[0]]), [[1, 0]])


def test_fit_resets_previous_class_vocabulary():
    tree = DecisionTreeClassifier(buffer_size=2)
    tree.partial_fit([[0], [1]], [0, 1])
    tree.fit([[0]], ["fresh"])
    np.testing.assert_array_equal(tree.classes_, ["fresh"])
    assert tree.predict_proba([[0]]).shape == (1, 1)


def test_overshoot_uses_every_buffered_sample():
    tree = DecisionTreeClassifier(buffer_size=5, max_depth=0)
    tree.partial_fit([[0], [1], [2]], [0, 0, 0])
    tree.partial_fit([[3], [4], [5]], [0, 1, 1])
    assert tree.n_samples_fit_ == 6
    assert tree.pending_samples_ == 0
    np.testing.assert_array_equal(tree.predict([[9]]), [0])


def test_large_chunk_causes_one_retrain():
    tree = DecisionTreeClassifier(buffer_size=2)
    tree.partial_fit(np.arange(7).reshape(-1, 1), [0, 0, 0, 0, 1, 1, 1])
    assert tree.n_samples_fit_ == 7
    assert tree.n_retrains_ == 1
    assert tree.pending_samples_ == 0


def test_new_buffer_replaces_old_training_samples():
    tree = DecisionTreeClassifier(buffer_size=2, max_depth=0)
    tree.partial_fit(np.arange(10).reshape(-1, 1), np.zeros(10, dtype=int))
    tree.partial_fit([[0], [1]], [1, 1])
    assert tree.n_samples_fit_ == 2
    assert tree.n_retrains_ == 2
    np.testing.assert_array_equal(tree.predict([[0]]), [1])


def test_pending_chunks_are_independent_copies():
    X = np.array([[0.0], [1.0]])
    y = np.array([0, 0])
    tree = DecisionTreeClassifier(buffer_size=4)
    tree.partial_fit(X, y)
    X[:] = 99
    y[:] = 1
    tree.partial_fit([[8], [9]], [1, 1])
    np.testing.assert_array_equal(tree.predict([[0], [9]]), [0, 1])


def test_fit_trains_without_waiting_for_buffer():
    tree = DecisionTreeClassifier(buffer_size=100).fit([[5]], [7])
    assert tree.n_samples_fit_ == 1
    assert tree.n_retrains_ == 0
    np.testing.assert_array_equal(tree.predict([[-100], [100]]), [7, 7])


def test_prediction_leaves_stream_state_unchanged():
    tree = DecisionTreeClassifier(buffer_size=3).fit([[0], [1]], [0, 1])
    tree.partial_fit([[2]], [1])
    tree.predict([[0], [3]])
    tree.predict_proba([[0], [3]])
    assert tree.pending_samples_ == 1
    assert tree.n_retrains_ == 0


def test_depth_zero_uses_sorted_class_order_for_ties():
    tree = DecisionTreeClassifier(max_depth=0).fit([[0], [1]], ["z", "a"])
    np.testing.assert_array_equal(tree.predict([[0]]), ["a"])


def test_constant_feature_and_pure_labels_stop_growth():
    constant = DecisionTreeClassifier().fit([[1], [1], [1]], [1, 0, 1])
    pure = DecisionTreeClassifier().fit([[0], [1], [2]], [7, 7, 7])
    assert "value" in constant.root_
    assert "value" in pure.root_
    np.testing.assert_array_equal(constant.predict([[1]]), [1])


def test_min_samples_split_stops_growth():
    tree = DecisionTreeClassifier(min_samples_split=5).fit(
        [[1], [2], [8], [9]], [0, 0, 1, 1]
    )
    assert "value" in tree.root_


def test_duplicate_values_stay_together():
    tree = DecisionTreeClassifier().fit([[0], [0], [1], [1]], [0, 0, 1, 1])
    np.testing.assert_array_equal(tree.predict([[0], [1]]), [0, 1])


def test_empty_prediction_preserves_output_shapes():
    tree = DecisionTreeClassifier().fit([[0], [1]], [0, 1])
    X = np.empty((0, 1))
    assert tree.predict(X).shape == (0,)
    assert tree.predict_proba(X).shape == (0, 2)


@pytest.mark.parametrize("method", ["predict", "predict_proba"])
def test_unfitted_prediction_raises(method):
    with pytest.raises(RuntimeError, match="fit"):
        getattr(DecisionTreeClassifier(), method)([[0]])


@pytest.mark.parametrize("method", ["predict", "predict_proba"])
def test_prediction_rejects_changed_feature_count(method):
    tree = DecisionTreeClassifier().fit([[0], [1]], [0, 1])
    with pytest.raises(ValueError, match="features"):
        getattr(tree, method)([[0, 1]])


@pytest.mark.parametrize("method", ["fit", "partial_fit"])
@pytest.mark.parametrize(
    "X,y",
    [
        ([1, 2], [0, 1]),
        ([[1], [2]], [0]),
        ([[1]], [[0]]),
        (np.empty((0, 1)), []),
        (np.empty((1, 0)), [0]),
    ],
)
def test_training_rejects_invalid_shapes(method, X, y):
    with pytest.raises(ValueError):
        getattr(DecisionTreeClassifier(), method)(X, y)


@pytest.mark.parametrize("method", ["fit", "partial_fit"])
@pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf, 1 + 2j])
def test_training_rejects_invalid_features(method, value):
    with pytest.raises(ValueError):
        getattr(DecisionTreeClassifier(), method)([[value]], [0])


@pytest.mark.parametrize("method", ["fit", "partial_fit"])
@pytest.mark.parametrize("label", [np.nan, np.inf, None, 1 + 2j])
def test_training_rejects_missing_or_invalid_labels(method, label):
    with pytest.raises(ValueError):
        getattr(DecisionTreeClassifier(), method)([[0]], [label])


def test_object_string_labels_are_supported():
    tree = DecisionTreeClassifier().fit([[0]], np.array(["yes"], dtype=object))
    np.testing.assert_array_equal(tree.predict([[1]]), ["yes"])


def test_invalid_chunk_does_not_change_pending_state():
    tree = DecisionTreeClassifier(buffer_size=3)
    tree.partial_fit([[0]], [0])
    with pytest.raises(ValueError, match="features"):
        tree.partial_fit([[0, 1]], [1])
    assert tree.pending_samples_ == 1
    assert len(tree._X_chunks) == 1


def test_stream_rejects_switching_label_types():
    tree = DecisionTreeClassifier(buffer_size=3)
    tree.partial_fit([[0]], [0])
    with pytest.raises(ValueError, match="label"):
        tree.partial_fit([[1]], ["one"])
    assert tree.pending_samples_ == 1


def test_failed_replacement_keeps_active_tree_and_buffer(monkeypatch):
    tree = DecisionTreeClassifier(buffer_size=2)
    tree.partial_fit([[0], [1]], [0, 0])
    old_root = tree.root_

    def fail_training(self, X, y, depth):
        raise RuntimeError("simulated training failure")

    monkeypatch.setattr(DecisionTreeClassifier, "_grow", fail_training)
    with pytest.raises(RuntimeError, match="simulated"):
        tree.partial_fit([[2], [3]], [1, 1])
    assert tree.root_ is old_root
    assert tree.pending_samples_ == 2
    assert tree.n_retrains_ == 1
    np.testing.assert_array_equal(tree.predict([[0]]), [0])


def test_failed_fit_preserves_previous_model_and_buffer(monkeypatch):
    tree = DecisionTreeClassifier(buffer_size=3).fit([[0], [1]], [0, 1])
    tree.partial_fit([[2]], [1])
    old_root = tree.root_

    def fail_training(self, X, y, depth):
        raise RuntimeError("simulated training failure")

    monkeypatch.setattr(DecisionTreeClassifier, "_grow", fail_training)
    with pytest.raises(RuntimeError, match="simulated"):
        tree.fit([[4], [5]], ["a", "b"])
    assert tree.root_ is old_root
    assert tree.pending_samples_ == 1
    np.testing.assert_array_equal(tree.classes_, [0, 1])


def test_score_matches_correct_prediction_fraction():
    tree = DecisionTreeClassifier().fit([[0], [1]], [0, 1])
    assert tree.score([[0], [1]], [0, 0]) == 0.5
    with pytest.raises(ValueError):
        tree.score([[0]], [0, 1])


@pytest.mark.parametrize(
    "options",
    [
        {"max_depth": -1},
        {"max_depth": True},
        {"min_samples_split": 1},
        {"buffer_size": 0},
        {"buffer_size": 1.5},
        {"buffer_size": True},
        {"max_features": 0},
        {"max_features": "sqrt"},
        {"random_state": -1},
        {"random_state": True},
        {"criterion": "invalid"},
    ],
)
def test_invalid_parameters_fail_early(options):
    with pytest.raises(ValueError):
        DecisionTreeClassifier(**options)


@pytest.mark.parametrize("criterion", ["gini", "entropy"])
def test_split_matches_independent_brute_force(criterion):
    X = np.array([[4, 9], [1, 6], [7, 3], [2, 7], [8, 1], [5, 2]], float)
    y = np.array([0, 1, 2, 1, 2, 0])

    def impurity(labels):
        probabilities = np.unique(labels, return_counts=True)[1] / len(labels)
        if criterion == "gini":
            return 1 - sum(float(p) ** 2 for p in probabilities)
        return -sum(float(p) * np.log2(float(p)) for p in probabilities)

    candidates = []
    for feature in range(X.shape[1]):
        for threshold in np.unique(X[:, feature])[:-1]:
            left = X[:, feature] <= threshold
            loss = (
                left.sum() * impurity(y[left])
                + (~left).sum() * impurity(y[~left])
            ) / len(y)
            candidates.append((float(loss), feature, threshold))
    best_loss = min(item[0] for item in candidates)
    valid = [item for item in candidates if np.isclose(item[0], best_loss)]
    _, expected_feature, expected_threshold = min(valid, key=lambda item: (item[1], item[2]))
    tree = DecisionTreeClassifier(max_depth=1, criterion=criterion).fit(X, y)
    assert tree.root_["feature"] == expected_feature
    assert tree.root_["threshold"] == expected_threshold
