import numpy as np

import numcompute as nc


class FirstColumnThreshold:
    def fit(self, X, y=None):
        self.threshold_ = float(np.mean(np.asarray(X)[:, 0]))
        return self

    def predict(self, X):
        return (np.asarray(X)[:, 0] >= self.threshold_).astype(int)


def test_unified_public_api_exposes_all_workstreams():
    assert nc.io.load_csv is nc.load_csv
    assert nc.sort_search.topk is nc.topk
    assert nc.metrics.accuracy_score is nc.accuracy_score
    assert nc.pipeline.FeatureUnion is nc.FeatureUnion


def test_csv_reader_streams_bounded_chunks(tmp_path):
    path = tmp_path / "values.csv"
    path.write_text("a,b\n1,2\n3,4\n5,6\n", encoding="utf-8")
    reader = nc.CSVReader(path, skip_header=1)
    chunks = list(reader.iter_chunks(2))
    assert [chunk.shape for chunk in chunks] == [(2, 2), (1, 2)]
    np.testing.assert_allclose(np.vstack(chunks), [[1, 2], [3, 4], [5, 6]])


def test_feature_union_combines_hongyu_liu_transformers():
    X = np.array([[1.0, 10.0], [2.0, 20.0], [3.0, 30.0]])
    union = nc.make_union(nc.StandardScaler(), nc.MinMaxScaler())
    transformed = union.fit_transform(X)
    assert transformed.shape == (3, 4)
    np.testing.assert_allclose(transformed[:, 2:], [[0, 0], [0.5, 0.5], [1, 1]])


def test_pipeline_connects_imputation_scaling_and_estimation():
    X = np.array([[0.0, 1.0], [np.nan, 2.0], [4.0, 3.0]])
    model = nc.make_pipeline(
        nc.Imputer(fill_value=2.0),
        nc.StandardScaler(),
        FirstColumnThreshold(),
    )
    model.fit(X, np.array([0, 1, 1]))
    np.testing.assert_array_equal(model.predict(X), [0, 1, 1])


def test_binary_roc_auc_for_perfect_ranking():
    y_true = np.array([0, 0, 1, 1])
    y_score = np.array([0.1, 0.2, 0.8, 0.9])
    fpr, tpr, thresholds = nc.roc_curve(y_true, y_score)
    assert thresholds[0] == np.inf
    assert fpr[0] == tpr[0] == 0.0
    assert fpr[-1] == tpr[-1] == 1.0
    assert nc.roc_auc_score(y_true, y_score) == 1.0


def test_activation_and_topk_helpers():
    np.testing.assert_allclose(nc.sigmoid([-1000.0, 0.0, 1000.0]), [0.0, 0.5, 1.0])
    np.testing.assert_array_equal(nc.relu([-2, 0, 3]), [0, 0, 3])
    np.testing.assert_array_equal(nc.topk_indices([2, 9, 4, 7], 2), [1, 3])


def test_stream_trainer_evaluates_before_learning_current_labels():
    from numcompute_stream import DecisionTreeClassifier, StreamTrainer
    trainer = StreamTrainer(DecisionTreeClassifier(buffer_size=2))
    first = trainer.process_chunk([[0], [1]], [0, 0])
    second = trainer.process_chunk([[2], [3]], [1, 1])
    third = trainer.process_chunk([[4]], [1])
    assert first["warmup"]
    assert second["accuracy"] == 0.0
    assert second["retrained"]
    assert third["accuracy"] == 1.0
    assert third["evaluated_samples"] == 3
    np.testing.assert_allclose(third["cumulative_accuracy"], 1 / 3)


def test_stream_trainer_warmup_is_excluded_from_denominator():
    from numcompute_stream import DecisionTreeClassifier, StreamTrainer
    trainer = StreamTrainer(DecisionTreeClassifier(buffer_size=3))
    for value in range(5):
        trainer.process_chunk([[value]], [0])
    assert trainer.samples_seen_ == 5
    assert trainer.metrics.result()["samples"] == 2
    assert [row["warmup"] for row in trainer.logs_] == [True, True, True, False, False]
    assert trainer.logs_[-1]["cumulative_accuracy"] == 1
    assert trainer.logs_[-1]["memory_bytes"] > 0


def test_stream_trainer_accounts_for_unequal_evaluation_chunks():
    from numcompute_stream import DecisionTreeClassifier, StreamTrainer
    trainer = StreamTrainer(DecisionTreeClassifier(buffer_size=2))
    trainer.process_chunk([[0], [1]], [0, 0])
    trainer.process_chunk([[2]], [1])
    trainer.process_chunk([[3], [4], [5]], [1, 1, 1])
    final = trainer.process_chunk([[6]], [1])
    assert final["evaluated_samples"] == 5
    assert final["cumulative_accuracy"] == 0.2


def test_stream_end_to_end_mixed_csv_pipeline():
    from io import StringIO
    from numcompute_stream import (
        DecisionTreeClassifier, MixedPreprocessor, Pipeline, StreamTrainer, iter_csv_chunks,
    )
    text = "x,device,target\n0,a,0\n1,b,0\n2,c,1\n3,a,1\n4,b,1\n5,c,1\n"
    chunks = iter_csv_chunks(
        StringIO(text), "target", chunk_size=2, categorical_columns=("device",), target_dtype=int,
    )
    pipeline = Pipeline([
        ("prepare", MixedPreprocessor(categorical_columns=(1,))),
        ("tree", DecisionTreeClassifier(buffer_size=2)),
    ])
    trainer = StreamTrainer(pipeline)
    logs = trainer.run(chunks)
    assert len(logs) == 3
    assert logs[-1]["evaluated_samples"] == 4
    assert pipeline.n_retrains_ == 3
    assert trainer.last_y_pred_.shape == trainer.last_y_true_.shape == (2,)


def test_stream_visualisation_functions_return_axes():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from numcompute_stream import (
        plot_metric_over_time, compare_models, plot_predictions_vs_ground_truth,
    )
    first = plot_metric_over_time([np.nan, 0.5, 0.75], "Accuracy", "Score")
    second = compare_models([np.nan, 0.5], [np.nan, 0.75], labels=("small", "large"))
    third = plot_predictions_vs_ground_truth(["a", "b"], ["a", "a"])
    assert first.get_ylabel() == "Score"
    assert len(second.lines) == 2
    assert len(third.collections) == 2
    for axis in (first, second, third):
        plt.close(axis.figure)
