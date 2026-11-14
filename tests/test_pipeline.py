import numpy as np
import pytest

from numcompute.pipeline import Pipeline, make_pipeline


class Centerer:
    def __init__(self, offset=0.0):
        self.offset = offset

    def fit(self, X):
        self.center_ = np.mean(X, axis=0) + self.offset
        return self

    def transform(self, X):
        return np.asarray(X) - self.center_


class LinearRegressor:
    def fit(self, X, y):
        design = np.c_[np.ones(len(X)), np.asarray(X)]
        self.coef_ = np.linalg.lstsq(design, y, rcond=None)[0]
        return self

    def predict(self, X):
        design = np.c_[np.ones(len(X)), np.asarray(X)]
        return design @ self.coef_

    def score(self, X, y):
        prediction = self.predict(X)
        return 1.0 - np.sum((y - prediction) ** 2) / np.sum((y - np.mean(y)) ** 2)


class RecorderEstimator:
    def fit(self, X, y=None, *, token=None):
        self.token = token
        return self

    def predict(self, X):
        return np.zeros(len(X))


def regression_data():
    X = np.arange(8.0).reshape(-1, 1)
    y = 2.0 * X[:, 0] + 3.0
    return X, y


def test_pipeline_fits_transforms_and_predicts():
    X, y = regression_data()
    pipeline = Pipeline([("center", Centerer()), ("model", LinearRegressor())])
    np.testing.assert_allclose(pipeline.fit(X, y).predict(X), y, atol=1e-10)


def test_pipeline_passes_y_only_when_transformer_accepts_it():
    X, y = regression_data()
    pipeline = Pipeline([("center", Centerer()), ("model", LinearRegressor())])
    pipeline.fit(X, y)
    assert hasattr(pipeline["center"], "center_")


def test_fit_parameters_are_routed_to_named_step():
    X, y = regression_data()
    pipeline = Pipeline([("center", Centerer()), ("model", RecorderEstimator())])
    pipeline.fit(X, y, model__token="sentinel")
    assert pipeline["model"].token == "sentinel"


def test_predict_before_fit_has_clear_error():
    pipeline = Pipeline([("center", Centerer()), ("model", LinearRegressor())])
    with pytest.raises(RuntimeError, match="not fitted"):
        pipeline.predict([[1.0]])


def test_duplicate_names_are_rejected():
    with pytest.raises(ValueError, match="duplicate"):
        Pipeline([("step", Centerer()), ("step", LinearRegressor())])


def test_intermediate_step_requires_transform():
    with pytest.raises(TypeError, match="transform"):
        Pipeline([("model", LinearRegressor()), ("final", LinearRegressor())])


def test_set_params_updates_nested_plain_python_attributes():
    pipeline = Pipeline([("center", Centerer()), ("model", LinearRegressor())])
    pipeline.set_params(center__offset=2.5)
    assert pipeline["center"].offset == 2.5


def test_make_pipeline_generates_unique_names():
    pipeline = make_pipeline(Centerer(), Centerer(), LinearRegressor())
    assert list(pipeline.named_steps) == ["centerer", "centerer2", "linearregressor"]


def test_score_delegates_to_final_estimator():
    X, y = regression_data()
    pipeline = Pipeline([("center", Centerer()), ("model", LinearRegressor())]).fit(X, y)
    assert pipeline.score(X, y) == pytest.approx(1.0)


def test_stream_pipeline_keeps_active_scaler_and_tree_together():
    from numcompute_stream import Pipeline as StreamingPipeline
    from numcompute_stream import StandardScaler, DecisionTreeClassifier
    pipeline = StreamingPipeline([
        ("scale", StandardScaler()),
        ("tree", DecisionTreeClassifier(buffer_size=2)),
    ])
    pipeline.partial_fit([[0], [2]], [0, 1])
    probes = [[-1], [1], [3]]
    before = pipeline.predict(probes)
    active_before = pipeline.transform(probes)
    pipeline.partial_fit([[100]], [1])
    np.testing.assert_array_equal(pipeline.predict(probes), before)
    np.testing.assert_allclose(pipeline.transform(probes), active_before)
    assert pipeline.pending_samples_ == 1
    assert pipeline["tree"].pending_samples_ == 0
    pipeline.partial_fit([[101]], [1])
    assert pipeline.n_retrains_ == 2
    assert pipeline["tree"].n_samples_fit_ == 2


def test_stream_pipeline_handles_growing_onehot_schema_at_retrain():
    from numcompute_stream import Pipeline as StreamingPipeline
    from numcompute_stream import MixedPreprocessor, DecisionTreeClassifier
    pipeline = StreamingPipeline([
        ("prepare", MixedPreprocessor(categorical_columns=(1,))),
        ("tree", DecisionTreeClassifier(buffer_size=2)),
    ])
    pipeline.partial_fit(np.array([[0, "red"], [2, "blue"]], dtype=object), [0, 1])
    initial_width = pipeline.transform(np.array([[1, "red"]], dtype=object)).shape[1]
    pipeline.partial_fit(np.array([[3, "green"]], dtype=object), [1])
    assert pipeline.transform(np.array([[1, "green"]], dtype=object)).shape[1] == initial_width
    pipeline.partial_fit(np.array([[4, "green"]], dtype=object), [1])
    assert pipeline.transform(np.array([[1, "green"]], dtype=object)).shape[1] == initial_width + 1
    np.testing.assert_array_equal(pipeline.classes_, [0, 1])
    np.testing.assert_array_equal(pipeline.predict(np.array([[3, "green"]], dtype=object)), [1])


def test_stream_pipeline_fit_resets_raw_buffer_and_preprocessors():
    from numcompute_stream import Pipeline as StreamingPipeline
    from numcompute_stream import Imputer, DecisionTreeClassifier
    pipeline = StreamingPipeline([
        ("impute", Imputer()),
        ("tree", DecisionTreeClassifier(buffer_size=3)),
    ])
    pipeline.partial_fit([[0]], [0])
    pipeline.fit([[10], [20]], [1, 1])
    assert pipeline.pending_samples_ == pipeline.n_retrains_ == 0
    np.testing.assert_allclose(pipeline["impute"].statistics_, [15])
    np.testing.assert_array_equal(pipeline.predict([[np.nan]]), [1])


def test_stream_pipeline_invalid_chunk_is_atomic():
    from numcompute_stream import Pipeline as StreamingPipeline
    from numcompute_stream import StandardScaler, DecisionTreeClassifier
    pipeline = StreamingPipeline([
        ("scale", StandardScaler()),
        ("tree", DecisionTreeClassifier(buffer_size=3)),
    ])
    pipeline.partial_fit([[0]], [0])
    before = pipeline["scale"].mean_.copy()
    with pytest.raises(ValueError):
        pipeline.partial_fit([[np.inf]], [1])
    np.testing.assert_array_equal(pipeline["scale"].mean_, before)
    assert pipeline.pending_samples_ == 1


def test_stream_pipeline_predict_before_warmup_raises():
    from numcompute_stream import Pipeline as StreamingPipeline
    from numcompute_stream import DecisionTreeClassifier
    pipeline = StreamingPipeline([("tree", DecisionTreeClassifier(buffer_size=3))])
    pipeline.partial_fit([[0]], [0])
    with pytest.raises(RuntimeError):
        pipeline.predict([[0]])


@pytest.mark.parametrize("value", [np.nan, np.inf])
def test_stream_pipeline_rejects_invalid_data_before_buffering(value):
    from numcompute_stream import Pipeline as StreamingPipeline
    from numcompute_stream import DecisionTreeClassifier
    pipeline = StreamingPipeline([("tree", DecisionTreeClassifier(buffer_size=3))])
    with pytest.raises(ValueError):
        pipeline.partial_fit([[value]], [0])
    assert pipeline.pending_samples_ == 0
    assert pipeline.n_features_in_ is None
