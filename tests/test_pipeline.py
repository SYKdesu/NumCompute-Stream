import numpy as np
import pytest


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
