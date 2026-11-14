import numpy as np


def test_stream_scaler_matches_batch_with_missing_values():
    from numcompute_stream import StandardScaler as StreamingScaler
    X = np.array([[1, np.nan], [3, 2], [5, 4], [np.nan, 6]])
    scaler = StreamingScaler().partial_fit(X[:2]).partial_fit(X[2:])
    np.testing.assert_allclose(scaler.mean_, np.nanmean(X, axis=0))
    np.testing.assert_allclose(scaler.var_, np.nanvar(X, axis=0))
    np.testing.assert_array_equal(scaler.n_samples_seen_, [3, 3])
    np.testing.assert_allclose(scaler.transform(X), StreamingScaler().fit_transform(X), equal_nan=True)


def test_stream_scaler_handles_initially_missing_and_constant_features():
    from numcompute_stream import StandardScaler as StreamingScaler
    scaler = StreamingScaler().partial_fit([[1, np.nan], [1, np.nan]])
    np.testing.assert_array_equal(scaler.scale_, [1, 1])
    scaler.partial_fit([[1, 4], [1, 8]])
    np.testing.assert_allclose(scaler.mean_, [1, 6])
    np.testing.assert_allclose(scaler.var_, [0, 4])


def test_stream_scaler_preserves_small_variance_at_large_offsets():
    from numcompute_stream import StandardScaler as StreamingScaler
    X = 1e12 + np.arange(12, dtype=float).reshape(-1, 1)
    scaler = StreamingScaler().partial_fit(X[:5]).partial_fit(X[5:])
    np.testing.assert_allclose(scaler.var_, np.var(X, axis=0), rtol=1e-5)


def test_stream_scaler_fit_resets_and_transform_is_read_only():
    from numcompute_stream import StandardScaler as StreamingScaler
    scaler = StreamingScaler().partial_fit([[0], [10]])
    scaler.fit([[4], [4]])
    before = scaler.mean_.copy()
    assert scaler.transform(np.empty((0, 1))).shape == (0, 1)
    scaler.transform([[100]])
    np.testing.assert_array_equal(scaler.mean_, before)
    np.testing.assert_array_equal(scaler.n_samples_seen_, [2])


def test_stream_imputer_uses_only_observed_values_and_configured_fallback():
    from numcompute_stream import Imputer
    imputer = Imputer(fill_value=-1)
    imputer.partial_fit([[2, np.nan], [np.nan, np.nan]])
    np.testing.assert_allclose(imputer.transform([[np.nan, np.nan]]), [[2, -1]])
    imputer.partial_fit([[6, 10]])
    np.testing.assert_allclose(imputer.statistics_, [4, 10])
    np.testing.assert_array_equal(imputer._stats.n_samples_seen_, [2, 1])


def test_stream_constant_imputer():
    from numcompute_stream import Imputer
    imputer = Imputer(strategy="constant", fill_value=7).partial_fit([[1, np.nan]])
    np.testing.assert_allclose(imputer.transform([[np.nan, 2]]), [[7, 2]])


def test_stream_minmax_updates_extrema():
    from numcompute_stream import MinMaxScaler as StreamingMinMax
    scaler = StreamingMinMax(feature_range=(-1, 1))
    scaler.partial_fit([[1, np.nan], [3, 5]])
    scaler.partial_fit([[5, 5]])
    np.testing.assert_allclose(scaler.transform([[1, 5], [5, 5]]), [[-1, -1], [1, -1]])


def test_stream_onehot_appends_columns_without_moving_old_columns():
    from numcompute_stream import OneHotEncoder as StreamingEncoder
    encoder = StreamingEncoder().partial_fit([["red", "small"], ["blue", "large"]])
    before = encoder.transform([["red", "small"]])
    old_names = encoder.get_feature_names_out()
    encoder.partial_fit([["green", "medium"]])
    after = encoder.transform([["red", "small"]])
    np.testing.assert_array_equal(after[:, :before.shape[1]], before)
    np.testing.assert_array_equal(encoder.get_feature_names_out()[:len(old_names)], old_names)
    assert after.shape[1] == before.shape[1] + 2


def test_stream_onehot_missing_values_and_unknown_categories():
    from numcompute_stream import OneHotEncoder as StreamingEncoder
    encoder = StreamingEncoder().fit([[None], [np.nan], ["<missing>"]])
    assert encoder.n_output_features_ == 2
    np.testing.assert_array_equal(encoder.transform([[None]]), encoder.transform([[np.nan]]))
    np.testing.assert_array_equal(encoder.transform([["unseen"]]), [[0, 0]])


def test_stream_onehot_unknown_error_and_shape_validation():
    import pytest
    from numcompute_stream import OneHotEncoder as StreamingEncoder
    encoder = StreamingEncoder(handle_unknown="error").fit([["red"]])
    with pytest.raises(ValueError, match="Unknown"):
        encoder.transform([["blue"]])
    with pytest.raises(ValueError, match="features"):
        encoder.partial_fit([["red", "small"]])


def test_mixed_preprocessor_handles_missing_numeric_and_new_categories():
    from numcompute_stream import MixedPreprocessor
    processor = MixedPreprocessor(categorical_columns=(1,))
    processor.partial_fit(np.array([[1, "a"], [np.nan, "b"]], dtype=object))
    initial = processor.transform(np.array([[np.nan, "a"]], dtype=object))
    assert initial.shape == (1, 3)
    assert np.all(np.isfinite(initial))
    processor.partial_fit(np.array([[5, "c"]], dtype=object))
    assert len(processor.get_feature_names_out()) == 4
    assert processor.transform(np.array([[np.nan, "c"]], dtype=object)).shape == (1, 4)
