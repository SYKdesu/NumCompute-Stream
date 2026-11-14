import numpy as np


def test_stream_stats_match_full_nan_aware_reference():
    from numcompute_stream import StreamingStats
    X = np.array([[1, np.nan], [3, 2], [5, 4], [np.nan, 6]])
    stats = StreamingStats(ddof=1)
    stats.update_stats(X[:2]).update_stats(X[2:])
    result = stats.result()
    np.testing.assert_allclose(result["mean"], np.nanmean(X, axis=0))
    np.testing.assert_allclose(result["var"], np.nanvar(X, axis=0, ddof=1))
    np.testing.assert_allclose(result["quantiles"], np.nanquantile(X, [0.25, 0.5, 0.75], axis=0))
    np.testing.assert_array_equal(result["count"], [3, 3])


def test_stream_stats_use_global_quantiles_not_chunk_medians():
    from numcompute_stream import StreamingStats
    stats = StreamingStats(quantiles=(0.5,))
    stats.update_stats([[0], [100]]).update_stats([[1], [2], [3]])
    assert stats.result()["quantiles"][0, 0] == 2


def test_stream_stats_histogram_edges_and_out_of_range_values():
    from numcompute_stream import StreamingStats
    stats = StreamingStats(bins=[0, 1, 2])
    stats.update_stats([[-1], [0], [0.5], [1], [2], [3], [np.nan]])
    result = stats.result()
    np.testing.assert_array_equal(result["histogram"], [[2, 2]])
    np.testing.assert_array_equal(result["underflow"], [1])
    np.testing.assert_array_equal(result["overflow"], [1])


def test_stream_stats_window_is_measured_in_samples():
    from numcompute_stream import StreamingStats
    stats = StreamingStats(window_size=3)
    stats.update_stats([[0], [10], [20]])
    stats.update_stats([[30], [40]])
    result = stats.result()
    assert result["samples"] == 3
    assert result["total_samples"] == 5
    np.testing.assert_allclose(result["mean"], [30])
    np.testing.assert_allclose(result["quantiles"][1], [30])


def test_stream_stats_all_missing_feature_and_reset():
    from numcompute_stream import StreamingStats
    stats = StreamingStats().update_stats([[np.nan], [np.nan]])
    assert np.isnan(stats.result()["mean"][0])
    assert stats.result()["count"][0] == 0
    stats.reset().update_stats([[4]])
    np.testing.assert_allclose(stats.result()["mean"], [4])
