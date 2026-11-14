from collections import deque
import warnings

import numpy as np

from numcompute.stats import describe, histogram, maximum, mean, median, minimum, quantile, std, var

from .preprocessing import StandardScaler, _numeric_array, _feature_check
from .metrics import _window_size, _trim_chunks


class StreamingStats:

    def __init__(
        self,
        quantiles=(0.25, 0.5, 0.75),
        bins=10,
        histogram_range=(-5.0, 5.0),
        window_size=None,
        ddof=0,
    ):
        self.quantiles = np.asarray(quantiles, dtype=float)
        if (
            self.quantiles.ndim != 1
            or not np.all(np.isfinite(self.quantiles))
            or np.any((self.quantiles < 0) | (self.quantiles > 1))
        ):
            raise ValueError("quantiles must be a 1D array of values in [0, 1].")
        if type(ddof) is not int or ddof < 0:
            raise ValueError("ddof must be a non-negative integer.")
        if type(bins) is int:
            bounds = np.asarray(histogram_range, dtype=float)
            if (
                bins < 1
                or bounds.shape != (2,)
                or not np.all(np.isfinite(bounds))
                or bounds[0] >= bounds[1]
            ):
                raise ValueError("Histogram bins and range must define increasing finite edges.")
            self.bin_edges_ = np.linspace(bounds[0], bounds[1], bins + 1)
        else:
            self.bin_edges_ = np.asarray(bins, dtype=float)
            if (
                self.bin_edges_.ndim != 1
                or len(self.bin_edges_) < 2
                or not np.all(np.isfinite(self.bin_edges_))
                or np.any(np.diff(self.bin_edges_) <= 0)
            ):
                raise ValueError("Histogram edges must be finite and strictly increasing.")
        self.window_size = _window_size(window_size)
        self.ddof = ddof
        self.reset()

    def reset(self):
        self.n_features_in_ = None
        self.n_samples_ = 0
        self.total_samples_ = 0
        self._chunks = deque()
        self._scaler = StandardScaler()
        self._minimum = None
        self._maximum = None
        self.histogram_ = None
        self.underflow_ = None
        self.overflow_ = None
        return self

    def _accumulate(self, X):
        self._scaler.partial_fit(X)
        valid = ~np.isnan(X)
        minimum = np.where(valid, X, np.inf).min(axis=0)
        maximum = np.where(valid, X, -np.inf).max(axis=0)
        if self.histogram_ is None:
            self._minimum = minimum
            self._maximum = maximum
            self.histogram_ = np.zeros((X.shape[1], len(self.bin_edges_) - 1), dtype=np.int64)
            self.underflow_ = np.zeros(X.shape[1], dtype=np.int64)
            self.overflow_ = np.zeros(X.shape[1], dtype=np.int64)
        else:
            self._minimum = np.minimum(self._minimum, minimum)
            self._maximum = np.maximum(self._maximum, maximum)
        self.underflow_ += np.sum(valid & (X < self.bin_edges_[0]), axis=0)
        self.overflow_ += np.sum(valid & (X > self.bin_edges_[-1]), axis=0)
        for feature in range(X.shape[1]):
            self.histogram_[feature] += np.histogram(
                X[valid[:, feature], feature], bins=self.bin_edges_
            )[0]
        self.n_samples_ += len(X)

    def update_stats(self, X_chunk):
        X = _numeric_array(X_chunk)
        _feature_check(X, self.n_features_in_)
        self.n_features_in_ = X.shape[1]
        self._chunks.append((X.copy(),))
        if self.window_size is None:
            self._accumulate(X)
        else:
            _trim_chunks(self._chunks, self.n_samples_ + len(X), self.window_size)
            self.n_samples_ = 0
            self._scaler = StandardScaler()
            self.histogram_ = None
            for chunk in self._chunks:
                self._accumulate(chunk[0])
        self.total_samples_ += len(X)
        return self

    def update(self, X_chunk):
        return self.update_stats(X_chunk)

    def partial_fit(self, X_chunk):
        return self.update_stats(X_chunk)

    def fit(self, X):
        self.reset()
        return self.update_stats(X)

    def _require_fitted(self):
        if self.n_features_in_ is None:
            raise RuntimeError("No statistics are available before update_stats.")

    @property
    def count(self):
        self._require_fitted()
        return self._scaler.n_samples_seen_.copy()

    @property
    def mean(self):
        self._require_fitted()
        return np.where(self.count > 0, self._scaler.mean_, np.nan)

    @property
    def var(self):
        self._require_fitted()
        denominator = self.count - self.ddof
        return np.divide(
            self._scaler._m2,
            denominator,
            out=np.full(self.n_features_in_, np.nan),
            where=denominator > 0,
        )

    @property
    def std(self):
        return np.sqrt(self.var)

    def result(self):
        self._require_fitted()
        retained = np.concatenate([chunk[0] for chunk in self._chunks], axis=0)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            quantiles = np.nanquantile(retained, self.quantiles, axis=0)
        return {
            "samples": self.n_samples_,
            "total_samples": self.total_samples_,
            "count": self.count,
            "mean": self.mean,
            "var": self.var,
            "std": self.std,
            "min": np.where(self.count > 0, self._minimum, np.nan),
            "max": np.where(self.count > 0, self._maximum, np.nan),
            "quantile_levels": self.quantiles.copy(),
            "quantiles": quantiles,
            "histogram": self.histogram_.copy(),
            "bin_edges": self.bin_edges_.copy(),
            "underflow": self.underflow_.copy(),
            "overflow": self.overflow_.copy(),
        }


RunningStats = StreamingStats
