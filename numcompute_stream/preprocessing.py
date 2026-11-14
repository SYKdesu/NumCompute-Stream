import numpy as np


def _numeric_array(X, allow_empty=False):
    if np.iscomplexobj(X):
        raise ValueError("X must contain real numerical values.")
    X = np.asarray(X, dtype=float)
    if X.ndim != 2 or X.shape[1] == 0:
        raise ValueError("X must be a 2D array with at least one feature.")
    if not allow_empty and len(X) == 0:
        raise ValueError("X must not be empty.")
    if np.any(np.isinf(X)):
        raise ValueError("Infinite feature values are not supported.")
    return X


def _feature_check(X, expected):
    if expected is not None and X.shape[1] != expected:
        raise ValueError("X has a different number of features.")


class StandardScaler:

    def __init__(self):
        self.n_features_in_ = None
        self.n_samples_seen_ = None
        self.mean_ = None
        self.var_ = None
        self.scale_ = None
        self._m2 = None

    def fit(self, X):
        self.__init__()
        return self.partial_fit(X)

    def partial_fit(self, X):
        X = _numeric_array(X)
        _feature_check(X, self.n_features_in_)
        valid = ~np.isnan(X)
        counts = valid.sum(axis=0)
        sums = np.where(valid, X, 0.0).sum(axis=0)
        means = np.divide(sums, counts, out=np.zeros(X.shape[1]), where=counts > 0)
        centered = np.where(valid, X - means, 0.0)
        m2 = np.sum(centered ** 2, axis=0)

        if self.n_features_in_ is None:
            self.n_features_in_ = X.shape[1]
            self.n_samples_seen_ = counts
            self.mean_ = means
            self._m2 = m2
        else:
            old_counts = self.n_samples_seen_
            total = old_counts + counts
            delta = means - self.mean_
            ratio = np.divide(counts, total, out=np.zeros_like(means), where=total > 0)
            weight = np.divide(
                old_counts.astype(float) * counts,
                total,
                out=np.zeros_like(means),
                where=total > 0,
            )
            self.mean_ = self.mean_ + delta * ratio
            self._m2 = self._m2 + m2 + delta ** 2 * weight
            self.n_samples_seen_ = total

        self.var_ = np.divide(
            self._m2,
            self.n_samples_seen_,
            out=np.zeros_like(self.mean_),
            where=self.n_samples_seen_ > 0,
        )
        self.var_ = np.maximum(self.var_, 0.0)
        self.scale_ = np.sqrt(self.var_)
        self.scale_ = np.where(self.scale_ > 0, self.scale_, 1.0)
        return self

    def transform(self, X):
        if self.mean_ is None:
            raise RuntimeError("StandardScaler is not fitted.")
        X = _numeric_array(X, allow_empty=True)
        _feature_check(X, self.n_features_in_)
        return (X - self.mean_) / self.scale_

    def fit_transform(self, X):
        return self.fit(X).transform(X)


class MinMaxScaler:

    def __init__(self, feature_range=(0.0, 1.0)):
        bounds = np.asarray(feature_range, dtype=float)
        if bounds.shape != (2,) or not np.all(np.isfinite(bounds)) or bounds[0] >= bounds[1]:
            raise ValueError("feature_range must contain two increasing finite values.")
        self.feature_range = tuple(bounds)
        self.n_features_in_ = None
        self.n_samples_seen_ = None
        self.data_min_ = None
        self.data_max_ = None
        self.data_range_ = None

    def fit(self, X):
        self.__init__(self.feature_range)
        return self.partial_fit(X)

    def partial_fit(self, X):
        X = _numeric_array(X)
        _feature_check(X, self.n_features_in_)
        valid = ~np.isnan(X)
        counts = valid.sum(axis=0)
        minimum = np.where(valid, X, np.inf).min(axis=0)
        maximum = np.where(valid, X, -np.inf).max(axis=0)
        if self.n_features_in_ is None:
            self.n_features_in_ = X.shape[1]
            self.n_samples_seen_ = counts
            self._minimum = minimum
            self._maximum = maximum
        else:
            self.n_samples_seen_ = self.n_samples_seen_ + counts
            self._minimum = np.minimum(self._minimum, minimum)
            self._maximum = np.maximum(self._maximum, maximum)
        self.data_min_ = np.where(self.n_samples_seen_ > 0, self._minimum, 0.0)
        self.data_max_ = np.where(self.n_samples_seen_ > 0, self._maximum, 0.0)
        difference = self.data_max_ - self.data_min_
        self.data_range_ = np.where(difference > 0, difference, 1.0)
        return self

    def transform(self, X):
        if self.data_min_ is None:
            raise RuntimeError("MinMaxScaler is not fitted.")
        X = _numeric_array(X, allow_empty=True)
        _feature_check(X, self.n_features_in_)
        low, high = self.feature_range
        return (X - self.data_min_) / self.data_range_ * (high - low) + low

    def fit_transform(self, X):
        return self.fit(X).transform(X)


class Imputer:

    def __init__(self, strategy="mean", fill_value=0.0):
        if strategy not in ("mean", "constant"):
            raise ValueError("strategy must be 'mean' or 'constant'.")
        if not np.isscalar(fill_value) or not np.isfinite(float(fill_value)):
            raise ValueError("fill_value must be a finite number.")
        self.strategy = strategy
        self.fill_value = float(fill_value)
        self.n_features_in_ = None
        self.statistics_ = None
        self._stats = StandardScaler()

    def fit(self, X):
        self.__init__(self.strategy, self.fill_value)
        return self.partial_fit(X)

    def partial_fit(self, X):
        X = _numeric_array(X)
        _feature_check(X, self.n_features_in_)
        self._stats.partial_fit(X)
        self.n_features_in_ = X.shape[1]
        if self.strategy == "mean":
            self.statistics_ = np.where(
                self._stats.n_samples_seen_ > 0, self._stats.mean_, self.fill_value
            )
        else:
            self.statistics_ = np.full(X.shape[1], self.fill_value)
        return self

    def transform(self, X):
        if self.statistics_ is None:
            raise RuntimeError("Imputer is not fitted.")
        X = _numeric_array(X, allow_empty=True)
        _feature_check(X, self.n_features_in_)
        return np.where(np.isnan(X), self.statistics_, X)

    def fit_transform(self, X):
        return self.fit(X).transform(X)


SimpleImputer = Imputer


def _category(value):
    if isinstance(value, np.generic):
        value = value.item()
    if value is None or isinstance(value, float) and np.isnan(value):
        return ("missing", ""), "<missing>"
    if isinstance(value, (str, bool, int, float)):
        if isinstance(value, float) and not np.isfinite(value):
            raise ValueError("Infinite categories are not supported.")
        return (type(value).__name__, repr(value)), repr(value)
    raise ValueError("Categories must be strings, finite numbers, or missing values.")


class OneHotEncoder:

    def __init__(self, handle_unknown="ignore"):
        if handle_unknown not in ("error", "ignore"):
            raise ValueError("handle_unknown must be 'error' or 'ignore'.")
        self.handle_unknown = handle_unknown
        self.n_features_in_ = None
        self.categories_ = None
        self.n_output_features_ = 0
        self._mappings = []
        self._features = []

    def fit(self, X):
        self.__init__(self.handle_unknown)
        return self.partial_fit(X)

    def partial_fit(self, X):
        X = np.asarray(X, dtype=object)
        if X.ndim != 2 or len(X) == 0 or X.shape[1] == 0:
            raise ValueError("X must be a non-empty 2D categorical array.")
        _feature_check(X, self.n_features_in_)
        discovered = []
        for feature in range(X.shape[1]):
            values = {}
            for value in X[:, feature]:
                key, display = _category(value)
                values[key] = (value, display)
            discovered.append(values)

        if self.n_features_in_ is None:
            self.n_features_in_ = X.shape[1]
            self._mappings = [{} for _ in range(X.shape[1])]
            self.categories_ = [[] for _ in range(X.shape[1])]

        for feature, values in enumerate(discovered):
            for key in sorted(values):
                if key not in self._mappings[feature]:
                    value, display = values[key]
                    self._mappings[feature][key] = self.n_output_features_
                    self.categories_[feature].append(value)
                    self._features.append((feature, display))
                    self.n_output_features_ += 1
        return self

    def transform(self, X):
        if self.n_features_in_ is None:
            raise RuntimeError("OneHotEncoder is not fitted.")
        X = np.asarray(X, dtype=object)
        if X.ndim != 2:
            raise ValueError("X must be a 2D categorical array.")
        _feature_check(X, self.n_features_in_)
        result = np.zeros((len(X), self.n_output_features_), dtype=float)
        for feature in range(X.shape[1]):
            columns = np.fromiter(
                (self._mappings[feature].get(_category(value)[0], -1) for value in X[:, feature]),
                dtype=int,
                count=len(X),
            )
            known = columns >= 0
            if self.handle_unknown == "error" and not np.all(known):
                raise ValueError("Unknown category encountered.")
            result[np.flatnonzero(known), columns[known]] = 1.0
        return result

    def get_feature_names_out(self, input_features=None):
        if self.n_features_in_ is None:
            raise RuntimeError("OneHotEncoder is not fitted.")
        names = (
            [f"x{index}" for index in range(self.n_features_in_)]
            if input_features is None else list(input_features)
        )
        if len(names) != self.n_features_in_:
            raise ValueError("input_features has a different number of features.")
        return np.asarray([f"{names[feature]}={value}" for feature, value in self._features])

    def fit_transform(self, X):
        return self.fit(X).transform(X)


class MixedPreprocessor:

    def __init__(self, numeric_columns=None, categorical_columns=(), scale=True):
        self.numeric_columns = None if numeric_columns is None else tuple(numeric_columns)
        self.categorical_columns = tuple(categorical_columns)
        self.scale = bool(scale)
        self._reset()

    def _reset(self):
        self.n_features_in_ = None
        self.imputer_ = Imputer()
        self.scaler_ = StandardScaler()
        self.encoder_ = OneHotEncoder(handle_unknown="ignore")

    def _input(self, X, allow_empty=False):
        X = np.asarray(X, dtype=object)
        if X.ndim != 2 or X.shape[1] == 0 or not allow_empty and len(X) == 0:
            raise ValueError("X must be a 2D array with samples and features.")
        _feature_check(X, self.n_features_in_)
        return X

    def fit(self, X):
        self._reset()
        return self.partial_fit(X)

    def partial_fit(self, X):
        X = self._input(X)
        categorical = list(self.categorical_columns)
        numeric = (
            [index for index in range(X.shape[1]) if index not in categorical]
            if self.numeric_columns is None else list(self.numeric_columns)
        )
        columns = numeric + categorical
        if (
            not columns
            or any(type(column) is not int or not 0 <= column < X.shape[1] for column in columns)
            or len(columns) != len(set(columns))
        ):
            raise ValueError("Column selections must be disjoint valid integer indices.")
        if numeric:
            values = np.asarray(X[:, numeric], dtype=float)
            self.imputer_.partial_fit(values)
            if self.scale:
                self.scaler_.partial_fit(self.imputer_.transform(values))
        if categorical:
            self.encoder_.partial_fit(X[:, categorical])
        self.n_features_in_ = X.shape[1]
        self.numeric_columns_ = numeric
        self.categorical_columns_ = categorical
        return self

    def transform(self, X):
        if self.n_features_in_ is None:
            raise RuntimeError("MixedPreprocessor is not fitted.")
        X = self._input(X, allow_empty=True)
        parts = []
        if self.numeric_columns_:
            numeric = self.imputer_.transform(np.asarray(X[:, self.numeric_columns_], dtype=float))
            parts.append(self.scaler_.transform(numeric) if self.scale else numeric)
        if self.categorical_columns_:
            parts.append(self.encoder_.transform(X[:, self.categorical_columns_]))
        return np.concatenate(parts, axis=1)

    def get_feature_names_out(self):
        if self.n_features_in_ is None:
            raise RuntimeError("MixedPreprocessor is not fitted.")
        names = [f"x{index}" for index in self.numeric_columns_]
        if self.categorical_columns_:
            names.extend(
                self.encoder_.get_feature_names_out(
                    [f"x{index}" for index in self.categorical_columns_]
                ).tolist()
            )
        return np.asarray(names)

    def fit_transform(self, X):
        return self.fit(X).transform(X)
