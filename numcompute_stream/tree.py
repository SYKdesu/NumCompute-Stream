from copy import copy

import numpy as np


class DecisionTreeClassifier:

    def __init__(
        self,
        max_depth=5,
        min_samples_split=2,
        buffer_size=500,
        max_features=None,
        random_state=42,
        criterion="gini",
    ):
        if type(max_depth) is not int or max_depth < 0:
            raise ValueError("max_depth must be a non-negative integer.")
        if type(min_samples_split) is not int or min_samples_split < 2:
            raise ValueError("min_samples_split must be an integer >= 2.")
        if type(buffer_size) is not int or buffer_size < 1:
            raise ValueError("buffer_size must be a positive integer.")
        if max_features is not None:
            if type(max_features) is not int or max_features < 1:
                raise ValueError("max_features must be None or a positive integer.")
        if random_state is not None:
            if type(random_state) is not int or random_state < 0:
                raise ValueError("random_state must be None or a non-negative integer.")
        if criterion not in ("gini", "entropy"):
            raise ValueError("criterion must be 'gini' or 'entropy'.")

        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.buffer_size = buffer_size
        self.max_features = max_features
        self.random_state = random_state
        self.criterion = criterion
        self.root_ = None
        self.n_retrains_ = 0
        self.n_samples_fit_ = 0
        self._clear_buffer()

    def _clear_buffer(self):
        self._X_chunks = []
        self._y_chunks = []
        self.pending_samples_ = 0

    @staticmethod
    def _validate_X(X):
        if np.iscomplexobj(X):
            raise ValueError("X must contain real numerical features.")
        X = np.asarray(X, dtype=float)
        if X.ndim != 2 or X.shape[1] == 0:
            raise ValueError("X must be a 2D array with at least one feature.")
        if not np.all(np.isfinite(X)):
            raise ValueError("X must contain finite values; impute NaNs first.")
        return X

    def _validate_training_data(self, X, y):
        X = self._validate_X(X)
        if self.max_features is not None and self.max_features > X.shape[1]:
            raise ValueError("max_features must not exceed the number of features.")
        y = np.asarray(y)
        if y.ndim != 1 or len(y) != len(X):
            raise ValueError("y must be 1D and have the same rows as X.")
        if len(y) == 0:
            raise ValueError("Training data must not be empty.")
        if y.dtype.kind == "O":
            if not all(isinstance(label, str) for label in y):
                raise ValueError("y must contain finite numerical or string labels.")
            y = y.astype(str)
        if y.dtype.kind in "biuf":
            if not np.all(np.isfinite(y)):
                raise ValueError("y must contain finite labels without NaNs.")
        elif y.dtype.kind not in "US":
            raise ValueError("y must contain finite numerical or string labels.")
        return X, y

    @staticmethod
    def _gini(y):
        y = np.asarray(y)
        if y.ndim != 1 or len(y) == 0:
            raise ValueError("Encoded labels must be a non-empty 1D array.")
        counts = np.bincount(y)
        probabilities = counts / counts.sum()
        return float(1.0 - np.sum(probabilities ** 2))

    def _impurity(self, counts):
        counts = np.asarray(counts, dtype=float)
        totals = counts.sum(axis=-1, keepdims=True)
        probabilities = np.divide(
            counts, totals, out=np.zeros_like(counts), where=totals > 0
        )
        if self.criterion == "gini":
            result = 1.0 - np.sum(probabilities ** 2, axis=-1)
        else:
            logarithms = np.zeros_like(probabilities)
            np.log2(probabilities, out=logarithms, where=probabilities > 0)
            result = -np.sum(probabilities * logarithms, axis=-1)
        return np.where(totals[..., 0] > 0, result, 0.0)

    def _fit_tree(self, X, y, keep_classes=False):
        classes = np.unique(y)
        if keep_classes and self.root_ is not None:
            classes = np.union1d(self.classes_, classes)

        builder = copy(self)
        builder.classes_ = classes
        builder._rng = np.random.default_rng(self.random_state)
        root = builder._grow(X, np.searchsorted(classes, y), depth=0)

        self.root_ = root
        self.classes_ = classes
        self.n_features_in_ = X.shape[1]
        self.n_samples_fit_ = len(y)
        self._label_kind = self._label_type(y)
        self._rng = builder._rng

    @staticmethod
    def _label_type(y):
        return "numeric" if y.dtype.kind in "biuf" else "string"

    def fit(self, X, y):
        X, y = self._validate_training_data(X, y)
        self._fit_tree(X, y)
        self._clear_buffer()
        self.n_retrains_ = 0
        return self

    def partial_fit(self, X_chunk, y_chunk):
        X, y = self._validate_training_data(X_chunk, y_chunk)
        if hasattr(self, "n_features_in_") and X.shape[1] != self.n_features_in_:
            raise ValueError("X has a different number of features.")
        label_kind = self._label_type(y)
        if hasattr(self, "_label_kind") and label_kind != self._label_kind:
            raise ValueError("y must keep the same numerical or string label type.")

        self.n_features_in_ = X.shape[1]
        self._label_kind = label_kind
        self._X_chunks.append(X.copy())
        self._y_chunks.append(y.copy())
        self.pending_samples_ += len(y)

        if self.pending_samples_ >= self.buffer_size:
            X_buffer = np.concatenate(self._X_chunks, axis=0)
            y_buffer = np.concatenate(self._y_chunks, axis=0)
            self._fit_tree(X_buffer, y_buffer, keep_classes=True)
            self.n_retrains_ += 1
            self._clear_buffer()

        return self

    def _grow(self, X, y, depth):
        counts = np.bincount(y, minlength=len(self.classes_))
        node = {
            "value": int(np.argmax(counts)),
            "probabilities": tuple((counts / counts.sum()).tolist()),
        }
        if (
            depth >= self.max_depth
            or len(y) < self.min_samples_split
            or np.count_nonzero(counts) == 1
        ):
            return node

        split = self._best_split(X, y)
        if split is None:
            return node

        feature, threshold = split
        left_mask = X[:, feature] <= threshold
        return {
            "feature": feature,
            "threshold": threshold,
            "left": self._grow(X[left_mask], y[left_mask], depth + 1),
            "right": self._grow(X[~left_mask], y[~left_mask], depth + 1),
        }

    def _best_split(self, X, y):
        n_samples = len(y)
        parent_impurity = float(
            self._impurity(np.bincount(y, minlength=len(self.classes_)))
        )
        best_gain = 1e-12
        best_split = None
        indicators = y[:, None] == np.arange(len(self.classes_))
        n_features = X.shape[1]
        if self.max_features is None or self.max_features == n_features:
            features = np.arange(n_features)
        else:
            features = np.sort(
                self._rng.choice(n_features, size=self.max_features, replace=False)
            )

        n_left = np.arange(1, n_samples)
        n_right = n_samples - n_left
        for feature in features:
            order = np.argsort(X[:, feature], kind="stable")
            values = X[order, feature]
            valid = values[:-1] < values[1:]
            if not np.any(valid):
                continue

            cumulative = np.cumsum(indicators[order], axis=0)
            left_counts = cumulative[:-1]
            right_counts = cumulative[-1] - left_counts
            weighted_impurity = (
                n_left * self._impurity(left_counts)
                + n_right * self._impurity(right_counts)
            ) / n_samples
            gains = parent_impurity - weighted_impurity
            gains[~valid] = -np.inf
            index = int(np.argmax(gains))
            gain = float(gains[index])
            if gain > best_gain + 1e-12:
                best_gain = gain
                best_split = (int(feature), float(values[index]))

        return best_split

    def _prediction_input(self, X):
        if self.root_ is None:
            raise RuntimeError("Call fit or accumulate enough partial_fit data first.")
        X = self._validate_X(X)
        if X.shape[1] != self.n_features_in_:
            raise ValueError("X has a different number of features.")
        return X

    def predict(self, X):
        X = self._prediction_input(X)
        return self.classes_[self._predict_node(self.root_, X)]

    def predict_proba(self, X):
        X = self._prediction_input(X)
        return self._predict_node(self.root_, X, probabilities=True)

    def _predict_node(self, node, X, probabilities=False):
        if "value" in node:
            if probabilities:
                return np.tile(np.asarray(node["probabilities"]), (len(X), 1))
            return np.full(len(X), node["value"], dtype=int)

        if probabilities:
            predictions = np.empty((len(X), len(self.classes_)), dtype=float)
        else:
            predictions = np.empty(len(X), dtype=int)
        left_mask = X[:, node["feature"]] <= node["threshold"]
        if np.any(left_mask):
            predictions[left_mask] = self._predict_node(
                node["left"], X[left_mask], probabilities
            )
        if np.any(~left_mask):
            predictions[~left_mask] = self._predict_node(
                node["right"], X[~left_mask], probabilities
            )
        return predictions

    def score(self, X, y):
        prediction = self.predict(X)
        y = np.asarray(y)
        if y.ndim != 1 or y.shape != prediction.shape or len(y) == 0:
            raise ValueError("y must be non-empty and match the predicted labels.")
        return float(np.mean(prediction == y))
