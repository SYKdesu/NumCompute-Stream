from copy import copy, deepcopy

import numpy as np

from .metrics import _as_labels, _label_kind
from .tree import DecisionTreeClassifier


class Pipeline:

    def __init__(self, steps):
        self.steps = list(steps)
        if not self.steps:
            raise ValueError("steps must contain at least one named component.")
        names = []
        for item in self.steps:
            if not isinstance(item, tuple) or len(item) != 2:
                raise TypeError("Each step must be a (name, component) tuple.")
            name, component = item
            if not isinstance(name, str) or not name or "__" in name or name in names:
                raise ValueError("Step names must be unique non-empty strings.")
            if component is None:
                raise ValueError("Pipeline components must not be None.")
            names.append(name)
        for _, transformer in self.steps[:-1]:
            for method in ("fit", "partial_fit", "transform"):
                if not callable(getattr(transformer, method, None)):
                    raise TypeError("Transformers must implement fit, partial_fit, and transform.")
        if not isinstance(self.steps[-1][1], DecisionTreeClassifier):
            raise TypeError("The final component must be a DecisionTreeClassifier.")
        self.n_features_in_ = None
        self.n_retrains_ = 0
        self._active_transformers = None
        self._label_kind = None
        self._clear_buffer()

    def _clear_buffer(self):
        self._X_chunks = []
        self._y_chunks = []
        self.pending_samples_ = 0

    @property
    def named_steps(self):
        return dict(self.steps)

    @property
    def buffer_size(self):
        return self.steps[-1][1].buffer_size

    @property
    def is_fitted_(self):
        return self.steps[-1][1].root_ is not None and self._active_transformers is not None

    @property
    def classes_(self):
        if not self.is_fitted_:
            raise RuntimeError("Pipeline is not fitted.")
        return self.steps[-1][1].classes_

    @property
    def n_samples_fit_(self):
        return self.steps[-1][1].n_samples_fit_

    def __getitem__(self, key):
        return self.named_steps[key] if isinstance(key, str) else self.steps[key][1]

    def _input(self, X, allow_empty=False):
        X = np.asarray(X)
        if X.ndim != 2 or X.shape[1] == 0 or not allow_empty and len(X) == 0:
            raise ValueError("X must be a 2D array with samples and features.")
        return X

    def _training_input(self, X, y, streaming=False):
        X = self._input(X)
        y = _as_labels(y)
        if len(y) != len(X):
            raise ValueError("X and y must have the same number of samples.")
        if streaming:
            if self.n_features_in_ is not None and X.shape[1] != self.n_features_in_:
                raise ValueError("X has a different number of features.")
            if self._label_kind is not None and _label_kind(y) != self._label_kind:
                raise ValueError("Labels must keep the same type across chunks.")
        return X, y

    @staticmethod
    def _transform(transformers, X):
        transformed = X
        for _, transformer in transformers:
            transformed = transformer.transform(transformed)
        return transformed

    def _prepare_chunk(self, X, streaming):
        transformers = deepcopy(self.steps[:-1])
        for _, transformer in transformers:
            if streaming:
                transformer.partial_fit(X)
            else:
                transformer.fit(X)
            X = transformer.transform(X)
        return transformers, X

    def _set_steps(self, transformers, model, X, y):
        self.steps = transformers + [(self.steps[-1][0], model)]
        self.n_features_in_ = X.shape[1]
        self._label_kind = _label_kind(y)

    def fit(self, X, y):
        X, y = self._training_input(X, y)
        running, prepared = self._prepare_chunk(X, streaming=False)
        model = copy(self.steps[-1][1]).fit(prepared, y)
        self._active_transformers = deepcopy(running)
        self._set_steps(running, model, X, y)
        self.n_retrains_ = 0
        self._clear_buffer()
        return self

    def partial_fit(self, X_chunk, y_chunk):
        X, y = self._training_input(X_chunk, y_chunk, streaming=True)
        running, prepared = self._prepare_chunk(X, streaming=True)
        model = self.steps[-1][1]
        model._validate_training_data(prepared, y)

        if self.pending_samples_ + len(y) >= self.buffer_size:
            raw = np.concatenate(self._X_chunks + [X], axis=0)
            labels = np.concatenate(self._y_chunks + [y])
            active = deepcopy(running)
            prepared = self._transform(active, raw)
            prepared, labels = model._validate_training_data(prepared, labels)

            model = copy(model)
            model._fit_tree(prepared, labels, keep_classes=True)
            model._clear_buffer()
            self.n_retrains_ += 1
            model.n_retrains_ = self.n_retrains_
            self._active_transformers = active
            self._clear_buffer()
        else:
            self._X_chunks.append(X.copy())
            self._y_chunks.append(y.copy())
            self.pending_samples_ += len(y)

        self._set_steps(running, model, X, y)
        return self

    def transform(self, X):
        if not self.is_fitted_:
            raise RuntimeError("Pipeline is not fitted.")
        X = self._input(X, allow_empty=True)
        if X.shape[1] != self.n_features_in_:
            raise ValueError("X has a different number of features.")
        return self._transform(self._active_transformers, X)

    def predict(self, X):
        return self.steps[-1][1].predict(self.transform(X))

    def predict_proba(self, X):
        return self.steps[-1][1].predict_proba(self.transform(X))

    def score(self, X, y):
        prediction = self.predict(X)
        y = _as_labels(y)
        if y.shape != prediction.shape:
            raise ValueError("y must match the prediction shape.")
        return float(np.mean(prediction == y))


def make_pipeline(*components):
    counts = {}
    steps = []
    for component in components:
        base = type(component).__name__.lower()
        counts[base] = counts.get(base, 0) + 1
        name = base if counts[base] == 1 else f"{base}{counts[base]}"
        steps.append((name, component))
    return Pipeline(steps)
