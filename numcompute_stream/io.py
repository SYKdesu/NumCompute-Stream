import csv
from contextlib import nullcontext
from pathlib import Path

import numpy as np


def _chunk_size(value):
    if type(value) is not int or value < 1:
        raise ValueError("chunk_size must be a positive integer.")
    return value


def iter_chunks(X, y, chunk_size=100):
    chunk_size = _chunk_size(chunk_size)
    X = np.asarray(X)
    y = np.asarray(y)
    if X.ndim != 2 or X.shape[1] == 0:
        raise ValueError("X must be a 2D array with features.")
    if y.ndim != 1 or len(y) != len(X):
        raise ValueError("y must be 1D and have the same rows as X.")
    for start in range(0, len(X), chunk_size):
        stop = start + chunk_size
        yield X[start:stop].copy(), y[start:stop].copy()


def _column_index(column, header):
    if isinstance(column, str):
        if column not in header:
            raise ValueError(f"Column {column!r} is not in the CSV header.")
        return header.index(column)
    if type(column) is not int or not -len(header) <= column < len(header):
        raise ValueError("Column must be a header name or a valid integer index.")
    return column % len(header)


def iter_csv_chunks(
    filepath,
    target_column,
    chunk_size=100,
    categorical_columns=(),
    target_dtype=None,
    delimiter=",",
    missing_values=("", "NA", "NaN", "nan", "?"),
):
    chunk_size = _chunk_size(chunk_size)
    if not isinstance(delimiter, str) or len(delimiter) != 1:
        raise ValueError("delimiter must be one character.")
    source = (
        nullcontext(filepath)
        if hasattr(filepath, "read")
        else Path(filepath).open("r", encoding="utf-8-sig", newline="")
    )
    with source as handle:
        reader = csv.reader(handle, delimiter=delimiter)
        header = next(reader, None)
        if header is None or not header:
            raise ValueError("CSV must have a header.")
        header = [value.strip() for value in header]
        if any(not value for value in header) or len(set(header)) != len(header):
            raise ValueError("CSV header names must be non-empty and unique.")
        target = _column_index(target_column, header)
        categorical = {_column_index(column, header) for column in categorical_columns}
        if target in categorical:
            raise ValueError("The target cannot also be a categorical feature.")
        features = [index for index in range(len(header)) if index != target]
        if not features:
            raise ValueError("CSV must contain at least one feature column.")
        missing = set(missing_values)
        rows = []
        labels = []

        def convert():
            X = np.asarray(rows, dtype=object if categorical else float)
            y = np.asarray(labels, dtype=target_dtype)
            if y.dtype.kind in "biuf" and not np.all(np.isfinite(y)):
                raise ValueError("Target values must be finite.")
            return X, y

        for row in reader:
            if not row or all(not value.strip() for value in row):
                continue
            if len(row) != len(header):
                raise ValueError(f"CSV row {reader.line_num} has an incorrect number of columns.")
            row = [value.strip() for value in row]
            if row[target] in missing:
                raise ValueError(f"CSV row {reader.line_num} has a missing target.")
            values = []
            for feature in features:
                value = row[feature]
                if feature in categorical:
                    values.append(None if value in missing else value)
                else:
                    try:
                        number = np.nan if value in missing else float(value)
                    except ValueError as error:
                        raise ValueError(
                            f"CSV row {reader.line_num}, column {header[feature]!r} is not numeric."
                        ) from error
                    if np.isinf(number):
                        raise ValueError("Infinite feature values are not supported.")
                    values.append(number)
            rows.append(values)
            labels.append(row[target])
            if len(rows) == chunk_size:
                yield convert()
                rows = []
                labels = []
        if rows:
            yield convert()


def load_csv(filepath, target_column, **options):
    options = dict(options)
    options.setdefault("chunk_size", 4096)
    chunks = list(iter_csv_chunks(filepath, target_column, **options))
    if not chunks:
        raise ValueError("CSV has no data rows.")
    return (
        np.concatenate([X for X, _ in chunks], axis=0),
        np.concatenate([y for _, y in chunks], axis=0),
    )


class CSVReader:

    def __init__(self, filepath, target_column, **options):
        self.filepath = filepath
        self.target_column = target_column
        self.options = dict(options)

    def read(self):
        return load_csv(self.filepath, self.target_column, **self.options)

    def iter_chunks(self, chunk_size=100):
        options = dict(self.options)
        options["chunk_size"] = chunk_size
        return iter_csv_chunks(self.filepath, self.target_column, **options)
