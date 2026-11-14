# tests/test_io.py

import numpy as np

from numcompute.io import load_csv


def test_load_basic_csv(tmp_path):
    file = tmp_path / "data.csv"
    file.write_text(
        "1,2,3\n"
        "4,5,6\n"
    )

    result = load_csv(file)

    expected = np.array([
        [1, 2, 3],
        [4, 5, 6],
    ])

    assert np.array_equal(result, expected)


def test_custom_delimiter(tmp_path):
    file = tmp_path / "data.tsv"
    file.write_text(
        "1\t2\n"
        "3\t4\n"
    )

    result = load_csv(file, delimiter="\t")

    expected = np.array([
        [1, 2],
        [3, 4],
    ])

    assert np.array_equal(result, expected)


def test_missing_values(tmp_path):
    file = tmp_path / "data.csv"
    file.write_text(
        "1,2\n"
        "3,\n"
    )

    result = load_csv(file)

    assert np.isnan(result[1, 1])


def test_skip_header(tmp_path):
    file = tmp_path / "data.csv"
    file.write_text(
        "x,y\n"
        "1,2\n"
        "3,4\n"
    )

    result = load_csv(
        file,
        skip_header=1
    )

    expected = np.array([
        [1, 2],
        [3, 4],
    ])

    assert np.array_equal(result, expected)


def test_missing_file():
    try:
        load_csv("does_not_exist.csv")
    except FileNotFoundError:
        pass
    else:
        raise AssertionError(
            "Expected FileNotFoundError"
        )

def test_stream_csv_preserves_order_and_final_short_chunk():
    from io import StringIO
    from numcompute_stream.io import iter_csv_chunks
    source = StringIO("x,y\n0,0\n1,0\n2,1\n3,1\n4,1\n")
    chunks = list(iter_csv_chunks(source, "y", chunk_size=2, target_dtype=int))
    assert [len(y) for _, y in chunks] == [2, 2, 1]
    np.testing.assert_array_equal(np.concatenate([X[:, 0] for X, _ in chunks]), np.arange(5))


def test_stream_csv_numeric_missing_values_and_categorical_columns():
    from io import StringIO
    from numcompute_stream.io import load_csv as load_stream_csv
    source = StringIO('value,device,target\n1,"a,b",0\n?,red,1\n')
    X, y = load_stream_csv(source, "target", categorical_columns=("device",), target_dtype=int)
    assert X.shape == (2, 2)
    assert X[0, 1] == "a,b"
    assert np.isnan(X[1, 0])
    np.testing.assert_array_equal(y, [0, 1])


def test_stream_csv_rejects_missing_target_and_malformed_rows():
    from io import StringIO
    import pytest
    from numcompute_stream.io import load_csv as load_stream_csv
    for text in ("x,y\n1,\n", "x,y\n1,0,extra\n", "x,y\n"):
        with pytest.raises(ValueError):
            load_stream_csv(StringIO(text), "y")


def test_stream_array_chunks_are_independent_and_keep_remainder():
    from numcompute_stream.io import iter_chunks
    X = np.arange(5).reshape(-1, 1)
    y = np.arange(5)
    chunks = list(iter_chunks(X, y, chunk_size=2))
    chunks[0][0][0, 0] = 99
    assert X[0, 0] == 0
    assert [len(chunk[1]) for chunk in chunks] == [2, 2, 1]
