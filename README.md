# NumCompute-Stream

A modular classification framework built with Python, NumPy and matplotlib. It extends the original NumCompute group package with a decision tree written from scratch, chunk-wise preprocessing, buffered retraining, streaming statistics, classification metrics, and reusable plots.

The original package remains available as numcompute. The individual assignment is on the assignment-2.2 branch. The original group project is available at https://github.com/SYKdesu/NumCompute.

## Setup

Requires Python 3.10 or newer. NumPy and matplotlib are the runtime dependencies. Pytest and Jupyter are optional testing and presentation tools; package algorithms do not depend on pandas, scikit-learn, PyTorch, or other external data-processing libraries.

Run from the repository root:

    python -m venv .venv
    .\.venv\Scripts\python.exe -m pip install -e ".[dev,demo]"
    .\.venv\Scripts\python.exe -m ipykernel install --sys-prefix --name numcompute-stream --display-name "Python (NumCompute Stream)"

If .venv already exists, use it without recreating it. On macOS/Linux, replace .\.venv\Scripts\python.exe with .venv/bin/python.

Run the tests with pytest rather than executing an individual test file:

    .\.venv\Scripts\python.exe -m pytest -q

## Streaming example

    from numcompute_stream import (
        DecisionTreeClassifier, MixedPreprocessor, Pipeline,
        StreamTrainer, iter_csv_chunks,
    )

    pipeline = Pipeline([
        ("prepare", MixedPreprocessor(categorical_columns=(3,), scale=True)),
        ("tree", DecisionTreeClassifier(
            max_depth=5, buffer_size=300, max_features=None,
            random_state=42, criterion="gini",
        )),
    ])
    trainer = StreamTrainer(pipeline, window_size=500)

    stream = iter_csv_chunks(
        "demo/data/stream_data.csv",
        target_column="target",
        categorical_columns=("device",),
        target_dtype=int,
        chunk_size=100,
    )
    trainer.run(stream)
    print(trainer.metrics.result())
    print(trainer.logs_[-1])

iter_csv_chunks reads bounded chunks directly from the CSV. load_csv reads the complete dataset and returns (X, y); iter_chunks can then simulate a stream while preserving row order. Both paths use the package's own CSV reader.

## Architecture

| Module | Responsibilities |
|---|---|
| io.py | Header-aware CSV loading, missing-value parsing, categorical columns, chunk iteration |
| tree.py | Gini/entropy split search, probabilities, depth limits, feature sampling, buffered replacement |
| preprocessing.py | Incremental mean/variance scaling, min/max scaling, imputation, one-hot encoding, mixed columns |
| stats.py | Streaming moments, exact quantiles, fixed-edge histograms, optional sample windows |
| metrics.py | Cumulative and rolling accuracy, error, confusion matrix, precision, recall, F1, exact binary AUC |
| pipeline.py | Raw buffering and coordinated activation of preprocessing snapshots and trees |
| stream.py | Test-then-train processing, warm-up handling, timing, retained-state memory estimates, logs |
| visualise.py | Metric curves, model comparisons, predictions versus ground truth |

Existing NumCompute metric/statistical functions are reused for batch convenience APIs. Streaming state and the tree are implemented in numcompute_stream. The original sorting, ranking, optimisation and general pipeline modules remain available for the previous assignment.

## API conventions

X has shape (n_samples, n_features), and y has shape (n_samples,). Prediction returns one original class label per row. predict_proba returns (n_samples, n_classes), with columns matching the active model's classes_. Numeric statistics reduce rows, equivalent to axis=0.

fit resets learned state and returns self. partial_fit and metric update methods return self. predict, predict_proba, transform and result do not update training state. Empty training chunks, mismatched shapes, infinite features and missing targets are rejected. Empty prediction arrays are accepted if their feature count is correct.

### Tree behaviour

DecisionTreeClassifier supports max_depth, min_samples_split, buffer_size, max_features, random_state and criterion. max_features is None or a positive integer no larger than the transformed feature count. criterion is gini or entropy. Root depth is zero. A depth of zero gives a majority-class leaf.

Numeric splits use X[:, feature] <= threshold. A threshold is the lower observed value at a boundary between distinct sorted values. Candidate child impurities are weighted by sample count, and threshold evaluation uses cumulative NumPy class counts. Recursion and loops over features remain in Python.

Tied class counts use the first label in NumPy's sorted order. Split gains within the numerical tolerance retain the first feature and lowest candidate boundary. Equal random seeds reproduce feature selection for the same ordered input. Each replacement tree restarts the configured seed.

Each partial_fit call appends copies of the incoming chunk. Once the pending count is at least buffer_size, ALL pending rows train a replacement and the entire buffer is cleared. A buffer of 500 receiving three 200-row chunks therefore trains once on 600 rows, leaving zero pending rows.

Between retrains, predictions use the previous fitted tree. Before the first retrain, prediction raises RuntimeError. A final below-threshold buffer remains pending. n_retrains_, pending_samples_ and n_samples_fit_ expose this lifecycle. fit trains immediately and resets the pending buffer and retrain count.

Class vocabulary is preserved across stream retrains, including classes missing from a later buffer. New classes enter classes_ only when a replacement is activated. Calling fit starts a fresh vocabulary. Numeric and string label families cannot be mixed across chunks.

### Preprocessing consistency

StandardScaler combines per-feature valid counts, running means and sums of squared deviations. NaNs are omitted from estimates and preserved by transform; constant or entirely unobserved scales are one. Imputer learns observed-value means and uses fill_value for entirely missing columns. Imputed values are not fed back into its observed counts. A constant-imputation strategy is also supported.

OneHotEncoder appends new global output columns without moving existing columns. Missing categories have their own identity, distinct from a literal string such as "<missing>". Unknown categories can raise an error or produce zeros. Category vocabulary grows with distinct categories.

MixedPreprocessor applies numerical imputation/scaling and categorical encoding to selected columns. Numeric columns precede encoded columns. With numeric_columns=None, all columns not declared categorical are numeric; explicitly unselected columns are dropped.

Pipeline owns cloned running transformers and a separate active snapshot. It buffers RAW rows. At a retrain it transforms the whole pending raw buffer with one snapshot, builds a replacement tree, and activates both together. Between retrains, inference uses the previous snapshot. A growing category schema therefore cannot change the active tree's input dimension mid-cycle. The pipeline holds the raw buffer; the tree inside it has no second pending buffer.

### Statistics and metrics

StreamingStats uses per-feature valid counts for mean and variance. Quantiles are exact across retained observations, not averages of chunk quantiles. Histogram edges stay fixed; underflow and overflow counts are reported separately. NaNs are omitted.

Exact cumulative quantiles and exact cumulative AUC retain observations/scores and grow with stream length. window_size bounds retained sample history for rolling statistics and metrics. The window is measured in samples, including partial trimming of an old chunk.

ClassificationMetrics defaults to macro precision, recall and F1. Binary, micro, weighted and per-class averaging are available. Confusion-matrix rows are true labels and columns are predictions. The result includes the corresponding labels. Zero-division behaviour is configurable as zero or one.

Cumulative accuracy is total correct divided by total evaluated samples, so differently sized chunks receive the proper weight. Binary ROC AUC uses scores, gives half credit to ties, and returns NaN until both classes are available. Scores must correspond to pos_label, which defaults to 1. If scores are omitted, AUC uses only updates that supplied scores.

StreamTrainer evaluates before learning each chunk. Warm-up rows never enter accuracy denominators. Logs include chunk and cumulative accuracy, rolling accuracy, precision/recall/F1/AUC, evaluated counts, retrain events, buffer counts, training/prediction time and approximate retained-state memory. Memory is a recursive Python/NumPy object estimate, not process RSS.

## Dataset and demo

demo/data/stream_data.csv is synthetic and reproducible with seed 42. It has 2,400 ordered samples, three normally distributed numeric features, a categorical device feature and a binary target. About 4% of numeric entries are missing and 5% of labels are flipped. The generating boundary changes at row 1,200. Data order is preserved; it must not be shuffled for the adaptation experiment.

Regenerate the included data and benchmark results:

    .\.venv\Scripts\python.exe benchmark/benchmark_stream.py --generate-data --repeats 3

Open the notebook:

    .\.venv\Scripts\python.exe -m jupyterlab demo/stream_demo.ipynb

Select Python (NumCompute Stream), restart the kernel and run all cells. The notebook supports launch from the repository root or demo directory. It shows CSV input, partial_fit, warm-up and retrain logs, cumulative/rolling plots, buffer comparisons, latest pre-training predictions, numeric statistics and vectorisation measurements.

## Benchmarks

    .\.venv\Scripts\python.exe benchmark/benchmark_stream.py --repeats 3

benchmark/results_stream.json stores raw repeated measurements, per-chunk logs, environment details and summaries.

The loop/vectorised comparison searches the same Gini candidate boundaries and verifies the selected split before timing. Both implementations are warmed. Timings exclude data generation and report the median of repeated calls. The Python baseline counts labels with loops; the vectorised implementation uses sorted cumulative class counts.

Buffer comparisons use fresh pipelines, the same row order, seed, chunk size and data. In addition to each model's own prequential accuracy, a common-period score starts after every buffer size has completed warm-up. Its denominator is identical across models. Timing uses repeated runs; speed ratios and buffer rankings depend on the environment.

The measured speed comparison concerns split search, not the whole training pipeline. Smaller buffers can adapt sooner but train on less data; no buffer size is assumed to be universally best.

Example measurements on 14 November 2026 used Windows 11, Python 3.14.6 and NumPy 2.5.4. The 400-row split-search benchmark returned the same split in both implementations: median loop time 97.65 ms, vectorised time 0.45 ms, a 216.19x speedup for this operation.

| Buffer size | Common-period accuracy | Common evaluated samples | Retrains | Median total training time | Median time on retrain chunks |
|---|---:|---:|---:|---:|---:|
| 100 | 78.94% | 1,800 | 24 | 0.083 s | 0.083 s |
| 300 | 75.33% | 1,800 | 8 | 0.052 s | 0.046 s |
| 600 | 65.50% | 1,800 | 4 | 0.038 s | 0.032 s |

These are medians of three streaming runs on the included dataset. All common-period scores start after sample 600. The table is a reproducible example rather than a guaranteed ranking for other streams.

## Verification and limitations

Tests cover tree split correctness against an independent reference, buffered overshoot/replacement, class/probability behaviour, deterministic seeds, missing values, incremental moments, stable encoded columns, preprocessing snapshots, metric accumulation, CSV remainders and test-then-train evaluation.

The tree requires finite numerical inputs after preprocessing. This framework periodically rebuilds a single tree; it does not implement node-wise online or Hoeffding-tree updates. Exact history-dependent statistics and AUC have the memory limitations described above. Buffer-size accuracy comparisons need the common evaluation period to avoid different warm-up denominators.

The technical report and screen-recorded demonstration are separate submission deliverables. This repository's current work completes the code, tests, notebook and benchmark materials.

## Original group package

The original numcompute package includes CSV I/O, batch preprocessing, sorting/searching, ranking, descriptive statistics, classification metrics, numerical derivatives and composable pipelines.

| Group member | Original responsibilities |
|---|---|
| Hongyu Liu | I/O, preprocessing, utilities |
| Yi Su | Sorting, searching, ranking, statistics |
| Wenxuan Xue | Metrics, numerical differentiation, pipelines, integration |
