import numpy as np

from .metrics import _as_labels


def _axes(ax):
    if ax is not None:
        return ax
    import matplotlib.pyplot as plt
    return plt.subplots(figsize=(8, 4))[1]


def _finish(ax, save_path, show):
    ax.figure.tight_layout()
    if save_path is not None:
        ax.figure.savefig(save_path, dpi=150, bbox_inches="tight")
    if show:
        import matplotlib.pyplot as plt
        plt.show()
    return ax


def _metric_values(values):
    values = np.asarray(values, dtype=float)
    if values.ndim != 1 or len(values) == 0 or np.any(np.isinf(values)):
        raise ValueError("Metric values must be a non-empty 1D array without infinities.")
    return values


def plot_metric_over_time(
    metric_values,
    title="Streaming performance",
    ylabel="Accuracy",
    ax=None,
    save_path=None,
    show=False,
):
    values = _metric_values(metric_values)
    ax = _axes(ax)
    ax.plot(np.arange(1, len(values) + 1), values, marker="o", markersize=3)
    ax.set(title=title, xlabel="Chunk", ylabel=ylabel)
    ax.grid(alpha=0.25)
    return _finish(ax, save_path, show)


def compare_models(
    metric1,
    metric2,
    labels=("Model 1", "Model 2"),
    ax=None,
    save_path=None,
    show=False,
):
    if len(labels) != 2:
        raise ValueError("labels must contain exactly two model names.")
    first = _metric_values(metric1)
    second = _metric_values(metric2)
    ax = _axes(ax)
    ax.plot(np.arange(1, len(first) + 1), first, label=str(labels[0]))
    ax.plot(np.arange(1, len(second) + 1), second, label=str(labels[1]))
    ax.set(title="Buffer-size comparison", xlabel="Chunk", ylabel="Cumulative accuracy")
    ax.grid(alpha=0.25)
    ax.legend()
    return _finish(ax, save_path, show)


def plot_predictions_vs_ground_truth(
    y_true,
    y_pred,
    ax=None,
    save_path=None,
    show=False,
):
    true = _as_labels(y_true)
    pred = _as_labels(y_pred)
    if true.shape != pred.shape:
        raise ValueError("True and predicted labels must have the same shape.")
    if (true.dtype.kind in "biuf") != (pred.dtype.kind in "biuf"):
        raise ValueError("True and predicted label types must match.")
    labels = np.unique(np.concatenate((true, pred)))
    ax = _axes(ax)
    samples = np.arange(1, len(true) + 1)
    ax.scatter(samples, np.searchsorted(labels, true), marker="o", label="Ground truth", alpha=0.65)
    ax.scatter(samples, np.searchsorted(labels, pred), marker="x", label="Prediction")
    ax.set_yticks(np.arange(len(labels)), labels=[str(label) for label in labels])
    ax.set(title="Latest evaluated chunk", xlabel="Sample", ylabel="Class")
    ax.grid(alpha=0.25)
    ax.legend()
    return _finish(ax, save_path, show)
