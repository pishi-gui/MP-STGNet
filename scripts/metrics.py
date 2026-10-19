"""Shared raw-count metrics and baseline prediction observation."""

import numpy as np
from contextlib import contextmanager
from contextvars import ContextVar

_prediction_observer = ContextVar("prediction_observer", default=None)


@contextmanager
def observe_predictions(callback):
    """Observe legacy evaluation pairs without changing predictions or training."""
    token = _prediction_observer.set(callback)
    try:
        yield
    finally:
        _prediction_observer.reset(token)


def _observe(truth, prediction):
    callback = _prediction_observer.get()
    if callback is not None:
        callback(truth, prediction)


def _paired(truth, prediction):
    truth = np.asarray(truth, dtype=np.float64)
    prediction = np.asarray(prediction, dtype=np.float64)
    if truth.shape != prediction.shape:
        raise ValueError(f"Metric shapes differ: {truth.shape} vs {prediction.shape}")
    if not np.isfinite(truth).all() or not np.isfinite(prediction).all():
        raise ValueError("Metrics require finite truth and prediction arrays")
    return truth.reshape(-1), prediction.reshape(-1)


def compute_metrics(truth, prediction, mask="ALL"):
    """MAE/RMSE/COR over nodes, without implicit clipping or broadcasting.

    NZ selects truth > 0. Empty selections and undefined correlations return NaN.
    Callers must explicitly apply any prediction-clipping policy beforehand.
    """
    if mask not in {"ALL", "NZ"}:
        raise ValueError("mask must be ALL or NZ")
    truth, prediction = _paired(truth, prediction)
    if mask == "NZ":
        keep = truth > 0
        truth, prediction = truth[keep], prediction[keep]
    if truth.size == 0:
        return dict.fromkeys(("MAE", "RMSE", "COR"), float("nan"))
    error = prediction - truth
    cor = (
        float(np.corrcoef(truth, prediction)[0, 1])
        if truth.size > 1 and truth.std() > 0 and prediction.std() > 0
        else float("nan")
    )
    return {
        "MAE": float(np.abs(error).mean()),
        "RMSE": float(np.sqrt(np.square(error).mean())),
        "COR": cor,
    }


def mape_floor1(truth, prediction):
    """Legacy percentage error with denominator max(abs(truth), 1), not standard MAPE."""
    truth, prediction = _paired(truth, prediction)
    if not truth.size:
        return float("nan")
    return float((np.abs(prediction - truth) / np.maximum(np.abs(truth), 1.0)).mean() * 100)


def finite_mean(values, default=float("nan")):
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    return float(values.mean()) if values.size else float(default)


def legacy_metric_all(truth, prediction):
    """Legacy tuple order MAE/COR/RMSE/Zero_FP, with explicit nonnegative clipping."""
    truth, prediction = _paired(truth, prediction)
    _observe(truth, prediction)
    prediction = np.maximum(prediction, 0)
    result = compute_metrics(truth, prediction)
    zero = truth <= 0
    zero_fp = float(prediction[zero].mean()) if zero.any() else float("nan")
    return result["MAE"], result["COR"], result["RMSE"], zero_fp


def legacy_metric_nz(truth, prediction):
    result = compute_metrics(truth, np.maximum(prediction, 0), "NZ")
    return result["MAE"], result["COR"], result["RMSE"]


def legacy_metric_triplet(truth, prediction):
    _observe(truth, prediction)
    prediction = np.maximum(prediction, 0)
    result = compute_metrics(truth, prediction)
    return result["MAE"], mape_floor1(truth, prediction), result["RMSE"]


def legacy_pearson(truth, prediction, eps=1e-8):
    # eps is retained for legacy callers; undefined correlations remain NaN.
    return compute_metrics(truth, np.maximum(prediction, 0))["COR"]
