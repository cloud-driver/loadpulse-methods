"""Error and loss definitions [B1 §3.1; B2 §2]. Arrays must be aligned.

SOURCE_IDS = B1, B2. This is an independent implementation, not author code.
"""
from __future__ import annotations

import numpy as np
from ._validation import array

__all__ = ["forecast_error", "squared_loss", "absolute_loss", "loss_matrix",
           "loss_differentials"]


def forecast_error(actual, predicted):
    """Return actual - predicted; equal shapes required (no implicit broadcasting)."""
    y, p = array(actual, "actual"), array(predicted, "predicted")
    if y.shape != p.shape:
        raise ValueError("actual and predicted must have identical shapes.")
    return y - p


def squared_loss(actual, predicted):
    """Return elementwise squared forecast errors, in squared target units."""
    with np.errstate(over="raise", invalid="raise"):
        return np.square(forecast_error(actual, predicted))


def absolute_loss(actual, predicted):
    """Return absolute forecast errors, in target units."""
    return np.abs(forecast_error(actual, predicted))


def loss_matrix(actual, predictions, loss="squared"):
    """actual:(n,), predictions:(n,k) -> losses:(n,k); common targets/horizons.

    loss may be 'squared', 'absolute', or callable(actual_matrix, predictions).
    Custom criteria may be signed, provided smaller consistently means better.
    """
    y = array(actual, "actual", 1)
    p = array(predictions, "predictions", 2)
    if p.shape[0] != len(y):
        raise ValueError("Predictions must have exactly one row per actual target.")
    target = np.broadcast_to(y[:, None], p.shape)
    functions = {"squared": squared_loss, "absolute": absolute_loss}
    if callable(loss):
        function = loss
    elif loss in functions:
        function = functions[loss]
    else:
        raise ValueError("loss must be squared, absolute, or a callable.")
    values = array(function(target, p), "computed loss", 2)
    if values.shape != p.shape:
        raise ValueError("Custom loss must return shape (n,k).")
    return values


def loss_differentials(losses):
    """(n,k) -> (n,k,k), entry [t,i,j] = L[t,i]-L[t,j] [B2 §3.1].

    Quadratic in k; MCS itself avoids allocating this full time-by-pair tensor.
    """
    x = array(losses, "losses", 2)
    return x[:, :, None] - x[:, None, :]
