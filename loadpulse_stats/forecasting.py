"""Forecasting-scheme utilities [B1 §3.1]. OLS helpers are engineering glue.

X[t] predicts y[t], and MUST already be available at origin t-horizon.
Never pass contemporaneous measured weather as a future predictor.
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np
from ._validation import array, integer

__all__ = ["forecasting_splits", "rolling_origin_ols"]


def forecasting_splits(n_observations, initial_train_size, *, scheme="rolling", horizon=1):
    """Yield (training_indices, target_index) with an h-step information embargo.

    At origin o, only targets y[0:o+1] are available. Initial origin is m-1.
    The forecast target is o+h. Rolling keeps m eligible target rows; fixed
    always uses first m rows; recursive uses all eligible rows.
    """
    n = integer(n_observations, "n_observations", 3)
    m = integer(initial_train_size, "initial_train_size", 2, n - 1)
    h = integer(horizon, "horizon", 1, n - m)
    if scheme not in {"fixed", "rolling", "recursive"}:
        raise ValueError("scheme must be fixed, rolling, or recursive.")
    for origin in range(m - 1, n - h):
        start = origin - m + 1 if scheme == "rolling" else 0
        stop = m if scheme == "fixed" else origin + 1
        yield np.arange(start, stop), origin + h


@dataclass(frozen=True)
class ForecastResult:
    target_indices: np.ndarray
    predictions: np.ndarray
    actual: np.ndarray
    out_of_sample_loss: np.ndarray
    in_sample_mean_loss: np.ndarray
    coefficients: np.ndarray
    initial_in_sample_losses: np.ndarray
    intermediate_losses: np.ndarray
    initial_train_size: int
    horizon: int
    scheme: str

    @property
    def full_losses(self):
        """Algorithm-1 order: initial m, gap h-1, then n OOS losses."""
        return np.concatenate((self.initial_in_sample_losses,
                               self.intermediate_losses, self.out_of_sample_loss))


def _fit_ols(y, x):
    beta, _, rank, _ = np.linalg.lstsq(x, y, rcond=None)
    if rank != x.shape[1] or len(y) <= x.shape[1]:
        raise ValueError("OLS needs a full-rank design and more rows than coefficients.")
    return beta


def rolling_origin_ols(actual, features, initial_train_size, *, scheme="rolling",
                       horizon=1, add_intercept=True):
    """Leakage-conscious direct forecasts, using OLS + squared loss [B1].

    Supply target-aligned features known h steps before their target.
    Initial m rows are eligible targets, NOT m raw-origin rows of a lagged
    design. Initial NaNs from creating lags must be removed BEFORE this call.
    The same beta is used for OOS loss and its matching training mean.
    """
    y, x = array(actual, "actual", 1), array(features, "features", 2)
    if len(y) != len(x):
        raise ValueError("actual and features must have equal row counts.")
    if add_intercept:
        x = np.column_stack((np.ones(len(x)), x))
    splits = list(forecasting_splits(len(y), initial_train_size,
                                    scheme=scheme, horizon=horizon))
    first_beta = _fit_ols(y[:initial_train_size], x[:initial_train_size])
    initial = np.square(y[:initial_train_size] - x[:initial_train_size] @ first_beta)
    gap = []
    for target in range(initial_train_size, initial_train_size + horizon - 1):
        if scheme == "fixed":
            gap.append(0.0)  # Explicit convention in B1 Algorithm 1 footnote 3.
        else:
            start = target - initial_train_size + 1 if scheme == "rolling" else 0
            beta = _fit_ols(y[start:target + 1], x[start:target + 1])
            gap.append(float((y[target] - x[target] @ beta) ** 2))
    predictions, losses, means, betas, targets = [], [], [], [], []
    for train, target in splits:
        beta = _fit_ols(y[train], x[train])
        prediction = x[target] @ beta
        predictions.append(prediction)
        losses.append((y[target] - prediction) ** 2)
        means.append(np.square(y[train] - x[train] @ beta).mean())
        betas.append(beta)
        targets.append(target)
    target_indices = np.asarray(targets)
    return ForecastResult(target_indices, np.asarray(predictions), y[target_indices],
                          np.asarray(losses), np.asarray(means), np.asarray(betas),
                          initial, np.asarray(gap), initial_train_size, horizon, scheme)
