"""Time-dependent resampling [B2 §3], and Bartlett HAC [NW1987].

SOURCE_IDS = B2, NW1987, PR1994. Joint rows preserve cross-model pairing.
"""
from __future__ import annotations

import numpy as np
from ._validation import array, integer

__all__ = ["hac_long_run_covariance", "block_bootstrap_indices",
           "bootstrap_mean_samples"]


def hac_long_run_covariance(values, max_lag, *, center=True):
    """Bartlett/Newey-West covariance of sqrt(n)*mean(values).

    values:(n,) or (n,p), max_lag: explicit integer 0..n-1.
    Denominator is n at EVERY lag; Bartlett weight is 1-j/(max_lag+1).
    Set center=False ONLY for an already correctly centered/influence series.
    Returns float for a vector, (p,p) array for a matrix.
    """
    x = array(values, "values", min_rows=2)
    if x.ndim not in (1, 2):
        raise ValueError("values must be a vector or matrix.")
    was_vector = x.ndim == 1
    x = x[:, None] if was_vector else x.copy()
    n = len(x)
    lag = integer(max_lag, "max_lag", 0, n - 1)
    if center:
        x = x - x.mean(axis=0)
    covariance = x.T @ x / n
    for j in range(1, lag + 1):
        gamma = x[j:].T @ x[:-j] / n
        covariance += (1 - j / (lag + 1)) * (gamma + gamma.T)
    covariance = (covariance + covariance.T) / 2
    return float(covariance[0, 0]) if was_vector else covariance


def block_bootstrap_indices(n, n_resamples, block_length, *, method="stationary",
                            sample_size=None, random_state=None):
    """Return (B,sample_size) shared row indices.

    stationary: geometric block lengths, E[length]=block_length [PR1994].
    circular: fixed-length blocks with wraparound.
    moving: fixed-length overlapping blocks without wraparound [B2 §3.2.3].
    This utility does not determine an optimal block length for your data.
    """
    n = integer(n, "n", 2)
    b = integer(n_resamples, "n_resamples", 1)
    length = integer(block_length, "block_length", 1, n)
    size = n if sample_size is None else integer(sample_size, "sample_size", 1)
    if method not in {"stationary", "circular", "moving"}:
        raise ValueError("method must be stationary, circular, or moving.")
    rng = np.random.default_rng(random_state)
    indices = np.empty((b, size), dtype=np.int64)
    if method == "stationary":
        indices[:, 0] = rng.integers(n, size=b)
        for t in range(1, size):
            starts = rng.integers(n, size=b)
            restart = rng.random(b) < 1 / length
            indices[:, t] = np.where(restart, starts, (indices[:, t - 1] + 1) % n)
    else:
        blocks = (size + length - 1) // length
        high = n if method == "circular" else n - length + 1
        starts = rng.integers(high, size=(b, blocks, 1))
        indices[:] = ((starts + np.arange(length)) % n).reshape(b, -1)[:, :size]
    return indices


def bootstrap_mean_samples(values, n_resamples, block_length, *, method="stationary",
                           random_state=None, batch_size=64):
    """Joint block-bootstrap column means (B,p), generated in bounded batches.

    No null centering is imposed here; callers decide the null model.
    """
    x = array(values, "values", 2, 2)
    b = integer(n_resamples, "n_resamples", 2)
    batch_size = integer(batch_size, "batch_size", 1)
    rng = np.random.default_rng(random_state)
    output = np.empty((b, x.shape[1]))
    for start in range(0, b, batch_size):
        stop = min(b, start + batch_size)
        indices = block_bootstrap_indices(len(x), stop - start, block_length,
                                          method=method, random_state=rng)
        output[start:stop] = x[indices].mean(axis=1)
    return output
