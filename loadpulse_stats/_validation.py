"""Internal validation utilities. Engineering code, not a paper formula."""
from __future__ import annotations

import numpy as np


def array(value, name, ndim=None, min_rows=1):
    result = np.asarray(value, dtype=float)
    if ndim is not None and result.ndim != ndim:
        raise ValueError(f"{name} must have {ndim} dimensions; got {result.shape}.")
    if result.ndim == 0 or result.shape[0] < min_rows or result.size == 0:
        raise ValueError(f"{name} is empty or has too few observations.")
    if not np.isfinite(result).all():
        raise ValueError(f"{name} contains NaN or infinity; align and clean data explicitly.")
    return result


def integer(value, name, minimum=1, maximum=None):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer.")
    if value < minimum or (maximum is not None and value > maximum):
        raise ValueError(f"{name} must be >= {minimum}" +
                         (f" and <= {maximum}." if maximum is not None else "."))
    return int(value)


def probability(value, name="alpha"):
    value = float(value)
    if not np.isfinite(value) or not 0 < value < 1:
        raise ValueError(f"{name} must lie strictly between 0 and 1.")
    return value


def model_names(names, k):
    values = tuple(f"model_{i}" for i in range(k)) if names is None else tuple(names)
    if len(values) != k or any(not isinstance(v, str) or not v for v in values):
        raise ValueError("model_names must contain one nonempty string per column.")
    if len(set(values)) != k:
        raise ValueError("model_names must be unique.")
    return values
