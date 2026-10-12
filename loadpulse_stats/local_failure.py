"""LoadPulse extensions, NOT the two papers' original local-failure tests.

SOURCE_IDS = LP, B1(inspiration), B2(joint-resampling inspiration).
The bootstrap scan is CONDITIONAL on a separately fitted, frozen baseline.
It does not include baseline-estimation uncertainty or identify causality.
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np
from ._validation import array, integer, probability, model_names as validate_names
from .resampling import block_bootstrap_indices

__all__ = ["fit_calendar_baseline", "predict_calendar_baseline", "excess_loss",
           "window_excess_loss", "joint_window_scan", "consensus_failure",
           "merge_failure_events"]


def _keys(groups, n):
    if groups is None:
        return [()] * n
    values = np.asarray(groups, dtype=object)
    if values.ndim not in (1, 2) or len(values) != n:
        raise ValueError("calendar_groups must have shape (n,) or (n,p).")
    output = [tuple(row) for row in values] if values.ndim == 2 else [(v,) for v in values]
    for key in output:
        for item in key:
            if item is None or (isinstance(item, (float, np.floating)) and not np.isfinite(item)):
                raise ValueError("Calendar keys must not contain missing values.")
        try:
            hash(key)
        except TypeError as exc:
            raise ValueError("Calendar values must be hashable scalars.") from exc
    return output


@dataclass(frozen=True)
class CalendarBaseline:
    means: dict
    counts: dict
    global_mean: np.ndarray
    n_models: int
    min_count: int
    n_observations: int


def fit_calendar_baseline(calibration_losses, calendar_groups=None, *, min_count=20):
    """Fit conditional historical mean losses from OOS calibration predictions.

    Use data BEFORE the null-calibration and final-evaluation periods.
    Groups may be columns (hour,weekday,season), not measured future weather.
    No data are downloaded and no target-model training happens here.
    """
    x = array(calibration_losses, "calibration_losses", 2, 2)
    count_min = integer(min_count, "min_count", 2)
    keys = _keys(calendar_groups, len(x))
    positions = {}
    for i, key in enumerate(keys):
        positions.setdefault(key, []).append(i)
    means = {key: x[idx].mean(axis=0) for key, idx in positions.items()}
    counts = {key: len(idx) for key, idx in positions.items()}
    return CalendarBaseline(means, counts, x.mean(axis=0), x.shape[1], count_min, len(x))


def predict_calendar_baseline(baseline, calendar_groups=None, *, n_observations=None,
                             fallback="raise"):
    """Predict (n,k) expected loss. Unseen/sparse groups raise by default.

    fallback='global' is an explicit approximation using the training global
    mean; it must be documented in the analysis. It is never silently applied.
    """
    if fallback not in {"raise", "global"}:
        raise ValueError("fallback must be raise or global.")
    n = len(calendar_groups) if calendar_groups is not None else integer(n_observations, "n_observations")
    if n_observations is not None and n != n_observations:
        raise ValueError("Group count does not match n_observations.")
    keys = _keys(calendar_groups, n)
    output = np.empty((n, baseline.n_models))
    for t, key in enumerate(keys):
        if baseline.counts.get(key, 0) < baseline.min_count:
            if fallback == "raise":
                raise ValueError(f"Calendar group {key!r} is unseen or too sparse.")
            output[t] = baseline.global_mean
        else:
            output[t] = baseline.means[key]
    return output


def excess_loss(losses, expected_loss):
    """LP slide-11 definition: u[i,t]=L[i,t]-mu_hat[i](calendar[t])."""
    x, mean = array(losses, "losses", 2), array(expected_loss, "expected_loss", 2)
    if x.shape != mean.shape:
        raise ValueError("Expected losses must match (time,model) losses exactly.")
    return x - mean


def window_excess_loss(excess_losses, window_size, *, step=1):
    """LP slide-11 A_i(W); complete windows only, zero-based [start,stop).

    Returns starts, stops, means. No p-value or failure claim is made here.
    """
    x = array(excess_losses, "excess_losses", 2)
    size = integer(window_size, "window_size", 1, len(x))
    step = integer(step, "step", 1)
    starts = np.arange(0, len(x) - size + 1, step)
    stops = starts + size
    sums = np.vstack((np.zeros((1, x.shape[1])), np.cumsum(x, axis=0)))
    return starts, stops, (sums[stops] - sums[starts]) / size


def _scan_features(x, sizes, step):
    values, windows = [], []
    for size in sizes:
        starts, stops, means = window_excess_loss(x, size, step=step)
        values.append(means)
        windows.extend(zip(map(int, starts), map(int, stops), [size] * len(starts)))
    return np.vstack(values), tuple(windows)


@dataclass(frozen=True)
class WindowScanResult:
    windows: tuple[tuple[int, int, int], ...]
    mean_excess_loss: np.ndarray
    statistics: np.ndarray
    adjusted_pvalues: np.ndarray
    failure_mask: np.ndarray
    model_names: tuple[str, ...]
    alpha: float
    n_resamples: int
    block_length: int
    practical_threshold: np.ndarray
    inference_scope: str = "conditional_on_frozen_baseline"


def joint_window_scan(null_calibration_excess, evaluation_excess, *, window_sizes,
                     block_length, practical_threshold, alpha=0.05,
                     n_resamples=2000, step=1, model_names=None,
                     random_state=None, residual_stationarity_confirmed=False):
    """LP extension: one-sided max-statistic bootstrap over ALL models/windows.

    Null-calibration residuals and evaluation residuals MUST be from separate
    periods, using the SAME separately frozen baseline/model-column order.
    Calendar-adjusted residuals must have a stable joint null distribution;
    this function cannot establish it. It also cannot carry baseline-fitting
    uncertainty, data-revision uncertainty or model-selection uncertainty.
    These restrictions are exposed in inference_scope and the manual.
    No weather is used. Duplicate overlapping windows are not separate events.
    """
    if not residual_stationarity_confirmed:
        raise ValueError("Confirm joint null-residual stationarity before this conditional scan.")
    calibration = array(null_calibration_excess, "null_calibration_excess", 2, 3)
    actual = array(evaluation_excess, "evaluation_excess", 2, 3)
    if calibration.shape[1] != actual.shape[1]:
        raise ValueError("Calibration and evaluation model columns must be identical.")
    names = validate_names(model_names, actual.shape[1])
    alpha = probability(alpha)
    b = integer(n_resamples, "n_resamples", 99)
    integer(block_length, "block_length", 1, len(calibration))
    step = integer(step, "step", 1)
    sizes = tuple(window_sizes)
    if not sizes or len(set(sizes)) != len(sizes):
        raise ValueError("window_sizes must be nonempty and unique.")
    sizes = tuple(integer(s, "window_size", 2, len(actual)) for s in sizes)
    threshold = np.asarray(practical_threshold, dtype=float)
    if threshold.ndim == 0:
        threshold = np.repeat(threshold, actual.shape[1])
    if threshold.shape != (actual.shape[1],) or not np.isfinite(threshold).all() or np.any(threshold < 0):
        raise ValueError("practical_threshold must be scalar or (k,), finite/nonnegative.")
    observed, windows = _scan_features(actual, sizes, step)
    # Recenter to impose no deterioration; baseline remains frozen throughout.
    residuals = calibration - calibration.mean(axis=0)
    # Estimate scale on calibration only, using n-length jointly resampled paths.
    # Save per-feature mean/square sums and per-replicate maxima only: bounded RAM.
    seeds = np.random.default_rng(random_state).integers(0, 2**63 - 1, size=b)
    sums, sums2 = np.zeros_like(observed), np.zeros_like(observed)
    for seed in seeds:
        idx = block_bootstrap_indices(len(residuals), 1, block_length,
                                      sample_size=len(actual), random_state=int(seed))[0]
        features, _ = _scan_features(residuals[idx], sizes, step)
        sums += features
        sums2 += features * features
    variance = np.maximum(0, (sums2 - sums * sums / b) / (b - 1))
    if np.any(variance <= 0):
        raise ValueError("At least one scan statistic has zero null-bootstrap variance.")
    sd = np.sqrt(variance)
    statistics = observed / sd
    null_max = np.empty(b)
    for j, seed in enumerate(seeds):
        idx = block_bootstrap_indices(len(residuals), 1, block_length,
                                      sample_size=len(actual), random_state=int(seed))[0]
        features, _ = _scan_features(residuals[idx], sizes, step)
        null_max[j] = (features / sd).max()
    ordered = np.sort(null_max)
    # single-step maxT adjustment, including every requested window/model/scale.
    counts = b - np.searchsorted(ordered, statistics, side="left")
    adjusted = (1 + counts) / (b + 1)
    failures = (adjusted < alpha) & (observed > threshold) & (observed > 0)
    return WindowScanResult(windows, observed, statistics, adjusted, failures,
                            names, alpha, b, block_length, threshold)


def consensus_failure(failure_mask, *, minimum_fraction):
    """Descriptive fraction threshold, NOT independent model votes or a p-value.

    Returns (per-window consensus mask, fraction). Model family selection and
    minimum_fraction must be fixed before inspecting final outcomes.
    """
    x = np.asarray(failure_mask)
    if x.ndim != 2 or x.size == 0 or x.dtype != bool:
        raise ValueError("failure_mask must be a nonempty (windows,models) boolean array.")
    fraction = float(minimum_fraction)
    if not np.isfinite(fraction) or not 0 < fraction <= 1:
        raise ValueError("minimum_fraction must lie in (0,1].")
    proportions = x.mean(axis=1)
    return proportions >= fraction, proportions


def merge_failure_events(windows, selected, *, max_gap=0):
    """Merge selected [start,stop) windows into descriptive index intervals.

    Returns list[(start,stop)]. Touching/overlapping windows merge; max_gap is
    measured in rows, not hours. Boundaries are window coverage, not a proven
    physical onset/recovery. Sorts windows so multiple scales can be combined.
    """
    flags = np.asarray(selected)
    if flags.dtype != bool or flags.ndim != 1 or len(flags) != len(windows):
        raise ValueError("selected must be a boolean vector matching windows.")
    gap = integer(max_gap, "max_gap", 0)
    chosen = []
    for window, hit in zip(windows, flags):
        if len(window) < 2:
            raise ValueError("Each window needs a start and stop.")
        start, stop = integer(window[0], "start", 0), integer(window[1], "stop", 1)
        if stop <= start:
            raise ValueError("Windows must have stop>start.")
        if hit:
            chosen.append((start, stop))
    events = []
    for start, stop in sorted(chosen):
        if events and start <= events[-1][1] + gap:
            events[-1] = (events[-1][0], max(events[-1][1], stop))
        else:
            events.append((start, stop))
    return events
