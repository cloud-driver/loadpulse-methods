"""Hansen, Lunde & Nason MCS [B2 §2.3 and §3.1.2].

TR is paired ONLY with eR=argmax_i max_j tij.
Tmax is paired ONLY with emax=argmax_i ti_dot.
Bootstrap loss differences are recentered under the EPA null hypothesis.
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np
from ._validation import array, integer, probability, model_names as validate_names
from .resampling import bootstrap_mean_samples

__all__ = ["mcs_statistics", "model_confidence_set", "MCSResult"]


def _standardize(numerator, variance, *, diagonal=False):
    num, var = np.asarray(numerator), np.asarray(variance)
    output = np.zeros_like(num)
    mask = np.ones(num.shape, dtype=bool)
    if diagonal:
        np.fill_diagonal(mask, False)
    if np.any(var[mask] <= 0) or not np.isfinite(var[mask]).all():
        raise ValueError("A loss differential has zero bootstrap variance. Remove identical "
                         "forecasts or constant-offset duplicates; B2 Assumption 2 excludes this.")
    output[mask] = num[mask] / np.sqrt(var[mask])
    return output


def mcs_statistics(losses, bootstrap_means, *, statistic="R"):
    """Compute observed statistic, bootstrap-null distribution, and removal index.

    losses:(n,k), bootstrap_means:(B,k) in matching column order.
    For R: se(dbar_ij)^2 = sample variance of paired bootstrap mean differences.
    For max: use deviations relative to the CURRENT set's mean.
    Return dict; worst_index refers to the current (not original) columns.
    """
    x, boot = array(losses, "losses", 2, 2), array(bootstrap_means, "bootstrap_means", 2, 2)
    if x.shape[1] != boot.shape[1] or x.shape[1] < 2:
        raise ValueError("Need >=2 aligned model columns in losses and bootstrap means.")
    mean = x.mean(axis=0)
    if statistic == "R":
        difference = mean[:, None] - mean[None, :]
        bd = boot[:, :, None] - boot[:, None, :]
        variance = bd.var(axis=0, ddof=1)
        t = _standardize(difference, variance, diagonal=True)
        inv_se = np.zeros_like(variance)
        mask = ~np.eye(len(mean), dtype=bool)
        inv_se[mask] = 1 / np.sqrt(variance[mask])
        null_t = (bd - difference[None, :, :]) * inv_se
        observed = float(np.max(np.abs(t)))
        null = np.max(np.abs(null_t), axis=(1, 2))
        worst = int(np.argmax(np.max(t, axis=1)))
    elif statistic == "max":
        difference = mean - mean.mean()
        bd = boot - boot.mean(axis=1, keepdims=True)
        variance = bd.var(axis=0, ddof=1)
        t = _standardize(difference, variance)
        null_t = (bd - difference) / np.sqrt(variance)
        observed = float(t.max())
        null = null_t.max(axis=1)
        worst = int(np.argmax(t))
    else:
        raise ValueError("statistic must be R or max.")
    return {"statistic": observed, "bootstrap_statistics": null,
            "worst_index": worst, "t_statistics": t, "mean_loss": mean}


@dataclass(frozen=True)
class MCSResult:
    included: tuple[str, ...]
    excluded: tuple[str, ...]
    pvalues: dict[str, float]
    elimination_order: tuple[str, ...]
    mean_losses: dict[str, float]
    history: tuple[dict, ...]
    alpha: float
    statistic: str
    n_observations: int
    n_resamples: int
    block_length: int
    bootstrap_method: str


def _complete_path(names, test_current, alpha):
    """Common sequence engine. Full path needed to obtain ALL model p-values."""
    remaining = list(range(len(names)))
    history, removed, pvalues = [], [], {}
    running_p = 0.0
    while len(remaining) > 1:
        observed, null, worst = test_current(remaining)
        p = float((1 + np.count_nonzero(null >= observed)) / (len(null) + 1))
        running_p = max(running_p, p)
        model = names[remaining[worst]]
        pvalues[model] = running_p
        history.append({"models": tuple(names[j] for j in remaining),
                        "test_statistic": observed, "test_pvalue": p,
                        "mcs_pvalue": running_p, "next_eliminated": model,
                        "removed_at_requested_alpha": running_p < alpha})
        removed.append(model)
        remaining.pop(worst)
    last = names[remaining[0]]
    pvalues[last] = 1.0
    order = tuple(removed + [last])
    included = tuple(name for name in names if pvalues[name] >= alpha)
    excluded = tuple(name for name in names if pvalues[name] < alpha)
    return included, excluded, pvalues, order, tuple(history)


def model_confidence_set(losses, *, model_names=None, alpha=0.05, statistic="R",
                          n_resamples=2000, block_length, bootstrap_method="stationary",
                          random_state=None):
    """B2 forecast MCS. Rows=time, columns=model, lower loss=better.

    Compute both exact coherent choices R/eR and max/emax. A SINGLE joint
    bootstrap is reused across all subsets. alpha=.05 means a 95% MCS.
    MCS p-values are running maxima along the COMPLETE elimination path.
    +1 correction prevents a Monte-Carlo p-value of exactly zero.
    Exact duplicate columns are collapsed and all duplicates share a p-value.
    Other zero-variance pair differences raise rather than fabricate inference.
    """
    x = array(losses, "losses", 2, 3)
    alpha = probability(alpha)
    b = integer(n_resamples, "n_resamples", 99)
    integer(block_length, "block_length", 1, len(x))
    names = validate_names(model_names, x.shape[1])
    if statistic not in {"R", "max"}:
        raise ValueError("statistic must be R or max.")
    representatives, groups = [], []
    for j in range(x.shape[1]):
        group = next((g for g, r in enumerate(representatives) if np.array_equal(x[:, j], x[:, r])), None)
        if group is None:
            representatives.append(j)
            groups.append([j])
        else:
            groups[group].append(j)
    reduced = x[:, representatives]
    # Remove common time shocks before bootstrap; equivalent for every pair
    # and stabilizes comparisons when the common component is very large.
    reduced = reduced - reduced[:, [0]]
    boot = bootstrap_mean_samples(reduced, b, block_length,
                                 method=bootstrap_method, random_state=random_state)
    reduced_names = tuple(names[j] for j in representatives)
    def test_current(remaining):
        result = mcs_statistics(reduced[:, remaining], boot[:, remaining], statistic=statistic)
        return result["statistic"], result["bootstrap_statistics"], result["worst_index"]
    _, _, reduced_p, reduced_order, history = _complete_path(reduced_names, test_current, alpha)
    pvalues = {names[j]: reduced_p[reduced_names[g]] for g, members in enumerate(groups) for j in members}
    order = tuple(names[j] for name in reduced_order
                  for g, representative_name in enumerate(reduced_names) if name == representative_name
                  for j in groups[g])
    included = tuple(name for name in names if pvalues[name] >= alpha)
    excluded = tuple(name for name in names if pvalues[name] < alpha)
    means = {name: float(x[:, j].mean()) for j, name in enumerate(names)}
    return MCSResult(included, excluded, pvalues, order, means, history, alpha,
                     statistic, len(x), b, block_length, bootstrap_method)
