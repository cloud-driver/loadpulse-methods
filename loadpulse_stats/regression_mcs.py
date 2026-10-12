"""Regression-criterion MCS [B2 §3.1.1, §3.2.1-3.2.4].

Forecast-loss MCS and IN-SAMPLE regression MCS answer different questions.
This module implements Gaussian linear regressions only, not arbitrary ML.
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np
from scipy.stats import chi2
from ._validation import array, integer, probability, model_names as validate_names
from .forecasting import _fit_ols
from .resampling import block_bootstrap_indices, hac_long_run_covariance
from .model_confidence_set import _complete_path

__all__ = ["equal_predictive_ability_test", "gaussian_deviance",
           "effective_degrees_of_freedom", "regression_model_confidence_set"]


def equal_predictive_ability_test(losses, *, max_lag, alpha=0.05):
    """Standalone B2 §3.1.1 quadratic-form EPA test with Bartlett HAC.

    Uses k-1 contrasts against the last model; statistic n*xbar' Sigma# xbar.
    NOT an MCS: a TQ rejection does not identify an inferior model and needs
    closed testing before elimination. Rank-zero/unsupported null directions
    are explicitly rejected as undefined input.
    """
    x = array(losses, "losses", 2, 3)
    if x.shape[1] < 2:
        raise ValueError("EPA test needs at least two model columns.")
    alpha = probability(alpha)
    contrast = x[:, :-1] - x[:, [-1]]
    covariance = hac_long_run_covariance(contrast, max_lag)
    mean = contrast.mean(axis=0)
    rank = int(np.linalg.matrix_rank(covariance))
    if rank == 0 or not np.allclose(covariance @ np.linalg.pinv(covariance) @ mean, mean):
        raise ValueError("Contrasts contain a deterministic nonzero or unidentifiable direction.")
    statistic = float(len(x) * mean @ np.linalg.pinv(covariance) @ mean)
    pvalue = float(chi2.sf(statistic, rank))
    return {"statistic": statistic, "df": rank, "pvalue": pvalue,
            "reject": pvalue < alpha, "alpha": alpha, "contrast_mean": mean,
            "long_run_covariance": covariance, "is_mcs": False}


def gaussian_deviance(actual, design, coefficients, error_variance):
    """B2 §3.2.2 Q=-2*Gaussian quasi-loglikelihood, excluding common constant.

    Q=n*log(sigma^2)+sum((y-X beta)^2)/sigma^2. No intercept is added here.
    All candidates must use the same y rows; scale/units must be consistent.
    """
    y, x = array(actual, "actual", 1), array(design, "design", 2)
    beta = array(coefficients, "coefficients", 1)
    var = float(error_variance)
    if x.shape != (len(y), len(beta)) or not np.isfinite(var) or var <= 0:
        raise ValueError("Need aligned y/X/beta and positive error variance.")
    return float(len(y) * np.log(var) + np.sum(np.square(y - x @ beta)) / var)


def effective_degrees_of_freedom(original_on_bootstrap_deviance, refitted_bootstrap_deviance):
    """B2 §3.2.3 B3 estimator: mean_b[Q(Z*b,theta_hat)-Q(Z*b,theta_hat*b)].

    Arrays:(B,k), each bootstrap fits every candidate on the SAME rows.
    Includes the variance parameter. No 'number of regressors' substitution.
    """
    fixed = array(original_on_bootstrap_deviance, "original_on_bootstrap_deviance", 2, 2)
    fitted = array(refitted_bootstrap_deviance, "refitted_bootstrap_deviance", 2, 2)
    if fixed.shape != fitted.shape or np.any(fixed < fitted - 1e-8):
        raise ValueError("Refitted likelihood should not be worse than the original fit on the resample.")
    return np.maximum(0, (fixed - fitted).mean(axis=0))


@dataclass(frozen=True)
class RegressionMCSResult:
    included: tuple[str, ...]
    excluded: tuple[str, ...]
    pvalues: dict[str, float]
    elimination_order: tuple[str, ...]
    history: tuple[dict, ...]
    criterion: str
    alpha: float
    effective_df: dict[str, float]
    criterion_values: dict[str, float]
    n_observations: int
    n_resamples: int
    block_length: int


def regression_model_confidence_set(actual, designs, *, model_names=None,
                                     criterion="KLIC", alpha=0.05,
                                     n_resamples=2000, block_length,
                                     random_state=None, add_intercept=True):
    """B2 §3.2 regression MCS with moving-block refits and coherent range rule.

    designs: sequence of (n,p_i) matrices, all target-aligned and full rank.
    criterion in KLIC, AIC, BIC, AIC_star, BIC_star.
    KLIC uses Q+k_star; remaining penalties follow B2 §3.2.3.
    Null bootstrap is EXACTLY eq.(3): Q_boot_refit+k_star-Q_original,
    not a naive bootstrap of a fixed loss matrix. Nested-model KLIC bootstrap
    can be conservative as explained in §3.2.4.
    """
    y = array(actual, "actual", 1, 3)
    matrices = [array(x, f"designs[{i}]", 2) for i, x in enumerate(designs)]
    if not matrices or any(len(x) != len(y) for x in matrices):
        raise ValueError("All design matrices need one row per common y observation.")
    if add_intercept:
        matrices = [np.column_stack((np.ones(len(y)), x)) for x in matrices]
    names = validate_names(model_names, len(matrices))
    alpha = probability(alpha)
    b = integer(n_resamples, "n_resamples", 99)
    integer(block_length, "block_length", 1, len(y))
    if criterion not in {"KLIC", "AIC", "BIC", "AIC_star", "BIC_star"}:
        raise ValueError("Unknown criterion.")
    betas, variances, original_q, dimensions = [], [], [], []
    for x in matrices:
        beta = _fit_ols(y, x)
        var = float(np.square(y - x @ beta).mean())
        original_q.append(gaussian_deviance(y, x, beta, var))
        betas.append(beta)
        variances.append(var)
        dimensions.append(x.shape[1] + 1)  # Include Gaussian variance parameter.
    q = np.asarray(original_q)
    q_fixed, q_refitted = np.empty((b, len(matrices))), np.empty((b, len(matrices)))
    rng = np.random.default_rng(random_state)
    for start in range(0, b, 64):
        stop = min(b, start + 64)
        indices = block_bootstrap_indices(len(y), stop - start, block_length,
                                           method="moving", random_state=rng)
        for row, idx in enumerate(indices, start=start):
            boot_y = y[idx]
            for j, x in enumerate(matrices):
                boot_x = x[idx]
                beta = _fit_ols(boot_y, boot_x)
                var = float(np.square(boot_y - boot_x @ beta).mean())
                q_refitted[row, j] = gaussian_deviance(boot_y, boot_x, beta, var)
                q_fixed[row, j] = gaussian_deviance(boot_y, boot_x, betas[j], variances[j])
    k_star = effective_degrees_of_freedom(q_fixed, q_refitted)
    penalties = {"KLIC": k_star, "AIC": 2 * np.asarray(dimensions),
                 "BIC": np.log(len(y)) * np.asarray(dimensions),
                 "AIC_star": 2 * k_star, "BIC_star": np.log(len(y)) * k_star}
    values = q + penalties[criterion]
    null = q_refitted + k_star - q
    def test_current(remaining):
        observed = float(np.ptp(values[remaining]))
        bootstrap_statistics = np.ptp(null[:, remaining], axis=1)
        worst = int(np.argmax(values[remaining]))
        return observed, bootstrap_statistics, worst
    included, excluded, pvalues, order, history = _complete_path(names, test_current, alpha)
    return RegressionMCSResult(included, excluded, pvalues, order, history, criterion,
                               alpha, dict(zip(names, map(float, k_star))),
                               dict(zip(names, map(float, values))), len(y), b, block_length)
