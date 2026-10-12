"""Giacomini & Rossi forecast breakdown [B1, ECB WP 638, 2006].

Independent implementation of §3.3, Algorithm 1 / Theorem 2 / Corollaries
3-4 (§3.5), linear-model overfit correction (§5), and predictive regression
point estimates (§6). See SOURCES.md for version and applicability.
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np
from scipy.stats import chi2, norm
from ._validation import array, integer, probability
from .resampling import hac_long_run_covariance

__all__ = ["surprise_loss", "forecast_weights", "weighted_breakdown_variance",
           "scheme_variance_factor", "stationary_breakdown_variance",
           "linear_overfit_correction", "iid_overfit_correction",
           "forecast_breakdown_test", "fit_surprise_regression",
           "predict_surprise_regression", "predictive_wald_test",
           "predictive_cross_factor", "predictive_covariance_from_moments",
           "surprise_regression_lower_bound"]


def surprise_loss(out_of_sample_loss, in_sample_mean_loss):
    """B1 equation (1). in_sample_mean_loss is scalar or exactly shape (n,).

    Each training mean must be evaluated using the SAME fitted parameters
    that produce its corresponding OOS forecast, not a different full fit.
    """
    losses = array(out_of_sample_loss, "out_of_sample_loss", 1)
    baseline = np.asarray(in_sample_mean_loss, dtype=float)
    if baseline.ndim > 1 or (baseline.ndim == 1 and baseline.shape != losses.shape):
        raise ValueError("in_sample_mean_loss must be scalar or match losses exactly.")
    if not np.isfinite(baseline).all():
        raise ValueError("in_sample_mean_loss must be finite.")
    return losses - baseline


def forecast_weights(m, n, *, scheme, horizon=1, derivative_times_influence=None):
    """Algorithm-1 weights (w_L, w_h) without fragile case-by-case formulas.

    T=m+n+h-1. For origin o=m-1,...,m+n-2, add +1 at o+h and
    -1 / training_size at each estimation-window index.
    derivative_times_influence:(n,q) is D_(o+h) @ B_o, if uncertainty matters.
    w_h:(T,q), or None. The weights operate on B1's full loss/score series,
    not on the already differenced surprise-loss series.
    """
    m, n, h = integer(m, "m", 2), integer(n, "n", 2), integer(horizon, "horizon", 1)
    if scheme not in {"fixed", "rolling", "recursive"}:
        raise ValueError("scheme must be fixed, rolling, or recursive.")
    total = m + n + h - 1
    w_loss = np.zeros(total)
    products = None
    if derivative_times_influence is not None:
        products = array(derivative_times_influence, "derivative_times_influence", 2)
        if len(products) != n:
            raise ValueError("D @ B products need one row per forecast.")
    w_score = None if products is None else np.zeros((total, products.shape[1]))
    for j, origin in enumerate(range(m - 1, m + n - 1)):
        start = origin - m + 1 if scheme == "rolling" else 0
        stop = m if scheme == "fixed" else origin + 1
        w_loss[origin + h] += 1
        w_loss[start:stop] -= 1 / (stop - start)
        if w_score is not None:
            w_score[start:stop] += products[j] / (stop - start)
    return w_loss, w_score


def weighted_breakdown_variance(full_losses, m, n, *, scheme, max_lag,
                               horizon=1, estimation_irrelevant=False,
                               scores=None, surprise_gradients=None,
                               influence_matrices=None):
    """B1 Algorithm 1, equations (5)-(9), Theorem 2: estimate sigma^2_m,n.

    full_losses length T=m+n+h-1 must be ordered as in Algorithm 1.
    equal estimation/evaluation loss: explicitly set estimation_irrelevant=True.
    General branch needs scores:(T,q), surprise_gradients D:(n,k), and
    influence_matrices B:(n,k,q). The last h score rows MUST be zero.
    The combined influence is w_L*(L-mean(L)) + row_sum(w_h*scores).
    Apply Bartlett HAC WITHOUT a second centering and multiply by T/n.
    Users remain responsible for A1-A7 and the influence representation A3.
    """
    full = array(full_losses, "full_losses", 1, 2)
    wl, _ = forecast_weights(m, n, scheme=scheme, horizon=horizon)
    if len(full) != len(wl):
        raise ValueError("full_losses length must equal m+n+horizon-1.")
    influence = wl * (full - full.mean())
    if estimation_irrelevant:
        if any(v is not None for v in (scores, surprise_gradients, influence_matrices)):
            raise ValueError("Do not supply general score inputs in equal-loss branch.")
    else:
        if any(v is None for v in (scores, surprise_gradients, influence_matrices)):
            raise ValueError("General variance needs scores, D gradients, and B matrices; "
                             "or explicitly justify estimation_irrelevant=True.")
        s = array(scores, "scores", 2)
        d = array(surprise_gradients, "surprise_gradients", 2)
        b = array(influence_matrices, "influence_matrices", 3)
        if s.shape[0] != len(full) or d.shape[0] != n or b.shape != (n, d.shape[1], s.shape[1]):
            raise ValueError("Expected scores:(T,q), gradients:(n,k), B:(n,k,q).")
        if np.any(s[-horizon:] != 0) or (scheme == "fixed" and np.any(s[m:] != 0)):
            raise ValueError("Score padding must match B1 Algorithm 1 footnote 4.")
        products = np.einsum("nk,nkq->nq", d, b)
        _, wh = forecast_weights(m, n, scheme=scheme, horizon=horizon,
                                 derivative_times_influence=products)
        influence += np.sum(wh * s, axis=1)
    result = len(full) / n * hac_long_run_covariance(influence, max_lag, center=False)
    if not np.isfinite(result) or result <= 0:
        raise ValueError("Estimated variance is nonpositive; inference is undefined.")
    return float(result)


def scheme_variance_factor(m, n, scheme):
    """B1 Corollary 4 equation (10), under covariance stationarity/equal loss."""
    m, n = integer(m, "m", 2), integer(n, "n", 2)
    if scheme == "fixed":
        return 1 + n / m
    if scheme == "rolling":
        return 1 - (n / m) ** 2 / 3 if n < m else 2 * m / (3 * n)
    if scheme == "recursive":
        return 1.0
    raise ValueError("scheme must be fixed, rolling, or recursive.")


def stationary_breakdown_variance(out_of_sample_losses, m, *, scheme, max_lag,
                                  assumptions_confirmed=False):
    """sigma^2=lambda * HAC(OOS losses), B1 Corollary 4 / equation (10).

    Requires equal estimation/evaluation loss (or the Theorem 2(a) condition),
    covariance-stationary loss under H0, A1-A7, and a fixed horizon.
    This function does NOT test these assumptions from the data.
    """
    if not assumptions_confirmed:
        raise ValueError("Confirm Corollary-4 assumptions explicitly before using this shortcut.")
    x = array(out_of_sample_losses, "out_of_sample_losses", 1, 2)
    result = scheme_variance_factor(m, len(x), scheme) * hac_long_run_covariance(x, max_lag)
    if result <= 0 or not np.isfinite(result):
        raise ValueError("Estimated variance is nonpositive; inference is undefined.")
    return float(result)


def _gamma(n, m, scheme):
    integer(n, "n", 2)
    integer(m, "m", 2)
    if scheme in {"fixed", "rolling"}:
        return np.sqrt(n) / m
    if scheme == "recursive":
        return np.log1p(n / m) / np.sqrt(n)
    raise ValueError("scheme must be fixed, rolling, or recursive.")


def linear_overfit_correction(n, m, *, scheme, mean_xtx, parameter_asymptotic_covariance):
    """B1 §5 eq.(13): c=2*gamma*trace(mean(X'X)*asyvar(sqrt(T)*beta)).

    Applicable to linear OLS / squared loss with stationary regressors.
    Covariance is NOT Cov(beta); multiply a full-sample Cov(beta) by T first.
    """
    q = array(mean_xtx, "mean_xtx", 2)
    v = array(parameter_asymptotic_covariance, "parameter_asymptotic_covariance", 2)
    if q.shape != v.shape or q.shape[0] != q.shape[1]:
        raise ValueError("Both matrices must have identical square shape.")
    for name, matrix in (("mean_xtx", q), ("parameter covariance", v)):
        if not np.allclose(matrix, matrix.T) or np.linalg.eigvalsh(matrix).min() < -1e-10:
            raise ValueError(f"{name} must be symmetric positive semidefinite.")
    return float(2 * _gamma(n, m, scheme) * np.trace(q @ v))


def iid_overfit_correction(n, m, *, scheme, error_variance, n_parameters):
    """B1 eq.(14): c=2*gamma*sigma_error^2*k. Include intercept in k.

    Use only when asymptotic OLS covariance is sigma^2*(E[XX'])^-1.
    """
    k = integer(n_parameters, "n_parameters", 1)
    variance = float(error_variance)
    if not np.isfinite(variance) or variance < 0:
        raise ValueError("error_variance must be finite and nonnegative.")
    return float(2 * _gamma(n, m, scheme) * variance * k)


@dataclass(frozen=True)
class BreakdownResult:
    n: int
    mean_surprise_loss: float
    asymptotic_variance: float
    correction: float
    statistic: float
    pvalue: float
    alpha: float
    reject: bool
    alternative: str
    variance_source: str


def forecast_breakdown_test(surprise_losses, *, asymptotic_variance,
                             variance_source, alpha=0.05, correction=0.0,
                             alternative="greater"):
    """B1 eq.(4)/(13): (sqrt(n)*mean(SL)-c)/sqrt(sigma^2).

    Variance is REQUIRED; no naive sample-standard-deviation fallback.
    variance_source records the caller's estimator/assumptions for auditing.
    Reject uses strict p<alpha; two-sided testing describes change, not
    necessarily deterioration. c is numerator-scale correction, not mean-scale.
    """
    sl = array(surprise_losses, "surprise_losses", 1, 2)
    alpha = probability(alpha)
    variance, c = float(asymptotic_variance), float(correction)
    if not np.isfinite(variance) or variance <= 0 or not np.isfinite(c) or c < 0:
        raise ValueError("variance must be positive and correction finite/nonnegative.")
    if not isinstance(variance_source, str) or not variance_source.strip():
        raise ValueError("Record the variance estimator in variance_source.")
    statistic = float((np.sqrt(len(sl)) * sl.mean() - c) / np.sqrt(variance))
    if alternative == "greater":
        pvalue = float(norm.sf(statistic))
    elif alternative == "two-sided":
        pvalue = float(2 * norm.sf(abs(statistic)))
    else:
        raise ValueError("alternative must be greater or two-sided.")
    return BreakdownResult(len(sl), float(sl.mean()), variance, c, statistic,
                           pvalue, alpha, pvalue < alpha, alternative, variance_source)


@dataclass(frozen=True)
class SurpriseRegression:
    coefficients: np.ndarray
    fitted_values: np.ndarray
    residuals: np.ndarray
    n: int
    add_intercept: bool
    n_features: int


def fit_surprise_regression(surprise_losses, predictors, *, add_intercept=True):
    """B1 §6 eq.(15), OLS POINT estimates. Predictors must be known at origin.

    No automatic Wald p-value: B1 Proposition 7 requires a scheme-corrected
    covariance. Generic OLS or residual-only HAC would not reproduce it.
    """
    y, x = array(surprise_losses, "surprise_losses", 1, 3), array(predictors, "predictors", 2)
    if len(y) != len(x):
        raise ValueError("predictors and surprise losses must be row-aligned.")
    features = x.shape[1]
    if add_intercept:
        x = np.column_stack((np.ones(len(x)), x))
    coefficients, _, rank, _ = np.linalg.lstsq(x, y, rcond=None)
    if rank < x.shape[1] or len(x) <= x.shape[1]:
        raise ValueError("Regression must have a full-rank design with residual degrees of freedom.")
    fitted = x @ coefficients
    return SurpriseRegression(coefficients, fitted, y - fitted, len(y), add_intercept, features)


def predict_surprise_regression(fit, predictors):
    """Out-of-sample predictions from a previously fitted §6 loss regression."""
    x = array(predictors, "predictors", 2)
    if x.shape[1] != fit.n_features:
        raise ValueError("predictors must use the same feature columns as the fit.")
    if fit.add_intercept:
        x = np.column_stack((np.ones(len(x)), x))
    return x @ fit.coefficients


def predictive_wald_test(fit, asymptotic_covariance, *, covariance_source, alpha=0.05):
    """B1 §6 Wald n*delta' Omega^-1 delta. Caller supplies Proposition-7 Omega.

    Omega is covariance of sqrt(n)*delta, not covariance of delta itself.
    This low-level function DOES NOT construct Omega or verify stationarity.
    """
    alpha = probability(alpha)
    omega = array(asymptotic_covariance, "asymptotic_covariance", 2)
    k = len(fit.coefficients)
    if omega.shape != (k, k) or not np.allclose(omega, omega.T):
        raise ValueError("Omega must be a symmetric (k,k) matrix.")
    if np.linalg.eigvalsh(omega).min() <= 0:
        raise ValueError("Omega must be positive definite; no silent pseudoinverse.")
    if not isinstance(covariance_source, str) or not covariance_source.strip():
        raise ValueError("Record the covariance estimator in covariance_source.")
    statistic = float(fit.n * fit.coefficients @ np.linalg.solve(omega, fit.coefficients))
    pvalue = float(chi2.sf(statistic, k))
    return {"statistic": statistic, "df": k, "pvalue": pvalue,
            "reject": pvalue < alpha, "alpha": alpha, "covariance_source": covariance_source}


def predictive_cross_factor(m, n, scheme):
    """B1 Proposition 7: Lambda cross-covariance adjustment by scheme."""
    m, n = integer(m, "m", 2), integer(n, "n", 2)
    ratio = n / m
    if scheme == "fixed":
        return 1.0
    if scheme == "rolling":
        return 1 - ratio / 2 if n <= m else 1 / (2 * ratio)
    if scheme == "recursive":
        return np.log1p(ratio) / ratio
    raise ValueError("scheme must be fixed, rolling, or recursive.")


def predictive_covariance_from_moments(predictor_mean, predictor_moment,
                                      score_long_run_covariance, loss_score_cross,
                                      breakdown_variance, m, n, *, scheme):
    """B1 Proposition 7 eq.(16), LOW-LEVEL assembly of Omega.

    zbar:(p,), S_zz:(p,p), S_zL,zL:(p,p), S_zL,L:(p,), sigma^2:scalar.
    Caller must construct moments under Proposition-7 fourth-order stationarity
    and verify B3. This API deliberately does not guess the S_zz estimator:
    the public WP states a raw moment in B3 and a HAC expression after (16).
    That version ambiguity is documented; verify the final article before
    choosing a general autocorrelated-regressor moment estimator.
    """
    z = array(predictor_mean, "predictor_mean", 1)
    moment = array(predictor_moment, "predictor_moment", 2)
    score_cov = array(score_long_run_covariance, "score_long_run_covariance", 2)
    cross = array(loss_score_cross, "loss_score_cross", 1)
    p = len(z)
    if moment.shape != (p, p) or score_cov.shape != (p, p) or cross.shape != (p,):
        raise ValueError("Moments must have matching predictor dimensions.")
    if not np.allclose(moment, moment.T) or np.linalg.eigvalsh(moment).min() <= 0:
        raise ValueError("predictor_moment must be symmetric positive definite.")
    if not np.allclose(score_cov, score_cov.T):
        raise ValueError("score covariance must be symmetric.")
    variance = float(breakdown_variance)
    if not np.isfinite(variance) or variance <= 0:
        raise ValueError("breakdown_variance must be positive.")
    inv = np.linalg.inv(moment)
    transform = np.zeros((p + 1, p + 1))
    transform[0, 0], transform[0, 1:], transform[1:, 1:] = 1, -z @ inv, inv
    joint = np.zeros_like(transform)
    joint[0, 0], joint[1:, 1:] = variance, score_cov
    adjusted_cross = predictive_cross_factor(m, n, scheme) * cross
    joint[0, 1:], joint[1:, 0] = adjusted_cross, adjusted_cross
    if np.linalg.eigvalsh(joint).min() < -1e-10:
        raise ValueError("Supplied adjusted moments are not a valid positive semidefinite covariance.")
    omega = transform @ joint @ transform.T
    return (omega + omega.T) / 2


def surprise_regression_lower_bound(fit, predictors, asymptotic_covariance, *, alpha=0.05):
    """B1 §6 pointwise one-sided mean-loss bound: fitted-z_alpha*sqrt(Z'Omega Z/n).

    This is a pointwise CONFIDENCE bound for the fitted conditional mean,
    not a simultaneous band or prediction interval for individual surprises.
    Generated predictors can require additional variance adjustments.
    """
    alpha = probability(alpha)
    x = array(predictors, "predictors", 2)
    if x.shape[1] != fit.n_features:
        raise ValueError("predictors must match the fitted feature columns.")
    if fit.add_intercept:
        x = np.column_stack((np.ones(len(x)), x))
    omega = array(asymptotic_covariance, "asymptotic_covariance", 2)
    if omega.shape != (x.shape[1], x.shape[1]) or not np.allclose(omega, omega.T):
        raise ValueError("Omega must be a symmetric coefficient covariance matrix.")
    if np.linalg.eigvalsh(omega).min() < -1e-10:
        raise ValueError("Omega must be positive semidefinite.")
    se = np.sqrt(np.maximum(0, np.einsum("ni,ij,nj->n", x, omega, x)) / fit.n)
    return x @ fit.coefficients - norm.ppf(1 - alpha) * se
