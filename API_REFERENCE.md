# LoadPulse 函式與回傳值完整參考

所有 signature 與 docstring 由本版程式產生。來源定位見 SOURCES.md。
陣列的 row 是時間／共同目標，column 是模型或 feature。

## losses.py

Error and loss definitions [B1 §3.1; B2 §2]. Arrays must be aligned.

SOURCE_IDS = B1, B2. This is an independent implementation, not author code.

### forecast_error

```python
from loadpulse_stats.losses import forecast_error
forecast_error(actual, predicted)
```

Return actual - predicted; equal shapes required (no implicit broadcasting).

### squared_loss

```python
from loadpulse_stats.losses import squared_loss
squared_loss(actual, predicted)
```

Return elementwise squared forecast errors, in squared target units.

### absolute_loss

```python
from loadpulse_stats.losses import absolute_loss
absolute_loss(actual, predicted)
```

Return absolute forecast errors, in target units.

### loss_matrix

```python
from loadpulse_stats.losses import loss_matrix
loss_matrix(actual, predictions, loss='squared')
```

actual:(n,), predictions:(n,k) -> losses:(n,k); common targets/horizons.

loss may be 'squared', 'absolute', or callable(actual_matrix, predictions).
Custom criteria may be signed, provided smaller consistently means better.

### loss_differentials

```python
from loadpulse_stats.losses import loss_differentials
loss_differentials(losses)
```

(n,k) -> (n,k,k), entry [t,i,j] = L[t,i]-L[t,j] [B2 §3.1].

Quadratic in k; MCS itself avoids allocating this full time-by-pair tensor.

## resampling.py

Time-dependent resampling [B2 §3], and Bartlett HAC [NW1987].

SOURCE_IDS = B2, NW1987, PR1994. Joint rows preserve cross-model pairing.

### hac_long_run_covariance

```python
from loadpulse_stats.resampling import hac_long_run_covariance
hac_long_run_covariance(values, max_lag, *, center=True)
```

Bartlett/Newey-West covariance of sqrt(n)*mean(values).

values:(n,) or (n,p), max_lag: explicit integer 0..n-1.
Denominator is n at EVERY lag; Bartlett weight is 1-j/(max_lag+1).
Set center=False ONLY for an already correctly centered/influence series.
Returns float for a vector, (p,p) array for a matrix.

### block_bootstrap_indices

```python
from loadpulse_stats.resampling import block_bootstrap_indices
block_bootstrap_indices(n, n_resamples, block_length, *, method='stationary', sample_size=None, random_state=None)
```

Return (B,sample_size) shared row indices.

stationary: geometric block lengths, E[length]=block_length [PR1994].
circular: fixed-length blocks with wraparound.
moving: fixed-length overlapping blocks without wraparound [B2 §3.2.3].
This utility does not determine an optimal block length for your data.

### bootstrap_mean_samples

```python
from loadpulse_stats.resampling import bootstrap_mean_samples
bootstrap_mean_samples(values, n_resamples, block_length, *, method='stationary', random_state=None, batch_size=64)
```

Joint block-bootstrap column means (B,p), generated in bounded batches.

No null centering is imposed here; callers decide the null model.

## forecasting.py

Forecasting-scheme utilities [B1 §3.1]. OLS helpers are engineering glue.

X[t] predicts y[t], and MUST already be available at origin t-horizon.
Never pass contemporaneous measured weather as a future predictor.

### forecasting_splits

```python
from loadpulse_stats.forecasting import forecasting_splits
forecasting_splits(n_observations, initial_train_size, *, scheme='rolling', horizon=1)
```

Yield (training_indices, target_index) with an h-step information embargo.

At origin o, only targets y[0:o+1] are available. Initial origin is m-1.
The forecast target is o+h. Rolling keeps m eligible target rows; fixed
always uses first m rows; recursive uses all eligible rows.

### rolling_origin_ols

```python
from loadpulse_stats.forecasting import rolling_origin_ols
rolling_origin_ols(actual, features, initial_train_size, *, scheme='rolling', horizon=1, add_intercept=True)
```

Leakage-conscious direct forecasts, using OLS + squared loss [B1].

Supply target-aligned features known h steps before their target.
Initial m rows are eligible targets, NOT m raw-origin rows of a lagged
design. Initial NaNs from creating lags must be removed BEFORE this call.
The same beta is used for OOS loss and its matching training mean.

### 回傳容器 ForecastResult

```text
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
```

## forecast_breakdown.py

Giacomini & Rossi forecast breakdown [B1, ECB WP 638, 2006].

Independent implementation of §3.3, Algorithm 1 / Theorem 2 / Corollaries
3-4 (§3.5), linear-model overfit correction (§5), and predictive regression
point estimates (§6). See SOURCES.md for version and applicability.

### surprise_loss

```python
from loadpulse_stats.forecast_breakdown import surprise_loss
surprise_loss(out_of_sample_loss, in_sample_mean_loss)
```

B1 equation (1). in_sample_mean_loss is scalar or exactly shape (n,).

Each training mean must be evaluated using the SAME fitted parameters
that produce its corresponding OOS forecast, not a different full fit.

### forecast_weights

```python
from loadpulse_stats.forecast_breakdown import forecast_weights
forecast_weights(m, n, *, scheme, horizon=1, derivative_times_influence=None)
```

Algorithm-1 weights (w_L, w_h) without fragile case-by-case formulas.

T=m+n+h-1. For origin o=m-1,...,m+n-2, add +1 at o+h and
-1 / training_size at each estimation-window index.
derivative_times_influence:(n,q) is D_(o+h) @ B_o, if uncertainty matters.
w_h:(T,q), or None. The weights operate on B1's full loss/score series,
not on the already differenced surprise-loss series.

### weighted_breakdown_variance

```python
from loadpulse_stats.forecast_breakdown import weighted_breakdown_variance
weighted_breakdown_variance(full_losses, m, n, *, scheme, max_lag, horizon=1, estimation_irrelevant=False, scores=None, surprise_gradients=None, influence_matrices=None)
```

B1 Algorithm 1, equations (5)-(9), Theorem 2: estimate sigma^2_m,n.

full_losses length T=m+n+h-1 must be ordered as in Algorithm 1.
equal estimation/evaluation loss: explicitly set estimation_irrelevant=True.
General branch needs scores:(T,q), surprise_gradients D:(n,k), and
influence_matrices B:(n,k,q). The last h score rows MUST be zero.
The combined influence is w_L*(L-mean(L)) + row_sum(w_h*scores).
Apply Bartlett HAC WITHOUT a second centering and multiply by T/n.
Users remain responsible for A1-A7 and the influence representation A3.

### scheme_variance_factor

```python
from loadpulse_stats.forecast_breakdown import scheme_variance_factor
scheme_variance_factor(m, n, scheme)
```

B1 Corollary 4 equation (10), under covariance stationarity/equal loss.

### stationary_breakdown_variance

```python
from loadpulse_stats.forecast_breakdown import stationary_breakdown_variance
stationary_breakdown_variance(out_of_sample_losses, m, *, scheme, max_lag, assumptions_confirmed=False)
```

sigma^2=lambda * HAC(OOS losses), B1 Corollary 4 / equation (10).

Requires equal estimation/evaluation loss (or the Theorem 2(a) condition),
covariance-stationary loss under H0, A1-A7, and a fixed horizon.
This function does NOT test these assumptions from the data.

### linear_overfit_correction

```python
from loadpulse_stats.forecast_breakdown import linear_overfit_correction
linear_overfit_correction(n, m, *, scheme, mean_xtx, parameter_asymptotic_covariance)
```

B1 §5 eq.(13): c=2*gamma*trace(mean(X'X)*asyvar(sqrt(T)*beta)).

Applicable to linear OLS / squared loss with stationary regressors.
Covariance is NOT Cov(beta); multiply a full-sample Cov(beta) by T first.

### iid_overfit_correction

```python
from loadpulse_stats.forecast_breakdown import iid_overfit_correction
iid_overfit_correction(n, m, *, scheme, error_variance, n_parameters)
```

B1 eq.(14): c=2*gamma*sigma_error^2*k. Include intercept in k.

Use only when asymptotic OLS covariance is sigma^2*(E[XX'])^-1.

### forecast_breakdown_test

```python
from loadpulse_stats.forecast_breakdown import forecast_breakdown_test
forecast_breakdown_test(surprise_losses, *, asymptotic_variance, variance_source, alpha=0.05, correction=0.0, alternative='greater')
```

B1 eq.(4)/(13): (sqrt(n)*mean(SL)-c)/sqrt(sigma^2).

Variance is REQUIRED; no naive sample-standard-deviation fallback.
variance_source records the caller's estimator/assumptions for auditing.
Reject uses strict p<alpha; two-sided testing describes change, not
necessarily deterioration. c is numerator-scale correction, not mean-scale.

### fit_surprise_regression

```python
from loadpulse_stats.forecast_breakdown import fit_surprise_regression
fit_surprise_regression(surprise_losses, predictors, *, add_intercept=True)
```

B1 §6 eq.(15), OLS POINT estimates. Predictors must be known at origin.

No automatic Wald p-value: B1 Proposition 7 requires a scheme-corrected
covariance. Generic OLS or residual-only HAC would not reproduce it.

### predict_surprise_regression

```python
from loadpulse_stats.forecast_breakdown import predict_surprise_regression
predict_surprise_regression(fit, predictors)
```

Out-of-sample predictions from a previously fitted §6 loss regression.

### predictive_wald_test

```python
from loadpulse_stats.forecast_breakdown import predictive_wald_test
predictive_wald_test(fit, asymptotic_covariance, *, covariance_source, alpha=0.05)
```

B1 §6 Wald n*delta' Omega^-1 delta. Caller supplies Proposition-7 Omega.

Omega is covariance of sqrt(n)*delta, not covariance of delta itself.
This low-level function DOES NOT construct Omega or verify stationarity.

### predictive_cross_factor

```python
from loadpulse_stats.forecast_breakdown import predictive_cross_factor
predictive_cross_factor(m, n, scheme)
```

B1 Proposition 7: Lambda cross-covariance adjustment by scheme.

### predictive_covariance_from_moments

```python
from loadpulse_stats.forecast_breakdown import predictive_covariance_from_moments
predictive_covariance_from_moments(predictor_mean, predictor_moment, score_long_run_covariance, loss_score_cross, breakdown_variance, m, n, *, scheme)
```

B1 Proposition 7 eq.(16), LOW-LEVEL assembly of Omega.

zbar:(p,), S_zz:(p,p), S_zL,zL:(p,p), S_zL,L:(p,), sigma^2:scalar.
Caller must construct moments under Proposition-7 fourth-order stationarity
and verify B3. This API deliberately does not guess the S_zz estimator:
the public WP states a raw moment in B3 and a HAC expression after (16).
That version ambiguity is documented; verify the final article before
choosing a general autocorrelated-regressor moment estimator.

### surprise_regression_lower_bound

```python
from loadpulse_stats.forecast_breakdown import surprise_regression_lower_bound
surprise_regression_lower_bound(fit, predictors, asymptotic_covariance, *, alpha=0.05)
```

B1 §6 pointwise one-sided mean-loss bound: fitted-z_alpha*sqrt(Z'Omega Z/n).

This is a pointwise CONFIDENCE bound for the fitted conditional mean,
not a simultaneous band or prediction interval for individual surprises.
Generated predictors can require additional variance adjustments.

### 回傳容器 BreakdownResult

```text
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
```

### 回傳容器 SurpriseRegression

```text
coefficients: np.ndarray
fitted_values: np.ndarray
residuals: np.ndarray
n: int
add_intercept: bool
n_features: int
```

## model_confidence_set.py

Hansen, Lunde & Nason MCS [B2 §2.3 and §3.1.2].

TR is paired ONLY with eR=argmax_i max_j tij.
Tmax is paired ONLY with emax=argmax_i ti_dot.
Bootstrap loss differences are recentered under the EPA null hypothesis.

### mcs_statistics

```python
from loadpulse_stats.model_confidence_set import mcs_statistics
mcs_statistics(losses, bootstrap_means, *, statistic='R')
```

Compute observed statistic, bootstrap-null distribution, and removal index.

losses:(n,k), bootstrap_means:(B,k) in matching column order.
For R: se(dbar_ij)^2 = sample variance of paired bootstrap mean differences.
For max: use deviations relative to the CURRENT set's mean.
Return dict; worst_index refers to the current (not original) columns.

### model_confidence_set

```python
from loadpulse_stats.model_confidence_set import model_confidence_set
model_confidence_set(losses, *, model_names=None, alpha=0.05, statistic='R', n_resamples=2000, block_length, bootstrap_method='stationary', random_state=None)
```

B2 forecast MCS. Rows=time, columns=model, lower loss=better.

Compute both exact coherent choices R/eR and max/emax. A SINGLE joint
bootstrap is reused across all subsets. alpha=.05 means a 95% MCS.
MCS p-values are running maxima along the COMPLETE elimination path.
+1 correction prevents a Monte-Carlo p-value of exactly zero.
Exact duplicate columns are collapsed and all duplicates share a p-value.
Other zero-variance pair differences raise rather than fabricate inference.

### 回傳容器 MCSResult

```text
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
```

## regression_mcs.py

Regression-criterion MCS [B2 §3.1.1, §3.2.1-3.2.4].

Forecast-loss MCS and IN-SAMPLE regression MCS answer different questions.
This module implements Gaussian linear regressions only, not arbitrary ML.

### equal_predictive_ability_test

```python
from loadpulse_stats.regression_mcs import equal_predictive_ability_test
equal_predictive_ability_test(losses, *, max_lag, alpha=0.05)
```

Standalone B2 §3.1.1 quadratic-form EPA test with Bartlett HAC.

Uses k-1 contrasts against the last model; statistic n*xbar' Sigma# xbar.
NOT an MCS: a TQ rejection does not identify an inferior model and needs
closed testing before elimination. Rank-zero/unsupported null directions
are explicitly rejected as undefined input.

### gaussian_deviance

```python
from loadpulse_stats.regression_mcs import gaussian_deviance
gaussian_deviance(actual, design, coefficients, error_variance)
```

B2 §3.2.2 Q=-2*Gaussian quasi-loglikelihood, excluding common constant.

Q=n*log(sigma^2)+sum((y-X beta)^2)/sigma^2. No intercept is added here.
All candidates must use the same y rows; scale/units must be consistent.

### effective_degrees_of_freedom

```python
from loadpulse_stats.regression_mcs import effective_degrees_of_freedom
effective_degrees_of_freedom(original_on_bootstrap_deviance, refitted_bootstrap_deviance)
```

B2 §3.2.3 B3 estimator: mean_b[Q(Z*b,theta_hat)-Q(Z*b,theta_hat*b)].

Arrays:(B,k), each bootstrap fits every candidate on the SAME rows.
Includes the variance parameter. No 'number of regressors' substitution.

### regression_model_confidence_set

```python
from loadpulse_stats.regression_mcs import regression_model_confidence_set
regression_model_confidence_set(actual, designs, *, model_names=None, criterion='KLIC', alpha=0.05, n_resamples=2000, block_length, random_state=None, add_intercept=True)
```

B2 §3.2 regression MCS with moving-block refits and coherent range rule.

designs: sequence of (n,p_i) matrices, all target-aligned and full rank.
criterion in KLIC, AIC, BIC, AIC_star, BIC_star.
KLIC uses Q+k_star; remaining penalties follow B2 §3.2.3.
Null bootstrap is EXACTLY eq.(3): Q_boot_refit+k_star-Q_original,
not a naive bootstrap of a fixed loss matrix. Nested-model KLIC bootstrap
can be conservative as explained in §3.2.4.

### 回傳容器 RegressionMCSResult

```text
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
```

## local_failure.py

LoadPulse extensions, NOT the two papers' original local-failure tests.

SOURCE_IDS = LP, B1(inspiration), B2(joint-resampling inspiration).
The bootstrap scan is CONDITIONAL on a separately fitted, frozen baseline.
It does not include baseline-estimation uncertainty or identify causality.

### fit_calendar_baseline

```python
from loadpulse_stats.local_failure import fit_calendar_baseline
fit_calendar_baseline(calibration_losses, calendar_groups=None, *, min_count=20)
```

Fit conditional historical mean losses from OOS calibration predictions.

Use data BEFORE the null-calibration and final-evaluation periods.
Groups may be columns (hour,weekday,season), not measured future weather.
No data are downloaded and no target-model training happens here.

### predict_calendar_baseline

```python
from loadpulse_stats.local_failure import predict_calendar_baseline
predict_calendar_baseline(baseline, calendar_groups=None, *, n_observations=None, fallback='raise')
```

Predict (n,k) expected loss. Unseen/sparse groups raise by default.

fallback='global' is an explicit approximation using the training global
mean; it must be documented in the analysis. It is never silently applied.

### excess_loss

```python
from loadpulse_stats.local_failure import excess_loss
excess_loss(losses, expected_loss)
```

LP slide-11 definition: u[i,t]=L[i,t]-mu_hat[i](calendar[t]).

### window_excess_loss

```python
from loadpulse_stats.local_failure import window_excess_loss
window_excess_loss(excess_losses, window_size, *, step=1)
```

LP slide-11 A_i(W); complete windows only, zero-based [start,stop).

Returns starts, stops, means. No p-value or failure claim is made here.

### joint_window_scan

```python
from loadpulse_stats.local_failure import joint_window_scan
joint_window_scan(null_calibration_excess, evaluation_excess, *, window_sizes, block_length, practical_threshold, alpha=0.05, n_resamples=2000, step=1, model_names=None, random_state=None, residual_stationarity_confirmed=False)
```

LP extension: one-sided max-statistic bootstrap over ALL models/windows.

Null-calibration residuals and evaluation residuals MUST be from separate
periods, using the SAME separately frozen baseline/model-column order.
Calendar-adjusted residuals must have a stable joint null distribution;
this function cannot establish it. It also cannot carry baseline-fitting
uncertainty, data-revision uncertainty or model-selection uncertainty.
These restrictions are exposed in inference_scope and the manual.
No weather is used. Duplicate overlapping windows are not separate events.

### consensus_failure

```python
from loadpulse_stats.local_failure import consensus_failure
consensus_failure(failure_mask, *, minimum_fraction)
```

Descriptive fraction threshold, NOT independent model votes or a p-value.

Returns (per-window consensus mask, fraction). Model family selection and
minimum_fraction must be fixed before inspecting final outcomes.

### merge_failure_events

```python
from loadpulse_stats.local_failure import merge_failure_events
merge_failure_events(windows, selected, *, max_gap=0)
```

Merge selected [start,stop) windows into descriptive index intervals.

Returns list[(start,stop)]. Touching/overlapping windows merge; max_gap is
measured in rows, not hours. Boundaries are window coverage, not a proven
physical onset/recovery. Sorts windows so multiple scales can be combined.

### 回傳容器 CalendarBaseline

```text
means: dict
counts: dict
global_mean: np.ndarray
n_models: int
min_count: int
n_observations: int
```

### 回傳容器 WindowScanResult

```text
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
inference_scope: str
```

公開函式總數：37。
