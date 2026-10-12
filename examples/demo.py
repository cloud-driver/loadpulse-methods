"""Runnable synthetic demonstration. Not Denmark data or paper replication.

Run: python -m examples.demo
"""
from dataclasses import asdict
import json
import numpy as np
from loadpulse_stats import (
    rolling_origin_ols, surprise_loss, weighted_breakdown_variance,
    forecast_breakdown_test, iid_overfit_correction, model_confidence_set,
    fit_calendar_baseline, predict_calendar_baseline, excess_loss,
    joint_window_scan, consensus_failure, merge_failure_events,
    regression_model_confidence_set,
)


def run_demo():
    rng = np.random.default_rng(42)
    n = 420
    x = rng.normal(size=(n, 2))
    y = 5+x @ np.array([1., 2.])+rng.normal(size=n)
    y[300:] += rng.normal(scale=4, size=n-300)
    fit = rolling_origin_ols(y, x, 160, scheme="fixed")
    sl = surprise_loss(fit.out_of_sample_loss, fit.in_sample_mean_loss)
    variance = weighted_breakdown_variance(fit.full_losses, 160, len(sl), scheme="fixed",
                    max_lag=8, estimation_irrelevant=True)
    correction = iid_overfit_correction(len(sl), 160, scheme="fixed",
                    error_variance=fit.initial_in_sample_losses.sum()/(160-3), n_parameters=3)
    b1 = forecast_breakdown_test(sl, asymptotic_variance=variance, correction=correction,
                    variance_source="B1 Algorithm 1 / Corollary 3; OLS + squared loss")

    losses = rng.gamma(shape=2, scale=1, size=(250, 3))+[0, .05, 2]
    b2 = model_confidence_set(losses, model_names=["calendar", "calendar_lags", "weak_model"],
                    block_length=6, n_resamples=499, random_state=3)

    # Four separate roles: model training (external), baseline fitting,
    # null-calibration, final evaluation. No evaluation-based fitting.
    baseline_period = rng.gamma(2, 1, size=(240, 3))
    null_period = rng.gamma(2, 1, size=(240, 3))
    evaluation = rng.gamma(2, 1, size=(80, 3)); evaluation[30:55] += 7
    baseline = fit_calendar_baseline(baseline_period, min_count=20)
    null_u = excess_loss(null_period, predict_calendar_baseline(baseline, n_observations=len(null_period)))
    eval_u = excess_loss(evaluation, predict_calendar_baseline(baseline, n_observations=len(evaluation)))
    scan = joint_window_scan(null_u, eval_u, window_sizes=[6, 12], block_length=6,
                    practical_threshold=1, n_resamples=199, random_state=9,
                    residual_stationarity_confirmed=True)
    consensus, _ = consensus_failure(scan.failure_mask, minimum_fraction=2/3)
    events = merge_failure_events(scan.windows, consensus)

    reg_x = rng.normal(size=(180, 1)); unrelated = rng.normal(size=(180, 1))
    reg_y = 3*reg_x[:, 0]+rng.normal(size=180)
    regression = regression_model_confidence_set(reg_y, [reg_x, unrelated],
                    model_names=["signal", "unrelated"], block_length=5,
                    n_resamples=199, criterion="AIC_star", random_state=3)
    return {"B1": asdict(b1), "B2_included": b2.included, "B2_pvalues": b2.pvalues,
            "regression_MCS_included": regression.included,
            "LP_events_start_stop_exclusive": events, "LP_inference_scope": scan.inference_scope}


if __name__ == "__main__":
    print(json.dumps(run_demo(), indent=2, ensure_ascii=False))
