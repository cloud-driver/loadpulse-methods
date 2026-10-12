"""Small reproducible finite-sample diagnostic, not a universal guarantee.

Runs null/alternative MCS and null B1 fixed-OLS studies. Designs differ from
the original papers. Reports empirical rates and binomial Wilson intervals.
"""
import json
import numpy as np
from loadpulse_stats import (model_confidence_set, rolling_origin_ols, surprise_loss,
    weighted_breakdown_variance, iid_overfit_correction, forecast_breakdown_test)


def wilson_interval(hits, total):
    p, z = hits / total, 1.959963984540054
    d = 1 + z*z/total
    center = (p+z*z/(2*total))/d
    half = z*np.sqrt(p*(1-p)/total + z*z/(4*total**2))/d
    return [round(center-half, 4), round(center+half, 4)]


def run_check(repetitions=100, seed=20261012):
    rng = np.random.default_rng(seed)
    counts = {"MCS_R_null_any_exclusion": 0, "MCS_max_null_any_exclusion": 0,
              "MCS_R_shift_bad_model_excluded": 0, "MCS_R_AR1_null_any_exclusion": 0,
              "B1_fixed_OLS_null_rejection": 0}
    for _ in range(repetitions):
        x = np.square(rng.normal(size=(240, 3)))
        bs_seed = int(rng.integers(1, 2**31))
        for stat in ("R", "max"):
            result = model_confidence_set(x, statistic=stat, block_length=5,
                                         n_resamples=499, random_state=bs_seed)
            counts[f"MCS_{stat}_null_any_exclusion"] += int(bool(result.excluded))
        alternative = x + [0, 0, 1.5]
        result = model_confidence_set(alternative, block_length=5,
                                     n_resamples=499, random_state=bs_seed)
        counts["MCS_R_shift_bad_model_excluded"] += int("model_2" in result.excluded)
        innovations = rng.normal(size=(340, 3)) * np.sqrt(1-.5**2)
        latent = np.zeros_like(innovations)
        for t in range(1, len(latent)):
            latent[t] = .5*latent[t-1]+innovations[t]
        serial_losses = np.square(latent[100:])
        serial_result = model_confidence_set(serial_losses, block_length=8,
                       n_resamples=499, random_state=bs_seed)
        counts["MCS_R_AR1_null_any_exclusion"] += int(bool(serial_result.excluded))
        features = rng.normal(size=(480, 2))
        y = 1 + features @ np.array([.5, 1.5]) + rng.normal(size=480)
        fit = rolling_origin_ols(y, features, 240, scheme="fixed")
        sl = surprise_loss(fit.out_of_sample_loss, fit.in_sample_mean_loss)
        variance = weighted_breakdown_variance(fit.full_losses, 240, len(sl),
                     scheme="fixed", max_lag=5, estimation_irrelevant=True)
        error_variance = fit.initial_in_sample_losses.sum()/(240-3)
        c = iid_overfit_correction(len(sl), 240, scheme="fixed",
                                   error_variance=error_variance, n_parameters=3)
        result = forecast_breakdown_test(sl, asymptotic_variance=variance,
                    correction=c, variance_source="Algorithm 1 iid OLS null diagnostic")
        counts["B1_fixed_OLS_null_rejection"] += int(result.reject)
    return {"repetitions": repetitions, "seed": seed, "alpha": .05,
            "design": "MCS: n=240, 3 iid squared Gaussian losses, B=499, block=5; "
                      "AR1: rho=.5, burn-in=100, square loss, block=8; "
                      "B1: T=480,m=n=240,fixed OLS,squared loss,HAC lag=5,iid correction",
            "results": {name: {"hits": hits, "rate": hits/repetitions,
                                "wilson_95pct": wilson_interval(hits, repetitions)}
                        for name, hits in counts.items()}}


if __name__ == "__main__":
    print(json.dumps(run_check(), indent=2))
