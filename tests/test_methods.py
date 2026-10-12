"""Meaningful formula, leakage, invariance and degenerate-input checks.

Run from project root: python -m unittest discover -s tests -v
No extra test dependency is required.
"""
import unittest
import numpy as np
from numpy.testing import assert_allclose, assert_array_equal
from scipy.stats import norm
from loadpulse_stats import (
    forecast_error, squared_loss, absolute_loss, loss_matrix, loss_differentials,
    hac_long_run_covariance, block_bootstrap_indices, bootstrap_mean_samples,
    forecasting_splits, rolling_origin_ols, surprise_loss, forecast_weights,
    weighted_breakdown_variance, scheme_variance_factor, stationary_breakdown_variance,
    linear_overfit_correction, iid_overfit_correction, forecast_breakdown_test,
    fit_surprise_regression, predict_surprise_regression, predictive_wald_test,
    predictive_cross_factor, predictive_covariance_from_moments,
    surprise_regression_lower_bound,
    mcs_statistics, model_confidence_set, equal_predictive_ability_test,
    gaussian_deviance, effective_degrees_of_freedom, regression_model_confidence_set,
    fit_calendar_baseline, predict_calendar_baseline, excess_loss,
    window_excess_loss, joint_window_scan, consensus_failure, merge_failure_events,
)
from loadpulse_stats.model_confidence_set import _complete_path


class FormulaTests(unittest.TestCase):
    def test_errors_and_losses_have_correct_sign_and_units(self):
        assert_array_equal(forecast_error([2, 4], [3, 1]), [-1, 3])
        assert_array_equal(squared_loss([2, 4], [3, 1]), [1, 9])
        assert_array_equal(absolute_loss([2, 4], [3, 1]), [1, 3])
        assert_array_equal(loss_matrix([2, 4], [[3, 2], [1, 5]]), [[1, 0], [9, 1]])
        self.assertEqual(loss_differentials([[10, 12]])[0, 0, 1], -2)

    def test_nan_and_accidental_broadcasting_are_rejected(self):
        for actual, prediction in [([1, 2], [1]), ([1, np.nan], [1, 2])]:
            with self.assertRaises(ValueError):
                squared_loss(actual, prediction)

    def test_hac_bartlett_matches_hand_computation(self):
        self.assertAlmostEqual(hac_long_run_covariance([1, 2, 4], 0), 14 / 9)
        self.assertAlmostEqual(hac_long_run_covariance([1, 2, 4], 1), 41 / 27)
        x = np.column_stack(([1, 2, 4], [2, 4, 8]))
        assert_allclose(hac_long_run_covariance(x, 1), 41 / 27 * np.array([[1, 2], [2, 4]]))

    def test_fixed_and_rolling_weights_match_hand_computation(self):
        assert_allclose(forecast_weights(3, 2, scheme="fixed")[0], [-2/3]*3+[1, 1])
        assert_allclose(forecast_weights(3, 2, scheme="rolling")[0], [-1/3, -2/3, -2/3, 2/3, 1])
        assert_allclose(forecast_weights(3, 3, scheme="rolling")[0], [-1/3, -2/3, -1, 1/3, 2/3, 1])

    def test_recursive_weights_match_hand_computation(self):
        assert_allclose(forecast_weights(3, 2, scheme="recursive")[0], [-7/12]*3+[3/4, 1])

    def test_weighted_loss_identity_all_schemes_and_horizons(self):
        rng = np.random.default_rng(2)
        m, n = 7, 11
        for scheme in ("fixed", "rolling", "recursive"):
            for h in (1, 2, 5):
                full = rng.normal(size=m+n+h-1)
                wl, _ = forecast_weights(m, n, scheme=scheme, horizon=h)
                expected = 0
                for origin in range(m-1, m+n-1):
                    if scheme == "fixed":
                        training = full[:m]
                    elif scheme == "rolling":
                        training = full[origin-m+1:origin+1]
                    else:
                        training = full[:origin+1]
                    expected += full[origin+h] - training.mean()
                self.assertAlmostEqual(wl @ full, expected)
                self.assertAlmostEqual(wl.sum(), 0)

    def test_general_variance_equals_explicit_combined_influence(self):
        m, n = 4, 5
        full = np.linspace(1, 3, m+n) ** 2
        scores = np.ones((m+n, 1)); scores[-1] = 0
        d = np.arange(1, n+1)[:, None]
        b = np.ones((n, 1, 1))
        wl, wh = forecast_weights(m, n, scheme="rolling", derivative_times_influence=d)
        influence = wl*(full-full.mean()) + (wh*scores).sum(axis=1)
        # Independent lag-0 variance expression, avoiding reuse of tested HAC.
        expected = np.sum(influence**2) / n
        value = weighted_breakdown_variance(full, m, n, scheme="rolling", max_lag=0,
                                           scores=scores, surprise_gradients=d, influence_matrices=b)
        self.assertAlmostEqual(value, expected)

    def test_equal_loss_variance_uses_weighted_full_series(self):
        m, n = 3, 2
        full = np.array([1, 2, 4, 3, 8.])
        wl, _ = forecast_weights(m, n, scheme="fixed")
        expected = np.sum((wl*(full-full.mean()))**2)/n
        value = weighted_breakdown_variance(full, m, n, scheme="fixed", max_lag=0,
                                           estimation_irrelevant=True)
        self.assertAlmostEqual(value, expected)
        with self.assertRaises(ValueError):
            weighted_breakdown_variance(full, m, n, scheme="fixed", max_lag=0)

    def test_scheme_factors_and_confirmed_shortcut(self):
        self.assertAlmostEqual(scheme_variance_factor(10, 5, "fixed"), 1.5)
        self.assertAlmostEqual(scheme_variance_factor(10, 5, "rolling"), 11/12)
        self.assertAlmostEqual(scheme_variance_factor(5, 10, "rolling"), 1/3)
        self.assertEqual(scheme_variance_factor(10, 5, "recursive"), 1)
        with self.assertRaises(ValueError):
            stationary_breakdown_variance([1, 2, 3], 3, scheme="fixed", max_lag=0)

    def test_overfit_general_equals_iid_special_case(self):
        q = np.array([[2., .5], [.5, 1.]])
        v = 3*np.linalg.inv(q)
        for scheme in ("fixed", "rolling", "recursive"):
            a = linear_overfit_correction(100, 80, scheme=scheme, mean_xtx=q,
                                          parameter_asymptotic_covariance=v)
            b = iid_overfit_correction(100, 80, scheme=scheme, error_variance=3, n_parameters=2)
            self.assertAlmostEqual(a, b)

    def test_corrected_test_is_one_sided_and_scales_c_correctly(self):
        sl = np.repeat(1., 100)
        result = forecast_breakdown_test(sl, asymptotic_variance=4, variance_source="hand", correction=2)
        self.assertEqual(result.statistic, 4)
        self.assertAlmostEqual(result.pvalue, norm.sf(4))
        negative = forecast_breakdown_test(-sl, asymptotic_variance=4, variance_source="hand")
        self.assertFalse(negative.reject)
        with self.assertRaises(ValueError):
            forecast_breakdown_test(sl, asymptotic_variance=0, variance_source="degenerate")

    def test_predictive_regression_and_external_wald_covariance(self):
        x = np.arange(10.)[:, None]
        y = 2 + 3*x[:, 0]
        fit = fit_surprise_regression(y, x)
        assert_allclose(fit.coefficients, [2, 3])
        assert_allclose(predict_surprise_regression(fit, [[10], [11]]), [32, 35])
        result = predictive_wald_test(fit, np.eye(2), covariance_source="hand")
        self.assertAlmostEqual(result["statistic"], 130)

    def test_proposition7_covariance_assembly_and_mean_lower_bound(self):
        self.assertAlmostEqual(predictive_cross_factor(100, 50, "rolling"), .75)
        self.assertAlmostEqual(predictive_cross_factor(50, 100, "rolling"), .25)
        self.assertAlmostEqual(predictive_cross_factor(100, 100, "recursive"), np.log(2))
        omega = predictive_covariance_from_moments([0.], [[1.]], [[3.]], [0.], 2.,
                                                  100, 100, scheme="fixed")
        assert_allclose(omega, np.diag([2., 3.]))
        fit = fit_surprise_regression(2+np.arange(10.), np.arange(10.)[:, None])
        bound = surprise_regression_lower_bound(fit, [[0]], omega)
        assert_allclose(bound, [2-norm.ppf(.95)*np.sqrt(2/10)])


class ForecastingTests(unittest.TestCase):
    def test_embargo_and_schemes_have_no_future_training_targets(self):
        for scheme in ("fixed", "rolling", "recursive"):
            splits = list(forecasting_splits(20, 5, horizon=3, scheme=scheme))
            self.assertEqual(splits[0][1], 7)
            for train, target in splits:
                self.assertLessEqual(train.max(), target-3)
                if scheme in ("fixed", "rolling"):
                    self.assertEqual(len(train), 5)

    def test_changing_future_targets_cannot_change_past_forecast(self):
        rng = np.random.default_rng(23)
        x = rng.normal(size=(30, 1)); y = 2*x[:, 0]+rng.normal(size=30)
        before = rolling_origin_ols(y, x, 10, horizon=2)
        after_y = y.copy(); after_y[20:] += 5000
        after = rolling_origin_ols(after_y, x, 10, horizon=2)
        assert_allclose(before.predictions[:9], after.predictions[:9])
        self.assertEqual(len(before.full_losses), len(y))


class MCSTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(4)
        self.losses = np.square(rng.normal(size=(250, 3))) + [0, .1, 2]

    def test_bootstrap_is_reproducible_and_block_paired(self):
        for method in ("moving", "circular", "stationary"):
            a = block_bootstrap_indices(20, 4, 3, method=method, random_state=1)
            b = block_bootstrap_indices(20, 4, 3, method=method, random_state=1)
            assert_array_equal(a, b)
            self.assertTrue(np.all((0 <= a) & (a < 20)))
        idx = block_bootstrap_indices(20, 4, 3, method="moving", random_state=1)
        assert_array_equal(np.diff(idx[:, :3]), np.ones((4, 2), dtype=int))

    def test_mcs_eliminates_obviously_inferior_model_both_statistics(self):
        for statistic in ("R", "max"):
            result = model_confidence_set(self.losses, model_names=["A", "B", "bad"],
                                           statistic=statistic, block_length=5,
                                           n_resamples=499, random_state=1)
            self.assertIn("bad", result.excluded)
            self.assertEqual(max(result.pvalues.values()), 1)
            ordered_p = [result.pvalues[name] for name in result.elimination_order]
            self.assertTrue(np.all(np.diff(ordered_p) >= 0))

    def test_identical_models_all_survive(self):
        x = np.repeat(self.losses[:, [0]], 3, axis=1)
        result = model_confidence_set(x, block_length=4, n_resamples=99, random_state=0)
        self.assertEqual(len(result.included), 3)
        self.assertTrue(all(p == 1 for p in result.pvalues.values()))

    def test_common_time_shock_cancels_from_mcs(self):
        shock = np.linspace(100, 500, len(self.losses))[:, None]
        for statistic in ("R", "max"):
            a = model_confidence_set(self.losses, block_length=5, n_resamples=199,
                                     random_state=7, statistic=statistic)
            b = model_confidence_set(self.losses+shock, block_length=5, n_resamples=199,
                                     random_state=7, statistic=statistic)
            self.assertEqual(a.included, b.included)
            self.assertEqual(a.pvalues, b.pvalues)

    def test_r_elimination_uses_standardized_worst_pair_not_raw_mean(self):
        # Highest raw mean is model 0, but model 1 is significantly inferior to 2.
        x = np.array([[10., 2., 0.], [10., 2., 0.], [10., 2., 0.]])
        boot = np.array([[-90, 1.9, 0], [10, 2.0, .01], [110, 2.1, -.01]])
        result = mcs_statistics(x, boot, statistic="R")
        self.assertEqual(result["worst_index"], 1)

    def test_zero_variance_nonidentical_difference_raises(self):
        x = np.column_stack((np.arange(20.), np.arange(20.)+10))
        with self.assertRaises(ValueError):
            model_confidence_set(x, block_length=3, n_resamples=99, random_state=0)

    def test_running_max_pvalue_definition_from_paper_table(self):
        names = ("A", "B", "C", "D")
        counts = {4: 1, 3: 0, 2: 7}
        def test_current(remaining):
            null = np.r_[np.ones(counts[len(remaining)]), np.zeros(99-counts[len(remaining)])]
            return 1.0, null, 0
        included, excluded, pvalues, _, _ = _complete_path(names, test_current, .05)
        self.assertEqual(pvalues, {"A": .02, "B": .02, "C": .08, "D": 1})
        self.assertEqual(included, ("C", "D"))

    def test_gaussian_deviance_and_effective_df(self):
        value = gaussian_deviance([1, 2], [[1], [1]], [0], 2)
        self.assertAlmostEqual(value, 2*np.log(2)+2.5)
        assert_allclose(effective_degrees_of_freedom([[5, 7], [8, 10]], [[3, 6], [6, 7]]), [2, 2])

    def test_quadratic_epa_detects_inferior_alternative(self):
        result = equal_predictive_ability_test(self.losses, max_lag=3)
        self.assertTrue(result["reject"])
        self.assertFalse(result["is_mcs"])

    def test_regression_mcs_uses_joint_refits(self):
        rng = np.random.default_rng(11)
        x = rng.normal(size=(140, 1)); unrelated = rng.normal(size=(140, 1))
        y = 4*x[:, 0]+rng.normal(size=140)
        for criterion in ("KLIC", "AIC_star", "BIC"):
            result = regression_model_confidence_set(y, [x, unrelated], model_names=["signal", "noise"],
                     criterion=criterion, block_length=4, n_resamples=199, random_state=2)
            self.assertIn("signal", result.included)
            self.assertIn("noise", result.excluded)
            self.assertGreater(result.effective_df["signal"], 0)


class LocalFailureTests(unittest.TestCase):
    def test_calendar_baseline_does_not_fit_on_evaluation(self):
        baseline = fit_calendar_baseline([[1], [3], [10], [12]], [0, 0, 1, 1], min_count=2)
        assert_allclose(predict_calendar_baseline(baseline, [0, 1]), [[2], [11]])
        with self.assertRaises(ValueError):
            predict_calendar_baseline(baseline, [2])
        assert_allclose(predict_calendar_baseline(baseline, [2], fallback="global"), [[6.5]])

    def test_slide_11_example_and_window_endpoints(self):
        u = excess_loss([[14], [19], [15]], [[10], [10], [10]])
        start, stop, mean = window_excess_loss(u, 3)
        assert_array_equal(start, [0]); assert_array_equal(stop, [3])
        assert_allclose(mean, [[6]])

    def test_event_union_does_not_count_overlapping_windows_twice(self):
        windows = [(0, 5, 5), (4, 9, 5), (20, 25, 5), (24, 29, 5)]
        self.assertEqual(merge_failure_events(windows, np.array([True, True, False, True])), [(0, 9), (24, 29)])
        flags, fractions = consensus_failure(np.array([[True, True, False], [True, False, False]]), minimum_fraction=2/3)
        assert_array_equal(flags, [True, False]); assert_allclose(fractions, [2/3, 1/3])

    def test_joint_scan_detects_common_shift_without_using_mcs(self):
        rng = np.random.default_rng(17)
        calibration = rng.normal(size=(160, 3))
        evaluation = rng.normal(size=(50, 3)); evaluation[15:35] += 6
        result = joint_window_scan(calibration, evaluation, window_sizes=[5, 10], block_length=4,
                     practical_threshold=1, n_resamples=199, random_state=6,
                     residual_stationarity_confirmed=True)
        self.assertTrue(result.failure_mask.any())
        self.assertEqual(result.inference_scope, "conditional_on_frozen_baseline")
        self.assertTrue(np.all(result.adjusted_pvalues >= 1/200))
        self.assertTrue(np.all(result.failure_mask <= (result.mean_excess_loss > 1)))


if __name__ == "__main__":
    unittest.main()
