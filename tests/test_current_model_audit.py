import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))
from audit_current_model import (block_indices, bootstrap_metric_arrays,
    reconstruct_coefficients, recent_training_subset, hac_moment_normality)
from minute_experiment import estimator
from optimize_minute_svr import calendars, metrics


class CurrentModelAuditTests(unittest.TestCase):
    def test_circular_draw_contains_contiguous_calendar_blocks(self):
        indices = block_indices(83, 35, 30, 17)
        self.assertEqual(indices.shape, (30, 83))
        for start, end in [(0, 35), (35, 70), (70, 83)]:
            np.testing.assert_array_equal(np.diff(indices[:, start:end], axis=1) % 83,
                                          np.ones((30, end-start-1), dtype=int))
        np.testing.assert_array_equal(indices, block_indices(83, 35, 30, 17))

    def test_bootstrap_recomputes_r2_and_horizon_rmse_on_same_draw(self):
        rng = np.random.default_rng(19)
        actual = rng.uniform(.2, 6, (83, 3, 7))
        predictions = actual+rng.normal(0, .2, actual.shape)
        indices = block_indices(83, 35, 4, 12)
        result, identical = bootstrap_metric_arrays(actual, [predictions, predictions.copy()], indices)
        np.testing.assert_array_equal(result, identical)
        for i, draw in enumerate(indices):
            for config in range(3):
                manual = metrics(actual[draw, config], predictions[draw, config])
                np.testing.assert_allclose(result[i, config],
                    [manual[k] for k in ['r2', 'rmse', 'mae', 'mse', 'mape']], rtol=1e-12)
        # Cross-asset alignment is preserved: scaling both actual and prediction
        # changes RMSE by the scale, with identical R2 in every shared date draw.
        y = np.repeat(actual[:, :1, :], 2, axis=1)
        p = np.repeat(predictions[:, :1, :], 2, axis=1)
        y[:, 1] *= 10
        p[:, 1] *= 10
        value = bootstrap_metric_arrays(y, [p], indices)[0]
        np.testing.assert_allclose(value[:, 0, 0], value[:, 1, 0], atol=1e-12)
        np.testing.assert_allclose(value[:, 1, 1], 10*value[:, 0, 1], atol=1e-12)

    def test_two_scalers_are_reversed_to_engineered_input_and_target_units(self):
        rng = np.random.default_rng(4)
        X = rng.normal(size=(110, 4))*[2, 100, .002, 7]+[10, -50, 3, 6]
        y = X @ rng.normal(size=(4, 7))*.5+np.arange(7)*10
        model = make_pipeline(StandardScaler(), estimator(.1, .01)).fit(X, y)
        weights, intercept, one_sd = reconstruct_coefficients(model)
        probe = X[:12]+rng.normal(size=(12, 4))*.01
        np.testing.assert_allclose(probe @ weights+intercept, model.predict(probe), atol=1e-10)
        for j in range(4):
            altered = probe.copy()
            altered[:, j] += model.steps[0][1].scale_[j]
            np.testing.assert_allclose(model.predict(altered)-model.predict(probe),
                np.tile(one_sd[j], (len(probe), 1)), atol=1e-10)

    def test_learning_histories_are_nested_and_end_before_validation_targets(self):
        dates = pd.date_range('2023-01-01', '2025-12-31', tz='UTC')
        folds, _, _ = calendars(pd.DataFrame({'x': 1}, index=dates))
        for tr, va in folds:
            previous = set()
            for fraction in [.25, .5, .75, 1.0]:
                selected = recent_training_subset(tr, fraction, validation=va)
                self.assertTrue(previous.issubset(set(selected)))
                self.assertEqual(selected[-1], tr[-1])
                self.assertLess(selected[-1]+7, va[0])
                self.assertLess(dates[va[-1]+7], pd.Timestamp('2025-01-01', tz='UTC'))
                previous = set(selected)
        with self.assertRaises(ValueError):
            recent_training_subset(np.arange(30), .5, validation=np.array([31]))

    def test_hac_normality_diagnostic_is_location_scale_invariant(self):
        rng = np.random.default_rng(18)
        e = rng.standard_t(5, size=358)
        np.testing.assert_allclose(hac_moment_normality(e), hac_moment_normality(7*e+10), atol=1e-10)


if __name__ == '__main__':
    unittest.main()
