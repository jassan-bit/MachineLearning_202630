"""Meaningful causal-support checks for the new local HAR reference."""
import sys
from pathlib import Path
import unittest
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from professor_feedback_har_baseline import har_inputs, fit_reference, forecast


class HARReferenceTests(unittest.TestCase):
    def test_future_perturbation_cannot_change_historical_features(self):
        values = np.arange(1., 101.)
        changed = values.copy()
        changed[60:] = 1e9
        np.testing.assert_allclose(har_inputs(values)[:60], har_inputs(changed)[:60], equal_nan=True)

    def test_calendar_gaps_invalidate_rolling_inputs_instead_of_compressing_time(self):
        values = np.ones(80)
        values[40] = np.nan
        X = har_inputs(values)
        self.assertTrue(np.isnan(X[40:68, 2]).all())
        self.assertEqual(X[68, 2], 1.)

    def test_training_scaler_and_coefficients_do_not_use_test_labels(self):
        X = har_inputs(np.arange(1., 101.))
        y = np.column_stack([2 * X[:, 0] + h for h in range(7)])
        train = np.arange(30, 60)
        first = fit_reference(X, y, train, 1.)
        y[60:] = 1e12
        changed = X.copy()
        changed[80:] = 1e12
        second = fit_reference(changed, y, train, 1.)
        np.testing.assert_allclose(forecast(first, X, np.arange(60, 70)),
                                   forecast(second, X, np.arange(60, 70)))


if __name__ == '__main__':
    unittest.main()
