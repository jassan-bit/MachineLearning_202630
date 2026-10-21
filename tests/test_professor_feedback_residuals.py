"""Temporal information boundaries for horizon-specific residual corrections."""
from pathlib import Path
import sys
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
import professor_feedback_residuals as residuals


class MatureResidualTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        rng = np.random.default_rng(18102026)
        cls.errors = rng.normal(size=(350, 7))
        for t in range(10, len(cls.errors)):
            cls.errors[t] += .7*cls.errors[t-7]
        cls.train = np.arange(30, 201)
        cls.bundle = residuals.fit_corrector(cls.errors, cls.train, 3, 10., 207)

    def test_horizon_seven_never_uses_immature_error(self):
        t = 250
        before = residuals.causal_lags(self.errors, np.array([t]), 7, 3)
        changed = self.errors.copy()
        changed[t-6:, 6] += 1e9
        after = residuals.causal_lags(changed, np.array([t]), 7, 3)
        np.testing.assert_array_equal(before, after)
        # The origin t-7 target ends at t and becomes observable after close.
        changed[t-7, 6] += 5.
        self.assertNotEqual(before[0, 0], residuals.causal_lags(changed, np.array([t]), 7, 3)[0, 0])

    def test_every_forecast_is_invariant_to_labels_ending_after_issuance(self):
        t = 250
        changed = self.errors.copy()
        for h in range(1, 8):
            # Error at origin s can depend on an unknown actual iff s+h>t.
            changed[t-h+1:, h-1] += 1e8
        expected = residuals.correction(self.bundle, self.errors, np.array([t]))
        observed = residuals.correction(self.bundle, changed, np.array([t]))
        np.testing.assert_array_equal(expected, observed)
        np.testing.assert_array_equal(residuals.last_correction(self.errors, np.array([t])),
                                      residuals.last_correction(changed, np.array([t])))

    def test_outer_fit_ignores_all_labels_outside_training_cutoff(self):
        changed = self.errors.copy()
        changed[201:] += 1e8
        second = residuals.fit_corrector(changed, self.train, 3, 10., 207)
        np.testing.assert_array_equal(residuals.correction(self.bundle, self.errors, np.array([240])),
                                      residuals.correction(second, self.errors, np.array([240])))
        with self.assertRaisesRegex(ValueError, 'cutoff'):
            residuals.fit_corrector(self.errors, self.train, 3, 10., 206)

    def test_missing_calendar_lag_disables_correction_without_compressing_time(self):
        changed = self.errors.copy()
        t = 250
        changed[t-7, 6] = np.nan
        lag = residuals.causal_lags(changed, np.array([t]), 7, 3)
        self.assertTrue(np.isnan(lag[0, 0]))
        np.testing.assert_array_equal(lag[0, 1:], self.errors[t-9:t-7, 6][::-1])
        self.assertEqual(residuals.correction(self.bundle, changed, np.array([t]))[0, 6], 0.)

    def test_offline_recursion_reads_no_new_observed_labels(self):
        changed = self.errors.copy()
        changed[220:] += 1e9
        origins = np.arange(220, 260)
        expected = residuals.offline_correction(self.bundle, self.errors, origins, 220)
        actual = residuals.offline_correction(self.bundle, changed, origins, 220)
        np.testing.assert_array_equal(expected, actual)
        # Online forecasts respond once a changed label has matured.
        self.assertFalse(np.array_equal(residuals.correction(self.bundle, self.errors, origins),
                                       residuals.correction(self.bundle, changed, origins)))

    def test_disabling_correction_reproduces_nonnegative_frozen_base(self):
        base = np.abs(self.errors[:10])
        np.testing.assert_array_equal(
            residuals.completed_forecast(base, self.errors[20:30], 0.), base)


if __name__ == '__main__':
    unittest.main()
