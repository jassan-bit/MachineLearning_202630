"""Regression checks for duplicate keys, missing calendar pairs and MAD limits."""
from pathlib import Path
import sys
import unittest

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))
from audit_current_dataset import FIELDS, duplicate_counts, effective_size, robust_limits


class QualityAuditTests(unittest.TestCase):
    def frame(self, unit):
        factor = 1000 if unit == 'ms' else 1000000
        open_time = int(pd.Timestamp('2023-01-01', tz='UTC').timestamp()*factor)
        row = [open_time, 100., 101., 99., 100., 2.,
               open_time+60*factor-1, 200., 10, 1., 100., 0]
        later = row.copy()
        later[0] += 60*factor
        later[6] += 60*factor
        near = row.copy()
        near[4] += .000001
        conflicting = row.copy()
        conflicting[4] = 100.5
        return pd.DataFrame([row, row.copy(), later, near, conflicting], columns=FIELDS)

    def test_repeated_price_different_minute_is_not_duplicate(self):
        for unit in ['ms', 'us']:
            frame = self.frame(unit).iloc[[0, 2]]
            self.assertEqual(duplicate_counts(frame, unit)['duplicate_minute_extra_rows'], 0)

    def test_exact_near_and_conflicting_rows_are_separate(self):
        for unit in ['ms', 'us']:
            counts = duplicate_counts(self.frame(unit), unit)
            self.assertEqual(counts['exact_duplicate_extra_rows'], 1)
            self.assertEqual(counts['near_duplicate_extra_rows'], 1)
            self.assertEqual(counts['conflicting_minute_extra_rows'], 1)

    def test_ess_does_not_compress_calendar_gaps(self):
        values = np.array([1., np.nan, -1., np.nan, 1., np.nan, -1.])
        result = effective_size(values, 3)
        self.assertEqual(result['n'], 4)
        self.assertEqual(result['used_lags'], 0)
        self.assertEqual(result['n_effective'], 4.)

    def test_ess_of_correlated_series_is_reduced(self):
        rng = np.random.default_rng(42)
        values = np.zeros(30000)
        innovations = rng.normal(size=len(values))
        for i in range(1, len(values)):
            values[i] = .8*values[i-1]+innovations[i]
        result = effective_size(values, 90)
        # For stationary AR(1), ESS/n approaches (1-rho)/(1+rho) = 1/9.
        self.assertGreater(result['n_effective']/len(values), .085)
        self.assertLess(result['n_effective']/len(values), .145)

    def test_mad_robust_to_single_large_observation(self):
        center, scale, low, high = robust_limits([-2, -1, 0, 1, 2, 1000, np.nan])
        self.assertAlmostEqual(center, .5)
        self.assertAlmostEqual(scale, 1.4826*1.5)
        self.assertLess(high, 1000)
        self.assertLess(low, center)


if __name__ == '__main__':
    unittest.main()
