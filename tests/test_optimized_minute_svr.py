import sys
import unittest
from pathlib import Path
import numpy as np
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))
from optimize_minute_svr import features, minute_features, calendars


class OptimizedMinuteTests(unittest.TestCase):
    def test_future_minutes_cannot_change_past_features(self):
        rng = np.random.default_rng(4)
        prices = 100*np.exp(np.cumsum(rng.normal(0, .0001, 90*1440)))
        close = pd.Series(prices.reshape(-1, 1440)[:, -1])
        X, y, b = features(close, minute_features(prices), 28, 28)
        altered = prices.copy()
        altered[61*1440:] *= np.exp(np.arange(len(altered)-61*1440)*.00001)
        close2 = pd.Series(altered.reshape(-1, 1440)[:, -1])
        X2, y2, b2 = features(close2, minute_features(altered), 28, 28)
        np.testing.assert_allclose(X[28:61], X2[28:61])
        np.testing.assert_allclose(b[28:61], b2[28:61])
        self.assertFalse(np.allclose(y[60], y2[60]))

    def test_intraminute_change_affects_features_even_with_same_daily_close(self):
        prices = np.exp(np.linspace(0, .1, 4*1440))
        original = minute_features(prices)
        prices[1440+600] *= 1.01
        changed = minute_features(prices)
        self.assertNotEqual(original[1,0], changed[1,0])
        np.testing.assert_allclose(original[2:], changed[2:])

    def test_expanding_cuts_are_distinct_and_never_use_future_labels(self):
        dates = pd.date_range('2023-01-01', '2025-12-31', tz='UTC')
        panel = pd.DataFrame({'x': np.ones(len(dates))}, index=dates)
        folds, eligible, audit = calendars(panel)
        self.assertEqual(len(folds), 6)
        for tr, va in folds:
            self.assertLess(tr.max()+7, va.min())
            self.assertLess(dates[va.max()+7], pd.Timestamp('2025-01-01', tz='UTC'))
        self.assertTrue(all(len(folds[i][0]) < len(folds[i+1][0]) for i in range(5)))
        self.assertEqual(audit.query("method == 'forwardChaining'").future_train_origins.sum(), 0)
        self.assertGreater(audit.query("method == 'kFold'").future_train_origins.sum(), 0)
        self.assertGreater(audit.query("method == 'groupKFold'").future_train_origins.sum(), 0)


if __name__ == '__main__':
    unittest.main()
