import sys
import unittest
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from volatility_experiment import configuration, calendar_folds, targets, arrays, estimator, fit_model, metric_rows
from download_minute_history import aggregate


class DailyExperimentTests(unittest.TestCase):
    def setUp(self):
        self.cfg = configuration()
        self.grid = pd.date_range('2020-01-01', '2025-12-31', tz='UTC')
        rng = np.random.default_rng(42)
        self.panel = pd.DataFrame(np.exp(4 + np.cumsum(rng.normal(0, .01, (len(self.grid), 4)), axis=0)),
                                  index=self.grid, columns=self.cfg['symbols'])

    def test_calendar_and_common_windows(self):
        self.panel.iloc[100, 0] = np.nan
        folds, audit = calendar_folds(self.panel, self.cfg)
        for fold in folds:
            self.assertLess(fold['train'].max()+7, fold['val'].min())
            self.assertLess(fold['val'].max()+7, fold['test'].min())
            for anchors in fold.values():
                self.assertFalse(((anchors-28 <= 100) & (anchors+7 >= 100)).any())
                for lag in (7, 14, 21, 28):
                    for w in (7, 14, 21, 28):
                        X, y, baseline = arrays(self.panel.iloc[:, 0], targets(self.panel.iloc[:, 0], w), anchors, lag)
                        self.assertEqual(X.shape, (len(anchors), lag))
                        self.assertEqual(y.shape, (len(anchors), 7))
                        a = anchors[0]
                        expected = np.log(self.panel.iloc[a+2-w:a+2, 0].to_numpy()/self.panel.iloc[a+1-w:a+1, 0].to_numpy()).std(ddof=0)*100
                        self.assertAlmostEqual(y[0, 0], expected)

    def test_future_changes_do_not_change_training(self):
        folds, _ = calendar_folds(self.panel, self.cfg)
        before = arrays(self.panel.iloc[:, 0], targets(self.panel.iloc[:, 0], 28), folds[0]['train'], 28)
        changed = self.panel.copy()
        changed.loc['2024':] *= 10
        after = arrays(changed.iloc[:, 0], targets(changed.iloc[:, 0], 28), folds[0]['train'], 28)
        for a, b in zip(before, after): np.testing.assert_array_equal(a, b)

    def test_scalers_and_multistep(self):
        anchors = np.arange(30, 130)
        X, y, _ = arrays(self.panel.iloc[:, 0], targets(self.panel.iloc[:, 0], 7), anchors, 14)
        m = fit_model(estimator(.1, .01), X, y)
        np.testing.assert_allclose(m['x_scale'].mean_, X.mean(axis=0))
        np.testing.assert_allclose(m['regressor'].transformer_.mean_, y.mean(axis=0))
        self.assertEqual(m.predict(X[:2]).shape, (2, 7))

    def test_incomplete_day_and_microseconds(self):
        for unit, multiplier in [('ms', 1000), ('us', 1000000)]:
            times = pd.date_range('2025-01-01', periods=1440, freq='min', tz='UTC')
            raw = (times.as_unit('ns').asi8 // (10**9 // multiplier))
            frame = pd.DataFrame({i: np.ones(1440) for i in range(12)})
            frame[0] = raw; frame[6] = raw + 60*multiplier-1
            daily, actual_unit = aggregate(frame, 'BTCUSDT', '2025-01')
            self.assertEqual(actual_unit, unit)
            self.assertTrue(daily.iloc[0].complete)
            self.assertTrue(pd.isna(daily.iloc[1].close))
            daily, _ = aggregate(frame.iloc[:-1], 'BTCUSDT', '2025-01')
            self.assertFalse(daily.iloc[0].complete)

    def test_metrics_units_and_zero_target(self):
        y = np.ones((2, 7)); p = y+2
        summary = metric_rows(y, p)[-1]
        self.assertEqual(summary['rmse'], 2)
        self.assertEqual(summary['mse'], 4)
        self.assertEqual(summary['mape'], 200)
        y[0, 0] = 0
        self.assertEqual(metric_rows(y, p)[0]['mape_zero_targets'], 1)


if __name__ == '__main__':
    unittest.main()
