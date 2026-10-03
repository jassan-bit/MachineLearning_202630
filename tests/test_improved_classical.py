import sys
import tempfile
import unittest
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from improve_classical_forecast import choose_blend, har_features, ridge
from optimize_minute_svr import metrics, minute_features


class ImprovedClassicalTests(unittest.TestCase):
    def test_future_data_cannot_change_har_features(self):
        rng = np.random.default_rng(12)
        prices = 100*np.exp(np.cumsum(rng.normal(0, .0002, 90*1440)))
        close = pd.Series(prices.reshape(-1, 1440)[:, -1])
        X, y, base = har_features(close, minute_features(prices), 14)
        changed = prices.copy()
        changed[61*1440:] *= np.exp(np.arange(len(changed)-61*1440)*.00001)
        X2, y2, base2 = har_features(pd.Series(changed.reshape(-1, 1440)[:, -1]),
                                    minute_features(changed), 14)
        np.testing.assert_allclose(X[28:61], X2[28:61])
        np.testing.assert_allclose(base[28:61], base2[28:61])
        self.assertFalse(np.allclose(y[60], y2[60]))

    def test_selection_can_keep_xgboost_when_har_is_worse(self):
        actual = np.ones((20, 7))
        boosted = actual.copy()
        best, _ = choose_blend(actual, boosted, {1.: actual*2})
        self.assertEqual(best['ridge_weight'], 0)
        self.assertEqual(best['validation_rmse'], 0)

    def test_selection_can_improve_complementary_errors(self):
        actual = np.ones((20, 7))
        boosted = actual*.8
        best, _ = choose_blend(actual, boosted, {1.: actual*1.2})
        self.assertEqual(best['ridge_weight'], .5)
        self.assertLess(best['validation_rmse'], metrics(actual, boosted)['rmse'])

    def test_scaler_uses_training_only_and_serialization_preserves_predictions(self):
        rng = np.random.default_rng(3)
        X = rng.normal(size=(60, 8))
        y = rng.normal(size=(60, 7))
        model = ridge(10).fit(X[:40], y[:40])
        np.testing.assert_allclose(model[0].mean_, X[:40].mean(axis=0))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'ridge.joblib'
            joblib.dump(model, path)
            np.testing.assert_allclose(model.predict(X[40:]), joblib.load(path).predict(X[40:]))


if __name__ == '__main__':
    unittest.main()
