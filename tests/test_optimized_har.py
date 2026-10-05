"""Causality and loss-alignment checks for the new HAR search."""
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from optimize_har_xgboost import select_horizons, matrices, training_target, decode, component_configuration, feature_names
from optimize_minute_svr import minute_features


class OptimizedHarTests(unittest.TestCase):
    def test_horizons_choose_different_components_and_preserve_exact_booster(self):
        actual = np.ones((20, 7))
        booster = actual.copy()
        booster[:, 1:] = .8
        early_ridge = np.full_like(actual, 2.)
        late_ridge = np.full_like(actual, 1.2)
        late_ridge[:, 0] = 3
        choices, _, predicted = select_horizons(actual, {'boost__source_relative': booster},
                                               {'ridge__relative__1': early_ridge, 'ridge__level__1': late_ridge})
        self.assertEqual(choices[0]['ridge_weight'], 0)
        self.assertEqual(choices[1]['ridge_key'], 'ridge__level__1')
        self.assertAlmostEqual(choices[1]['ridge_weight'], .5)
        np.testing.assert_allclose(predicted, actual)

    def test_absolute_loss_weighting_and_decoding_are_consistent(self):
        rng = np.random.default_rng(4)
        y = rng.uniform(.2, 3, size=(30, 7))
        base = rng.uniform(.1, 4, size=30)
        decay = rng.uniform(.1, 3, size=(30, 7))
        train = np.arange(30)
        for key in ['ridge__relative__10', 'ridge__mse_relative__10', 'ridge__level__10',
                    'ridge__decay__10', 'boost__mse_decay', 'boost__level_correction']:
            config = component_configuration(key)
            encoded = training_target(config, y, base, decay, train)
            np.testing.assert_allclose(decode(config, encoded, base, decay), y)
        relative = y / base[:, None] - 1
        estimate = rng.normal(size=y.shape)
        np.testing.assert_allclose(base[:, None] ** 2 * (relative - estimate) ** 2,
                                   (y - base[:, None] * (1 + estimate)) ** 2)

    def test_future_prices_do_not_change_features_or_decay(self):
        rng = np.random.default_rng(19)
        prices = 100 * np.exp(np.cumsum(rng.normal(0, .0003, 95 * 1440)))
        close = pd.Series(prices.reshape(-1, 1440)[:, -1])
        values, y, base, decay = matrices(close, minute_features(prices), 14, 7)
        changed = prices.copy()
        changed[61 * 1440:] *= np.exp(np.linspace(0, 2, len(changed) - 61 * 1440))
        altered, y2, base2, decay2 = matrices(pd.Series(changed.reshape(-1, 1440)[:, -1]), minute_features(changed), 14, 7)
        for name in values:
            np.testing.assert_allclose(values[name][28:61], altered[name][28:61])
        np.testing.assert_allclose(decay[28:61], decay2[28:61])
        np.testing.assert_allclose(base[28:61], base2[28:61])
        self.assertFalse(np.allclose(y[60], y2[60]))
        self.assertEqual(len(feature_names('H')), values['H'].shape[1])
        self.assertEqual(len(feature_names('D')), values['D'].shape[1])

    def test_alignment_errors_fail_before_selection(self):
        actual = np.ones((20, 7))
        with self.assertRaises(ValueError):
            select_horizons(actual, {'a': actual[:-1]}, {'b': actual})
        broken = actual.copy()
        broken[3, 2] = np.nan
        with self.assertRaises(ValueError):
            select_horizons(actual, {'a': broken}, {'b': actual})


if __name__ == '__main__':
    unittest.main()
