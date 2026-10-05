import unittest

import numpy as np
import pandas as pd

from src.report_optimized_har import HAR, assert_alignment, har_results, outcome_counts


class OptimizedHarReportTests(unittest.TestCase):
    def test_sign_orientation_and_holm_boundary(self):
        table = pd.DataFrame([
            dict(modelo_a=HAR, modelo_b="A", diferencia_mse=-1., p_holm=.01),
            dict(modelo_a="A", modelo_b=HAR, diferencia_mse=1., p_holm=.01),
            dict(modelo_a="A", modelo_b=HAR, diferencia_mse=-1., p_holm=.01),
            dict(modelo_a=HAR, modelo_b="A", diferencia_mse=-1., p_holm=.05),
            dict(modelo_a=HAR, modelo_b="A", diferencia_mse=-1., p_holm=np.nan),
            dict(modelo_a="A", modelo_b="B", diferencia_mse=-1., p_holm=.01),
        ])
        selected = har_results(table)
        self.assertEqual(selected.outcome.tolist(), [
            "wins", "wins", "losses", "no_significant_difference", "not_evaluable"
        ])
        counts = outcome_counts(selected)
        self.assertEqual(counts["wins"], 2)
        self.assertEqual(counts["losses"], 1)
        self.assertEqual(counts["lower_mse"], 4)
        self.assertEqual(counts["total"], 5)

    def test_alignment_rejects_missing_keys_duplicates_and_small_target_changes(self):
        original = pd.DataFrame({
            "symbol": ["BTCUSDT", "BTCUSDT"], "volatility_window": [7, 7],
            "origin": pd.to_datetime(["2025-01-01", "2025-01-02"], utc=True),
            "horizon": [1, 1], "actual": [1., 2.], "forecast": [1.1, 2.1],
        })
        changed = original.iloc[::-1].copy()
        changed["forecast"] += .1
        assert_alignment(original, changed)
        with self.assertRaises(ValueError):
            assert_alignment(original, changed.iloc[:1])
        with self.assertRaises(ValueError):
            assert_alignment(original, pd.concat([changed, changed.iloc[:1]]))
        changed.loc[changed.index[0], "actual"] += 1e-12
        with self.assertRaises(ValueError):
            assert_alignment(original, changed)


if __name__ == "__main__":
    unittest.main()
