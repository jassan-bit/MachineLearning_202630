"""Invariantes de calendario y del bootstrap pareado, con datos sintéticos."""
from pathlib import Path
import importlib.util
import unittest
import numpy as np
import pandas as pd

spec = importlib.util.spec_from_file_location('diagnostics', Path(__file__).resolve().parents[1]/'src/19_extended_diagnostics.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class ExtendedDiagnostics(unittest.TestCase):
    def test_acf_respects_missing_hour_and_common_denominator(self):
        index = pd.date_range('2024-01-01', periods=6, freq='h', tz='UTC')
        full = pd.Series([1., 4., np.nan, 9., 3., 2.], index=index)
        rows = module.calendar_correlations(full.dropna(), 2)
        centered = full-full.mean()
        self.assertEqual(rows[0][2], 3)
        self.assertEqual(rows[1][2], 2)
        self.assertAlmostEqual(rows[0][1],
            (centered.iloc[0]*centered.iloc[1]+centered.iloc[3]*centered.iloc[4]+
             centered.iloc[4]*centered.iloc[5])/(centered**2).sum())

    def test_identical_models_have_exact_zero_paired_difference_with_holes(self):
        idx = pd.date_range('2024-01-01', periods=70, freq='h', tz='UTC').delete([8, 9, 25])
        t = np.arange(len(idx))
        parts = []
        for fold in [1, 2]:
            for symbol in ['A', 'B']:
                parts.append(pd.DataFrame(dict(time=idx+pd.Timedelta(days=fold*10),
                    fold=fold, symbol=symbol, y=2+np.sin(t), svr=np.full(len(t), 2.),
                    persistence=np.full(len(t), 2.))))
        frame = pd.concat(parts, ignore_index=True)
        result = module.block_bootstrap(frame, 12, reps=31)
        delta = result.iloc[-1]
        np.testing.assert_allclose(delta[['estimate', 'low', 'high']].to_numpy(dtype=float), 0, atol=0)
        expected = np.sqrt(np.mean(np.sin(t)**2))
        self.assertAlmostEqual(result.iloc[0].estimate, expected)
        repeated = module.block_bootstrap(frame, 12, reps=31)
        pd.testing.assert_frame_equal(result, repeated)


if __name__ == '__main__':
    unittest.main()
