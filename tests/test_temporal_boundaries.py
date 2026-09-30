"""Regresión de ventanas temporales; datos sintéticos, sin leer TEST."""
from pathlib import Path
import importlib.util
import unittest
import numpy as np
import pandas as pd

spec = importlib.util.spec_from_file_location('base_model', Path(__file__).resolve().parents[1]/'src/18_base_model.py')
base = importlib.util.module_from_spec(spec)
spec.loader.exec_module(base)


class TemporalBoundaries(unittest.TestCase):
    def setUp(self):
        self.grid = pd.date_range('2020-01-01', periods=1200, freq='h', tz='UTC')
        rng = np.random.default_rng(42)
        self.df = pd.concat([pd.DataFrame({
            'symbol': symbol, 'open_time': self.grid,
            'close_time': self.grid+pd.Timedelta(hours=1)-pd.Timedelta(milliseconds=1),
            'close': 100*np.exp(np.cumsum(rng.normal(0,.01,len(self.grid))))
        }) for symbol in ['A','B']], ignore_index=True)

    def test_exact_boundaries_and_independent_block(self):
        frames = base.prepare(self.df, self.grid)
        start, end = self.grid[500], self.grid[799]
        selected = base.contained_block(frames['A'], start, end)
        self.assertEqual(selected.index.min(), self.grid[667])
        self.assertEqual(selected.index.max(), self.grid[775])
        self.assertEqual(len(selected), 109)
        # Recalcular desde el bloque aislado produce exactamente las mismas
        # entradas, etiquetas y persistencia: ninguna depende del exterior.
        isolated = self.df.loc[self.df.open_time.between(start,end)]
        independent = base.prepare(isolated, pd.date_range(start,end,freq='h'))['A']
        pd.testing.assert_frame_equal(selected, independent, check_freq=False, rtol=1e-10, atol=1e-10)

    def test_future_mutation_does_not_change_features_or_persistence(self):
        before = base.prepare(self.df, self.grid)['A']
        anchor = self.grid[700]
        changed = self.df.copy()
        changed.loc[(changed.symbol=='A') & (changed.open_time>anchor),'close'] *= 2
        after = base.prepare(changed, self.grid)['A']
        cols = [f'lag_{k}' for k in range(168)]+['persistence']
        np.testing.assert_allclose(before.loc[anchor,cols],after.loc[anchor,cols])
        self.assertNotAlmostEqual(before.loc[anchor,'y'],after.loc[anchor,'y'])

    def test_missing_hour_invalidates_past_and_future_windows(self):
        damaged = self.df.loc[~((self.df.symbol=='A') & (self.df.open_time==self.grid[600]))]
        frames = base.prepare(damaged, self.grid)
        for frame in frames.values():
            self.assertFalse(frame.index.isin(self.grid[576:768]).any())
            self.assertIn(self.grid[575],frame.index)
            self.assertIn(self.grid[768],frame.index)

    def test_train_validation_and_prefix_containment(self):
        data = base.prepare(self.df,self.grid)['A']
        train = base.contained_block(data,self.grid[0],self.grid[499])
        val = base.contained_block(data,self.grid[500],self.grid[799])
        self.assertLess(train.index.max()+pd.Timedelta(hours=24), self.grid[500])
        self.assertGreaterEqual(val.index.min()-pd.Timedelta(hours=167),self.grid[500])
        self.assertLessEqual(val.index.max()+pd.Timedelta(hours=24),self.grid[799])
        prefix = base.contained_block(data,self.grid[0],self.grid[249])
        self.assertLessEqual(prefix.index.max()+pd.Timedelta(hours=24),self.grid[249])
        with self.assertRaises(ValueError):
            base.contained_block(data,self.grid[500],self.grid[550])


if __name__ == '__main__':
    unittest.main()
