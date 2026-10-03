import unittest
import numpy as np
import pandas as pd
from scipy.stats import t
from volatility_dashboard.diebold_mariano import dm_test, comparisons


class DieboldMarianoTests(unittest.TestCase):
    def test_one_step_matches_paired_t_with_dm_variance_convention(self):
        d = np.array([1., 2., -1., 3., -2., 2.])
        result = dm_test(d)
        statistic = d.mean() / (d.std(ddof=1) / np.sqrt(len(d)))
        self.assertAlmostEqual(result['dm'], statistic)
        self.assertAlmostEqual(result['p'], 2*t.sf(abs(statistic), len(d)-1))

    def test_sign_symmetry_and_degenerate_variance(self):
        d = np.random.default_rng(8).normal(.3, 1, 100)
        a, b = dm_test(d, 7, 20), dm_test(-d, 7, 20)
        self.assertAlmostEqual(a['dm'], -b['dm'])
        self.assertEqual(a['p'], b['p'])
        self.assertIsNone(dm_test(np.zeros(30))['p'])

    def test_daily_aggregation_holm_and_alignment(self):
        rows = []
        for model, scale in [('A', .1), ('B', 1), ('C', 2)]:
            for i, origin in enumerate(pd.date_range('2025-01-01', periods=60)):
                for h in [1, 2]:
                    rows.append(dict(model=model, origin=origin, symbol='BTC', horizon=h,
                                     volatility_window=7, actual=3., forecast=3+scale*(1+i%5)))
        frame = pd.DataFrame(rows)
        result = comparisons(frame, 'TODOS', 7, 'TODOS', ['A', 'B', 'C'])
        self.assertTrue(result.n.eq(60).all())
        self.assertTrue((result.p_holm >= result.p).all())
        self.assertTrue(result.rezagos_hac.eq(7).all())
        with self.assertRaises(ValueError):
            comparisons(frame.iloc[1:], 'TODOS', 7, 'TODOS', ['A', 'B'])
        frame.loc[0, 'actual'] = 9
        with self.assertRaises(ValueError):
            comparisons(frame, 'TODOS', 7, 'TODOS', ['A', 'B'])


if __name__ == '__main__':
    unittest.main()
