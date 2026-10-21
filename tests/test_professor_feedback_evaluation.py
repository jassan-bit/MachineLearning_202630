"""Protect the inferential unit and the frozen comparison family."""
from pathlib import Path
import sys
import unittest
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from evaluate_professor_feedback import daily_losses, comparison_family, normalized


class FeedbackEvaluationTests(unittest.TestCase):
    def test_series_are_averaged_within_day_before_statistical_test(self):
        days = pd.date_range('2026-01-01', periods=100, tz='UTC')
        frame = pd.DataFrame([dict(origin=day, model=model, actual=0., forecast=error)
            for day in days for model, error in [('a', 1.), ('a', 3.), ('b', 2.), ('b', 4.)]])
        losses = daily_losses(frame)
        self.assertEqual(losses.shape, (100, 2))
        np.testing.assert_array_equal(losses.a, 5.)
        np.testing.assert_array_equal(losses.b, 10.)

    def test_holm_adjusts_the_declared_family_and_preserves_paired_day_count(self):
        rng = np.random.default_rng(42)
        daily = pd.DataFrame({'candidate': rng.uniform(.1, .3, 100),
                              'ref1': rng.uniform(.2, .4, 100),
                              'ref2': rng.uniform(.1, .4, 100)})
        output = comparison_family(daily, 'candidate', ['ref1', 'ref2'], 14)
        self.assertTrue(output.n.eq(100).all())
        self.assertTrue((output.p_holm >= output.p).all())
        self.assertEqual(len(output), 2)

    def test_incomplete_panel_is_rejected_instead_of_silently_dropping_dates(self):
        frame = pd.DataFrame([dict(model='a', symbol='BTCUSDT', volatility_window=7,
            origin='2026-01-01', horizon=1, actual=1., forecast=1.)])
        with self.assertRaisesRegex(ValueError, 'calendar'):
            normalized(frame, 2026)


if __name__ == '__main__':
    unittest.main()
