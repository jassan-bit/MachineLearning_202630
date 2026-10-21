"""Verify causal daily-HAR inputs, warmup exclusions and fixed OLS fitting."""
from pathlib import Path
import sys
import unittest
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))
from professor_feedback_daily_har import daily_har_inputs, finite_train, fit_reference, forecast


class DailyHARTests(unittest.TestCase):
    def test_feature_values_and_future_perturbation(self):
        values = np.arange(1., 101.)
        X = daily_har_inputs(values)
        np.testing.assert_allclose(X[30], [31., np.mean(values[24:31]), np.mean(values[3:31])])
        changed = values.copy()
        changed[61:] = 1e9
        np.testing.assert_allclose(X[:61], daily_har_inputs(changed)[:61], equal_nan=True)

    def test_initial_and_post_gap_history_excludes_train_without_compressing_time(self):
        values = np.ones(110)
        values[:7] = np.nan
        values[50] = np.nan
        X = daily_har_inputs(values)
        y = np.ones((110,7))
        used = finite_train(X,y,np.arange(10,100))
        np.testing.assert_array_equal(used,np.r_[np.arange(34,50),np.arange(78,100)])
        self.assertTrue(np.isnan(X[50:78,2]).all())
        self.assertEqual(X[78,2],1.)

    def test_fit_does_not_read_future_labels_or_inputs(self):
        rng = np.random.default_rng(219)
        X = daily_har_inputs(rng.uniform(.1,4,140))
        y = np.column_stack([X@np.array([.4,.3,.2])+h*.1 for h in range(7)])
        train = np.arange(28,80)
        first, used = fit_reference(X,y,train)
        changed_X, changed_y = X.copy(), y.copy()
        changed_X[80:] = 1e12
        changed_y[80:] = -1e12
        second, used_again = fit_reference(changed_X,changed_y,train)
        np.testing.assert_array_equal(used,used_again)
        np.testing.assert_allclose(first.coef_,second.coef_)
        np.testing.assert_allclose(first.intercept_,second.intercept_)
        self.assertTrue(first.fit_intercept)
        self.assertEqual(first.coef_.shape,(7,3))

    def test_incomplete_test_history_fails_instead_of_changing_test_sample(self):
        X = daily_har_inputs(np.arange(1.,101.))
        y = np.ones((100,7))
        model,_ = fit_reference(X,y,np.arange(28,60))
        with self.assertRaisesRegex(ValueError,'test origin'):
            forecast(model,X,np.array([2,65]))


if __name__ == '__main__':
    unittest.main()
