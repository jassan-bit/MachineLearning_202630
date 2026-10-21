"""Check leakage boundaries, calendar gaps and the numerical diagnostics."""
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
import professor_feedback_diagnostics as audit
from improve_classical_forecast import har_features
from optimize_minute_svr import minute_features


class FeedbackDiagnosticsTests(unittest.TestCase):
    def test_output_is_new_local_results_directory(self):
        with tempfile.TemporaryDirectory(dir=audit.ROOT / 'results') as directory:
            output = Path(directory) / 'new_audit'
            self.assertEqual(audit.prepare_output_directory(output), output.resolve())
            (output / 'metadata.json').write_text('{}', encoding='utf-8')
            with self.assertRaises(FileExistsError):
                audit.prepare_output_directory(output)
        with self.assertRaises(ValueError):
            audit.prepare_output_directory(audit.ROOT / 'delivery' / 'unsafe_output')
        with self.assertRaises(ValueError):
            audit.prepare_output_directory(audit.ROOT / 'results')

    def test_prefix_does_not_depend_on_test_values(self):
        with tempfile.TemporaryDirectory(dir=audit.ROOT) as directory:
            path = Path(directory) / 'prices.npy'
            values = np.arange(3*1440, dtype=float) + 1
            np.save(path, values)
            before, before_manifest = audit.development_prefix(path, 2)
            values[2*1440:] = 1e99
            np.save(path, values)
            after, after_manifest = audit.development_prefix(path, 2)
            np.testing.assert_array_equal(before, after)
            self.assertEqual(before_manifest, after_manifest)

    def test_vif_matches_auxiliary_ols_and_marks_constants_and_dependencies(self):
        rng = np.random.default_rng(204)
        X = rng.normal(size=(250, 3))
        X[:, 1] += 2 * X[:, 0]
        rows, summary = audit.vif_diagnostics(X, ['a', 'b', 'c'])
        for j, row in enumerate(rows):
            other = np.delete(X, j, axis=1)
            r2 = LinearRegression().fit(other, X[:,j]).score(other, X[:,j])
            self.assertAlmostEqual(row['vif'], 1/(1-r2))
        self.assertEqual(summary['centered_rank'], 3)
        duplicated = np.column_stack([X, X[:, 0], np.ones(len(X))])
        rows, summary = audit.vif_diagnostics(duplicated, ['a','b','c','duplicate','constant'])
        self.assertTrue(np.isinf(rows[0]['vif']))
        self.assertEqual(rows[-1]['status'], 'constant_undefined')
        self.assertEqual(summary['centered_rank'], 3)

    def test_mi_finds_nonlinear_signal_and_never_fits_constant_feature(self):
        rng = np.random.default_rng(113)
        signal = rng.uniform(-2, 2, 600)
        noise = rng.normal(size=600)
        X = np.column_stack([signal, noise, np.ones(600)])
        rows = audit.mutual_information(X, signal**2+.02*rng.normal(size=600), ['signal','noise','constant'])
        self.assertGreater(rows[0]['mi_nats'], rows[1]['mi_nats']+1)
        self.assertEqual(rows[2]['mi_nats'], 0.)
        self.assertTrue(rows[2]['constant_feature'])

    def test_har_names_match_actual_columns_and_zero_decay(self):
        rng = np.random.default_rng(5)
        values = 100*np.exp(np.cumsum(rng.normal(0,.0004,60*1440)))
        index = pd.date_range('2023-01-01', periods=60, tz='UTC')
        close = audit.daily_closes(values, index)
        X, _, base = har_features(close, minute_features(values), 7)
        self.assertEqual(X.shape[1], len(audit.har_names()))
        j = audit.har_names().index('daily_return_mean_1d_over_base')
        expected = (100*np.log(close).diff()).to_numpy()/base
        np.testing.assert_allclose(X[:,j], expected, equal_nan=True)
        # At span=1, downside=max(-r,0)=(abs(r)-r)/2 is an exact
        # linear dependence, rather than evidence that the data are corrupt.
        np.testing.assert_allclose(X[:,2], (X[:,1]-X[:,0])/2, equal_nan=True, atol=1e-14)
        good = np.isfinite(X).all(axis=1)
        self.assertFalse(audit.nonconstant_columns(X[good])[-1])

    def test_daily_and_intraday_gaps_are_preserved(self):
        rng = np.random.default_rng(42)
        index = pd.date_range('2023-12-28', periods=9, tz='UTC')
        values = 100*np.exp(np.cumsum(rng.normal(0,.0003,len(index)*1440)))
        values[3*1440+81] = np.nan
        closes = audit.daily_closes(values,index)
        self.assertTrue(np.isnan(closes.iloc[3]))
        rows, hours, summaries = audit.intraday_profiles(values,index,repeats=25)
        minute = pd.DataFrame(rows).query("period == 'development_2023_2024'").set_index('minute_of_day_utc')
        self.assertEqual(minute.loc[81,'n_finite_returns'],8)
        self.assertEqual(minute.loc[82,'n_finite_returns'],8)
        self.assertEqual(minute.loc[80,'n_finite_returns'],9)
        self.assertEqual(summaries[0]['n_complete_return_days'],7)
        hourly = pd.DataFrame(hours).query("period == 'development_2023_2024'")
        self.assertAlmostEqual(hourly.mean_daily_normalized_squared_return.mean(),1.)
        self.assertEqual(set(hourly.n_calendar_days), {9})
        positions = audit.calendar_block_indices(9,repeats=20,block=3)
        np.testing.assert_array_equal((np.diff(positions[:,:3],axis=1)%9), np.ones((20,2),int))

    def test_training_mask_excludes_labels_at_or_after_cutoff(self):
        index = pd.date_range('2023-01-01','2025-01-01',inclusive='left',tz='UTC')
        panel = pd.DataFrame({'a':np.ones(len(index)), 'b':np.ones(len(index))},index=index)
        panel.iloc[82,1] = np.nan
        eligible = audit.shared_eligible(panel)
        self.assertFalse(eligible[82-7:82+28].any())
        train = eligible & (index+pd.Timedelta(days=7) < audit.TRAIN_END)
        self.assertLess((index[train]+pd.Timedelta(days=7)).max(),audit.TRAIN_END)


if __name__ == '__main__':
    unittest.main()
