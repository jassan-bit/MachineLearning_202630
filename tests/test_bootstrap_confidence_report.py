"""Publication inputs must retain complete paired consecutive daily panels."""
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from src.build_bootstrap_confidence_report import LEGACY, read_daily, view_details


class BootstrapReportTests(unittest.TestCase):
    def test_release_identifies_bootstrap_and_the_actual_runtime_commit(self):
        from volatility_dashboard.app import create_app
        with patch.dict('os.environ', {'RENDER_GIT_COMMIT': 'test-runtime-commit'}):
            response = create_app().server.test_client().get('/bootstrap-release.json')
        self.assertEqual(response.status_code, 200)
        details = response.get_json()
        self.assertEqual(details['method'], 'stationary_bootstrap_percentile')
        self.assertEqual(details['deployed_commit'], 'test-runtime-commit')
        self.assertEqual(details['highlight'], 'yellow')
        self.assertEqual(details['repetitions'], 5000)
        self.assertEqual(details['views'], 10)

    def test_all_ten_saved_panels_have_the_declared_calendars_and_block_lengths(self):
        for year, n in [(2025, 358), (2026, 236)]:
            for scope in ['global', 7, 14, 21, 28]:
                with self.subTest(year=year, scope=scope):
                    daily = read_daily(LEGACY / str(year) / str(scope) / 'daily_loss.csv', year)
                    details = view_details(daily, year, scope, 5000, 42)
                    self.assertEqual(len(daily), n)
                    self.assertEqual(details['bootstrap_block_length'],
                                     34 if scope == 'global' else scope+6)

    def test_corrupt_loss_panels_fail_instead_of_silently_dropping_or_reordering_dates(self):
        good = pd.read_csv(LEGACY / '2025/7/daily_loss.csv')
        nan = good.copy()
        nan.iloc[0, 1] = np.nan
        negative = good.copy()
        negative.iloc[0, 1] = -1
        wrong_year = good.copy()
        wrong_year['origin'] = pd.to_datetime(wrong_year.origin, utc=True) + pd.DateOffset(years=1)
        bad_cases = [good.iloc[::-1], good.iloc[1:],
                     pd.concat([good.iloc[:-1], good.iloc[[0]]]),
                     good.drop(columns=['Ridge']), nan, negative, wrong_year]
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / 'daily_loss.csv'
            for bad in bad_cases:
                with self.subTest(rows=len(bad), columns=len(bad.columns)):
                    bad.to_csv(target, index=False)
                    with self.assertRaises(ValueError):
                        read_daily(target, 2025)


if __name__ == '__main__':
    unittest.main()
