"""The compact web calendar preserves the audited experimental dates."""
from pathlib import Path
import shutil
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from optimize_minute_svr import calendars, load_panel
from volatility_dashboard.calendar import saved_calendar


class DashboardCalendarTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.panel = load_panel()
        cls.out = ROOT / 'results/optimized_minute_2023_2025'

    def test_matches_original_native_calendar_exactly(self):
        original_folds, original_eligible, original_audit = calendars(self.panel)
        folds, eligible, audit = saved_calendar(self.panel, self.out)
        np.testing.assert_array_equal(eligible, original_eligible)
        pd.testing.assert_frame_equal(audit, original_audit)
        self.assertEqual(len(folds), len(original_folds))
        for actual, expected in zip(folds, original_folds):
            np.testing.assert_array_equal(actual[0], expected[0])
            np.testing.assert_array_equal(actual[1], expected[1])

    def test_rejects_changed_target_end(self):
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory)
            shutil.copyfile(self.out / 'native_cv_audit.csv', out / 'native_cv_audit.csv')
            calendar = pd.read_csv(self.out / 'calendar.csv')
            calendar.loc[0, 'target_end'] = calendar.loc[0, 'origin']
            calendar.to_csv(out / 'calendar.csv', index=False)
            with self.assertRaisesRegex(ValueError, 'target_end'):
                saved_calendar(self.panel, out)

    def test_rejects_changed_native_code(self):
        with patch('volatility_dashboard.calendar.inspect.getsource', return_value='changed source'):
            with self.assertRaisesRegex(ValueError, 'código nativo cambió'):
                saved_calendar(self.panel, self.out)

    def test_rejects_removed_training_origin(self):
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory)
            shutil.copyfile(self.out / 'native_cv_audit.csv', out / 'native_cv_audit.csv')
            calendar = pd.read_csv(self.out / 'calendar.csv').iloc[1:]
            calendar.to_csv(out / 'calendar.csv', index=False)
            with self.assertRaisesRegex(ValueError, 'Fechas de train modificadas'):
                saved_calendar(self.panel, out)


if __name__ == '__main__':
    unittest.main()
