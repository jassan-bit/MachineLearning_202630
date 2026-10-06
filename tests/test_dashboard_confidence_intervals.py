"""Focused checks for the published seven-method HAC forest and its filters."""
import json
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from volatility_dashboard.confidence_intervals import (
    BENCHMARK, MODELS, RIVALS, _holm, confidence_intervals,
)
from volatility_dashboard.diebold_mariano import dm_test


def predictions(year=2025, n=70):
    rows = []
    for day, origin in enumerate(pd.date_range(f'{year}-01-01', periods=n, tz='UTC')):
        for asset_index, asset in enumerate(['BTCUSDT', 'ETHUSDT']):
            for horizon in range(1, 8):
                actual = 3+asset_index*.4+horizon*.02+day*.01
                error = .8+.16*np.sin(day/6)+asset_index*.06+horizon*.005
                for index, model in enumerate(MODELS):
                    rows.append(dict(model=model, symbol=asset, volatility_window=7,
                                     origin=origin, horizon=horizon, actual=actual,
                                     forecast=actual-error*(1+index*.09)))
    return pd.DataFrame(rows)


class DashboardConfidenceIntervalTests(unittest.TestCase):
    def test_paired_daily_sample_size_and_figure_use_only_original_methods(self):
        frame = predictions()
        foreign = frame[frame.model.eq(BENCHMARK)].copy()
        foreign['model'] = 'HAR-Ridge + XGBoost residual'
        foreign['forecast'] = np.nan
        frame = pd.concat([frame, foreign], ignore_index=True)
        untouched = frame.copy(deep=True)
        figure, table, details = confidence_intervals(frame)
        self.assertEqual(details['n'], 70)
        self.assertEqual(details['cells_per_origin'], 14)
        self.assertEqual(details['lags'], 12)
        self.assertEqual(details['family_size'], 6)
        self.assertEqual(table.model.tolist(), list(RIVALS))
        self.assertTrue(table.n.eq(70).all())
        self.assertEqual(len(table), 6)
        self.assertNotIn('residual', figure.to_json())
        self.assertEqual(len(figure.layout.shapes), 1)
        pd.testing.assert_frame_equal(frame, untouched)
        json.loads(figure.to_json())

    def test_asset_and_single_horizon_filters_preserve_true_pairing(self):
        frame = predictions()
        _, table, details = confidence_intervals(frame, 'BTCUSDT', 7, 2, year=2025)
        self.assertEqual(details['cells_per_origin'], 1)
        self.assertEqual(details['horizon_hln'], 2)
        self.assertEqual(details['lags'], 7)
        selected = frame[frame.symbol.eq('BTCUSDT') & frame.horizon.eq(2)].copy()
        selected['loss'] = (selected.actual-selected.forecast)**2
        pivot = selected.pivot(index='origin', columns='model', values='loss')
        for row in table.itertuples():
            gain = pivot[row.model]-pivot[BENCHMARK]
            self.assertAlmostEqual(row.gain, gain.mean())
            test = dm_test(gain, horizon=2, lags=7)
            self.assertAlmostEqual(row.dm_gain, test['dm'])
            self.assertAlmostEqual(row.p_raw_DM_HLN, test['p'])

    def test_hac_interval_matches_independent_bartlett_quadratic_form(self):
        _, table, details = confidence_intervals(predictions(), 'ETHUSDT', 7, 1)
        frame = predictions()
        frame = frame[frame.symbol.eq('ETHUSDT') & frame.horizon.eq(1)].copy()
        frame['loss'] = (frame.actual-frame.forecast)**2
        pivot = frame.pivot(index='origin', columns='model', values='loss')
        gain = (pivot['Ridge']-pivot[BENCHMARK]).to_numpy()
        centered = gain-gain.mean()
        distances = np.abs(np.arange(len(gain))[:, None]-np.arange(len(gain))[None, :])
        kernel = np.maximum(0., 1-distances/(details['lags']+1))
        expected_se = np.sqrt((centered@kernel@centered)/len(gain)**2)
        row = table[table.model.eq('Ridge')].iloc[0]
        self.assertAlmostEqual(row.se_HAC, expected_se)
        self.assertAlmostEqual(row.ci_high_gain-row.gain, 1.959963984540054*expected_se)
        self.assertAlmostEqual(row.gain-row.ci_low_gain, 1.959963984540054*expected_se)

    def test_stars_follow_holm6_and_not_the_individual_interval(self):
        raw = [.002308, .150037, .000027, .000106, .017882, .004809]
        def fake_dm(differential, horizon, lags):
            index = fake_dm.calls
            fake_dm.calls += 1
            return dict(n=len(differential), horizonte_hln=horizon, rezagos_hac=lags,
                        diferencia_mse=float(np.mean(differential)), dm=3., p=raw[index], estado='OK')
        fake_dm.calls = 0
        with patch('volatility_dashboard.confidence_intervals.dm_test', side_effect=fake_dm):
            figure, table, details = confidence_intervals(predictions())
        self.assertEqual(int(table.star.sum()), 5)
        self.assertEqual(details['significant_wins'], 5)
        self.assertFalse(bool(table.iloc[1].star))
        self.assertGreater(table.iloc[1].ci_low_gain, 0)
        self.assertEqual(sum('*' in label for trace in figure.data for label in trace.y), 5)

    def test_a_better_rival_is_shown_with_negative_gain_and_its_own_winner(self):
        frame = predictions()
        selected = frame.model.eq('Ridge')
        frame.loc[selected, 'forecast'] = (frame.loc[selected, 'actual']
                                          -.2*(frame.loc[selected, 'actual']-frame.loc[selected, 'forecast']))
        figure, table, details = confidence_intervals(frame)
        rival = table[table.model.eq('Ridge')].iloc[0]
        self.assertLess(rival.gain, 0)
        self.assertLess(rival.ci_high_gain, 0)
        self.assertTrue(rival.star)
        self.assertEqual(rival.winner, 'Ridge')
        self.assertEqual(details['significant_losses'], 1)
        trace = next(item for item in figure.data if item.name == 'Rival favorecido · Holm6')
        self.assertEqual(list(trace.y), ['Ridge *'])
        self.assertEqual(trace.marker.color, '#b91c1c')

    def test_planned_degenerate_tests_stay_in_holm_family(self):
        values = _holm([.01, .02, None, .5, .7, .8], 6)
        self.assertAlmostEqual(values[0], .06)
        self.assertAlmostEqual(values[1], .1)
        self.assertTrue(np.isnan(values[2]))
        frame = predictions()
        reference = frame[frame.model.eq(BENCHMARK)].set_index(['origin', 'symbol', 'horizon']).forecast
        frame['forecast'] = [reference.loc[(row.origin, row.symbol, row.horizon)] for row in frame.itertuples()]
        figure, table, details = confidence_intervals(frame)
        self.assertTrue(table.p_holm6.isna().all())
        self.assertTrue(table.ci_low_gain.isna().all())
        self.assertFalse(table.star.any())
        self.assertEqual(details['not_evaluable'], 6)
        self.assertEqual(figure.data[0].name, 'IC no evaluable')

    def test_common_years_require_explicit_year_and_filter_saved_data(self):
        frame = pd.concat([predictions(2025), predictions(2026)], ignore_index=True)
        with self.assertRaisesRegex(ValueError, 'único año'):
            confidence_intervals(frame)
        _, table, details = confidence_intervals(frame, year=2026)
        self.assertEqual(details['year'], 2026)
        self.assertEqual(details['start'], '2026-01-01')
        self.assertTrue(table.n.eq(70).all())

    def test_mismatched_targets_duplicates_missing_cells_and_calendar_are_rejected(self):
        good = predictions()
        wrong = good.copy()
        wrong.loc[0, 'actual'] += .5
        duplicate = pd.concat([good, good.iloc[[0]]], ignore_index=True)
        missing_one = good.drop(good.index[0])
        missing_all_models_cell = good[~(good.origin.eq(good.origin.min()) & good.symbol.eq('BTCUSDT') & good.horizon.eq(1))]
        missing_day = good[~good.origin.eq(good.origin.sort_values().unique()[3])]
        missing_model = good[~good.model.eq('Lasso')]
        for bad in [wrong, duplicate, missing_one, missing_all_models_cell, missing_day, missing_model]:
            with self.subTest(rows=len(bad)), self.assertRaises(ValueError):
                confidence_intervals(bad)

    def test_bad_filters_and_nonfinite_selected_forecasts_are_rejected(self):
        frame = predictions()
        for arguments in [dict(window=8), dict(horizon=0), dict(horizon=True), dict(alpha=0),
                          dict(alpha=1), dict(alpha=True), dict(year=2030), dict(symbol='UNKNOWN')]:
            with self.subTest(arguments=arguments), self.assertRaises(ValueError):
                confidence_intervals(frame, **arguments)
        frame.loc[0, 'forecast'] = np.nan
        with self.assertRaisesRegex(ValueError, 'finitos'):
            confidence_intervals(frame)


if __name__ == '__main__':
    unittest.main()
