"""Published dashboard callbacks preserve originals and expose exact HAC bounds."""
from pathlib import Path
from itertools import combinations
import sys
import unittest
import warnings
from unittest.mock import patch

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from xgboost import XGBRegressor

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from volatility_dashboard import confidence_intervals as confidence
from volatility_dashboard import data_loader as loader
from volatility_dashboard import runtime_audit
from volatility_dashboard.app import create_app


def components(node, kind):
    if isinstance(node, dict):
        if node.get('type') == kind:
            yield node.get('props', {})
        for value in node.values():
            yield from components(value, kind)
    elif isinstance(node, list):
        for value in node:
            yield from components(value, kind)


def forest_points(figure):
    """Recover drawn endpoints from Plotly's asymmetric error bars."""
    points = {}
    for trace in figure['data']:
        if 'error_x' not in trace:
            continue
        errors = trace['error_x']
        for label, gain, upper, lower in zip(trace['y'], trace['x'],
                errors['array'], errors['arrayminus']):
            name = label[:-2] if label.endswith(' *') else label
            if name in points:
                raise AssertionError('A rival appears twice in the forest.')
            points[name] = dict(gain=gain, ci_low_gain=gain-lower,
                                ci_high_gain=gain+upper, star=label.endswith(' *'))
    return points


class DashboardConfidencePublicationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.client = create_app().server.test_client()
        cls.predictions, _, _ = loader.repository()
        cls.saved = pd.read_csv(ROOT / 'results/har_original_ml_confidence_20261006/2025/7/comparisons.csv')
        cls.saved = cls.saved.set_index('model').loc[list(confidence.RIVALS)]

    def callback(self, models=None):
        values = [('models-symbol', 'TODOS'), ('models-window', 7), ('models-horizon', 'TODOS'),
            ('models-selected', list(loader.FAMILIES) if models is None else models),
            ('models-metric', 'rmse'), ('models-focus', 'XGBoost'),
            ('models-asset', 'BTCUSDT'), ('models-detail-h', 1)]
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', category=DeprecationWarning,
                                    module=r'dash\.development\.base_component')
            response = self.client.post('/_dash-update-component', json={
                'output': 'models-content.children',
                'outputs': {'id': 'models-content', 'property': 'children'},
                'inputs': [{'id': name, 'property': 'value', 'value': value} for name, value in values],
                'state': [], 'changedPropIds': ['models-selected.value']})
        self.assertEqual(response.status_code, 200)
        self.assertNotIn('No se pudo cargar la comparación', response.get_data(as_text=True))
        payload = response.get_json()['response']['models-content']['children']
        response.close()
        tables = [pd.DataFrame(component['data']) for component in components(payload, 'DataTable')
                  if component.get('data')]
        graphs = [component['figure'] for component in components(payload, 'Graph')
                  if 'Ganancia de MSE' in component.get('figure', {}).get('layout', {})
                  .get('xaxis', {}).get('title', {}).get('text', '')]
        self.assertEqual(len(graphs), 1)
        return payload, tables, graphs[0]

    def assert_forest_matches_saved(self, figure):
        points = forest_points(figure)
        self.assertEqual(set(points), set(confidence.RIVALS))
        self.assertEqual(len(points), 6)
        for name, point in points.items():
            with self.subTest(rival=name):
                expected = self.saved.loc[name]
                np.testing.assert_allclose(
                    [point[field] for field in ['gain', 'ci_low_gain', 'ci_high_gain']],
                    expected[['gain', 'ci_low_gain', 'ci_high_gain']].to_numpy(dtype=float),
                    rtol=1e-12, atol=1e-13)
                self.assertEqual(point['star'], bool(expected.star_holm6))
        self.assertEqual(sum(point['star'] for point in points.values()), 5)

    def test_callback_forest_and_six_rival_table_match_saved_hac_results(self):
        payload, tables, figure = self.callback()
        self.assert_forest_matches_saved(figure)
        confidence_tables = [table for table in tables if 'Rival' in table.columns]
        self.assertEqual(len(confidence_tables), 1)
        observed = confidence_tables[0].set_index('Rival').sort_index()
        fields = {'mse_original': 'MSE original', 'mse_rival': 'MSE rival',
            'gain': 'Ganancia MSE', 'ci_low_gain': 'IC 95 % inferior',
            'ci_high_gain': 'IC 95 % superior', 'p_holm6': 'p Holm6',
            'conclusion_5pct': 'Conclusión Holm6'}
        expected = self.saved[list(fields)].rename(columns=fields).round(6).sort_index()
        expected.index.name = 'Rival'
        pd.testing.assert_frame_equal(observed, expected, check_dtype=False)
        self.assertEqual(int(observed['p Holm6'].lt(.05).sum()), 5)
        links = list(components(payload, 'A'))
        self.assertTrue(any(link.get('href') == '/confidence-intervals/' for link in links))

    def test_original_twenty_one_pair_dm_table_remains_unchanged(self):
        _, tables, _ = self.callback()
        dm = [table for table in tables if 'modelo_a' in table.columns]
        self.assertEqual(len(dm), 1)
        observed = dm[0].set_index(['modelo_a', 'modelo_b']).sort_index()
        self.assertEqual(len(observed), 21)
        self.assertEqual(set(observed.index), set(combinations(sorted(loader.FAMILIES), 2)))
        self.assertTrue(observed.n.eq(358).all())
        self.assertTrue(observed.horizonte_hln.eq(7).all())
        self.assertTrue(observed.rezagos_hac.eq(12).all())
        self.assertTrue(observed.estado.eq('OK').all())
        # The published confidence CSV independently freezes every original
        # model's MSE and the original's six DM/Holm21 comparisons. No local
        # research-only DM export is needed to run this test in the checkout.
        means = self.saved.mse_rival.to_dict()
        means[confidence.BENCHMARK] = float(self.saved.mse_original.iloc[0])
        for (a, b), row in observed.iterrows():
            with self.subTest(pair=(a, b)):
                self.assertAlmostEqual(row.mse_a, round(means[a], 6), places=12)
                self.assertAlmostEqual(row.mse_b, round(means[b], 6), places=12)
                self.assertAlmostEqual(row.diferencia_mse, round(means[a]-means[b], 6), places=12)
        har = observed.loc[confidence.BENCHMARK]
        expected = pd.DataFrame(dict(mse_a=self.saved.mse_original, mse_b=self.saved.mse_rival,
            n=self.saved.n, horizonte_hln=self.saved.horizonte_hln, rezagos_hac=self.saved.rezagos_hac,
            diferencia_mse=-self.saved.gain, dm=-self.saved.dm_gain, p=self.saved.p_raw_DM_HLN,
            estado=self.saved.estado, p_holm=self.saved.p_holm21,
            conclusion_5pct=np.where(self.saved.p_holm21.lt(.05),
                'Favorece ' + confidence.BENCHMARK, 'Sin diferencia significativa')))
        expected.index.name = 'modelo_b'
        expected = expected[har.columns].round(6).sort_index()
        pd.testing.assert_frame_equal(har.sort_index(), expected, check_dtype=False)
        self.assertEqual(len(har), 6)
        self.assertTrue(har.mse_a.eq(1.186401).all())
        self.assertTrue(har.n.eq(358).all())
        self.assertEqual(int(har.p_holm.lt(.05).sum()), 3)

    def test_model_subset_filters_ranking_while_forest_keeps_six_rivals(self):
        subset = ['Ridge', 'XGBoost']
        _, tables, figure = self.callback(models=subset)
        self.assert_forest_matches_saved(figure)
        rankings = [table for table in tables if any(str(column).startswith('posici')
                    for column in table.columns)]
        self.assertEqual(len(rankings), 1)
        self.assertEqual(set(rankings[0].model), set(subset))
        self.assertEqual(len(rankings[0]), 2)
        dm = [table for table in tables if 'modelo_a' in table.columns]
        self.assertEqual(len(dm), 1)
        self.assertEqual(len(dm[0]), 1)
        self.assertEqual(set(dm[0][['modelo_a', 'modelo_b']].iloc[0]), set(subset))

    def test_report_png_and_global_csv_routes_serve_exact_saved_artifacts(self):
        artifacts = {'': 'report.html', '2025/7/forest_plot.png': '2025/7/forest_plot.png',
                     '2026/global/comparisons.csv': '2026/global/comparisons.csv'}
        for suffix, relative in artifacts.items():
            with self.subTest(path=relative):
                with self.client.get('/confidence-intervals/' + suffix) as response:
                    self.assertEqual(response.status_code, 200)
                    saved = ROOT / 'results/har_original_ml_confidence_20261006' / relative
                    self.assertEqual(response.data, saved.read_bytes())
                    if relative.endswith('.png'):
                        self.assertEqual(response.data[:8], b'\x89PNG\r\n\x1a\n')
                        self.assertIn('image/png', response.content_type)
                    if relative.endswith('.html'):
                        self.assertIn('text/html', response.content_type)

    def test_confidence_static_route_rejects_directory_traversal(self):
        for path in ['/confidence-intervals/../../README.md',
                     '/confidence-intervals/%2e%2e/%2e%2e/README.md',
                     '/confidence-intervals/..%2F..%2Fsrc/mse_hac_intervals.py',
                     '/confidence-intervals/nonexistent.png']:
            with self.subTest(path=path):
                self.assertEqual(self.client.get(path).status_code, 404)

    def test_runtime_confidence_module_never_replays_models_fits_or_repository_audits(self):
        before = self.predictions.copy(deep=True)
        forbidden = AssertionError('Runtime confidence graphs must only use supplied forecasts.')
        with patch.object(loader, 'repository', side_effect=forbidden), \
             patch.object(loader, 'dataset', side_effect=forbidden), \
             patch.object(loader, 'artifact_for', side_effect=forbidden), \
             patch.object(loader, 'predict', side_effect=forbidden), \
             patch.object(loader, 'predict_bundle', side_effect=forbidden), \
             patch.object(runtime_audit, 'read_audit', side_effect=forbidden), \
             patch.object(Ridge, 'fit', side_effect=forbidden), \
             patch.object(XGBRegressor, 'fit', side_effect=forbidden):
            figure, table, details = confidence.confidence_intervals(
                self.predictions, symbol='TODOS', window=7, horizon='TODOS')
        self.assertEqual(len(table), 6)
        self.assertEqual(details['n'], 358)
        self.assertEqual(details['cells_per_origin'], 28)
        self.assertEqual(details['lags'], 12)
        self.assertEqual(details['significant_wins'], 5)
        self.assertEqual(details['family_size'], 6)
        self.assert_forest_matches_saved(figure.to_plotly_json())
        pd.testing.assert_frame_equal(self.predictions, before)


if __name__ == '__main__':
    unittest.main()
