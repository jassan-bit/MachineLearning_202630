"""The final review displays frozen results without modifying production data."""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from dash import Dash, html
import pandas as pd

from volatility_dashboard import final_review_data as data
from volatility_dashboard.tabs import revision_final as view


def components(node):
    if hasattr(node, 'to_plotly_json'):
        node = node.to_plotly_json()
    if isinstance(node, dict):
        yield node
        for value in node.values():
            yield from components(value)
    elif isinstance(node, (list, tuple)):
        for value in node:
            yield from components(value)


class FinalReviewTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.bundle = Path(self.temp.name)
        self.patch = patch.object(data, 'BUNDLE_ROOT', self.bundle)
        self.patch.start()
        self.addCleanup(self.patch.stop)
        self._fixture()

    def _csv(self, relative, rows):
        path = data.artifact_path(relative)
        path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows).to_csv(path, index=False)

    def _fixture(self):
        image = data.artifact_path(f'{data.FEEDBACK}/diagnostics/intraday_development.png')
        image.parent.mkdir(parents=True, exist_ok=True)
        image.write_bytes(b'PNG fixture: image is served separately')
        metrics = []
        macro = []
        directed = []
        for year in data.YEARS:
            labels = [data.REFERENCE, *data._benchmark_labels('all'),
                      *[f'HAR-XG [{method}]' for method in data.OPTIMIZERS]]
            for i, model in enumerate(labels):
                values = {'model': model, 'mse': .4 + .002*i + (year-2025)*.1,
                          'rmse': .6 + .001*i, 'mae': .3, 'r2': .8}
                macro.append({'year': year, **values})
                for window in data.WINDOWS:
                    metrics.append({'year': year, 'volatility_window': window,
                                    **values, 'mse': values['mse'] + window/1000})
            for method in data.OPTIMIZERS:
                directed.append({'year': year, 'model_a': f'HAR-XG [{method}]',
                    'model_b': data.REFERENCE, 'family_size': 4, 'p': .1,
                    'p_holm': .4, 'diferencia_mse': -.01})
        self._csv(f'{data.RESULTS}/evaluation/macro_by_window.csv', metrics)
        self._csv(f'{data.RESULTS}/evaluation/macro_metrics.csv', macro)
        self._csv(f'{data.RESULTS}/evaluation/directed_dm_all_years.csv', directed)
        units = {'mse': 'percentage_points_squared', 'rmse': 'percentage_points',
                 'mae': 'percentage_points'}
        self._csv(f'{data.RESULTS}/evaluation/metric_intervals.csv', [
            {'year': row['year'], 'model': row['model'], 'metric': metric,
             'estimate': row[metric], 'ci_lower': row[metric]-.01,
             'ci_upper': row[metric]+.01, 'nominal_coverage': .95,
             'ci_method': 'paired stationary bootstrap percentile', 'unit': unit}
            for row in macro for metric, unit in units.items()])
        self._csv(f'{data.RESULTS}/evaluation/paired_effects_vs_original.csv', [
            {'year': row['year'], 'model': row['model'], 'reference': data.REFERENCE,
             'mse_gain': .4+(row['year']-2025)*.1-row['mse'],
             'gain_ci_lower': .4+(row['year']-2025)*.1-row['mse']-.02,
             'gain_ci_upper': .4+(row['year']-2025)*.1-row['mse']+.02}
            for row in macro if row['model'].startswith('HAR-XG [')])
        protocol_path = data.artifact_path(f'{data.RESULTS}/evaluation/protocol.json')
        protocol_path.write_text(json.dumps({'intervals': {'reps': 2000,
            'block_length': 34, 'coverage': .95,
            'method': 'paired stationary bootstrap percentile; individual, not simultaneous'},
            'seed': 42}), encoding='utf-8')
        dm = []
        for window in data.WINDOWS:
            for method in data.OPTIMIZERS:
                for family in data.FAMILIES:
                    dm.append({'volatility_window': window, 'method': method,
                        'family': family, 'model_a': data.REFERENCE,
                        'model_b': f'{family} [{method}]', 'mse_a': .4, 'mse_b': .5,
                        'n': 358, 'diferencia_mse': -.1, 'dm': -2.4,
                        'p': 1.7e-9, 'p_holm7': .03, 'p_holm28': .2,
                        'p_holm112': .8, 'estado': 'OK'})
        self._csv(f'{data.WINDOW_REVIEW}/dm_all_112.csv', dm)
        self._csv(f'{data.WINDOW_REVIEW}/window_summary.csv', [
            {'volatility_window': window, 'reference_rmse': .6, 'reference_r2': .8,
             'original_lower_mse': 28, 'original_wins_holm7': 21,
             'original_wins_holm28': 7, 'original_wins_holm112': 0}
            for window in data.WINDOWS])

    def test_filters_use_year_window_and_optimizer_keep_original(self):
        selected = data.metrics(2026, 'genetic', 14)
        self.assertEqual(len(selected), 8)
        self.assertIn(data.REFERENCE, selected.model.values)
        self.assertTrue(all(model == data.REFERENCE or model.endswith('[genetic]')
                            for model in selected.model))
        self.assertAlmostEqual(selected.iloc[0].mse, .514)
        self.assertEqual(len(data.metrics(2025, 'all', 7)), 29)
        with self.assertRaises(ValueError):
            data.metrics(2027, 'grid', 7)
        with self.assertRaises(ValueError):
            data.metrics(2025, 'grid', 8)

    def test_fixed_p_families_are_not_recomputed_after_filtering(self):
        full = data.dm_2025('all', 7)
        grid = data.dm_2025('grid', 7)
        self.assertEqual(len(full), 28)
        self.assertEqual(len(grid), 7)
        expected = full.loc[full.method.eq('grid')].reset_index(drop=True)
        pd.testing.assert_frame_equal(expected, grid)
        self.assertEqual(data.dm_decision(grid.iloc[0], 7), 'Favorece al original')
        self.assertEqual(data.dm_decision(grid.iloc[0], 28), 'No rechaza H₀')
        self.assertEqual(data.dm_decision(grid.iloc[0], 112), 'No rechaza H₀')
        positive = grid.iloc[0].copy()
        positive['diferencia_mse'] = .1
        self.assertEqual(data.dm_decision(positive, 7), 'Favorece al benchmark')
        grid.loc[0, 'p_holm112'] = 0
        self.assertEqual(data.dm_2025('grid', 7).iloc[0].p_holm112, .8)

    def test_four_har_tests_are_global_and_separate(self):
        frame = data.candidate_global(2026)
        self.assertEqual(len(frame), 4)
        self.assertTrue(frame.p_holm.eq(.4).all())
        self.assertTrue(frame.mse_original.eq(.5).all())
        self.assertTrue(frame.decision_holm4.eq('No rechaza H₀').all())

    def test_2026_never_loads_2025_dm_panel(self):
        with patch.object(data, 'dm_2025', side_effect=AssertionError('Wrong year')) as dm:
            rendered = view.render(2026, 'random', 21)
        dm.assert_not_called()
        self.assertNotEqual(getattr(rendered, 'className', None), 'notice')
        columns = [column['name'] for node in components(rendered)
                   if node.get('type') == 'DataTable'
                   for column in node['props']['columns']]
        self.assertIn('p Holm4', columns)
        self.assertNotIn('p Holm112', columns)

    def test_callback_serializes_and_only_reads_bundle(self):
        def hashes():
            return {str(path): hashlib.sha256(path.read_bytes()).hexdigest()
                    for path in self.bundle.rglob('*') if path.is_file()}
        before = hashes()
        app = Dash(__name__, suppress_callback_exceptions=True)
        app.layout = html.Div(view.layout())
        view.register(app)
        with patch.object(pd.DataFrame, 'to_csv', side_effect=AssertionError('Read only')):
            response = app.server.test_client().post('/_dash-update-component', json={
                'output': 'final-review-content.children',
                'outputs': {'id': 'final-review-content', 'property': 'children'},
                'inputs': [
                    {'id': 'final-review-year', 'property': 'value', 'value': 2025},
                    {'id': 'final-review-optimizer', 'property': 'value', 'value': 'grid'},
                    {'id': 'final-review-window', 'property': 'value', 'value': 7}],
                'state': [], 'changedPropIds': ['final-review-window.value']})
        self.assertEqual(response.status_code, 200)
        payload = response.get_json()
        serialized = json.dumps(payload, ensure_ascii=False)
        self.assertIn('p Holm112', serialized)
        self.assertIn('1.7e-09', serialized)
        self.assertNotIn('NaN', serialized)
        self.assertEqual(before, hashes())

    def test_missing_bundle_has_controls_and_visible_fallback(self):
        with patch.object(data, 'metrics', side_effect=OSError('Missing bundle')) as loader:
            controls = view.layout()
            loader.assert_not_called()
            with self.assertLogs(view.LOGGER, level='ERROR'):
                rendered = view.render(2025, 'all', 7)
        ids = {node['props']['id']: node['props'] for node in components(controls)
               if 'id' in node.get('props', {})}
        self.assertEqual(ids['final-review-window']['value'], 7)
        self.assertEqual(rendered.className, 'notice')
        self.assertIn('no está disponible', rendered.children[0].children)

    def test_confined_artifact_paths_and_urls(self):
        for value in ('../secret', '/etc/passwd', 'results/../../secret',
                      'C:/secret', 'models/test.joblib'):
            with self.assertRaises(ValueError):
                data.artifact_path(value)
        self.assertEqual(data.public_url(f'{data.RESULTS}/evaluation/report.html'),
                         f'/final-review/{data.RESULTS}/evaluation/report.html')


if __name__ == '__main__':
    unittest.main()
