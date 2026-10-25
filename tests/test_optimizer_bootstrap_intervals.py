"""Optimizer-specific global bootstrap intervals preserve the published study."""
import json
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from volatility_dashboard import final_review_data as data
from volatility_dashboard.tabs import revision_final as view


def nodes(node):
    if hasattr(node, 'to_plotly_json'):
        node = node.to_plotly_json()
    if isinstance(node, dict):
        yield node
        for value in node.values():
            yield from nodes(value)
    elif isinstance(node, (tuple, list)):
        for value in node:
            yield from nodes(value)


class OptimizerBootstrapTests(unittest.TestCase):
    def test_all_methods_and_years_have_exact_published_global_metric_bounds(self):
        source = data.csv(f'{data.RESULTS}/evaluation/metric_intervals.csv')
        for year in data.YEARS:
            self.assertEqual(len(data.optimizer_intervals(year, 'all')), 29*3)
            for method in data.OPTIMIZERS:
                with self.subTest(year=year, method=method):
                    selected = data.optimizer_intervals(year, method)
                    self.assertEqual(len(selected), 8*3)
                    self.assertEqual(set(selected.model), {data.REFERENCE, *data._benchmark_labels(method)})
                    expected = source[source.year.eq(year) & source.model.isin(selected.model)
                                      & source.metric.isin(['mse', 'rmse', 'mae'])]
                    pd.testing.assert_frame_equal(selected,
                        expected.sort_values(['metric', 'estimate', 'model']).reset_index(drop=True))
                    self.assertTrue(selected.year.eq(year).all())

    def test_protocol_uses_recorded_replicates_and_does_not_claim_grid_original_settings(self):
        self.assertEqual(data.bootstrap_protocol(),
            dict(repetitions=2000, block_length=34, seed=42, coverage=.95))

    def test_plot_draws_exact_endpoints_for_every_optimizer_without_significance_stars(self):
        for year in data.YEARS:
            for method in ['all', *data.OPTIMIZERS]:
                section = view._bootstrap_section(year, method)
                graphs = [node['props']['figure'] for node in nodes(section) if node.get('type') == 'Graph']
                self.assertEqual(len(graphs), 1)
                figure = graphs[0]
                expected = data.optimizer_intervals(year, method).query("metric == 'mse'").set_index('model')
                traces = [trace for trace in figure.data if trace.meta
                          and trace.meta.get('role') == 'optimizer_bootstrap_interval']
                self.assertEqual(len(traces), len(expected))
                for trace in traces:
                    row = expected.loc[trace.meta['model']]
                    np.testing.assert_allclose(trace.x, [row.ci_lower, row.ci_upper], rtol=0, atol=0)
                    self.assertFalse(any('*' in label for label in trace.y))
                serialized = json.dumps(section, default=lambda value: value.to_plotly_json(), ensure_ascii=False)
                self.assertIn('no cambia con el filtro superior de ventana', serialized)
                self.assertIn('bootstrap-update', serialized)
                self.assertIn('no constituye un contraste', serialized)

    def test_global_section_is_identical_when_only_window_filter_changes(self):
        first = view._render(2025, 'random', 7)
        second = view._render(2025, 'random', 28)
        def section(tree):
            return next(node for node in nodes(tree)
                        if node.get('props', {}).get('id') == 'optimizer-bootstrap-update')
        one = json.dumps(section(first), default=lambda value: value.to_plotly_json(), sort_keys=True)
        two = json.dumps(section(second), default=lambda value: value.to_plotly_json(), sort_keys=True)
        self.assertEqual(one, two)

    def test_corrupt_selected_intervals_are_rejected(self):
        path = f'{data.RESULTS}/evaluation/metric_intervals.csv'
        original = data.csv
        good = original(path)
        target = good.index[good.year.eq(2025) & good.model.eq('Ridge [random]') & good.metric.eq('mse')][0]
        variants = [good.drop(target), pd.concat([good, good.loc[[target]]], ignore_index=True)]
        for column, value in [('ci_lower', np.inf), ('ci_lower', 100),
                              ('estimate', 100), ('nominal_coverage', .9),
                              ('ci_method', 'normal'), ('unit', 'wrong')]:
            corrupt = good.copy()
            corrupt.loc[target, column] = value
            variants.append(corrupt)
        for corrupt in variants:
            def read(relative, *args, **kwargs):
                return corrupt.copy() if relative == path else original(relative, *args, **kwargs)
            with self.subTest(rows=len(corrupt)), patch.object(data, 'csv', side_effect=read):
                with self.assertRaises(ValueError):
                    data.optimizer_intervals(2025, 'random')

    def test_har_gain_orientation_is_original_minus_candidate_for_each_method_and_year(self):
        for year in data.YEARS:
            frame = data.candidate_global(year)
            self.assertEqual(len(frame), 4)
            np.testing.assert_allclose(frame.mse_gain, frame.mse_original-frame.mse)
            # These frozen individual intervals include zero for all four HAR candidates.
            self.assertTrue(frame.gain_ci_lower.lt(0).all())
            self.assertTrue(frame.gain_ci_upper.gt(0).all())
            self.assertTrue(frame.decision_holm4.eq('No rechaza H₀').all())

    def test_missing_global_source_has_explicit_fallback_not_a_normal_interval(self):
        with patch.object(data, 'optimizer_intervals', side_effect=OSError('Missing source')):
            with self.assertLogs(view.LOGGER, level='WARNING'):
                section = view._bootstrap_section(2026, 'bayesian')
        self.assertEqual(section.id, 'optimizer-bootstrap-update')
        self.assertFalse(any(node.get('type') == 'Graph' for node in nodes(section)))


if __name__ == '__main__':
    unittest.main()
