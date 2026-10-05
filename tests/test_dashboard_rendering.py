"""Rendering resources must be available before dynamic tabs are opened."""
from html.parser import HTMLParser
import unittest
from unittest.mock import patch

import plotly.graph_objects as go

from volatility_dashboard.app import create_app
from volatility_dashboard.tabs import modelos
from volatility_dashboard.utils import graph_card


class ScriptSources(HTMLParser):
    def __init__(self):
        super().__init__()
        self.sources = []

    def handle_starttag(self, tag, attrs):
        if tag == 'script':
            source = dict(attrs).get('src')
            if source:
                self.sources.append(source)


class DashboardRenderingTests(unittest.TestCase):
    def test_comparison_controls_do_not_require_cold_repository_audit(self):
        client = create_app().server.test_client()
        with patch.object(modelos, 'repository', side_effect=RuntimeError('Cold audit must not run')) as repository:
            response = client.post('/_dash-update-component', json={
                'output': 'tab-content.children',
                'outputs': {'id': 'tab-content', 'property': 'children'},
                'inputs': [{'id': 'main-tabs', 'property': 'value', 'value': 'modelos'}],
                'state': [], 'changedPropIds': ['main-tabs.value']})
        self.assertEqual(response.status_code, 200)
        repository.assert_not_called()
        payload = response.get_json()['response']['tab-content']['children']

        def components(node):
            if isinstance(node, dict):
                yield node
                for value in node.values():
                    yield from components(value)
            elif isinstance(node, list):
                for value in node:
                    yield from components(value)

        props = {component['props']['id']: component['props'] for component in components(payload)
                 if 'id' in component.get('props', {})}
        self.assertIn('models-window', props)
        self.assertIn('models-symbol', props)
        self.assertEqual(props['models-selected']['value'], list(modelos.FAMILIES))
        self.assertIsNotNone(props['models-content']['children'])

    def test_comparison_callback_failure_returns_visible_notice(self):
        with patch.object(modelos, 'repository', side_effect=OSError('Missing saved results')):
            with self.assertLogs(modelos.LOGGER, level='ERROR'):
                view = modelos.render('TODOS', 7, 'TODOS', list(modelos.FAMILIES),
                                      'rmse', 'XGBoost', 'BTCUSDT', 1)
        self.assertEqual(view.className, 'notice')
        self.assertEqual(view.children[0].children, 'No se pudo cargar la comparación')

    def test_dynamic_graph_resources_are_loaded_in_initial_page(self):
        client = create_app().server.test_client()
        parser = ScriptSources()
        parser.feed(client.get('/').get_data(as_text=True))
        for resource in ('async-graph', '/plotly/package_data/plotly', 'async-table'):
            paths = [p for p in parser.sources if resource in p]
            self.assertTrue(paths, resource)
            for path in paths:
                response = client.get(path)
                self.assertEqual(response.status_code, 200, path)
                self.assertIn('javascript', response.content_type)

    def test_graphs_reserve_space_and_keep_timeline_height(self):
        for height in (None, 650):
            figure = go.Figure(go.Scatter(y=[1, 2, 3]))
            if height:
                figure.update_layout(height=height)
            graph = graph_card('Serie', figure, 'Datos').children[1]
            self.assertEqual(graph.style['height'], f'{height or 450}px')
            self.assertTrue(graph.responsive)
            self.assertEqual(len(graph.figure.data), 1)


if __name__ == '__main__':
    unittest.main()
