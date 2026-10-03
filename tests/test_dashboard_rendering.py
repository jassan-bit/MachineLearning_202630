"""Rendering resources must be available before dynamic tabs are opened."""
from html.parser import HTMLParser
import unittest

import plotly.graph_objects as go

from volatility_dashboard.app import create_app
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
