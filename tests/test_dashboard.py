"""Comprueba filtros reales, correspondencia numérica y rutas Dash."""
import unittest
import app as dashboard


class DashboardTests(unittest.TestCase):
    def test_routes(self):
        client = dashboard.server.test_client()
        for route in ['/', '/_dash-layout', '/_dash-dependencies', '/assets/dashboard.css']:
            with client.get(route) as response:
                self.assertEqual(response.status_code, 200, route)

    def test_eda_for_each_asset(self):
        for symbol in dashboard.SYMBOLS:
            figures = dashboard.update_eda(symbol)
            expected = dashboard.summary.set_index('symbol').loc[symbol, 'median']
            self.assertAlmostEqual(figures[2].data[0].y[2], expected)
            for fig in [figures[0], *figures[2:]]:
                self.assertTrue(len(fig.data))
                fig.to_json()

    def test_every_model_filter_matches_source(self):
        for symbol in dashboard.SYMBOLS:
            for fold in ['all', '1', '2', '3', '4', '5']:
                for metric in ['rmse', 'mae', 'mape', 'r2']:
                    fig, trend, message = dashboard.update_models(symbol, fold, metric)
                    rows = dashboard.metrics[dashboard.metrics.symbol == symbol]
                    if fold != 'all':
                        rows = rows[rows.fold == int(fold)]
                    expected = sorted(rows.groupby('model')[metric].mean())
                    actual = sorted(float(t.y[0]) for t in fig.data)
                    for a, b in zip(actual, expected):
                        self.assertAlmostEqual(a, b)
                    self.assertEqual(len(trend.data), 2)
                    self.assertTrue(all(len(t.x) == 5 for t in trend.data))
                    self.assertIn(symbol, message)


if __name__ == '__main__':
    unittest.main()
