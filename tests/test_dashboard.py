"""Check active dashboard routes and displayed metric values."""
import json
import unittest
import pandas as pd
import dashboard


class DashboardTests(unittest.TestCase):
    def test_routes(self):
        client=dashboard.server.test_client()
        for route in ['/', '/_dash-layout', '/_dash-dependencies']:
            self.assertEqual(client.get(route).status_code,200,route)

    def test_model_filter_matches_source(self):
        minute=dashboard.ROOT/'results/minute_2023_2025'
        active=(minute/'status.json').exists() and json.loads((minute/'status.json').read_text())['status']=='complete'
        if active:
            callback=next(iter(dashboard.app.callback_map.values()))['callback'].__wrapped__
            figure,trend,message,overview=callback('ForwardChaining','all_available_test','BTCUSDT',7,7)
            source=pd.read_csv(minute/'all_metrics.csv').query(
                "method == 'ForwardChaining' and scope == 'all_available_test' and symbol == 'BTCUSDT' and volatility_window == 7")
            for trace in figure.data:
                expected=source[source.model==trace.name].sort_values('input_window').rmse.to_numpy()
                self.assertEqual(list(trace.y),list(expected))
            self.assertIn('10,080',message)
        else:
            figure,trend,message=dashboard.update('BTCUSDT',7,7)
            source=dashboard.metrics.query('symbol == "BTCUSDT" and volatility_window == 7 and input_window == 7 and split == "test" and horizon == 0')
            for trace in figure.data:
                self.assertEqual(list(trace.y),list(source[source.model==trace.name].rmse))
        self.assertEqual(len(trend.data),2)


if __name__=='__main__': unittest.main()
