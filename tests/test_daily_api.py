import unittest
from unittest.mock import patch
import numpy as np
from fastapi.testclient import TestClient
from app.api import app


class DummyModel:
    def predict(self, X):
        return np.ones((len(X), 7))


class DailyApiTests(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(app)

    def test_prediction_contract_and_length(self):
        artifact = dict(model=DummyModel(), input_window=14, fitted_through='2025-12-31')
        with patch('app.api.load_model', return_value=artifact):
            payload = dict(symbol='BTCUSDT', volatility_window=7, lags=[100.0]*14)
            r = self.client.post('/predict', json=payload)
            self.assertEqual(r.status_code, 200)
            self.assertEqual(len(r.json()['prediction']), 7)
            payload['lags'] = [100.0]*7
            self.assertEqual(self.client.post('/predict', json=payload).status_code, 422)

    def test_invalid_symbol_and_prices(self):
        for symbol, price in [('SOLUSDT', 10), ('BTCUSDT', -1)]:
            r = self.client.post('/predict', json=dict(symbol=symbol, volatility_window=7, lags=[price]*7))
            self.assertEqual(r.status_code, 422)


if __name__ == '__main__': unittest.main()
