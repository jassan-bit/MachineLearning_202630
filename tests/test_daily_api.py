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

    def test_full_minute_coefficients(self):
        size=7*1440
        artifact=dict(weights=np.full((size,7),1/size),intercept=np.arange(7),
            input_window=7,n_features=size,input_frequency='1min',fitted_through='2023-12-31')
        with patch('app.api.load_model',return_value=artifact):
            response=self.client.post('/predict',json=dict(symbol='BTCUSDT',volatility_window=7,lags=[100.0]*size))
            self.assertEqual(response.status_code,200)
            np.testing.assert_allclose(response.json()['prediction'],100+np.arange(7))
            self.assertEqual(response.json()['input_frequency'],'1min')


if __name__ == '__main__': unittest.main()
