"""Serve the validation-selected daily volatility models."""
from functools import lru_cache
from pathlib import Path
from typing import Literal
import joblib
import json
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, field_validator

ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT/'results/models'
MINUTE = ROOT/'results/minute_2023_2025'


def minute_active():
    path = MINUTE/'status.json'
    return path.exists() and json.loads(path.read_text())['status'] == 'complete'
app = FastAPI(title='SVR lineal — volatilidad diaria', version='3.0.0')


class Request(BaseModel):
    symbol: Literal['BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'XRPUSDT']
    volatility_window: Literal[7, 14, 21, 28]
    lags: list[float] = Field(min_length=7, max_length=40320,
        description='Consecutive closes, oldest first; frequency and length required by /models.')

    @field_validator('lags')
    @classmethod
    def valid_prices(cls, value):
        if not all(np.isfinite(v) and v > 0 for v in value):
            raise ValueError('Closes must be finite and positive')
        return value


def load_model(symbol, window):
    path = MODEL_DIR/f'{symbol}_v{window}_deployment.joblib'
    if minute_active():
        path = MINUTE/'models'/f'{symbol}_v{window}.joblib'
    if not path.exists():
        raise HTTPException(503, 'Model not yet trained. Run the experiment first.')
    return read_artifact(path)


@lru_cache(maxsize=32)
def read_artifact(path):
    return joblib.load(path)


@app.get('/health')
def health():
    directory = MINUTE/'models' if minute_active() else MODEL_DIR
    return {'status': 'ok', 'models_available': len(list(directory.glob('*.joblib'))),
            'input_frequency': '1min' if minute_active() else '1D'}


@app.get('/models')
def models():
    path = ROOT/'results/model_registry.csv'
    if minute_active(): path = MINUTE/'model_registry.csv'
    if not path.exists(): raise HTTPException(503, 'Experiment not yet completed')
    frame = pd.read_csv(path)
    frame['input_frequency'] = '1min' if minute_active() else '1D'
    return frame.to_dict('records')


@app.post('/predict')
def predict(data: Request):
    artifact = load_model(data.symbol, data.volatility_window)
    expected = artifact.get('n_features', artifact['input_window'])
    frequency = artifact.get('input_frequency', '1D')
    if len(data.lags) != expected:
        raise HTTPException(422, f'This model requires {expected} closes at {frequency}, oldest first')
    X = np.asarray(data.lags).reshape(1, -1)
    result = (X @ artifact['weights'] + artifact['intercept'])[0] if 'weights' in artifact else artifact['model'].predict(X)[0]
    if result.shape != (7,) or not np.isfinite(result).all():
        raise HTTPException(500, 'Invalid model output')
    return dict(symbol=data.symbol, volatility_window=data.volatility_window,
        prediction=result.tolist(), horizons=list(range(1, 8)), unit='percentage_points',
        annualized=False, negative_predictions=int((result < 0).sum()),
        input_frequency=frequency, horizon_unit='days', forecast_origin_frequency='1D',
        fitted_through=artifact['fitted_through'])
