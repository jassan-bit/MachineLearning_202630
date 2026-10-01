"""Serve the validation-selected daily volatility models."""
from functools import lru_cache
from pathlib import Path
from typing import Literal
import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, field_validator

ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT/'results/models'
app = FastAPI(title='SVR lineal — volatilidad diaria', version='2.0.0')


class Request(BaseModel):
    symbol: Literal['BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'XRPUSDT']
    volatility_window: Literal[7, 14, 21, 28]
    lags: list[float] = Field(min_length=7, max_length=28,
        description='Consecutive daily UTC closes, oldest first; length required by /models.')

    @field_validator('lags')
    @classmethod
    def valid_prices(cls, value):
        if not all(np.isfinite(v) and v > 0 for v in value):
            raise ValueError('Closes must be finite and positive')
        return value


@lru_cache(maxsize=16)
def load_model(symbol, window):
    path = MODEL_DIR/f'{symbol}_v{window}_deployment.joblib'
    if not path.exists():
        raise HTTPException(503, 'Model not yet trained. Run the experiment first.')
    return joblib.load(path)


@app.get('/health')
def health():
    return {'status': 'ok', 'models_available': len(list(MODEL_DIR.glob('*_deployment.joblib')))}


@app.get('/models')
def models():
    path = ROOT/'results/model_registry.csv'
    if not path.exists(): raise HTTPException(503, 'Experiment not yet completed')
    return pd.read_csv(path).to_dict('records')


@app.post('/predict')
def predict(data: Request):
    artifact = load_model(data.symbol, data.volatility_window)
    expected = artifact['input_window']
    if len(data.lags) != expected:
        raise HTTPException(422, f'This model requires {expected} daily closes, oldest first')
    result = artifact['model'].predict(np.asarray(data.lags).reshape(1, -1))[0]
    if result.shape != (7,) or not np.isfinite(result).all():
        raise HTTPException(500, 'Invalid model output')
    return dict(symbol=data.symbol, volatility_window=data.volatility_window,
        prediction=result.tolist(), horizons=list(range(1, 8)), unit='percentage_points',
        annualized=False, negative_predictions=int((result < 0).sum()),
        fitted_through=artifact['fitted_through'])
