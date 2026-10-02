"""Metrics use complete aligned observations; charts never determine scores."""
import numpy as np
import pandas as pd


def scores(actual, forecast):
    actual, forecast = np.asarray(actual, float), np.asarray(forecast, float)
    residual = actual-forecast
    mse = np.mean(residual**2)
    denominator = np.sum((actual-actual.mean())**2)
    return dict(mae=float(np.mean(np.abs(residual))), rmse=float(np.sqrt(mse)),
                mse=float(mse), r2=float(1-np.sum(residual**2)/denominator) if denominator > 0 else None)


def metric_table(predictions):
    rows = []
    for key, group in predictions.groupby(['model', 'symbol', 'volatility_window', 'horizon']):
        rows.append(dict(zip(['model','symbol','volatility_window','horizon'], key),
                         n_test=len(group), **scores(group.actual, group.forecast)))
    return pd.DataFrame(rows)


def aggregate(table, symbol, window, horizon, models):
    selected = table[table.model.isin(models) & table.volatility_window.eq(window)]
    if symbol != 'TODOS':
        selected = selected[selected.symbol.eq(symbol)]
    if horizon != 'TODOS':
        selected = selected[selected.horizon.eq(int(horizon))]
    return selected.groupby(['model','symbol'], as_index=False)[['mae','rmse','mse','r2']].mean()
