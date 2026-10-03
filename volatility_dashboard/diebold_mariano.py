"""Paired forecast accuracy tests on daily origins, never flattened horizons."""
from itertools import combinations

import numpy as np
import pandas as pd
from scipy.stats import t


def dm_test(differential, horizon=1, lags=None):
    """Two-sided DM with Bartlett HAC and Harvey-Leybourne-Newbold correction."""
    d = np.asarray(differential, dtype=float)
    n = len(d)
    if n < 3 or not np.isfinite(d).all() or not 1 <= horizon < n:
        raise ValueError('Se requieren pérdidas finitas y un horizonte menor que n.')
    lags = horizon - 1 if lags is None else int(lags)
    if not 0 <= lags < n:
        raise ValueError('Rezagos HAC fuera del rango de la muestra.')
    mean = float(d.mean())
    centered = d - mean
    variance = float(centered @ centered / n)
    for lag in range(1, lags + 1):
        variance += 2 * (1 - lag / (lags + 1)) * float(centered[lag:] @ centered[:-lag] / n)
    result = dict(n=n, horizonte_hln=horizon, rezagos_hac=lags, diferencia_mse=mean)
    if variance <= np.finfo(float).eps * max(float(np.mean(d**2)), np.finfo(float).tiny):
        return dict(result, dm=None, p=None, estado='Varianza degenerada')
    correction = np.sqrt((n + 1 - 2*horizon + horizon*(horizon-1)/n) / n)
    statistic = mean / np.sqrt(variance / n) * correction
    return dict(result, dm=float(statistic), p=float(2*t.sf(abs(statistic), n-1)), estado='OK')


def comparisons(predictions, symbol, window, horizon, models):
    """Average squared losses within each origin, retaining cross-series dependence."""
    selected = predictions[predictions.model.isin(models) & predictions.volatility_window.eq(int(window))].copy()
    if symbol != 'TODOS':
        selected = selected[selected.symbol.eq(symbol)]
    if horizon != 'TODOS':
        selected = selected[selected.horizon.eq(int(horizon))]
    keys = ['origin', 'symbol', 'horizon']
    if selected.duplicated(['model'] + keys).any():
        raise ValueError('Predicciones duplicadas.')
    selected['loss'] = (selected.actual - selected.forecast)**2
    loss = selected.pivot(index=keys, columns='model', values='loss')
    actual = selected.pivot(index=keys, columns='model', values='actual')
    if loss.isna().any().any() or actual.isna().any().any():
        raise ValueError('Los modelos requieren las mismas observaciones.')
    if len(actual.columns) and not np.allclose(actual.to_numpy(), actual.iloc[:, [0]].to_numpy(), rtol=1e-10, atol=1e-10):
        raise ValueError('Objetivos diferentes entre modelos.')
    daily = loss.groupby(level='origin').mean().sort_index()
    dates = pd.to_datetime(daily.index, utc=True)
    if len(dates) > 1 and not np.all(np.diff(dates.asi8) == pd.Timedelta(days=1).value):
        raise ValueError('La prueba requiere orígenes diarios consecutivos.')
    h = int(selected.horizon.max()) if len(selected) else 1
    # Moving volatility targets overlap beyond h-1; automatic bandwidth is a lower bound.
    lags = max(int(window) + h - 2, int(np.floor(4*(len(daily)/100)**(2/9))))
    rows = []
    for a, b in combinations(sorted(daily.columns), 2):
        test = dm_test(daily[a] - daily[b], h, lags)
        rows.append(dict(modelo_a=a, modelo_b=b, mse_a=float(daily[a].mean()),
                         mse_b=float(daily[b].mean()), **test))
    result = pd.DataFrame(rows)
    if result.empty:
        return result
    result['p_holm'] = np.nan
    valid = result.p.dropna().sort_values()
    adjusted = np.maximum.accumulate(valid.to_numpy() * (len(valid)-np.arange(len(valid))))
    result.loc[valid.index, 'p_holm'] = np.minimum(adjusted, 1)
    result['conclusion_5pct'] = [
        'No evaluable' if pd.isna(row.p_holm) else
        ('Sin diferencia significativa' if row.p_holm >= .05 else
         'Favorece ' + (row.modelo_a if row.diferencia_mse < 0 else row.modelo_b))
        for row in result.itertuples()]
    return result
