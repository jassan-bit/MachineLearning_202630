"""Paired HAC intervals for the original hybrid against six classical methods.

Only the seven published original model names enter this view. Squared losses
are averaged across selected assets and horizons within each daily origin;
the effective sample size is the number of dates. Normal HAC intervals are
individual, and the stars use a separate DM-HLN/t test with Holm over the six
benchmark comparisons. Research scripts, fitted artifacts and remote services
are never imported or accessed by this module.
"""
from numbers import Integral, Real

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from scipy.stats import norm

from .diebold_mariano import dm_test


BENCHMARK = 'HAR-Ridge + XGBoost'
RIVALS = ('k-NN', 'Ridge', 'Lasso', 'Random Forest', 'XGBoost', 'SVR Lineal')
MODELS = (BENCHMARK,) + RIVALS
SOURCES = {
    'HAC de Newey–West': 'https://doi.org/10.2307/1913610',
    'DM con corrección HLN': 'https://pkg.robjhyndman.com/forecast/reference/dm.test.html',
    'Ajuste Holm': 'https://www.jstor.org/stable/4615733',
}


def _holm(p_values, family_size):
    """Retain every predeclared comparison even when its test is not evaluable."""
    values = np.asarray(p_values, dtype=float)
    adjusted = np.full(len(values), np.nan)
    valid = np.flatnonzero(np.isfinite(values))
    order = valid[np.argsort(values[valid], kind='stable')]
    if len(order):
        adjusted[order] = np.minimum(1.0, np.maximum.accumulate(
            values[order]*(family_size-np.arange(len(order)))))
    return adjusted


def _integer(value, label, allowed=None):
    if isinstance(value, bool):
        raise ValueError(f'{label} debe ser un entero válido.')
    if isinstance(value, str):
        if not value.isdigit():
            raise ValueError(f'{label} debe ser un entero válido.')
        value = int(value)
    if not isinstance(value, Integral) or (allowed is not None and value not in allowed):
        raise ValueError(f'{label} fuera del rango permitido.')
    return int(value)


def _paired_daily(predictions, symbol, window, horizon, year):
    required = {'model', 'symbol', 'volatility_window', 'origin', 'horizon', 'actual', 'forecast'}
    if not isinstance(predictions, pd.DataFrame) or not required.issubset(predictions.columns):
        raise ValueError('Faltan las columnas necesarias para comparar pronósticos emparejados.')
    window = _integer(window, 'Ventana', (7, 14, 21, 28))
    selected_horizons = list(range(1, 8)) if horizon == 'TODOS' else [_integer(horizon, 'Horizonte', range(1, 8))]
    frame = predictions[predictions.model.isin(MODELS)
                        & predictions.volatility_window.eq(window)
                        & predictions.horizon.isin(selected_horizons)].copy()
    if symbol != 'TODOS':
        frame = frame[frame.symbol.eq(symbol)].copy()
    if frame.empty:
        raise ValueError('No hay pronósticos para estos filtros.')
    frame['origin'] = pd.to_datetime(frame.origin, utc=True, errors='raise')
    if frame.origin.isna().any():
        raise ValueError('Los orígenes diarios deben ser fechas válidas.')
    if year is not None:
        year = _integer(year, 'Año')
        frame = frame[frame.origin.dt.year.eq(year)].copy()
    if frame.empty:
        raise ValueError('No hay pronósticos guardados para el año seleccionado.')
    years = sorted(frame.origin.dt.year.unique())
    if len(years) != 1:
        raise ValueError('Selecciona un único año; las pruebas no mezclan periodos de evaluación.')
    if set(frame.model) != set(MODELS):
        raise ValueError('Se requieren el modelo original y los seis métodos clásicos en el mismo test.')
    if not np.isfinite(frame[['actual', 'forecast']].to_numpy(dtype=float)).all():
        raise ValueError('Los objetivos y pronósticos seleccionados deben ser finitos.')
    keys = ['origin', 'symbol', 'horizon']
    if frame.duplicated(['model']+keys).any():
        raise ValueError('Hay pronósticos duplicados para las mismas observaciones.')
    actual = frame.pivot(index=keys, columns='model', values='actual').sort_index()
    if actual.isna().any().any():
        raise ValueError('Todos los métodos deben contener exactamente las mismas observaciones.')
    reference = actual[BENCHMARK].to_numpy()[:, None]
    if not np.allclose(actual[list(MODELS)].to_numpy(), reference, rtol=1e-10, atol=1e-10):
        raise ValueError('Los objetivos reales difieren entre los métodos comparados.')
    origins = pd.DatetimeIndex(frame.origin.unique()).sort_values()
    if len(origins) < 3 or not np.all(np.diff(origins.asi8) == pd.Timedelta(days=1).value):
        raise ValueError('La comparación requiere al menos tres orígenes diarios consecutivos.')
    assets = sorted(frame.symbol.unique())
    expected = pd.MultiIndex.from_product([origins, assets, selected_horizons], names=keys).sort_values()
    if not actual.index.equals(expected):
        raise ValueError('El panel diario debe tener los mismos activos y horizontes en cada fecha.')
    frame['loss'] = (frame.actual-frame.forecast)**2
    if not np.isfinite(frame.loss.to_numpy()).all():
        raise ValueError('Las pérdidas cuadráticas desbordan la escala numérica.')
    daily = frame.pivot(index=keys, columns='model', values='loss').groupby(level='origin').mean()
    daily = daily.loc[origins, list(MODELS)]
    hln_horizon = max(selected_horizons)
    lags = max(window+hln_horizon-2, int(np.floor(4*(len(daily)/100)**(2/9))))
    if lags >= len(daily):
        raise ValueError('Hay pocos orígenes para estimar la dependencia del objetivo seleccionado.')
    details = dict(year=int(years[0]), n=len(daily), start=str(origins[0].date()),
                   end=str(origins[-1].date()), symbol=symbol, window=window,
                   horizon=horizon, horizon_hln=hln_horizon, lags=lags,
                   assets=len(assets), cells_per_origin=len(assets)*len(selected_horizons),
                   benchmark=BENCHMARK, rivals=list(RIVALS), family_size=len(RIVALS))
    return daily, details


def _interval(differential, lags, alpha):
    d = np.asarray(differential, dtype=float)
    n = len(d)
    mean = float(d.mean())
    centered = d-mean
    variance = float(centered@centered/n)
    for lag in range(1, lags+1):
        variance += 2*(1-lag/(lags+1))*float(centered[lag:]@centered[:-lag]/n)
    scale = float(np.mean(d**2))
    if not np.isfinite([mean, variance, scale]).all():
        raise ValueError('Las diferencias de pérdida desbordan la escala numérica.')
    if variance <= np.finfo(float).eps*max(scale, np.finfo(float).tiny):
        return dict(gain=mean, se_HAC=None, ci_low_gain=None, ci_high_gain=None)
    error = float(np.sqrt(variance/n))
    half_width = float(norm.ppf(1-alpha/2))*error
    return dict(gain=mean, se_HAC=error, ci_low_gain=mean-half_width, ci_high_gain=mean+half_width)


def _comparison_rows(daily, details, alpha):
    rows = []
    for rival in RIVALS:
        gain = daily[rival].to_numpy()-daily[BENCHMARK].to_numpy()
        interval = _interval(gain, details['lags'], alpha)
        test = dm_test(gain, horizon=details['horizon_hln'], lags=details['lags'])
        rows.append(dict(model=rival, benchmark=BENCHMARK,
                         mse_original=float(daily[BENCHMARK].mean()), mse_rival=float(daily[rival].mean()),
                         **interval, deltaMSE_original_minus_rival=-interval['gain'],
                         dm_gain=test['dm'], p_raw_DM_HLN=test['p'], estado=test['estado'],
                         n=test['n'], horizonte_hln=test['horizonte_hln'], rezagos_hac=test['rezagos_hac']))
    result = pd.DataFrame(rows)
    result['p_holm6'] = _holm(result.p_raw_DM_HLN, len(RIVALS))
    result['star'] = np.isfinite(result.p_holm6) & (result.p_holm6 < alpha)
    result['winner'] = [
        (BENCHMARK if row.gain > 0 else row.model) if row.star else None
        for row in result.itertuples()]
    result['conclusion_5pct'] = [
        'No evaluable' if not np.isfinite(row.p_holm6) else
        'Sin diferencia significativa' if row.winner is None else 'Favorece '+row.winner
        for row in result.itertuples()]
    return result


def _forest(table, alpha):
    figure = go.Figure()
    categories = [name+(' *' if bool(table.loc[table.model.eq(name), 'star'].iloc[0]) else '')
                  for name in RIVALS]
    groups = [
        ('Original favorecido · Holm6', '#0f766e', table.star & table.winner.eq(BENCHMARK)),
        ('Rival favorecido · Holm6', '#b91c1c', table.star & ~table.winner.eq(BENCHMARK)),
        ('Sin diferencia significativa', '#64748b', ~table.star),
    ]
    for title, color, selection in groups:
        selected = table[selection & table.ci_low_gain.notna()]
        if selected.empty:
            continue
        custom = selected[['ci_low_gain', 'ci_high_gain', 'p_holm6', 'p_raw_DM_HLN', 'n', 'rezagos_hac']].to_numpy()
        figure.add_trace(go.Scatter(
            x=selected.gain.tolist(), y=[row.model+(' *' if row.star else '') for row in selected.itertuples()],
            mode='markers', name=title, marker=dict(color=color, size=10),
            error_x=dict(type='data', symmetric=False,
                         array=(selected.ci_high_gain-selected.gain).tolist(),
                         arrayminus=(selected.gain-selected.ci_low_gain).tolist(), color=color, thickness=2, width=5),
            customdata=custom,
            hovertemplate=('%{y}<br>Ganancia MSE: %{x:.6f}<br>IC individual: '
                           '[%{customdata[0]:.6f}, %{customdata[1]:.6f}]'
                           '<br>p Holm6: %{customdata[2]:.6f}<br>p DM-HLN: %{customdata[3]:.6f}'
                           '<br>Orígenes: %{customdata[4]:.0f}<br>Rezagos HAC: %{customdata[5]:.0f}'
                           '<extra></extra>')))
    invalid = table[table.ci_low_gain.isna()]
    if not invalid.empty:
        figure.add_trace(go.Scatter(x=invalid.gain.tolist(), y=invalid.model.tolist(), mode='markers',
                                   marker=dict(color='#64748b', size=11, symbol='x'), name='IC no evaluable',
                                   hovertemplate='%{y}<br>Ganancia descriptiva: %{x:.6f}<br>Varianza degenerada<extra></extra>'))
    figure.add_vline(x=0, line_dash='dash', line_color='#334155', line_width=1)
    figure.update_layout(
        template='plotly_white', height=450,
        xaxis_title='Ganancia de MSE: rival − original (puntos porcentuales²)',
        yaxis_title='', yaxis=dict(categoryorder='array', categoryarray=list(reversed(categories))),
        margin=dict(l=120, r=35, t=30, b=105), legend=dict(orientation='h', y=-.24),
        hovermode='closest')
    figure.add_annotation(x=0, y=1.06, xref='x', yref='paper', text='Igual MSE', showarrow=False,
                          font=dict(size=11, color='#475569'))
    return figure


def confidence_intervals(predictions, symbol='TODOS', window=7, horizon='TODOS', year=None, alpha=.05):
    """Return an interactive forest, six comparison rows and explanatory details.

    The models selector used by other dashboard sections does not alter this
    fixed family. ``year=None`` accepts a single saved evaluation year; when
    multiple years are present the caller must select one. Inputs are untouched.
    """
    if isinstance(alpha, bool) or not isinstance(alpha, Real) or not np.isfinite(alpha) or not 0 < alpha < 1:
        raise ValueError('alpha debe estar entre cero y uno.')
    daily, details = _paired_daily(predictions, symbol, window, horizon, year)
    result = _comparison_rows(daily, details, float(alpha))
    details.update(alpha=float(alpha), confidence_level=float(1-alpha), interval_simultaneous=False,
                   significant_wins=int((result.star & result.winner.eq(BENCHMARK)).sum()),
                   significant_losses=int((result.star & ~result.winner.eq(BENCHMARK)).sum()),
                   not_evaluable=int(result.p_holm6.isna().sum()), sources=dict(SOURCES))
    confidence = f'{100*(1-alpha):g} %'
    details['note'] = (
        f"{details['year']} · {details['start']} a {details['end']} · {details['n']} orígenes diarios. "
        f"Activo: {symbol}; ventana: {details['window']}; horizonte: {horizon}. "
        'Una ganancia positiva favorece HAR-Ridge + XGBoost original. '
        f'Las barras son intervalos HAC normales individuales al {confidence}. '
        f'El asterisco indica DM-HLN significativo después de Holm entre los seis rivales (α = {alpha:g}). '
        'El intervalo y el asterisco usan criterios diferentes. '
        f"Rezagos HAC: {details['lags']}; horizonte HLN: {details['horizon_hln']}. "
        'Activos y horizontes se promedian dentro de cada día. '
        'Sin diferencia significativa no demuestra equivalencia.')
    return _forest(result, alpha), result, details
