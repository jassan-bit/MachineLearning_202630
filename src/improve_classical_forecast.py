"""Causal HAR Ridge/XGBoost blend for the existing seven-day volatility task.

Run from the repository root: python src/improve_classical_forecast.py
Selection uses 2024 only. Existing experiments and dashboard remain reproducible.
"""
import hashlib
import json
import time

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from optimize_minute_svr import (
    ROOT, DATA, LAGS, calendars, features, load_panel, metrics, minute_features, predict,
)
from optimize_minute_xgboost import estimator as boosted_estimator

OUT = ROOT / 'results/improved_classical_2023_2025'
SOURCE = ROOT / 'results/optimized_minute_xgboost_2023_2025'
ALPHAS = [1., 10., 100., 1000., 10000.]
WEIGHTS = [0., .25, .5, .75, 1.]


def har_features(close, minute, window):
    """Daily/weekly/monthly volatility levels, downside risk and decay.

    All rolling windows include the completed origin day; no fitted statistics.
    """
    _, y, base = features(close, minute, 28, window)
    returns = 100 * np.log(close).diff()
    columns = []
    for span in [1, 3, 7, 14, 28]:
        r = returns.rolling(span)
        columns.extend([r.mean().to_numpy() / base,
                        np.sqrt((returns**2).rolling(span).mean()).to_numpy() / base,
                        np.sqrt((returns.clip(upper=0)**2).rolling(span).mean()).to_numpy() / base])
        for j in range(minute.shape[1]):
            columns.append(pd.Series(minute[:, j]).rolling(span).mean().to_numpy() / base)
    # Current volatility level and variability of recent squared returns.
    columns.append(np.log(base))
    columns.append((returns**2).rolling(28).std(ddof=0).to_numpy() / base**2)
    # Existing features encode exactly which observed returns leave each window.
    X, _, _ = features(close, minute, 7, window)
    columns.extend(X[:, -7:].T)
    return np.column_stack(columns), y, base


def ridge(alpha):
    return make_pipeline(StandardScaler(), Ridge(alpha=alpha, solver='cholesky'))


def choose_blend(actual, boosted, ridge_predictions):
    """One weight shared by all horizons; includes unchanged XGBoost."""
    options = []
    for alpha, p in ridge_predictions.items():
        for weight in WEIGHTS:
            forecast = (1-weight)*boosted + weight*p
            options.append(dict(alpha=alpha, ridge_weight=weight,
                                validation_rmse=metrics(actual, forecast)['rmse']))
    return min(options, key=lambda r: (r['validation_rmse'], r['ridge_weight'], r['alpha'])), options


def predict_bundle(bundle, close, minutes, origins):
    """Rebuild causal features and reproduce exported forecasts."""
    X, _, base = features(close, minutes, bundle['input_window'], bundle['volatility_window'])
    H, _, _ = har_features(close, minutes, bundle['volatility_window'])
    if not np.isfinite(X[origins]).all() or not np.isfinite(H[origins]).all():
        raise ValueError('Incomplete causal history for prediction')
    weight = bundle['ridge_weight']
    return ((1-weight)*predict(bundle['boosted_model'], X[origins], base[origins])
            + weight*predict(bundle['ridge_model'], H[origins], base[origins]))


def run():
    started = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / 'models').mkdir(exist_ok=True)
    (OUT / 'status.json').write_text(json.dumps({'status': 'running'}), encoding='utf-8')
    panel = load_panel()
    folds, eligible, audit = calendars(panel)
    train = np.flatnonzero(eligible & (panel.index + pd.Timedelta(days=7) < pd.Timestamp('2025-01-01', tz='UTC')))
    audit.to_csv(OUT / 'native_cv_audit.csv', index=False)
    pd.DataFrame([dict(fold=i, split=split, origin=str(panel.index[a]),
                       target_end=str(panel.index[a+7]))
                  for i, (tr, va) in enumerate(folds, 1)
                  for split, anchors in [('train', tr), ('validation', va)]
                  for a in anchors]).to_csv(OUT / 'calendar.csv', index=False)
    source_choices = SOURCE / 'selected_inputs.csv'
    selected = pd.read_csv(source_choices)
    config = dict(model='HAR_Ridge_XGBoost', horizon_days=7,
                  target='Daily log-return rolling volatility; ddof=0; percent units',
                  source_frequency='1min', alphas=ALPHAS, ridge_weights=WEIGHTS,
                  har_spans=[1, 3, 7, 14, 28], volatility_windows=LAGS,
                  selection='Minimum pooled 2024 mean horizon RMSE; six purged expanding folds',
                  boosted_selection='Reuse XGBoost inputs and hyperparameters selected on 2024 only',
                  source_choices_sha256=hashlib.sha256(source_choices.read_bytes()).hexdigest(),
                  final_fit='All eligible labels ending before 2025; static model in 2025',
                  warning='2025 previously inspected: retrospective comparison, not an untouched test',
                  uncertainty='No prediction intervals; CV selection scores are not unbiased performance estimates')
    (OUT / 'configuration.json').write_text(json.dumps(config, indent=2), encoding='utf-8')
    choices, search, bundles = [], [], []
    minute_cache = {}
    for symbol in panel.columns:
        minute = minute_features(np.load(DATA / f'{symbol}.npy'))
        minute_cache[symbol] = minute
        for window in LAGS:
            source = selected.query('symbol == @symbol and volatility_window == @window').iloc[0]
            lag, candidate = int(source.input_window), int(source.candidate)
            X, y, base = features(panel[symbol], minute, lag, window)
            H, _, _ = har_features(panel[symbol], minute, window)
            required = np.concatenate([train] + [va for _, va in folds])
            if not np.isfinite(X[required]).all() or not np.isfinite(H[required]).all():
                raise ValueError(f'Missing features: {symbol}, {window}')
            actual, boosted = [], []
            hp = {alpha: [] for alpha in ALPHAS}
            for tr, va in folds:
                if tr.max()+7 >= va.min():
                    raise ValueError('Training labels overlap validation')
                model = boosted_estimator(candidate)
                model.fit(X[tr], y[tr]/base[tr, None]-1)
                boosted.append(predict(model, X[va], base[va]))
                actual.append(y[va])
                for alpha in ALPHAS:
                    model = ridge(alpha)
                    model.fit(H[tr], y[tr]/base[tr, None]-1)
                    hp[alpha].append(predict(model, H[va], base[va]))
            best, options = choose_blend(np.vstack(actual), np.vstack(boosted),
                                        {alpha: np.vstack(p) for alpha, p in hp.items()})
            best = dict(best)
            best.update(symbol=symbol, volatility_window=window, input_window=lag,
                        candidate=candidate,
                        xgboost_validation_rmse=metrics(np.vstack(actual), np.vstack(boosted))['rmse'])
            choices.append(best)
            search.extend(dict(symbol=symbol, volatility_window=window, **row) for row in options)
            boosted_model = boosted_estimator(candidate)
            boosted_model.fit(X[train], y[train]/base[train, None]-1)
            har_model = ridge(best['alpha'])
            har_model.fit(H[train], y[train]/base[train, None]-1)
            bundle = dict(**best, boosted_model=boosted_model, ridge_model=har_model,
                          training_origins=len(train), fitted_through=str(panel.index[train.max()+7]),
                          configuration=config)
            path = OUT / 'models' / f'{symbol}_v{window}.joblib'
            joblib.dump(bundle, path)
            bundles.append((best, path))
            print(f'{symbol} v{window}: HAR weight={best["ridge_weight"]}, CV RMSE={best["validation_rmse"]:.4f}', flush=True)
    pd.DataFrame(search).to_csv(OUT / 'search.csv', index=False)
    pd.DataFrame(choices).to_csv(OUT / 'selected_inputs.csv', index=False)
    frozen = hashlib.sha256((OUT / 'selected_inputs.csv').read_bytes()).hexdigest()
    # Test evaluation starts only after all choices have been frozen.
    test = np.flatnonzero(eligible & (panel.index >= pd.Timestamp('2025-01-01', tz='UTC')))
    rows, predictions = [], []
    for best, path in bundles:
        symbol, window = best['symbol'], best['volatility_window']
        bundle = joblib.load(path)
        X, y, base = features(panel[symbol], minute_cache[symbol], best['input_window'], window)
        p = predict_bundle(bundle, panel[symbol], minute_cache[symbol], test)
        original = predict(bundle['boosted_model'], X[test], base[test])
        baseline = np.repeat(base[test, None], 7, axis=1)
        for label, values in [('HAR_Ridge_XGBoost', p), ('XGBoost_reference', original), ('Persistence', baseline)]:
            rows.append(dict(**best, model=label, n_origins=len(test), **metrics(y[test], values)))
        for i, a in enumerate(test):
            for h in range(7):
                predictions.append(dict(symbol=symbol, volatility_window=window,
                                        origin=str(panel.index[a]), horizon=h+1, actual=y[a, h],
                                        forecast=p[i, h], xgboost=original[i, h], persistence=baseline[i, h]))
    result = pd.DataFrame(rows)
    result.to_csv(OUT / 'selected_metrics.csv', index=False)
    pd.DataFrame(predictions).to_csv(OUT / 'predictions.csv.gz', index=False, compression='gzip')
    cols = ['r2', 'rmse', 'mae', 'mse', 'mape']
    macro = pd.concat([result.groupby(['symbol', 'model'], as_index=False)[cols].mean(),
                       result.groupby('model', as_index=False)[cols].mean().assign(symbol='GLOBAL_MACRO')], ignore_index=True)
    macro.to_csv(OUT / 'macro_metrics.csv', index=False)
    (OUT / 'status.json').write_text(json.dumps(dict(status='complete', seconds=time.time()-started,
        selection_sha256=frozen, test_origins=len(test), serialized_models_verified=True), indent=2), encoding='utf-8')
    print(macro.to_string(index=False), flush=True)


if __name__ == '__main__':
    with threadpool_limits(limits=1):
        run()
