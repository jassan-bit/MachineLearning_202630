"""Select a seed ensemble and causal volatility blend on 2024 only."""
import hashlib
import json
import time
import warnings

import joblib
import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from optimize_minute_svr import ROOT, DATA, load_panel, calendars, minute_features, features, predict, metrics

PREVIOUS = ROOT / 'results/tuned_minute_mlp_2023_2025'
OUT = ROOT / 'results/ensemble_tuned_minute_mlp_2023_2025'
COLS = ['r2', 'rmse', 'mae', 'mse', 'mape']


def fit(config, seed, X, y, base, tr):
    model = make_pipeline(StandardScaler(), MLPRegressor(
        hidden_layer_sizes=tuple(config['layers']), alpha=config['alpha'],
        activation=config['activation'], learning_rate_init=config['rate'],
        solver='adam', batch_size=128, max_iter=3000, n_iter_no_change=30,
        tol=1e-4, early_stopping=False, random_state=seed))
    weights = base[tr] ** config['power']
    weights /= weights.mean()
    with warnings.catch_warnings(), threadpool_limits(limits=1):
        warnings.simplefilter('error', ConvergenceWarning)
        model.fit(X[tr], y[tr] / base[tr, None] - 1,
                  mlpregressor__sample_weight=weights)
    return model


def validate_seed(config, seed, X, y, base, folds):
    forecasts = []
    for tr, va in folds:
        assert tr.max() + 7 < va.min()
        forecasts.append(predict(fit(config, seed, X, y, base, tr), X[va], base[va]))
    return np.vstack(forecasts)


def predict_bundle(bundle, X, base):
    selection = bundle['selection']
    network = np.mean([predict(model, X, base) for model in bundle['models']], axis=0)
    weight = selection['blend_weight']
    if selection['blend'] == 'decay':
        reference = np.maximum(base[:, None] * (1 + X[:, -7:]), 0)
    else:
        reference = np.repeat(base[:, None], 7, axis=1)
    return (1 - weight) * network + weight * reference


def run():
    started = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / 'models').mkdir(exist_ok=True)
    (OUT / 'status.json').write_text(json.dumps({'status': 'running'}))
    protocol = dict(selection='Minimum pooled mean horizon RMSE, six expanding 2024 blocks',
        candidates='Original seed or mean of seeds 17, 42, 73; weights 0, .1, .2, .3, .5 with persistence or known-window decay',
        architecture='Previous selected architecture, regularization and sample weights retained',
        test='2025 retrospective, all choices frozen first; previously inspected',
        warning='Reused validation and retrospective test; confirm on unseen data',
        source_selection_sha256=hashlib.sha256((PREVIOUS / 'selection.json').read_bytes()).hexdigest())
    (OUT / 'configuration.json').write_text(json.dumps(protocol, indent=2))
    panel = load_panel()
    folds, eligible, audit = calendars(panel)
    audit.to_csv(OUT / 'calendar_audit.csv', index=False)
    entries = json.loads((PREVIOUS / 'selection.json').read_text())
    selected, search, bundles = [], [], []
    for symbol in panel.columns:
        minute = minute_features(np.load(DATA / f'{symbol}.npy'))
        for entry in (e for e in entries if e['symbol'] == symbol):
            config, window = entry['config'], entry['volatility_window']
            X, y, base = features(panel[symbol], minute, config['lag'], window)
            seeds = sorted(set([config['seed'], 17, 42, 73]))
            forecasts = joblib.Parallel(n_jobs=3)(joblib.delayed(validate_seed)(
                config, seed, X, y, base, folds) for seed in seeds)
            forecasts = dict(zip(seeds, forecasts))
            va = np.concatenate([v for _, v in folds])
            actual = y[va]
            original = forecasts[config['seed']]
            np.testing.assert_allclose(metrics(actual, original)['rmse'], entry['validation_rmse'], rtol=1e-8)
            decay = np.maximum(base[va, None] * (1 + X[va, -7:]), 0)
            persistence = np.repeat(base[va, None], 7, axis=1)
            options = []
            for members in [[config['seed']], seeds]:
                network = np.mean([forecasts[s] for s in members], axis=0)
                for blend, reference in [('persistence', persistence), ('decay', decay)]:
                    for weight in [0., .1, .2, .3, .5]:
                        score = metrics(actual, (1-weight)*network + weight*reference)
                        options.append(dict(symbol=symbol, volatility_window=window,
                            seeds=members, blend=blend, blend_weight=weight, **score))
            best = min(options, key=lambda option: option['rmse'])
            best.update(config=config, original_validation_rmse=entry['validation_rmse'])
            selected.append(best)
            search.extend(options)
            bundles.append((best, X, y, base))
            (OUT / 'selection.json').write_text(json.dumps(selected, indent=2))
            (OUT / 'search.json').write_text(json.dumps(search, indent=2))
            print(f"{symbol} v{window}: validation {entry['validation_rmse']:.6f} -> {best['rmse']:.6f}; seeds={best['seeds']}, {best['blend']}={best['blend_weight']}", flush=True)
    digest = hashlib.sha256((OUT / 'selection.json').read_bytes()).hexdigest()
    tr = np.flatnonzero(eligible & (panel.index + pd.Timedelta(days=7) < pd.Timestamp('2025-01-01', tz='UTC')))
    te = np.flatnonzero(eligible & (panel.index >= pd.Timestamp('2025-01-01', tz='UTC')))
    previous_predictions = pd.read_csv(PREVIOUS / 'predictions.csv.gz')
    rows, predictions, horizon_rows = [], [], []
    for best, X, y, base in bundles:
        symbol, window = best['symbol'], best['volatility_window']
        old = joblib.load(PREVIOUS / 'models' / f'{symbol}_v{window}.joblib')['model']
        models = [old if seed == best['config']['seed'] else fit(best['config'], seed, X, y, base, tr)
                  for seed in best['seeds']]
        for model in models:
            np.testing.assert_allclose(model.named_steps['standardscaler'].mean_, X[tr].mean(axis=0))
        bundle = dict(models=models, selection=best, fitted_through=str(panel.index[tr.max()+7]))
        path = OUT / 'models' / f'{symbol}_v{window}.joblib'
        joblib.dump(bundle, path)
        forecast = predict_bundle(bundle, X[te], base[te])
        np.testing.assert_allclose(forecast, predict_bundle(joblib.load(path), X[te], base[te]))
        original = predict(old, X[te], base[te])
        saved = previous_predictions.query('symbol == @symbol and volatility_window == @window')
        np.testing.assert_array_equal(pd.to_datetime(saved.origin.unique(), utc=True), panel.index[te])
        np.testing.assert_allclose(saved.pivot(index='origin', columns='horizon', values='forecast'), original)
        np.testing.assert_allclose(saved.pivot(index='origin', columns='horizon', values='actual'), y[te])
        for label, value in [('MLP ajustado', original), ('MLP ensemble', forecast)]:
            rows.append(dict(symbol=symbol, volatility_window=window, model=label, **metrics(y[te], value)))
            for h in range(7):
                horizon_rows.append(dict(symbol=symbol, volatility_window=window, model=label,
                    horizon=h+1, **metrics(y[te, h:h+1], value[:, h:h+1])))
        for i, a in enumerate(te):
            for h in range(7):
                predictions.append(dict(symbol=symbol, volatility_window=window, origin=str(panel.index[a]),
                    horizon=h+1, actual=y[a, h], original=original[i, h], forecast=forecast[i, h]))
    frame = pd.DataFrame(rows)
    frame.to_csv(OUT / 'metrics.csv', index=False)
    pd.DataFrame(horizon_rows).to_csv(OUT / 'horizon_metrics.csv', index=False)
    exported = pd.DataFrame(predictions)
    assert not exported.duplicated(['symbol', 'volatility_window', 'origin', 'horizon']).any()
    exported.to_csv(OUT / 'predictions.csv.gz', index=False, compression='gzip')
    finalize(digest, len(te), started)


def finalize(digest, test_origins, started):
    frame = pd.read_csv(OUT / 'metrics.csv')
    selected = json.loads((OUT / 'selection.json').read_text())
    reloaded = pd.read_csv(OUT / 'predictions.csv.gz')
    for (symbol, window), group in reloaded.groupby(['symbol', 'volatility_window']):
        actual = group.pivot(index='origin', columns='horizon', values='actual').to_numpy()
        forecast = group.pivot(index='origin', columns='horizon', values='forecast').to_numpy()
        saved = frame.query('symbol == @symbol and volatility_window == @window and model == "MLP ensemble"')
        recomputed = metrics(actual, forecast)
        np.testing.assert_allclose(saved[COLS].iloc[0], [recomputed[c] for c in COLS], rtol=1e-10)
    macro = frame.groupby('model')[COLS].mean()
    macro.to_csv(OUT / 'comparison.csv')
    reference = pd.read_csv(PREVIOUS / 'comparison.csv').set_index('model').loc['MLP ajustado', COLS]
    np.testing.assert_allclose(macro.loc['MLP ajustado'], reference)
    assert digest == hashlib.sha256((OUT / 'selection.json').read_bytes()).hexdigest()
    verification = dict(passed=True, selection_sha256=digest, baseline_validation_reproduced=True,
        baseline_test_reproduced=True, serialized_predictions_verified=True,
        exported_metrics_recomputed=True, train_only_scaling=True, test_origins=test_origins)
    (OUT / 'verification.json').write_text(json.dumps(verification, indent=2))
    html = '<!doctype html><meta charset="utf-8"><title>MLP: ensemble</title><style>body{font-family:Arial;max-width:1100px;margin:40px auto}td,th{padding:8px;border:1px solid #ddd}table{border-collapse:collapse}</style>'
    html += '<h1>MLP: ensemble y ajuste temporal</h1><p>Arquitectura y pesos del MLP ajustado conservados. Promedio de inicializaciones y mezcla con persistencia o salida conocida de la ventana de volatilidad. Selección por RMSE en seis bloques temporales de 2024; evaluación retrospectiva de 2025. Validación reutilizada: confirmar en datos nuevos.</p>'
    html += macro.round(6).to_html() + '<h2>Por activo</h2>'
    html += frame.groupby(['symbol', 'model'])[COLS].mean().round(6).to_html()
    html += '<h2>Selección en validación</h2>' + pd.DataFrame(selected).drop(columns='config').to_html(index=False)
    html += '<p><a href="../tuned_minute_mlp_2023_2025/index.html">Informe del MLP de referencia</a></p>'
    (OUT / 'index.html').write_text(html, encoding='utf-8')
    (OUT / 'status.json').write_text(json.dumps(dict(status='complete', seconds=time.time()-started)))
    print(macro.to_string(), flush=True)


if __name__ == '__main__':
    run()
