"""Temporal XGBoost search on the same residual volatility task as optimized SVR."""
import hashlib
import json
import time
import argparse

import joblib
import numpy as np
import pandas as pd
from xgboost import XGBRegressor
from threadpoolctl import threadpool_limits

from optimize_minute_svr import (
    ROOT, DATA, LAGS, load_panel, calendars, minute_features, features, predict, metrics,
)

CANDIDATES = list(range(4))
PARAMETERS = [dict(max_depth=depth, learning_rate=rate)
              for depth in [2, 4] for rate in [0.03, 0.1]]
FAMILY = 'xgboost'
OUT = ROOT / 'results/optimized_minute_xgboost_2023_2025'


def estimator(candidate):
    return XGBRegressor(n_estimators=100, objective='reg:squarederror',
                        tree_method='hist', max_bin=64, multi_strategy='one_output_per_tree',
                        subsample=0.8, colsample_bytree=0.8, min_child_weight=5,
                        reg_lambda=10, base_score=0, random_state=42, n_jobs=4,
                        **PARAMETERS[int(candidate)])


def run():
    started = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / 'models').mkdir(exist_ok=True)
    (OUT / 'status.json').write_text(json.dumps(dict(status='running')), encoding='utf-8')
    panel = load_panel()
    folds, eligible, audit = calendars(panel)
    audit.to_csv(OUT / 'native_cv_audit.csv', index=False)
    calendar = [dict(fold=f, split=split, origin=str(panel.index[a]),
                     target_end=str(panel.index[a+7]))
                for f, (tr, va) in enumerate(folds, 1)
                for split, anchors in [('train', tr), ('validation', va)] for a in anchors]
    pd.DataFrame(calendar).to_csv(OUT / 'calendar.csv', index=False)
    config = dict(model='XGBoost', input_windows=LAGS, volatility_windows=LAGS,
                  parameter_grid=PARAMETERS, n_estimators=100, tree_method='hist', max_bin=64,
                  multi_strategy='one_output_per_tree', subsample=0.8, colsample_bytree=0.8,
                  min_child_weight=5, reg_lambda=10, base_score=0, random_state=42,
                  source_frequency='1min', horizon_days=7,
                  features='Same causal daily and minute-return summaries as optimized SVR',
                  target='Relative correction to current daily-return rolling volatility; ddof=0; percent units',
                  selection='Minimum pooled out-of-fold mean horizon RMSE in six expanding 2024 blocks',
                  scaling='No fitted scaling; one boosted model per horizon fitted separately on each training block',
                  final_fit='All eligible labels ending before 2025; fixed model during 2025',
                  warning='Retrospective evaluation: 2025 was already inspected. No selection on 2025.')
    (OUT / 'configuration.json').write_text(json.dumps(config, indent=2), encoding='utf-8')
    train = np.flatnonzero(eligible & (panel.index + pd.Timedelta(days=7) < pd.Timestamp('2025-01-01', tz='UTC')))
    searches, choices, bundles = [], [], []
    for symbol in panel.columns:
        minute = minute_features(np.load(DATA / f'{symbol}.npy'))
        for window in LAGS:
            options = []
            for lag in LAGS:
                X, y, base = features(panel[symbol], minute, lag, window)
                assert np.isfinite(X[np.concatenate([train] + [va for _, va in folds])]).all()
                for candidate in CANDIDATES:
                    actual, forecasts = [], []
                    for tr, va in folds:
                        assert tr.max()+7 < va.min()
                        model = estimator(candidate)
                        model.fit(X[tr], y[tr]/base[tr, None]-1)
                        actual.append(y[va])
                        forecasts.append(predict(model, X[va], base[va]))
                    row = dict(symbol=symbol, volatility_window=window, input_window=lag,
                               candidate=candidate, **PARAMETERS[candidate],
                               validation_rmse=metrics(np.vstack(actual), np.vstack(forecasts))['rmse'])
                    searches.append(row)
                    options.append(row)
                print(f'Selection {symbol} volatility={window} input={lag}', flush=True)
            best = min(options, key=lambda r: (r['validation_rmse'], r['input_window'],
                                               r['candidate']))
            choices.append(best)
            X, y, base = features(panel[symbol], minute, best['input_window'], window)
            model = estimator(best['candidate'])
            model.fit(X[train], y[train]/base[train, None]-1)
            joblib.dump(dict(model=model, **best, training_origins=len(train),
                             fitted_through=str(panel.index[train.max()+7]), configuration=config),
                        OUT / 'models' / f'{symbol}_v{window}.joblib')
            bundles.append((best, X, y, base, model))
            pd.DataFrame(searches).to_csv(OUT / 'search.csv', index=False)
            pd.DataFrame(choices).to_csv(OUT / 'selected_inputs.csv', index=False)
    frozen = hashlib.sha256((OUT / 'selected_inputs.csv').read_bytes()).hexdigest()
    test = np.flatnonzero(eligible & (panel.index >= pd.Timestamp('2025-01-01', tz='UTC')))
    rows, predictions = [], []
    for best, X, y, base, model in bundles:
        forecasts = predict(model, X[test], base[test])
        baseline = np.repeat(base[test, None], 7, axis=1)
        for label, values in [('XGBoost'+'_optimized', forecasts), ('Persistence', baseline)]:
            rows.append(dict(**best, model=label, n_origins=len(test), **metrics(y[test], values)))
        for i, a in enumerate(test):
            for h in range(7):
                predictions.append(dict(symbol=best['symbol'], volatility_window=best['volatility_window'],
                                        origin=str(panel.index[a]), horizon=h+1, actual=y[a, h],
                                        forecast=forecasts[i, h], persistence=baseline[i, h]))
    result = pd.DataFrame(rows)
    result.to_csv(OUT / 'selected_metrics.csv', index=False)
    pd.DataFrame(predictions).to_csv(OUT / 'predictions.csv.gz', index=False, compression='gzip')
    cols = ['r2', 'rmse', 'mae', 'mse', 'mape']
    macro = pd.concat([result.groupby(['symbol', 'model'], as_index=False)[cols].mean(),
                       result.groupby('model', as_index=False)[cols].mean().assign(symbol='GLOBAL_MACRO')],
                      ignore_index=True)
    macro.to_csv(OUT / 'macro_metrics.csv', index=False)
    (OUT / 'status.json').write_text(json.dumps(dict(status='complete', seconds=time.time()-started,
        selection_sha256=frozen, test_origins=len(test), searched_configurations=len(searches)), indent=2), encoding='utf-8')
    print(macro.to_string(index=False), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', choices=['xgboost'], default='xgboost')
    FAMILY = parser.parse_args().model
    OUT = ROOT / f'results/optimized_minute_{FAMILY}_2023_2025'
    with threadpool_limits(limits=1):
        run()
