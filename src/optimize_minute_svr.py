"""Local-only residual LinearSVR experiment; never selects on 2025 metrics."""
import hashlib
import inspect
import json
from pathlib import Path
import time
import joblib
import numpy as np
import pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
from threadpoolctl import threadpool_limits
from tsxv import splitTrainVal as tsxv
from minute_experiment import DATA, ROOT, load_panel, estimator
from volatility_experiment import targets

OUT = ROOT / 'results/optimized_minute_2023_2025'
CS = [0.0001, 0.001, 0.01, 0.1, 1.0]
EPS = [0.01, 0.1]
LAGS = [7, 14, 21, 28]


def minute_features(values):
    """All minute returns contribute; no imputation or fitted statistics."""
    r = 100 * np.diff(np.log(values), prepend=np.nan).reshape(-1, 1440)
    return np.column_stack([np.sqrt(np.sum(r*r, axis=1)),
                            np.sum(np.abs(r), axis=1) / np.sqrt(1440),
                            np.sqrt(np.sum(np.minimum(r, 0)**2, axis=1)),
                            np.max(np.abs(r), axis=1)])


def features(close, minute, lag, window):
    returns = 100 * np.log(close).diff()
    vol = targets(close, window).to_numpy()
    base = np.maximum(vol, 1e-8)
    columns = []
    for offset in range(lag):
        columns.append(returns.shift(offset).to_numpy() / base)
        columns.append(returns.shift(offset).to_numpy()**2 / base**2)
        for j in range(minute.shape[1]):
            columns.append(pd.Series(minute[:, j]).shift(offset).to_numpy() / base)
    # Known returns leaving the volatility window explain predictable decay.
    # Future returns are NOT used: only the part of the old window retained.
    for h in range(1, 8):
        retained = window-h
        s = returns.rolling(retained).sum().to_numpy() if retained else np.zeros(len(close))
        q = (returns**2).rolling(retained).sum().to_numpy() if retained else np.zeros(len(close))
        expected_var = (q + h*base**2)/window - (s/window)**2
        columns.append(np.sqrt(np.maximum(expected_var, 0))/base - 1)
    X = np.column_stack(columns)
    y = np.column_stack([pd.Series(vol).shift(-h).to_numpy() for h in range(1, 8)])
    return X, y, base


def calendars(panel):
    """Keep each native training cut; extend its validation to two months.

    Only six forward cuts are fit, each independently with expanding history.
    KFold/GroupKFold are audited unchanged, not silently converted to forward CV.
    All library calls receive the development calendar ending in 2024.
    """
    dates = panel.index
    development = np.flatnonzero(dates < pd.Timestamp('2025-01-01', tz='UTC'))
    audit = []
    native_forward = None
    for name in ['forwardChaining', 'kFold', 'groupKFold']:
        function = getattr(tsxv, 'split_train_val_' + name)
        raw = function(development, 28, 7, 1)
        for key in raw[0]:
            tr = raw[0][key][:, -1].astype(int)
            va = raw[2][key][:, -1].astype(int)
            audit.append(dict(method=name, native_fold=int(key), n_train=len(tr), n_val=len(va),
                train_last=int(tr.max()), val_first=int(va.min()),
                future_train_origins=int(np.sum(tr > va.min())),
                train_labels_at_or_after_first_validation=int(np.sum(tr+7 >= va.min())),
                source_sha256=hashlib.sha256(inspect.getsource(function).encode()).hexdigest()))
        if name == 'forwardChaining': native_forward = raw
    finite = np.isfinite(panel.to_numpy()).all(axis=1)
    eligible = pd.Series(finite).rolling(36).sum().shift(-7).eq(36).to_numpy()
    folds = []
    for month in [1, 3, 5, 7, 9, 11]:
        start = pd.Timestamp(f'2024-{month:02d}-01', tz='UTC')
        end = start + pd.DateOffset(months=2)
        first = int(np.flatnonzero(dates == start)[0])
        keys = [k for k in native_forward[2] if int(native_forward[2][k][0, -1]) == first]
        assert len(keys) == 1
        tr = native_forward[0][keys[0]][:, -1].astype(int)
        tr = tr[eligible[tr] & (tr+7 < first)]
        va = np.flatnonzero(eligible & (dates >= start) & (dates < end)
                            & (dates + pd.Timedelta(days=7) < pd.Timestamp('2025-01-01', tz='UTC')))
        assert tr.max()+7 < va.min()
        folds.append((tr, va))
    return folds, eligible, pd.DataFrame(audit)


def metrics(y, p):
    e = p-y
    return dict(r2=float(r2_score(y, p, multioutput='uniform_average')),
                rmse=float(np.sqrt((e*e).mean(axis=0)).mean()),
                mse=float((e*e).mean()), mae=float(np.abs(e).mean()),
                mape=float((100*np.abs(e)/y).mean()))


def predict(model, X, base):
    return np.maximum(base[:, None]*(1+model.predict(X)), 0)


def run():
    started = time.time()
    OUT.mkdir(exist_ok=True, parents=True)
    (OUT/'models').mkdir(exist_ok=True)
    panel = load_panel()
    folds, eligible, audit = calendars(panel)
    audit.to_csv(OUT/'native_cv_audit.csv', index=False)
    calendar = []
    for f, (tr, va) in enumerate(folds, 1):
        for split, anchors in [('train', tr), ('validation', va)]:
            for a in anchors:
                calendar.append(dict(fold=f, split=split, origin=str(panel.index[a]),
                                     target_end=str(panel.index[a+7])))
    pd.DataFrame(calendar).to_csv(OUT/'calendar.csv', index=False)
    config = dict(source_frequency='1min', feature_representation='daily summaries of ALL minute returns plus daily return lags and known volatility-window decay',
        input_windows=LAGS, volatility_windows=LAGS, C=CS, epsilon=EPS, horizon_days=7,
        selection='Pooled out-of-fold 2024 mean horizon RMSE; six expanding fits; no 2025 selection',
        target='Relative correction to current daily-return rolling volatility, ddof=0, percent units',
        final_fit='All eligible labels ending before 2025; static model during 2025',
        warning='Retrospective test: 2025 was already inspected in previous experiments.',
        cv='Native tsxv forward training cuts, two-month validation extensions; native KFold/GroupKFold audited for future information, excluded from forecasting selection')
    (OUT/'configuration.json').write_text(json.dumps(config, indent=2), encoding='utf-8')
    searches, choices, bundles = [], [], []
    # Selection phase does not evaluate, rank, or export any 2025 metric.
    for symbol in panel.columns:
        minute = minute_features(np.load(DATA/f'{symbol}.npy'))
        for w in LAGS:
            options = []
            for lag in LAGS:
                X, y, base = features(panel[symbol], minute, lag, w)
                for C in CS:
                    for eps in EPS:
                        actual, predicted = [], []
                        for tr, va in folds:
                            assert np.isfinite(X[np.r_[tr, va]]).all()
                            model = make_pipeline(StandardScaler(), estimator(C, eps))
                            model.fit(X[tr], y[tr]/base[tr, None]-1)
                            actual.append(y[va]); predicted.append(predict(model, X[va], base[va]))
                        score = metrics(np.vstack(actual), np.vstack(predicted))['rmse']
                        row = dict(symbol=symbol, volatility_window=w, input_window=lag,
                                   C=C, epsilon=eps, validation_rmse=score)
                        searches.append(row); options.append(row)
                print(f'Selection {symbol} volatility={w} input={lag}', flush=True)
            best = min(options, key=lambda x:(x['validation_rmse'], x['input_window'], x['C'], x['epsilon']))
            choices.append(best)
            X, y, base = features(panel[symbol], minute, best['input_window'], w)
            tr = np.flatnonzero(eligible & (panel.index+pd.Timedelta(days=7) < pd.Timestamp('2025-01-01', tz='UTC')))
            model = make_pipeline(StandardScaler(), estimator(best['C'], best['epsilon']))
            model.fit(X[tr], y[tr]/base[tr, None]-1)
            artifact = dict(model=model, **best, training_origins=len(tr),
                            fitted_through=str(panel.index[tr.max()+7]), configuration=config)
            joblib.dump(artifact, OUT/'models'/f'{symbol}_v{w}.joblib')
            bundles.append((best, X, y, base, model))
            pd.DataFrame(searches).to_csv(OUT/'search.csv', index=False)
            pd.DataFrame(choices).to_csv(OUT/'selected_inputs.csv', index=False)
    # All choices are frozen before this point.
    frozen = hashlib.sha256((OUT/'selected_inputs.csv').read_bytes()).hexdigest()
    te = np.flatnonzero(eligible & (panel.index >= pd.Timestamp('2025-01-01', tz='UTC')))
    rows, predictions = [], []
    for best, X, y, base, model in bundles:
        p = predict(model, X[te], base[te])
        baseline = np.repeat(base[te, None], 7, axis=1)
        for label, value in [('SVR_optimized', p), ('Persistence', baseline)]:
            rows.append(dict(**best, model=label, n_origins=len(te), **metrics(y[te], value)))
        for i, a in enumerate(te):
            for h in range(7):
                predictions.append(dict(symbol=best['symbol'], volatility_window=best['volatility_window'],
                    origin=str(panel.index[a]), horizon=h+1, actual=y[a,h], svr=p[i,h], persistence=baseline[i,h]))
    result = pd.DataFrame(rows)
    result.to_csv(OUT/'selected_metrics.csv', index=False)
    pd.DataFrame(predictions).to_csv(OUT/'predictions.csv.gz', index=False, compression='gzip')
    cols = ['r2', 'rmse', 'mae', 'mse', 'mape']
    macro = result.groupby(['symbol', 'model'], as_index=False)[cols].mean()
    global_ = result.groupby('model', as_index=False)[cols].mean().assign(symbol='GLOBAL_MACRO')
    macro = pd.concat([macro, global_], ignore_index=True)
    macro.to_csv(OUT/'macro_metrics.csv', index=False)
    (OUT/'status.json').write_text(json.dumps(dict(status='complete', seconds=time.time()-started,
        selection_sha256=frozen, test_origins=len(te), searched_configurations=len(searches)), indent=2), encoding='utf-8')
    print(macro.to_string(index=False), flush=True)


if __name__ == '__main__':
    with threadpool_limits(limits=1):
        run()
