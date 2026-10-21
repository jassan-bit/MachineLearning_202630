"""Local HAR reference for the professor's feedback; never publish or replace.

Three causal realized-volatility scales (1, 7, 28 calendar days) predict the
UNCHANGED rolling daily-return volatility at each of seven horizons. This is
a HAR-inspired adaptation, not a replication of Corsi's realized-variance target.
OLS is fixed a priori. A separate Ridge reference selects alpha on 2024 only.
"""
from pathlib import Path
from datetime import datetime, timezone
import json
import os
import sys

for _variable in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[_variable] = '1'

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'src'))
from export_classical_holdout_baselines import (
    load_combined_panel, calendar, DATA, NEW_DATA, source_files, fingerprint, sha)
from optimize_minute_svr import minute_features
from volatility_experiment import targets
from volatility_dashboard.calendar import saved_calendar
from volatility_dashboard.metrics import metric_table

OUT = ROOT / 'results/professor_feedback_20261018/har_baseline'
ORIGINAL = ROOT / 'results/improved_classical_2023_2025'
OLS = 'HAR simple OLS (1/7/28)'
RIDGE = 'HAR simple Ridge (1/7/28)'
PERSISTENCE = 'Persistencia'
ALPHAS = (.1, 1., 10., 100., 1000.)
BOUNDARY = pd.Timestamp('2025-01-01', tz='UTC')
KEYS = ['symbol', 'volatility_window', 'origin', 'horizon']


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False,
                                    allow_nan=False) + '\n', encoding='utf-8')


def har_inputs(realized):
    """Every row includes only completed calendar days through its origin."""
    realized = pd.Series(np.asarray(realized, dtype=float))
    return np.column_stack([realized.rolling(span).mean().to_numpy()
                            for span in (1, 7, 28)])


def fit_reference(X, y, train, alpha=None):
    train = np.asarray(train, dtype=int)
    if not len(train) or not np.isfinite(X[train]).all() or not np.isfinite(y[train]).all():
        raise ValueError('HAR training support contains unavailable inputs/labels.')
    estimator = LinearRegression() if alpha is None else Ridge(alpha=alpha)
    model = make_pipeline(StandardScaler(), estimator)
    model.fit(X[train], y[train])
    return model


def forecast(model, X, origins):
    if not np.isfinite(X[origins]).all():
        raise ValueError('HAR prediction inputs unavailable.')
    return np.maximum(model.predict(X[origins]), 0.)


def export_rows(label, symbol, window, panel, origins, y, p):
    return pd.DataFrame(dict(model=label, symbol=symbol, volatility_window=window,
        origin=np.repeat(panel.index[origins].astype(str), 7),
        horizon=np.tile(np.arange(1, 8), len(origins)),
        target_date=panel.index[(origins[:, None] + np.arange(1, 8)).ravel()].astype(str),
        actual=y[origins].ravel(), forecast=p.ravel()))


def protected_files():
    paths = set(source_files())
    for directory, pattern in (('volatility_dashboard', '*.py'),
                               ('dashboard_data', '*'), ('book/sections', '*.md')):
        paths.update(p for p in (ROOT / directory).rglob(pattern) if p.is_file())
    paths.update(ROOT / p for p in ('app.py', 'dashboard.py', 'minute_dashboard.py',
        'render.yaml', 'Dockerfile', 'requirements-render.txt', 'DASHBOARD.md',
        'delivery/presentacion_volatilidad_ml.tex'))
    return sorted(paths)


def run():
    if (OUT / 'protocol.json').exists():
        raise RuntimeError('Local HAR baseline already exists; inspect without overwriting.')
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / 'models').mkdir(exist_ok=True)
    before = fingerprint(protected_files())
    write_json(OUT / 'protected_before.json', before)
    original, panel = load_combined_panel()
    folds, eligible_old, _ = saved_calendar(original, ORIGINAL)
    eligible = calendar(panel)
    train = np.flatnonzero(eligible & (panel.index + pd.Timedelta(days=7) < BOUNDARY))
    tests = {2025: np.flatnonzero(eligible_old & (original.index >= BOUNDARY)),
             2026: np.flatnonzero(eligible & (panel.index >= pd.Timestamp('2026-01-01', tz='UTC')))}
    if (len(train), len(tests[2025]), len(tests[2026])) != (660, 358, 236):
        raise ValueError('Original training/test calendar changed.')
    protocol = dict(publication='LOCAL ONLY', github_or_render_updated=False,
        feedback_date='2026-10-18', created_at_utc=datetime.now(timezone.utc).isoformat(),
        target='Unchanged population std of daily percent log returns, windows7/14/21/28, h1..7',
        features='sqrt(sum of minute percent log-return squares) today, trailing7 and trailing28 means',
        distinction='HAR-inspired OLS/Ridge for study target; not original HAR realized-variance response',
        frequency='Daily issuance at completed UTC close; original minute acquisition',
        ols='Fixed three scales, intercept, direct seven-output OLS; nonnegative clip',
        ridge_alphas=ALPHAS, ridge_selection='Pooled natural-unit MSE on six original purged2024 blocks',
        scaler='StandardScaler fit on train only in every fold and final fit',
        final_fit='660 original eligible origins; all labels strictly before2025',
        evaluation='Static model for2025 and2026; both already explored, retrospective',
        purpose='Stronger baseline than persistence, not automatic replacement of HAR-XG',
        code_sha256=sha(Path(__file__)),
        sources={'har':'https://doi.org/10.1093/jjfinec/nbp001',
                 'ridge':'https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Ridge.html'})
    write_json(OUT / 'protocol.json', protocol)
    choices, searches, validation, bundles = [], [], [], []
    for symbol in panel.columns:
        values = np.concatenate([np.load(DATA / f'{symbol}.npy'), np.load(NEW_DATA / f'{symbol}.npy')])
        realized = minute_features(values)[:, 0]
        X = har_inputs(realized)
        for window in (7, 14, 21, 28):
            vol = targets(panel[symbol], window).to_numpy()
            y = np.column_stack([pd.Series(vol).shift(-h).to_numpy() for h in range(1, 8)])
            candidates = {alpha: [] for alpha in ALPHAS}
            ols_rows, actual = [], []
            for tr, va in folds:
                if tr[-1] + 7 >= va[0] or panel.index[va[-1] + 7] >= BOUNDARY:
                    raise ValueError('Training/validation labels cross a time boundary.')
                actual.append(y[va])
                ols_rows.append(forecast(fit_reference(X, y, tr), X, va))
                for alpha in ALPHAS:
                    candidates[alpha].append(forecast(fit_reference(X, y, tr, alpha), X, va))
            actual = np.vstack(actual)
            scores = [(float(np.mean((np.vstack(rows) - actual) ** 2)), alpha)
                      for alpha, rows in candidates.items()]
            best_mse, best_alpha = min(scores)
            choices.append(dict(symbol=symbol, volatility_window=window, alpha=best_alpha,
                validation_mse=best_mse, n_train=len(train)))
            searches.extend(dict(symbol=symbol, volatility_window=window, alpha=a,
                                 validation_mse=s) for s, a in scores)
            anchors = np.concatenate([va for _, va in folds])
            for label, rows in ((OLS, ols_rows), (RIDGE, candidates[best_alpha])):
                validation.append(export_rows(label, symbol, window, panel, anchors, y, np.vstack(rows)))
            paths = {}
            for label, alpha, slug in ((OLS, None, 'ols'), (RIDGE, best_alpha, 'ridge')):
                model = fit_reference(X, y, train, alpha)
                path = OUT / 'models' / f'{symbol}_v{window}_{slug}.joblib'
                joblib.dump(model, path)
                paths[label] = path
            bundles.append((symbol, window, X, y, vol, paths))
        print(f'HAR simple {symbol}: trained all four target windows', flush=True)
    pd.DataFrame(choices).to_csv(OUT / 'selected_inputs.csv', index=False)
    pd.DataFrame(searches).to_csv(OUT / 'search.csv', index=False)
    pd.concat(validation).to_csv(OUT / 'validation_predictions.csv.gz', index=False,
        compression={'method': 'gzip', 'mtime': 0})
    freeze = dict(frozen_at_utc=datetime.now(timezone.utc).isoformat(),
        test_forecasts_or_metrics_computed=False, protocol_sha256=sha(OUT / 'protocol.json'),
        selected_inputs_sha256=sha(OUT / 'selected_inputs.csv'),
        model_sha256={p.name: sha(p) for p in sorted((OUT / 'models').glob('*.joblib'))})
    write_json(OUT / 'selection_freeze.json', freeze)
    metric_rows, alignments = [], []
    for year, origins in tests.items():
        rows = []
        for symbol, window, X, y, vol, paths in bundles:
            for label, path in paths.items():
                p = forecast(joblib.load(path), X, origins)
                rows.append(export_rows(label, symbol, window, panel, origins, y, p))
            rows.append(export_rows(PERSISTENCE, symbol, window, panel, origins, y,
                                    np.repeat(vol[origins, None], 7, axis=1)))
        output = pd.concat(rows, ignore_index=True)
        reference = pd.read_csv(ORIGINAL / 'predictions.csv.gz') if year == 2025 else pd.read_csv(
            ROOT / 'results/har_spca_research_20261005/baseline_2026/predictions.csv.gz').query(
                'model == "HAR-Ridge + XGBoost"')
        a = output[output.model.eq(OLS)].set_index(KEYS).actual.sort_index()
        b = reference.set_index(KEYS).actual.sort_index()
        if not a.index.equals(b.index):
            raise ValueError('HAR reference evaluation keys differ from published model.')
        np.testing.assert_allclose(a, b, rtol=1e-12, atol=1e-12)
        alignments.append(dict(year=year, exact_keys=True, rows=len(a),
                               max_abs_actual_error=float(np.max(np.abs(a-b)))))
        output.to_csv(OUT / f'predictions_{year}.csv.gz', index=False,
                      compression={'method': 'gzip', 'mtime': 0})
        table = metric_table(output)
        table['year'] = year
        table.to_csv(OUT / f'horizon_metrics_{year}.csv', index=False)
        metric_rows.append(table.groupby('model', as_index=False)[['mse', 'rmse', 'mae', 'r2']].mean().assign(year=year))
    summary = pd.concat(metric_rows)
    summary.to_csv(OUT / 'macro_metrics.csv', index=False)
    after = fingerprint([ROOT / p for p in before])
    if before != after:
        raise ValueError('Protected original inputs changed during HAR reference experiment.')
    write_json(OUT / 'verification.json', dict(status='passed', protected_files_unchanged=True,
        protected_files=len(before), alignment=alignments, training_origins=660,
        validation_origins=359, test_origins={2025: 358, 2026: 236},
        models=32, github_or_render_updated=False,
        frozen_models_verified=all(sha(OUT / 'models' / p) == h for p, h in freeze['model_sha256'].items())))
    print(summary.to_string(index=False), flush=True)


if __name__ == '__main__':
    with threadpool_limits(limits=1):
        run()
