"""LOCAL ONLY: fixed daily-volatility HAR-OLS reference, without tuning.

The original rolling daily-return standard deviation is both the response and
the historical signal. Its level and trailing 7/28-day means form three causal
HAR-inspired predictors. This is an explicit adaptation to the study's target,
not a replication of Corsi's intraday realized-variance response.
"""
from datetime import datetime, timezone
from pathlib import Path
import json
import os
import sys

for _name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[_name] = '1'

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'src'))
from export_classical_holdout_baselines import load_combined_panel, calendar, fingerprint, sha
from professor_feedback_har_baseline import protected_files, export_rows, KEYS
from volatility_experiment import targets
from volatility_dashboard.calendar import saved_calendar
from volatility_dashboard.metrics import metric_table

OUT = ROOT / 'results/professor_feedback_20261018/daily_har'
ORIGINAL = ROOT / 'results/improved_classical_2023_2025'
BASE2026 = ROOT / 'results/har_spca_research_20261005/baseline_2026/predictions.csv.gz'
LABEL = 'HAR diario OLS (1/7/28)'
BOUNDARY = pd.Timestamp('2025-01-01', tz='UTC')


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False,
                                    allow_nan=False) + '\n', encoding='utf-8')


def daily_har_inputs(volatility):
    """Causal means on exact calendar positions; every missing slot is retained."""
    series = pd.Series(np.asarray(volatility, dtype=float))
    return np.column_stack([series.rolling(span).mean().to_numpy() for span in (1, 7, 28)])


def finite_train(X, y, original_train):
    """Only remove unavailable training rows; never fill or shorten test support."""
    indices = np.asarray(original_train)
    if indices.ndim != 1 or indices.dtype.kind not in 'iu' or not len(indices):
        raise ValueError('Original training origins must be a nonempty integer vector.')
    if (indices < 0).any() or (indices >= len(X)).any() or len(X) != len(y):
        raise ValueError('Training origins or array alignment are invalid.')
    if (np.diff(indices) <= 0).any():
        raise ValueError('Training origins must be unique and increasing.')
    good = np.isfinite(X[indices]).all(axis=1) & np.isfinite(y[indices]).all(axis=1)
    result = indices[good]
    if len(result) < 4:
        raise ValueError('Insufficient complete rows for three predictors plus intercept.')
    return result


def fit_reference(X, y, original_train):
    train = finite_train(X, y, original_train)
    model = LinearRegression(fit_intercept=True)
    model.fit(X[train], y[train])
    return model, train


def forecast(model, X, origins):
    if not np.isfinite(X[origins]).all():
        raise ValueError('Do not discard a test origin with unavailable HAR history.')
    return np.maximum(model.predict(X[origins]), 0.)


def audit_support(symbol, window, fold, split, original, used, panel):
    return dict(symbol=symbol, volatility_window=window, fold=fold, split=split,
                n_original=len(original), n_used=len(used), n_excluded=len(original)-len(used),
                first_used_origin=str(panel.index[used[0]]), last_used_origin=str(panel.index[used[-1]]),
                last_used_label=str(panel.index[used[-1]+7]))


def run():
    if (OUT / 'protocol.json').exists():
        raise FileExistsError('Preserve the fixed daily HAR experiment; inspect existing artifacts.')
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / 'models').mkdir(exist_ok=True)
    before = fingerprint(protected_files())
    write_json(OUT / 'protected_before.json', before)
    original, panel = load_combined_panel()
    folds, eligible_old, native = saved_calendar(original, ORIGINAL)
    eligible = calendar(panel)
    original_train = np.flatnonzero(eligible & (panel.index+pd.Timedelta(days=7) < BOUNDARY))
    tests = {2025: np.flatnonzero(eligible_old & (original.index >= BOUNDARY)),
             2026: np.flatnonzero(eligible & (panel.index >= pd.Timestamp('2026-01-01', tz='UTC')))}
    if (len(original_train), len(tests[2025]), len(tests[2026])) != (660, 358, 236):
        raise ValueError('Original train/test support changed.')
    protocol = dict(publication='LOCAL ONLY', github_or_render_updates_authorized=False,
        created_at_utc=datetime.now(timezone.utc).isoformat(), model=LABEL,
        target='Unchanged population std of daily percent log returns; w7/14/21/28; seven direct daily horizons',
        inputs='sigma_t^(w), mean(sigma_(t-6:t)^(w)), mean(sigma_(t-27:t)^(w)); completed UTC days only',
        interpretation='Daily-volatility HAR-inspired adaptation to study target, not original intraday realized-variance HAR',
        fixed_before_fitting='OLS with intercept; three scales1/7/28; seven direct outputs; nonnegative clip',
        selection='None: no alpha, scales, input selection or hyperparameter search',
        train_rule='Start with660 original origins; remove only rows with unavailable causal HAR inputs/labels for each w',
        warmup='A trailing28 mean of rollingw volatility needs w+27 daily returns; extra initial/post-gap history is not imputed',
        validation='Six original purged2024 folds; exact validation origins retained, train finite rows audited separately',
        tests='All358 original2025 and236 original2026 origins retained; incomplete test histories fail instead of dropping rows',
        final_fit='All eligible finite training rows with seven labels ending strictly before2025; coefficients static thereafter',
        evaluation='Both2025 and2026 already inspected: retrospective comparison; no untouched confirmatory test',
        primary_reference='Daily HAR-OLS matches the historical quantity to the forecast target; intraday HAR remains a sensitivity',
        sources={'har':'https://doi.org/10.1093/jjfinec/nbp001'},
        code_sha256=sha(__file__))
    write_json(OUT / 'protocol.json', protocol)
    native.to_csv(OUT / 'original_outer_calendar_audit.csv', index=False)
    support, removed, validation, models = [], [], [], []
    for symbol in panel.columns:
        for window in (7, 14, 21, 28):
            vol = targets(panel[symbol], window).to_numpy()
            X = daily_har_inputs(vol)
            y = np.column_stack([pd.Series(vol).shift(-h).to_numpy() for h in range(1, 8)])
            for fold, (tr, va) in enumerate(folds, 1):
                if tr[-1]+7 >= va[0] or panel.index[va[-1]+7] >= BOUNDARY:
                    raise ValueError('An original fold reaches its future validation labels.')
                model, used = fit_reference(X, y, tr)
                support.append(audit_support(symbol, window, fold, 'train', tr, used, panel))
                support.append(audit_support(symbol, window, fold, 'validation', va, va, panel))
                missing = np.setdiff1d(tr, used)
                removed.extend(dict(symbol=symbol, volatility_window=window, fold=fold,
                    origin=str(panel.index[t]), reason='Unavailable initial/post-gap daily HAR history') for t in missing)
                validation.append(export_rows(LABEL, symbol, window, panel, va, y, forecast(model, X, va)))
            final, used = fit_reference(X, y, original_train)
            if panel.index[used[-1]+7] >= BOUNDARY:
                raise ValueError('Final training labels reach evaluation period.')
            support.append(audit_support(symbol, window, 0, 'final_train', original_train, used, panel))
            removed.extend(dict(symbol=symbol, volatility_window=window, fold=0,
                origin=str(panel.index[t]), reason='Unavailable initial/post-gap daily HAR history')
                for t in np.setdiff1d(original_train, used))
            path = OUT / 'models' / f'{symbol}_v{window}.joblib'
            joblib.dump(final, path)
            models.append((symbol, window, X, y, path))
        print(f'Daily HAR fixed OLS {symbol}: four models frozen', flush=True)
    pd.DataFrame(support).to_csv(OUT / 'training_support.csv', index=False)
    pd.DataFrame(removed).to_csv(OUT / 'excluded_training_origins.csv', index=False)
    pd.concat(validation, ignore_index=True).to_csv(OUT / 'validation_predictions.csv.gz', index=False,
        compression=dict(method='gzip', mtime=0))
    frozen = {path.name:sha(path) for *_, path in models}
    write_json(OUT / 'selection_freeze.json', dict(frozen_at_utc=datetime.now(timezone.utc).isoformat(),
        all16_models_frozen=True, hyperparameter_search_performed=False,
        test_forecasts_or_metrics_computed=False, protocol_sha256=sha(OUT/'protocol.json'),
        training_support_sha256=sha(OUT/'training_support.csv'), model_sha256=frozen))
    metrics, alignments, replay = [], [], []
    for year, origins in tests.items():
        output = pd.concat([export_rows(LABEL, symbol, window, panel, origins, y,
            forecast(joblib.load(path), X, origins)) for symbol, window, X, y, path in models], ignore_index=True)
        reference = pd.read_csv(ORIGINAL/'predictions.csv.gz') if year == 2025 else pd.read_csv(BASE2026).query(
            'model == "HAR-Ridge + XGBoost"')
        a = output.set_index(KEYS).actual.sort_index()
        b = reference.set_index(KEYS).actual.sort_index()
        if not a.index.is_unique or not a.index.equals(b.index):
            raise ValueError('Daily HAR keys do not match original evaluation support exactly.')
        np.testing.assert_allclose(a, b, rtol=1e-12, atol=1e-12)
        alignments.append(dict(year=year, exact_keys=True, rows=len(a),
            max_abs_actual_error=float(np.max(np.abs(a-b))), all_original_test_origins_retained=True))
        output.to_csv(OUT/f'predictions_{year}.csv.gz', index=False, compression=dict(method='gzip', mtime=0))
        table = metric_table(output).assign(year=year)
        table.to_csv(OUT/f'horizon_metrics_{year}.csv', index=False)
        metrics.append(table.groupby(['year', 'model'], as_index=False)[['mse','rmse','mae','r2']].mean())
        replay.append(dict(year=year, serialized_models_loaded=True, predictions_finite=bool(np.isfinite(output.forecast).all())))
    macro = pd.concat(metrics, ignore_index=True)
    macro.to_csv(OUT/'macro_metrics.csv', index=False)
    after = fingerprint([ROOT/path for path in before])
    if before != after:
        raise ValueError('An original dataset, model, dashboard or delivery artifact changed.')
    write_json(OUT/'verification.json', dict(status='passed', protected_files_unchanged=True,
        protected_files=len(before), original_training_origins=660, models=16,
        fold_validation_origins=359, alignment=alignments, replay=replay,
        frozen_models_verified=all(sha(OUT/'models'/name)==digest for name,digest in frozen.items()),
        github_or_render_updated=False, publication_authorized=False))
    print(macro.to_string(index=False), flush=True)


if __name__ == '__main__':
    with threadpool_limits(limits=1):
        run()
