"""LOCAL ONLY: causal AR correction of frozen HAR-Ridge + XGBoost errors.

The base hyperparameters were selected using all six 2024 blocks. Consequently
this is a CONDITIONAL search, not independent/nested model-selection validation.
Base training predictions are nevertheless honest expanding OOF forecasts. An
error of horizon h at origin s is observable only at s+h. Frozen AR coefficients
use errors e[t-h-j,h], never immature errors e[t-1,h] for h>1.
"""
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
import argparse
import hashlib
import json
import os
import sys
import time

for _name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[_name] = '1'
import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'src'))
import improve_classical_forecast as base_module
from optimize_minute_svr import DATA, features, minute_features, predict
from optimize_minute_xgboost import estimator
from export_classical_holdout_baselines import NEW_DATA, calendar, load_combined_panel, sha
from har_residual_core import residual_folds
from volatility_dashboard.calendar import saved_calendar
from volatility_dashboard.diebold_mariano import dm_test
from evaluate_har_spca_experiment import holm

OUT = ROOT / 'results/professor_feedback_20261018/residuals'
ORIGINAL = ROOT / 'results/improved_classical_2023_2025'
OOF = ROOT / 'results/har_ridge_xgboost_audit_20261005/blend_cv/oof_predictions.csv'
BASE2026 = ROOT / 'results/har_spca_research_20261005/baseline_2026/predictions.csv.gz'
BOUNDARY = pd.Timestamp('2025-01-01', tz='UTC')
ORDERS = (1, 3, 7)
ALPHAS = (1., 10., 100.)
ETAS = (0., .25, .5, 1.)
BASE = 'HAR-Ridge + XGBoost'
AR = 'HAR-XG + AR mature errors'
LAST = 'HAR-XG + last mature error'
OFFLINE = 'HAR-XG + AR no online labels'


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n', encoding='utf-8')


def causal_lags(errors, origins, horizon, order):
    """Read exact calendar positions of errors whose target has already ended.

    Missing calendar days stay missing. Reading residual s=t-h is permitted:
    forecasts are issued after the daily close, when its target is observable.
    """
    errors = np.asarray(errors, dtype=float)
    origins = np.asarray(origins)
    if errors.ndim != 2 or errors.shape[1] != 7 or origins.dtype.kind not in 'iu':
        raise ValueError('Expected daily seven-horizon error panel and integer origins.')
    if not 1 <= horizon <= 7 or order < 1 or (origins < 0).any() or (origins >= len(errors)).any():
        raise ValueError('Invalid horizon, order or origin.')
    positions = origins[:, None] - horizon - np.arange(order)[None, :]
    values = np.full(positions.shape, np.nan)
    valid = positions >= 0
    values[valid] = errors[positions[valid], horizon-1]
    return values


def fit_corrector(errors, train, order, alpha, label_cutoff):
    """Fit seven direct AR regressions on OOF errors with strict label cutoff.

    The base uses a common seven-target purge; retain it for AR response rows.
    Thus neither a training response nor any predictor reaches outer validation.
    """
    train = np.asarray(train, dtype=int)
    if len(train) == 0 or (train + 7 > label_cutoff).any():
        raise ValueError('An AR training label exceeds its permitted cutoff.')
    models, counts = [], []
    for h in range(1, 8):
        X = causal_lags(errors, train, h, order)
        z = errors[train, h-1]
        usable = np.isfinite(X).all(axis=1) & np.isfinite(z)
        if usable.sum() < 30:
            raise ValueError('Insufficient honest chronological AR residual rows.')
        model = make_pipeline(StandardScaler(), Ridge(alpha=alpha))
        model.fit(X[usable], z[usable])
        models.append(model)
        counts.append(int(usable.sum()))
    return dict(models=models, order=int(order), alpha=float(alpha), training_rows=counts,
                label_cutoff=int(label_cutoff))


def correction(bundle, errors, origins):
    result = np.zeros((len(origins), 7))
    for h, model in enumerate(bundle['models'], 1):
        X = causal_lags(errors, origins, h, bundle['order'])
        usable = np.isfinite(X).all(axis=1)
        if usable.any():
            result[usable, h-1] = model.predict(X[usable])
    return result


def last_correction(errors, origins):
    result = np.column_stack([causal_lags(errors, origins, h, 1)[:, 0] for h in range(1, 8)])
    return np.where(np.isfinite(result), result, 0.)


def completed_forecast(base, signed_correction, eta):
    return np.maximum(np.asarray(base) + float(eta) * np.asarray(signed_correction), 0.)


def offline_correction(bundle, history, prediction_origins, boundary):
    """Recursive expected errors, without reading actual labels after boundary.

    Online and offline models share coefficients and initial historical OOF state.
    Thereafter offline replaces unknown errors by its own unshrunk predictions.
    This separates static coefficients from access to newly matured observations.
    """
    state = np.asarray(history).copy()
    state[boundary:] = np.nan
    answer = np.zeros((len(prediction_origins), 7))
    for i, t in enumerate(prediction_origins):
        c = correction(bundle, state, np.array([t]))[0]
        state[t] = c
        answer[i] = c
    return answer


def fit_base(state, train, origins):
    X, H, y, current = (state[k] for k in ('X', 'H', 'y', 'current'))
    choice = state['choice']
    booster = estimator(int(choice['candidate'])).set_params(n_jobs=1)
    booster.fit(X[train], y[train]/current[train, None]-1.)
    ridge = base_module.ridge(float(choice['alpha']))
    ridge.fit(H[train], y[train]/current[train, None]-1.)
    weight = float(choice['ridge_weight'])
    return ((1.-weight)*predict(booster, X[origins], current[origins])
            + weight*predict(ridge, H[origins], current[origins]))


def load_states():
    original, panel = load_combined_panel()
    folds, eligible_old, audit = saved_calendar(original, ORIGINAL)
    eligible = calendar(panel)
    final_train = np.flatnonzero(eligible & (panel.index + pd.Timedelta(days=7) < BOUNDARY))
    tests = {2025: np.flatnonzero(eligible_old & (original.index >= BOUNDARY)),
             2026: np.flatnonzero(eligible & (panel.index >= pd.Timestamp('2026-01-01', tz='UTC')))}
    if [len(final_train), len(tests[2025]), len(tests[2026])] != [660, 358, 236]:
        raise ValueError('Changed original calendar.')
    oof = pd.read_csv(OOF)
    oof['origin'] = pd.to_datetime(oof.origin, utc=True)
    choices = pd.read_csv(ORIGINAL / 'selected_inputs.csv')
    states = []
    for symbol in panel.columns:
        values = np.concatenate([np.load(DATA/f'{symbol}.npy'), np.load(NEW_DATA/f'{symbol}.npy')])
        minutes = minute_features(values)
        for window in (7, 14, 21, 28):
            choice = choices[(choices.symbol == symbol) & (choices.volatility_window == window)].iloc[0].to_dict()
            X, y, current = features(panel[symbol], minutes, int(choice['input_window']), window)
            H, _, _ = base_module.har_features(panel[symbol], minutes, window)
            errors = np.full(y.shape, np.nan)
            base_oof = np.full(y.shape, np.nan)
            group = oof[(oof.symbol == symbol) & (oof.volatility_window == window)].sort_values(['origin', 'horizon'])
            origins = panel.index.get_indexer(pd.DatetimeIndex(group.origin.unique()))
            expected = np.concatenate([va for _, va in folds])
            np.testing.assert_array_equal(origins, expected)
            np.testing.assert_allclose(group.actual.to_numpy().reshape(-1, 7), y[origins], rtol=1e-12, atol=1e-12)
            weight = float(choice['ridge_weight'])
            values = ((1.-weight)*group.xgboost.to_numpy()
                      + weight*group[f'har_alpha_{choice["alpha"]:g}'].to_numpy()).reshape(-1, 7)
            base_oof[origins] = values
            errors[origins] = y[origins]-values
            states.append(dict(symbol=symbol, volatility_window=window, choice=choice,
                               X=X, H=H, y=y, current=current, errors=errors, base_oof=base_oof))
    return panel, folds, final_train, tests, states, audit


def prepare_archive(state, folds):
    """Complete 2023 OOF warmup; reuse exact purged 2024 component replays."""
    started = time.monotonic()
    pretrain = folds[0][0]
    inner = residual_folds(pretrain, min_train=120, block_size=60,
                           label_cutoff=int(pretrain[-1]+7))
    audit = []
    for number, (tr, va) in enumerate(inner, 1):
        forecast = fit_base(state, tr, va)
        state['base_oof'][va] = forecast
        state['errors'][va] = state['y'][va]-forecast
        audit.append(dict(symbol=state['symbol'], volatility_window=state['volatility_window'],
                          block=number, n_train=len(tr), n_prediction=len(va),
                          train_last=int(tr[-1]), train_label_last=int(tr[-1]+7),
                          prediction_first=int(va[0]), prediction_last=int(va[-1]),
                          strict_purge=bool(tr[-1]+7 < va[0])))
    return state, audit, time.monotonic()-started


def select_state(state, folds, train):
    errors, y, base = (state[k] for k in ('errors', 'y', 'base_oof'))
    anchors = np.concatenate([va for _, va in folds])
    bank, rows = {}, []
    for order in ORDERS:
        for alpha in ALPHAS:
            chunks = []
            for tr, va in folds:
                model = fit_corrector(errors, tr, order, alpha, int(va[0]-1))
                chunks.append(correction(model, errors, va))
            raw = np.vstack(chunks)
            for eta in ETAS:
                forecast = completed_forecast(base[anchors], raw, eta)
                mse = float(np.mean((y[anchors]-forecast)**2))
                row = dict(symbol=state['symbol'], volatility_window=state['volatility_window'],
                           order=order, alpha=alpha, eta=eta, validation_mse=mse)
                rows.append(row)
                bank[(order, alpha, eta)] = forecast
    best = min(rows, key=lambda x: (x['validation_mse'], x['eta'], x['order'], x['alpha']))
    last_raw = last_correction(errors, anchors)
    last_rows = [dict(eta=eta, validation_mse=float(np.mean(
        (y[anchors]-completed_forecast(base[anchors], last_raw, eta))**2))) for eta in ETAS]
    last = min(last_rows, key=lambda x: (x['validation_mse'], x['eta']))
    model = fit_corrector(errors, train, best['order'], best['alpha'], int(train[-1]+7))
    return dict(**best, last_eta=last['eta'], last_validation_mse=last['validation_mse']), rows, model


def metrics_frame(predictions):
    rows = []
    keys = ['year', 'model', 'symbol', 'volatility_window', 'horizon']
    for values, group in predictions.groupby(keys):
        error = group.actual.to_numpy()-group.forecast.to_numpy()
        denom = float(np.sum((group.actual-group.actual.mean())**2))
        rows.append(dict(zip(keys, values), n_origins=len(group), mse=float(np.mean(error**2)),
                         rmse=float(np.sqrt(np.mean(error**2))), mae=float(np.mean(np.abs(error))),
                         r2=float(1.-np.sum(error**2)/denom)))
    return pd.DataFrame(rows)


def evaluate(predictions, out):
    metrics = metrics_frame(predictions)
    cols = ['mse', 'rmse', 'mae', 'r2']
    macro = metrics.groupby(['year', 'model'], as_index=False)[cols].mean()
    metrics.to_csv(out/'horizon_metrics.csv', index=False)
    macro.to_csv(out/'macro_metrics.csv', index=False)
    metrics.groupby(['year', 'model', 'volatility_window'], as_index=False)[cols].mean().to_csv(out/'macro_by_window.csv', index=False)
    loss = predictions.assign(loss=(predictions.actual-predictions.forecast)**2).groupby(
        ['year', 'origin', 'model'], as_index=False).loss.mean().pivot(index=['year', 'origin'], columns='model', values='loss')
    loss.to_csv(out/'daily_loss_global.csv')
    dmrows = []
    for year in (2025, 2026):
        daily = loss.loc[year]
        for competitor in (BASE, LAST):
            result = dm_test(daily[AR]-daily[competitor], 7, 33)
            dmrows.append(dict(year=year, model_a=AR, model_b=competitor, **result))
    # The primary retrospective family includes AR against base and same-access
    # control in both inspected years (four comparisons), fixed before evaluation.
    adjusted = holm([row['p'] for row in dmrows], 4)
    for row, p in zip(dmrows, adjusted):
        row['p_holm4'] = float(p)
        row['significant_5pct'] = bool(p < .05)
    pd.DataFrame(dmrows).to_csv(out/'dm_global_holm4.csv', index=False)
    return macro, dmrows


def run(out=OUT, benchmark_only=False):
    started = time.monotonic()
    out = Path(out).resolve()
    if not out.is_relative_to(ROOT/'results/professor_feedback_20261018'):
        raise ValueError('Outputs must remain in the new local experiment.')
    out.mkdir(parents=True, exist_ok=True)
    if (out/'selection_freeze.json').exists():
        raise FileExistsError('Preserve the completed frozen experiment.')
    panel, folds, train, tests, states, native_audit = load_states()
    protected = [OOF, ORIGINAL/'selected_inputs.csv', ORIGINAL/'predictions.csv.gz', BASE2026]
    protected += sorted((ORIGINAL/'models').glob('*.joblib'))
    before = {str(p.relative_to(ROOT)): sha(p) for p in protected}
    pilot, audit, elapsed = prepare_archive(states[0], folds)
    cost = dict(pilot_2023_oof_seconds=elapsed, preparation_seconds=time.monotonic()-started-elapsed,
                estimated_remaining_seconds=elapsed*15/2*1.5, workers=2,
                required_new_xgboost_fits=16*len(audit), existing_2024_oof_reused=True)
    write_json(out/'cost_benchmark.json', cost)
    print(json.dumps(dict(stage='pilot', **cost)), flush=True)
    if benchmark_only:
        return
    protocol = dict(created_at_utc=datetime.now(timezone.utc).isoformat(), publication='LOCAL ONLY',
        github_or_render_updates_authorized=False, base_hyperparameters='Frozen original; selected on ALL six 2024 blocks',
        validation_limitation='CONDITIONAL on 2024-selected base; not nested/independent validation of base architecture',
        response='Actual minus honest out-of-fold base forecast, in original volatility percentage points',
        training_archive='2023 expanding 120-origin warmup/60-origin blocks, strict seven-day purge; original six purged 2024 OOF blocks',
        model='Seven direct Ridge AR regressions: e[t,h] ~ e[t-h,h],...,e[t-h-p+1,h]',
        coefficient_training='For each block only train origins with seven-day labels strictly before its first origin',
        scaling='StandardScaler on usable AR train rows only', orders=list(ORDERS), alphas=list(ALPHAS),
        shrinkage=list(ETAS), search_candidates_per_configuration=36,
        selection='Minimum pooled natural-unit MSE on six original 2024 validation blocks, common order/alpha/eta per asset/window',
        same_access_control='Last matured e[t-h,h], shrinkage from same 2024 validation grid',
        online='Coefficients fixed before 2025; history updates after close with only horizon-specific mature actuals',
        offline='Same frozen coefficients, 2024 OOF initial state; after 2025 recursively use predicted errors, no new observed labels',
        missing_history='No interpolation/compression; zero correction when exact calendar lags missing',
        evaluation='2025 and Jan01-Aug24 2026 already inspected; retrospective; no confirmatory holdout',
        primary='2026 global MSE and DM versus frozen base AND same-online-access control; report 2025 as secondary',
        dm='Paired daily loss averaged over 4 assets/4 windows/7 horizons; HLN7, Bartlett HAC33; Holm4 across two rivals/two years',
        replacement='No publication authorized; lower 2026 MSE and significant Holm4 wins versus both comparators required even for statistical evidence',
        input_sha256=before, training_source_sha256={str(Path(__file__).relative_to(ROOT)): sha(__file__)})
    write_json(out/'protocol.json', protocol)
    completed = [pilot]
    with ThreadPoolExecutor(max_workers=2) as pool:
        jobs = [pool.submit(prepare_archive, state, folds) for state in states[1:]]
        for future in as_completed(jobs):
            state, rows, seconds = future.result()
            completed.append(state)
            audit.extend(rows)
            print(f'OOF archive {len(completed)}/16: {state["symbol"]} w{state["volatility_window"]} ({seconds:.1f}s)', flush=True)
    completed.sort(key=lambda s: (s['symbol'], s['volatility_window']))
    pd.DataFrame(audit).to_csv(out/'inner_oof_calendar.csv', index=False)
    native_audit.to_csv(out/'original_outer_calendar_audit.csv', index=False)
    choices, search, models = [], [], {}
    (out/'models').mkdir(exist_ok=True)
    for state in completed:
        selected, rows, model = select_state(state, folds, train)
        choices.append(selected)
        search.extend(rows)
        key = (state['symbol'], state['volatility_window'])
        models[key] = model
        joblib.dump(dict(corrector=model, selected=selected), out/'models'/f'{key[0]}_v{key[1]}.joblib')
    pd.DataFrame(choices).to_csv(out/'selected_inputs.csv', index=False)
    pd.DataFrame(search).to_csv(out/'search_2024.csv', index=False)
    write_json(out/'selection_freeze.json', dict(frozen_at_utc=datetime.now(timezone.utc).isoformat(),
        protocol_sha256=sha(out/'protocol.json'), selected_inputs_sha256=sha(out/'selected_inputs.csv'),
        model_sha256={p.name:sha(p) for p in (out/'models').glob('*.joblib')},
        test_forecasts_or_metrics_computed=False))
    print('AR and same-access control choices frozen before 2025/2026 prediction.', flush=True)
    frames, replay = [], []
    boundary = int(panel.index.get_loc(BOUNDARY))
    for state, selected in zip(completed, choices):
        key = (state['symbol'], state['volatility_window'])
        model = models[key]
        bundle = joblib.load(ORIGINAL/'models'/f'{key[0]}_v{key[1]}.joblib')
        # Forecast every day with complete features, including dates outside the
        # seven-output comparison support, so later mature history has no gap.
        valid = np.flatnonzero((np.arange(len(panel)) >= boundary)
            & np.isfinite(state['X']).all(axis=1) & np.isfinite(state['H']).all(axis=1)
            & np.isfinite(state['current']))
        weight = bundle['ridge_weight']
        forecast = ((1.-weight)*predict(bundle['boosted_model'], state['X'][valid], state['current'][valid])
                    + weight*predict(bundle['ridge_model'], state['H'][valid], state['current'][valid]))
        forecast_panel = np.full(state['y'].shape, np.nan)
        forecast_panel[valid] = forecast
        errors = state['errors'].copy()
        errors[valid] = state['y'][valid]-forecast
        online_ar = correction(model, errors, valid)
        last = last_correction(errors, valid)
        offline = offline_correction(model, state['errors'], valid, boundary)
        for year, origins in tests.items():
            locations = np.searchsorted(valid, origins)
            np.testing.assert_array_equal(valid[locations], origins)
            variants = {BASE: forecast[locations],
                AR: completed_forecast(forecast[locations], online_ar[locations], selected['eta']),
                LAST: completed_forecast(forecast[locations], last[locations], selected['last_eta']),
                OFFLINE: completed_forecast(forecast[locations], offline[locations], selected['eta'])}
            for label, values in variants.items():
                frames.append(pd.DataFrame(dict(year=year, model=label, symbol=key[0], volatility_window=key[1],
                    origin=np.repeat(panel.index[origins].astype(str), 7), horizon=np.tile(np.arange(1,8), len(origins)),
                    target_date=panel.index[(origins[:,None]+np.arange(1,8)).ravel()].astype(str),
                    actual=state['y'][origins].ravel(), forecast=values.ravel())))
        saved = pd.read_csv(ORIGINAL/'predictions.csv.gz')
        saved = saved[(saved.symbol == key[0]) & (saved.volatility_window == key[1])].sort_values(['origin','horizon'])
        np.testing.assert_allclose(saved.forecast.to_numpy(), forecast_panel[tests[2025]].ravel(), rtol=1e-10, atol=1e-10)
        reference = pd.read_csv(BASE2026)
        reference = reference[(reference.model == BASE) & (reference.symbol == key[0]) & (reference.volatility_window == key[1])].sort_values(['origin','horizon'])
        np.testing.assert_allclose(reference.forecast.to_numpy(), forecast_panel[tests[2026]].ravel(), rtol=1e-10, atol=1e-10)
        replay.append(dict(symbol=key[0], volatility_window=key[1], frozen_base_replay_2025_2026=True))
    predictions = pd.concat(frames, ignore_index=True)
    predictions.to_csv(out/'predictions.csv.gz', index=False, compression=dict(method='gzip', mtime=0))
    for year in (2025, 2026):
        predictions[predictions.year.eq(year)].drop(columns='year').to_csv(
            out/f'predictions_{year}.csv.gz', index=False, compression=dict(method='gzip', mtime=0))
    pd.DataFrame(replay).to_csv(out/'frozen_base_replay.csv', index=False)
    macro, dmrows = evaluate(predictions, out)
    after = {str(p.relative_to(ROOT)): sha(p) for p in protected}
    if before != after:
        raise ValueError('A protected base artifact changed.')
    primary = macro[macro.year == 2026].set_index('model').mse
    wins = [row for row in dmrows if row['year'] == 2026 and row['p_holm4'] < .05 and row['dm'] < 0]
    summary = dict(status='complete', elapsed_seconds=time.monotonic()-started,
        publication='LOCAL ONLY', protected_artifacts_unchanged=True,
        conditional_validation_warning=protocol['validation_limitation'],
        primary_2026_mse={str(k):float(v) for k,v in primary.items()},
        lower_than_base_and_same_access_control=bool(primary[AR] < min(primary[BASE], primary[LAST])),
        primary_significant_wins=len(wins), evidence_gate_passed=bool(len(wins)==2),
        publication_authorized=False, retrospective_only=True)
    write_json(out/'summary.json', summary)
    print(macro.to_string(index=False), flush=True)
    print(json.dumps(summary), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--benchmark', action='store_true')
    args = parser.parse_args()
    with threadpool_limits(limits=1):
        run(benchmark_only=args.benchmark)
