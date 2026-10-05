"""Horizon-specific HAR/XGBoost selection on causal 2024 forecasts only.

The select phase cannot access 2025 targets. Evaluate runs once after the
complete selection and fitted artifacts have been frozen by SHA-256.
Existing experiment directories are never overwritten.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time

import joblib
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from improve_classical_forecast import har_features, ridge
from optimize_minute_svr import ROOT, DATA, LAGS, calendars, features, load_panel, metrics, minute_features
from optimize_minute_xgboost import estimator
from research_har_candidates import dynamic_features, continuous_blend

OUT = ROOT / 'results/optimized_har_xgboost_2023_2025'
RESEARCH = ROOT / 'results/har_candidate_research_2024'
SOURCE = ROOT / 'results/optimized_minute_xgboost_2023_2025'
END = pd.Timestamp('2025-01-01', tz='UTC')


def digest(path):
    result = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            result.update(block)
    return result.hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding='utf-8')


def matrices(close, minute, lag, window):
    X, y, base = features(close, minute, lag, window)
    H, _, _ = har_features(close, minute, window)
    D, _, _, decay = dynamic_features(close, minute, window)
    return {'X': X, 'H': H, 'D': D}, y, base, decay


def component_configuration(key):
    parts = key.split('__')
    if parts[0] == 'ridge' and len(parts) == 3:
        mode, alpha = parts[1], float(parts[2])
        if mode not in ['relative', 'mse_relative', 'level', 'decay']:
            raise ValueError(f'Unknown ridge mode: {mode}')
        return dict(kind='ridge', mode=mode, alpha=alpha,
                    feature_set='H' if mode in ['relative', 'mse_relative'] else 'D',
                    weighted=mode == 'mse_relative')
    if parts[0] == 'boost' and len(parts) == 2:
        mode = parts[1]
        if mode not in ['source_relative', 'mse_relative', 'mse_decay', 'mse_shallow', 'level_correction']:
            raise ValueError(f'Unknown boosting mode: {mode}')
        return dict(kind='boost', mode=mode, feature_set='X', weighted=mode.startswith('mse'))
    raise ValueError(f'Unknown component: {key}')


def training_target(config, y, base, decay, train):
    mode = config['mode']
    if mode in ['relative', 'source_relative', 'mse_relative']:
        return y[train] / base[train, None] - 1
    if mode == 'level':
        return y[train]
    if mode == 'decay':
        return y[train] - decay[train]
    if mode in ['mse_decay', 'mse_shallow']:
        return (y[train] - decay[train]) / base[train, None]
    if mode == 'level_correction':
        return y[train] - base[train, None]
    raise ValueError(mode)


def decode(config, output, base, decay):
    mode = config['mode']
    if mode in ['relative', 'source_relative', 'mse_relative']:
        output = base[:, None] * (1 + output)
    elif mode == 'decay':
        output = decay + output
    elif mode in ['mse_decay', 'mse_shallow']:
        output = decay + base[:, None] * output
    elif mode == 'level_correction':
        output = base[:, None] + output
    return np.maximum(output, 0)


def select_horizons(actual, boosted, ridges):
    """Select each horizon independently, with finite convex MSE optima."""
    if actual.ndim != 2 or actual.shape[1] != 7 or not np.isfinite(actual).all():
        raise ValueError('Expected seven finite validation targets')
    records, forecasts = [], {}
    for boost_key, boost_forecast in boosted.items():
        for ridge_key, ridge_forecast in ridges.items():
            if boost_forecast.shape != actual.shape or ridge_forecast.shape != actual.shape:
                raise ValueError('Validation predictions must have identical shapes')
            if not np.isfinite(boost_forecast).all() or not np.isfinite(ridge_forecast).all():
                raise ValueError('Non-finite validation forecasts')
            prediction, weights = continuous_blend(actual, boost_forecast, ridge_forecast)
            errors = np.mean((actual - prediction) ** 2, axis=0)
            forecasts[(boost_key, ridge_key)] = prediction
            for h in range(7):
                records.append(dict(horizon=h + 1, boosted_key=boost_key, ridge_key=ridge_key,
                                    ridge_weight=float(weights[h]), validation_mse=float(errors[h])))
    if not records:
        raise ValueError('At least one ridge and boosting candidate are required')
    search = pd.DataFrame(records)
    chosen, prediction = [], np.empty_like(actual, dtype=float)
    for h in range(7):
        row = min((r for r in records if r['horizon'] == h + 1),
                  key=lambda r: (r['validation_mse'], r['boosted_key'], r['ridge_weight'], r['ridge_key']))
        chosen.append(row)
        prediction[:, h] = forecasts[(row['boosted_key'], row['ridge_key'])][:, h]
    return chosen, search, prediction


def fit_component(key, values, y, base, decay, train, candidate):
    config = component_configuration(key)
    model = ridge(config['alpha']) if config['kind'] == 'ridge' else estimator(candidate).set_params(n_jobs=4)
    if config['mode'] == 'mse_shallow':
        model.set_params(max_depth=2, learning_rate=.03, min_child_weight=10, reg_lambda=30)
    kwargs = {}
    if config['weighted']:
        weights = base[train] ** 2
        weights /= weights.mean()
        kwargs['ridge__sample_weight' if config['kind'] == 'ridge' else 'sample_weight'] = weights
    model.fit(values[config['feature_set']][train], training_target(config, y, base, decay, train), **kwargs)
    return dict(config, model=model)


def predict_optimized_bundle(bundle, close, minutes, origins):
    values, _, base, decay = matrices(close, minutes, bundle['input_window'], bundle['volatility_window'])
    origins = np.asarray(origins, dtype=int)
    predictions = {}
    for key, component in bundle['components'].items():
        X = values[component['feature_set']][origins]
        if not np.isfinite(X).all():
            raise ValueError('Incomplete causal history for optimized HAR prediction')
        predictions[key] = decode(component, component['model'].predict(X), base[origins], decay[origins])
    result = np.empty((len(origins), 7))
    for h, choice in enumerate(bundle['horizons']):
        weight = choice['ridge_weight']
        result[:, h] = ((1 - weight) * predictions[choice['boosted_key']][:, h]
                        + weight * predictions[choice['ridge_key']][:, h])
    return result


def verify_training_components(bundle, close, minutes, train):
    values, _, _, _ = matrices(close, minutes, bundle['input_window'], bundle['volatility_window'])
    if bundle['training_origins'] != len(train) or pd.Timestamp(bundle['fitted_through']) >= END:
        raise ValueError('Optimized HAR final training calendar differs')
    for component in bundle['components'].values():
        if component['kind'] == 'ridge':
            mean = component['model'].named_steps['standardscaler'].mean_
            np.testing.assert_allclose(mean, values[component['feature_set']][train].mean(axis=0))


def feature_names(feature_set):
    names = []
    if feature_set == 'D':
        names = ['volatilidad_actual', 'log_volatilidad_actual']
    for span in [1, 3, 7, 14, 28]:
        names.extend(f'{name}_media_{span}d' for name in
                     ['retorno', 'retorno_rms', 'riesgo_negativo', 'volatilidad_minuto',
                      'retorno_absoluto_minuto', 'volatilidad_negativa', 'max_retorno_minuto'])
    if feature_set == 'H':
        return names + ['log_volatilidad_actual', 'variabilidad_retornos_cuadrados_28d'] + [f'decaimiento_conocido_h{h}' for h in range(1, 8)]
    if feature_set == 'D':
        names.append('variabilidad_retornos_cuadrados_28d')
        for h in range(1, 8):
            names.extend([f'retorno_retenido_h{h}', f'volatilidad_retenida_h{h}'])
        return names
    raise ValueError(feature_set)


def freeze_checks():
    frozen = json.loads((OUT / 'frozen_selection.json').read_text(encoding='utf-8'))
    for name, expected in frozen['files'].items():
        if digest(OUT / name.replace('\\', '/')) != expected:
            raise ValueError(f'Frozen selection or fitted model changed: {name}')
    return frozen


def select_and_fit():
    started = time.time()
    if (OUT / 'frozen_selection.json').exists():
        freeze_checks()
        print('Selection already frozen; use --evaluate or --verify.', flush=True)
        return
    panel = load_panel().loc[lambda frame: frame.index < END].copy()
    folds, eligible, audit = calendars(panel)
    origins = np.concatenate([va for _, va in folds])
    train = np.flatnonzero(eligible & (panel.index + pd.Timedelta(days=7) < END))
    cache_paths = [RESEARCH / 'oof' / f'{symbol}_v{window}.npz' for symbol in panel.columns for window in LAGS]
    if any(not path.exists() for path in cache_paths):
        raise ValueError('Complete the isolated 2024 research and its 16 OOF caches before selection')
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / 'models').mkdir(exist_ok=True)
    configuration = dict(model='HAR_Ridge_XGBoost', format_version=2, horizon_days=7,
        objective='Independent horizon MSE on six purged expanding 2024 validation blocks',
        selection='Select component pair and convex weight per horizon from 2024 forecasts only',
        research_configuration=json.loads((RESEARCH / 'configuration.json').read_text(encoding='utf-8')),
        code_sha256={name: digest(ROOT / 'src' / name) for name in ['optimize_har_xgboost.py', 'research_har_candidates.py']},
        oof_sha256={path.name: digest(path) for path in cache_paths},
        source_choices_sha256=digest(SOURCE / 'selected_inputs.csv'),
        final_fit='All eligible labels ending before 2025; static models throughout 2025',
        target='Rolling population volatility of daily percentage log returns; seven future horizons',
        warning='2024 reused for tuning; 2025 already inspected: retrospective evaluation, not an untouched test',
        dm='Existing bilateral DM, HLN, Bartlett HAC and Holm over 21 pairs per filter are unchanged',
        uncertainty='No guarantee of significant superiority; no correction for exploration across 160 overlapping filters')
    write_json(OUT / 'configuration.json', configuration)
    audit.to_csv(OUT / 'native_cv_audit.csv', index=False)
    pd.DataFrame([dict(fold=i, split=split, origin=str(panel.index[a]), target_end=str(panel.index[a + 7]))
                  for i, (tr, va) in enumerate(folds, 1)
                  for split, anchors in [('train', tr), ('validation', va)] for a in anchors]).to_csv(OUT / 'calendar.csv', index=False)
    write_json(OUT / 'status.json', dict(status='selecting', test_data_evaluated=False))
    choices, searches, validation = [], [], []
    for symbol in panel.columns:
        minutes = minute_features(np.load(DATA / f'{symbol}.npy', mmap_mode='r')[:len(panel) * 1440])
        for window in LAGS:
            with np.load(RESEARCH / 'oof' / f'{symbol}_v{window}.npz', allow_pickle=False) as cache:
                np.testing.assert_array_equal(cache['origins'], origins)
                actual = cache['actual']
                original = cache['baseline']
                boosted = {name: cache[name] for name in cache.files if name.startswith('boost__')}
                ridges = {name: cache[name] for name in cache.files if name.startswith('ridge__')}
                lag, candidate = int(cache['lag']), int(cache['candidate'])
                selected, search, p = select_horizons(actual, boosted, ridges)
            values, y, base, decay = matrices(panel[symbol], minutes, lag, window)
            np.testing.assert_allclose(actual, y[origins], rtol=1e-12, atol=1e-12)
            if not all(np.isfinite(matrix[np.r_[train, origins]]).all() for matrix in values.values()):
                raise ValueError('Non-finite development features')
            original_mse = np.mean((actual - original) ** 2, axis=0)
            if np.any(np.mean((actual - p) ** 2, axis=0) > original_mse + 1e-10):
                raise ValueError('Selected 2024 candidate loses to available legacy blend')
            needed = sorted({choice[key] for choice in selected for key in ['ridge_key', 'boosted_key']})
            components = {key: fit_component(key, values, y, base, decay, train, candidate) for key in needed}
            summary = dict(symbol=symbol, volatility_window=window, input_window=lag, candidate=candidate,
                           validation_mse=float(np.mean((actual - p) ** 2)), validation_rmse=metrics(actual, p)['rmse'],
                           original_validation_mse=float(original_mse.mean()), horizon_parameters=json.dumps(selected))
            bundle = dict(**summary, format_version=2, components=components, horizons=selected,
                          training_origins=len(train), fitted_through=str(panel.index[train.max() + 7]), configuration=configuration)
            joblib.dump(bundle, OUT / 'models' / f'{symbol}_v{window}.joblib')
            choices.append(summary)
            search.insert(0, 'volatility_window', window)
            search.insert(0, 'symbol', symbol)
            searches.append(search)
            for i, a in enumerate(origins):
                for h in range(7):
                    validation.append(dict(symbol=symbol, volatility_window=window, origin=str(panel.index[a]),
                                           horizon=h + 1, actual=actual[i, h], forecast=p[i, h], original=original[i, h]))
            print(f'{symbol} v{window}: validation MSE {summary["original_validation_mse"]:.6f} -> {summary["validation_mse"]:.6f}', flush=True)
    pd.DataFrame(choices).to_csv(OUT / 'selected_inputs.csv', index=False)
    pd.concat(searches, ignore_index=True).to_csv(OUT / 'search.csv', index=False)
    pd.DataFrame(validation).to_csv(OUT / 'validation_predictions.csv.gz', index=False, compression='gzip')
    paths = [OUT / name for name in ['configuration.json', 'selected_inputs.csv', 'search.csv', 'calendar.csv', 'native_cv_audit.csv']]
    paths.extend(sorted((OUT / 'models').glob('*.joblib')))
    write_json(OUT / 'frozen_selection.json', dict(test_data_evaluated=False, selection_sha256=digest(OUT / 'selected_inputs.csv'),
                                                 files={path.relative_to(OUT).as_posix(): digest(path) for path in paths}))
    write_json(OUT / 'status.json', dict(status='selected', seconds=time.time() - started, test_data_evaluated=False,
                                       selection_sha256=digest(OUT / 'selected_inputs.csv')))


def evaluate():
    freeze_checks()
    if (OUT / 'predictions.csv.gz').exists():
        print('Retrospective evaluation already exported; not repeating selection.', flush=True)
        return
    started = time.time()
    panel = load_panel()
    _, eligible, _ = calendars(panel)
    test = np.flatnonzero(eligible & (panel.index >= END))
    reference = pd.read_csv(SOURCE / 'predictions.csv.gz')
    selections = pd.read_csv(OUT / 'selected_inputs.csv')
    records, scores = [], []
    for symbol in panel.columns:
        minutes = minute_features(np.load(DATA / f'{symbol}.npy', mmap_mode='r'))
        for window in LAGS:
            bundle = joblib.load(OUT / 'models' / f'{symbol}_v{window}.joblib')
            _, actual, base, _ = matrices(panel[symbol], minutes, bundle['input_window'], window)
            p = predict_optimized_bundle(bundle, panel[symbol], minutes, test)
            ref = reference.query('symbol == @symbol and volatility_window == @window').copy()
            ref['origin'] = pd.to_datetime(ref.origin, utc=True)
            boosted = ref.pivot(index='origin', columns='horizon', values='forecast').reindex(panel.index[test]).to_numpy()
            np.testing.assert_allclose(ref.pivot(index='origin', columns='horizon', values='actual').reindex(panel.index[test]), actual[test])
            baseline = np.repeat(base[test, None], 7, axis=1)
            for label, predicted in [('HAR_Ridge_XGBoost', p), ('XGBoost_reference', boosted), ('Persistence', baseline)]:
                scores.append(dict(symbol=symbol, volatility_window=window, model=label, n_origins=len(test), **metrics(actual[test], predicted)))
            for i, a in enumerate(test):
                for h in range(7):
                    records.append(dict(symbol=symbol, volatility_window=window, origin=str(panel.index[a]), horizon=h + 1,
                                        actual=actual[a, h], forecast=p[i, h], xgboost=boosted[i, h], persistence=baseline[i, h]))
    result = pd.DataFrame(scores)
    result.to_csv(OUT / 'selected_metrics.csv', index=False)
    pd.DataFrame(records).to_csv(OUT / 'predictions.csv.gz', index=False, compression='gzip')
    columns = ['r2', 'rmse', 'mae', 'mse', 'mape']
    macro = pd.concat([result.groupby(['symbol', 'model'], as_index=False)[columns].mean(),
                       result.groupby('model', as_index=False)[columns].mean().assign(symbol='GLOBAL_MACRO')], ignore_index=True)
    macro.to_csv(OUT / 'macro_metrics.csv', index=False)
    freeze_checks()
    previous = json.loads((OUT / 'status.json').read_text(encoding='utf-8'))
    write_json(OUT / 'status.json', dict(status='complete', seconds=previous['seconds'] + time.time() - started,
        selection_sha256=digest(OUT / 'selected_inputs.csv'), test_origins=len(test), test_data_evaluated=True,
        models=len(selections), selection_used_test_data=False))
    print(macro.to_string(index=False), flush=True)


def verify():
    frozen = freeze_checks()
    manifest = json.loads((ROOT / 'results/minute_2023_2025/data_manifest.json').read_text(encoding='utf-8'))
    for name, expected in manifest.items():
        if digest(DATA / name) != expected:
            raise ValueError(f'Dataset changed: {name}')
    panel = load_panel()
    folds, eligible, audit = calendars(panel)
    train = np.flatnonzero(eligible & (panel.index + pd.Timedelta(days=7) < END))
    test = np.flatnonzero(eligible & (panel.index >= END))
    frame = pd.read_csv(OUT / 'predictions.csv.gz')
    if frame.duplicated(['symbol', 'volatility_window', 'origin', 'horizon']).any():
        raise ValueError('Duplicate forecasts')
    np.testing.assert_array_equal(sorted(frame.origin.unique()), sorted(str(date) for date in panel.index[test]))
    pd.testing.assert_frame_equal(pd.read_csv(OUT / 'native_cv_audit.csv'), audit, check_dtype=False)
    calendar = pd.read_csv(OUT / 'calendar.csv')
    if pd.to_datetime(calendar.target_end, utc=True).max() >= END:
        raise ValueError('Selection includes 2025 labels')
    for _, group in calendar.groupby('fold'):
        if pd.to_datetime(group.loc[group.split == 'train', 'target_end'], utc=True).max() >= pd.to_datetime(group.loc[group.split == 'validation', 'origin'], utc=True).min():
            raise ValueError('Training labels overlap validation')
    selected = pd.read_csv(OUT / 'selected_inputs.csv')
    search = pd.read_csv(OUT / 'search.csv')
    if len(selected) != len(panel.columns) * len(LAGS):
        raise ValueError('Incomplete horizon selection')
    for row in selected.itertuples():
        options = search[(search.symbol == row.symbol) & (search.volatility_window == row.volatility_window)]
        for choice in json.loads(row.horizon_parameters):
            if not np.isclose(choice['validation_mse'], options.loc[options.horizon == choice['horizon'], 'validation_mse'].min(), rtol=1e-12, atol=1e-12):
                raise ValueError('Selected component does not minimize 2024 horizon MSE')
    maximum_error = 0.
    for symbol in panel.columns:
        minutes = minute_features(np.load(DATA / f'{symbol}.npy', mmap_mode='r'))
        for window in LAGS:
            bundle = joblib.load(OUT / 'models' / f'{symbol}_v{window}.joblib')
            verify_training_components(bundle, panel[symbol], minutes, train)
            saved = frame.query('symbol == @symbol and volatility_window == @window').copy()
            saved['origin'] = pd.to_datetime(saved.origin, utc=True)
            expected = saved.pivot(index='origin', columns='horizon', values='forecast').reindex(panel.index[test]).to_numpy()
            observed = predict_optimized_bundle(bundle, panel[symbol], minutes, test)
            _, actual, _, _ = matrices(panel[symbol], minutes, bundle['input_window'], window)
            np.testing.assert_allclose(saved.pivot(index='origin', columns='horizon', values='actual').reindex(panel.index[test]), actual[test], rtol=1e-12, atol=1e-12)
            np.testing.assert_allclose(expected, observed, rtol=1e-12, atol=1e-12)
            if not np.isfinite(observed).all() or (observed < 0).any():
                raise ValueError('Invalid forecasts')
            maximum_error = max(maximum_error, float(np.max(np.abs(expected - observed))))
            selection_row = selected.query('symbol == @symbol and volatility_window == @window').iloc[0]
            if bundle['horizons'] != json.loads(selection_row.horizon_parameters):
                raise ValueError('Serialized bundle differs from frozen horizon selection')
    report = dict(status='passed', checked_predictions=len(frame), models=len(panel.columns) * len(LAGS),
                  serialization_max_absolute_error=maximum_error, temporal_folds=len(folds),
                  selection_sha256=frozen['selection_sha256'], selection_frozen_before_test=True,
                  training_scalers_verified=True, common_targets_verified=True, dataset_sha256_verified=True,
                  horizon_selection_minima_verified=True)
    write_json(OUT / 'verification.json', report)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--select', action='store_true')
    parser.add_argument('--evaluate', action='store_true')
    parser.add_argument('--verify', action='store_true')
    args = parser.parse_args()
    with threadpool_limits(limits=1):
        if args.select:
            select_and_fit()
        if args.evaluate:
            evaluate()
        if args.verify:
            verify()
        if not any([args.select, args.evaluate, args.verify]):
            parser.error('Use --select, --evaluate or --verify')
