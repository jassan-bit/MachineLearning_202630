"""Validate cached targets against saved metrics, then calculate TEST R2 only.

Run from any directory. Existing artifacts are read-only. The five new CSVs
must not already exist. No notebook, training, scaler or CV routine is executed.
Only the master's preparation cells and two selected functions are reused.
"""
import ast
import contextlib
import hashlib
import io
import json
import platform
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
from numpy.lib.stride_tricks import sliding_window_view
from sklearn.metrics import r2_score

ROOT = Path(__file__).resolve().parents[1]
RTOL, ATOL = 1e-8, 1e-12
OUTPUT_NAMES = [
    'volatility_r2_reconstruction_validation.csv',
    'volatility_r2_by_horizon.csv',
    'volatility_r2_by_fold.csv',
    'volatility_r2_summary.csv',
    'best_window_r2_by_symbol.csv',
]


def source_function(notebook, name):
    """Extract the literal function source, excluding all surrounding code."""
    matches = []
    for cell in notebook['cells']:
        if cell['cell_type'] != 'code':
            continue
        source = ''.join(cell['source'])
        for node in ast.parse(source).body:
            if isinstance(node, ast.FunctionDef) and node.name == name:
                matches.append(ast.get_source_segment(source, node))
    assert len(matches) == 1, name
    return matches[0]


def rebuild_targets():
    notebook = json.loads((ROOT / 'model_evaluation.ipynb').read_bytes())
    env = {'np': np, 'pd': pd, 'sliding_window_view': sliding_window_view,
           'display': lambda *args, **kwargs: None, 'MAPE_EPSILON': 1e-8}
    # Reuse literal constants from the master's configuration cell.
    required = {'SYMBOLS', 'INPUT_WINDOWS', 'N_STEPS_FORECAST', 'N_STEPS_JUMP'}
    for cell in notebook['cells']:
        if cell['cell_type'] != 'code':
            continue
        for node in ast.parse(''.join(cell['source'])).body:
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name) and target.id in required:
                        env[target.id] = ast.literal_eval(node.value)
    assert env['INPUT_WINDOWS'] == [7, 14, 21, 28]
    assert env['N_STEPS_FORECAST'] == 7 and env['N_STEPS_JUMP'] == 1
    env['df'] = pd.read_csv(ROOT / 'data/processed/crypto_binance_master_1h.csv')
    sources = []
    # These cells only transform data in memory and display diagnostics.
    for index, marker in [(5, 'PERIODO COMUN Y CONTINUIDAD TEMPORAL'),
                          (7, 'VOLATILITY_WINDOW = 30')]:
        source = ''.join(notebook['cells'][index]['source'])
        assert marker in source
        forbidden = {'fit', 'fit_transform', 'to_csv', 'savefig', 'savez',
                     'savez_compressed', 'split_train_val_test_groupKFold',
                     'expanding_temporal_indices', 'MLPRegressor'}
        for node in ast.walk(ast.parse(source)):
            if isinstance(node, ast.Call):
                name = getattr(node.func, 'attr', getattr(node.func, 'id', ''))
                assert name not in forbidden, name
        with contextlib.redirect_stdout(io.StringIO()):
            exec(compile(source, f'master-readonly-cell-{index + 1}', 'exec'), env)
        sources.append(hashlib.sha256(source.encode()).hexdigest())
    for name in ['build_segment_windows', 'metric_vectors']:
        source = source_function(notebook, name)
        exec(compile(source, f'master-function-{name}', 'exec'), env)
        sources.append(hashlib.sha256(source.encode()).hexdigest())
    print('Reused source SHA256:', json.dumps(sources))
    return env


def finite_mean(values):
    values = np.asarray(values, dtype=float)
    return float(np.mean(values)) if np.isfinite(values).all() else np.nan


def finite_std(values):
    values = np.asarray(values, dtype=float)
    return float(np.std(values, ddof=1)) if np.isfinite(values).all() else np.nan


def main():
    print('NO MODEL TRAINING PERFORMED', flush=True)
    outputs = {name: ROOT / 'results' / name for name in OUTPUT_NAMES}
    existing = [str(p) for p in outputs.values() if p.exists()]
    if existing:
        raise FileExistsError('Refusing to overwrite existing files: ' + ', '.join(existing))
    protected = [ROOT / 'model_evaluation.ipynb',
                 ROOT / 'data/processed/crypto_binance_master_1h.csv']
    protected += [p for p in (ROOT / 'results').rglob('*') if p.is_file()]
    protected += [p for p in (ROOT / 'notebooks/figs').rglob('*') if p.is_file()]
    hashes = {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in protected}
    env = rebuild_targets()
    symbols, windows = env['SYMBOLS'], env['INPUT_WINDOWS']
    horizon_metrics = pd.read_csv(ROOT / 'results/volatility_metrics_by_horizon.csv')
    fold_metrics = pd.read_csv(ROOT / 'results/volatility_metrics_by_fold.csv')
    horizon_metrics = horizon_metrics.loc[horizon_metrics.dataset.str.lower().eq('test')]
    fold_metrics = fold_metrics.loc[fold_metrics.dataset.str.lower().eq('test')]
    assert len(horizon_metrics) == 700 and len(fold_metrics) == 100
    keys = ['symbol', 'input_window', 'fold']
    assert not horizon_metrics.duplicated(keys + ['horizon']).any()
    assert not fold_metrics.duplicated(keys).any()
    horizon_metrics = horizon_metrics.set_index(keys + ['horizon'])
    fold_metrics = fold_metrics.set_index(keys)
    expected_files = {f'{s}_w{w}_fold{f}.npz' for s in symbols for w in windows for f in range(1, 6)}
    prediction_dir = ROOT / 'results/volatility_predictions'
    assert {p.name for p in prediction_dir.glob('*.npz')} == expected_files
    validation_rows, pending, fold_matches = [], [], []
    for symbol in symbols:
        asset = env['df_volatility'].loc[env['df_volatility'].symbol.eq(symbol)].copy()
        for window in windows:
            _, observed_y, _, _ = env['build_segment_windows'](asset, window)
            for fold in range(1, 6):
                key = (symbol, window, fold)
                with np.load(prediction_dir / f'{symbol}_w{window}_fold{fold}.npz', allow_pickle=False) as saved:
                    assert set(saved.files) == {'pred_train', 'pred_val', 'pred_test',
                                                'train_indices', 'validation_indices', 'test_indices'}
                    indices = saved['test_indices']
                    prediction = saved['pred_test'].copy()
                assert indices.ndim == 1 and np.issubdtype(indices.dtype, np.integer)
                assert len(indices) > 1 and np.all(np.diff(indices) > 0)
                assert indices.min() >= 0 and indices.max() < len(observed_y)
                truth = observed_y[indices]
                assert truth.shape == prediction.shape and truth.shape[1] == 7
                assert np.isfinite(truth).all() and np.isfinite(prediction).all()
                metrics = env['metric_vectors'](truth, prediction)
                fold_checks = {}
                for metric in ['MAE', 'MSE', 'RMSE', 'MAPE']:
                    stored = float(fold_metrics.loc[key, metric + '_mean'])
                    reconstructed = float(np.mean(metrics[metric]))
                    match = bool(np.isclose(stored, reconstructed, rtol=RTOL, atol=ATOL))
                    fold_checks[metric] = (stored, reconstructed, match)
                    fold_matches.append(match)
                for h in range(1, 8):
                    row = dict(symbol=symbol, input_window=window, fold=fold, horizon=h)
                    for metric in ['RMSE', 'MAE', 'MSE', 'MAPE']:
                        stored = float(horizon_metrics.loc[key + (h,), metric])
                        reconstructed = float(metrics[metric][h - 1])
                        row['stored_' + metric] = stored
                        row['reconstructed_' + metric] = reconstructed
                        row[metric + '_abs_difference'] = abs(stored - reconstructed)
                        row[metric + '_match'] = bool(np.isclose(stored, reconstructed, rtol=RTOL, atol=ATOL))
                        sf, rf, mf = fold_checks[metric]
                        row['fold_stored_' + metric] = sf
                        row['fold_reconstructed_' + metric] = rf
                        row['fold_' + metric + '_match'] = mf
                    row['abs_difference'] = row['RMSE_abs_difference']
                    row['match'] = all(v for k, v in row.items() if k.endswith('_match'))
                    validation_rows.append(row)
                pending.append((key, truth, prediction))
    validation = pd.DataFrame(validation_rows)
    validation.to_csv(outputs[OUTPUT_NAMES[0]], index=False, mode='x')
    # GLOBAL GATE: no R2 is calculated until every stored metric matches.
    assert len(validation) == 700 and validation['match'].all() and all(fold_matches), (
        'Reconstruction mismatch: R2 calculation stopped; inspect validation report.')
    print('VALIDATION PASSED: 700 horizon cases, 2800 horizon metric comparisons, '
          '400 fold metric comparisons; all match=True', flush=True)
    print('Maximum RMSE absolute difference:', validation.RMSE_abs_difference.max(), flush=True)
    r2_rows, fold_rows, degenerate = [], [], []
    for (symbol, window, fold), truth, prediction in pending:
        scores = []
        for h in range(7):
            constant = bool(np.all(truth[:, h] == truth[0, h]))
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter('always')
                score = float(r2_score(truth[:, h], prediction[:, h], force_finite=False))
            if not np.isfinite(score):
                assert constant, 'Unexpected nonfinite R2 with nonconstant target'
                degenerate.append(dict(symbol=symbol, input_window=window, fold=fold,
                                       horizon=h + 1, R2=str(score),
                                       warnings=[str(w.message) for w in caught]))
            scores.append(score)
            r2_rows.append(dict(symbol=symbol, input_window=window, fold=fold,
                                dataset='TEST', horizon=h + 1, R2=score))
        row = dict(symbol=symbol, input_window=window, fold=fold)
        row.update({f'R2_h{h + 1}': score for h, score in enumerate(scores)})
        row.update(R2_mean=finite_mean(scores), R2_std=finite_std(scores))
        fold_rows.append(row)
    by_horizon, by_fold = pd.DataFrame(r2_rows), pd.DataFrame(fold_rows)
    assert len(by_horizon) == 700 and len(by_fold) == 100
    assert by_horizon.groupby(keys).horizon.apply(lambda h: set(h) == set(range(1, 8))).all()
    assert by_fold.groupby(['symbol', 'input_window']).fold.apply(lambda f: set(f) == set(range(1, 6))).all()
    summary_rows = []
    for (symbol, window), group in by_fold.groupby(['symbol', 'input_window'], sort=False):
        row = dict(symbol=symbol, input_window=window, R2_mean=finite_mean(group.R2_mean),
                   R2_std=finite_std(group.R2_mean))
        row.update({f'R2_h{h}_mean': finite_mean(group[f'R2_h{h}']) for h in range(1, 8)})
        summary_rows.append(row)
    summary = pd.DataFrame(summary_rows)
    assert len(summary) == 20
    selected = pd.read_csv(ROOT / 'results/best_window_by_symbol.csv')[['symbol', 'best_input_window']]
    assert dict(zip(selected.symbol, selected.best_input_window)) == dict(
        BTCUSDT=28, ETHUSDT=28, BNBUSDT=7, XRPUSDT=28, SOLUSDT=7)
    best = selected.merge(summary, left_on=['symbol', 'best_input_window'],
                          right_on=['symbol', 'input_window'], validate='one_to_one').drop(columns='input_window')
    assert len(best) == 5
    for frame, name in zip([by_horizon, by_fold, summary, best], OUTPUT_NAMES[1:]):
        frame.to_csv(outputs[name], index=False, mode='x')
    assert all(hashlib.sha256(p.read_bytes()).hexdigest() == digest for p, digest in hashes.items())
    print('Versions:', json.dumps(dict(python=platform.python_version(), numpy=np.__version__,
                                      pandas=pd.__version__, sklearn=sklearn.__version__)))
    print('R2 calculated:', len(by_horizon))
    print('Negative R2 horizons:', int(by_horizon.R2.lt(0).sum()))
    print('R2 horizons >= 0.80:', int(by_horizon.R2.ge(.80).sum()))
    print('Fold R2 means >= 0.80:', int(by_fold.R2_mean.ge(.80).sum()))
    print('Configuration R2 means >= 0.80:', int(summary.R2_mean.ge(.80).sum()))
    print('Degenerate cases:', json.dumps(degenerate))
    print('Selected windows:\n' + best.to_string(index=False))
    print('NO MODEL TRAINING PERFORMED')


if __name__ == '__main__':
    main()
