"""Persistence on the existing selected TEST samples; no model fitting.

Uses the read-only reconstruction helpers already validated for R2. Creates
four new CSVs, refuses to overwrite files, and preserves previous artifacts.
"""
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import r2_score

from calculate_r2_from_saved_predictions import rebuild_targets, finite_mean, finite_std

ROOT = Path(__file__).resolve().parents[1]
NAMES = ['persistence_metrics_by_horizon.csv', 'persistence_metrics_by_fold.csv',
         'persistence_summary.csv', 'mlp_vs_persistence.csv']
METRICS = ['MAE', 'MSE', 'RMSE', 'MAPE', 'R2']


def main():
    print('NO MODEL TRAINING PERFORMED', flush=True)
    outputs = [ROOT / 'results' / name for name in NAMES]
    if any(p.exists() for p in outputs):
        raise FileExistsError('Refusing to overwrite existing persistence CSVs')
    protected = [ROOT / 'model_evaluation.ipynb',
                 ROOT / 'data/processed/crypto_binance_master_1h.csv']
    for directory in ['results', 'notebooks/figs']:
        protected.extend(p for p in (ROOT / directory).rglob('*') if p.is_file())
    hashes = {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in protected}
    selected = pd.read_csv(ROOT / 'results/best_window_by_symbol.csv', float_precision='round_trip')
    expected = dict(BTCUSDT=28, ETHUSDT=28, BNBUSDT=7, XRPUSDT=28, SOLUSDT=7)
    assert dict(zip(selected.symbol, selected.best_input_window)) == expected
    existing = pd.read_csv(ROOT / 'results/volatility_metrics_by_horizon.csv', float_precision='round_trip')
    existing = existing.loc[existing.dataset.str.lower().eq('test')]
    keys = ['symbol', 'input_window', 'fold', 'horizon']
    assert not existing.duplicated(keys).any()
    existing = existing.set_index(keys)
    saved_r2 = pd.read_csv(ROOT / 'results/volatility_r2_by_horizon.csv', float_precision='round_trip')
    assert saved_r2.dataset.eq('TEST').all() and not saved_r2.duplicated(keys).any()
    saved_r2 = saved_r2.set_index(keys)
    env = rebuild_targets()
    horizon_rows, fold_rows = [], []
    maximum_rmse_difference = 0.0
    comparison_count = 0
    for symbol, window in expected.items():
        asset = env['df_volatility'].loc[env['df_volatility'].symbol.eq(symbol)].copy()
        observed_x, observed_y, metadata, _ = env['build_segment_windows'](asset, window)
        assert observed_x.shape[1] == window and observed_y.shape[1] == 7
        assert (metadata.input_end < metadata.target_start).all()
        for fold in range(1, 6):
            path = ROOT / 'results/volatility_predictions' / f'{symbol}_w{window}_fold{fold}.npz'
            with np.load(path, allow_pickle=False) as saved:
                indices = saved['test_indices'].copy()
                pred_mlp = saved['pred_test'].copy()
            assert indices.ndim == 1 and np.issubdtype(indices.dtype, np.integer)
            assert len(indices) > 1 and np.all(np.diff(indices) > 0)
            assert indices.min() >= 0 and indices.max() < len(observed_y)
            truth = observed_y[indices]
            assert truth.shape == pred_mlp.shape and truth.shape[1] == 7
            assert np.isfinite(truth).all() and np.isfinite(pred_mlp).all()
            # Check alignment against existing MLP metrics before using targets.
            reconstructed = env['metric_vectors'](truth, pred_mlp)
            for h in range(1, 8):
                key = (symbol, window, fold, h)
                for metric in METRICS[:-1]:
                    stored = float(existing.loc[key, metric])
                    value = float(reconstructed[metric][h - 1])
                    np.testing.assert_allclose(value, stored, rtol=1e-8, atol=1e-12)
                    if metric == 'RMSE':
                        maximum_rmse_difference = max(maximum_rmse_difference, abs(value - stored))
                    comparison_count += 1
                np.testing.assert_allclose(
                    r2_score(truth[:, h - 1], pred_mlp[:, h - 1], force_finite=False),
                    saved_r2.loc[key, 'R2'], rtol=1e-8, atol=1e-12)
                comparison_count += 1
            # Last OBSERVED value in X, identical prediction for all horizons.
            prediction = np.repeat(observed_x[indices, -1, None], 7, axis=1)
            assert prediction.shape == truth.shape and np.isfinite(prediction).all()
            np.testing.assert_array_equal(prediction[:, 0], observed_x[indices, -1])
            assert np.all(prediction == prediction[:, :1])
            metrics = env['metric_vectors'](truth, prediction)
            metrics['R2'] = np.array([r2_score(truth[:, h], prediction[:, h], force_finite=False)
                                      for h in range(7)])
            if not np.isfinite(metrics['R2']).all():
                raise ValueError(f'Degenerate R2: {symbol}, window={window}, fold={fold}; no CSVs written')
            for h in range(1, 8):
                row = dict(symbol=symbol, input_window=window, fold=fold, dataset='TEST', horizon=h)
                row.update({metric: float(metrics[metric][h - 1]) for metric in METRICS})
                horizon_rows.append(row)
            row = dict(symbol=symbol, input_window=window, fold=fold, dataset='TEST')
            for metric in METRICS:
                row[metric + '_mean'] = finite_mean(metrics[metric])
                row[metric + '_std'] = finite_std(metrics[metric])
            fold_rows.append(row)
    by_horizon, by_fold = pd.DataFrame(horizon_rows), pd.DataFrame(fold_rows)
    assert len(by_horizon) == 175 and len(by_fold) == 25
    assert by_horizon.groupby(keys[:-1]).horizon.apply(lambda x: set(x) == set(range(1, 8))).all()
    assert by_fold.groupby(['symbol', 'input_window']).fold.apply(lambda x: set(x) == set(range(1, 6))).all()
    summaries = []
    for (symbol, window), group in by_fold.groupby(['symbol', 'input_window'], sort=False):
        row = dict(symbol=symbol, best_input_window=window, dataset='TEST')
        for metric in METRICS:
            row[metric + '_mean'] = finite_mean(group[metric + '_mean'])
            row[metric + '_std'] = finite_std(group[metric + '_mean'])
        summaries.append(row)
    summary = pd.DataFrame(summaries)
    mlp_r2 = pd.read_csv(ROOT / 'results/best_window_r2_by_symbol.csv', float_precision='round_trip').set_index('symbol')
    comparisons = []
    for symbol, window in expected.items():
        mlp = selected.loc[selected.symbol.eq(symbol)].iloc[0]
        baseline = summary.loc[summary.symbol.eq(symbol)].iloc[0]
        assert int(mlp_r2.loc[symbol, 'best_input_window']) == window
        for model, data in [('MLP', mlp), ('Persistence', baseline)]:
            row = dict(symbol=symbol, best_input_window=window, model=model)
            for metric in ['RMSE', 'MAE', 'MAPE']:
                row[metric + '_mean'] = float(data[metric + '_mean'])
            row['R2_mean'] = float(mlp_r2.loc[symbol, 'R2_mean'] if model == 'MLP' else data['R2_mean'])
            for metric in ['RMSE', 'MAE']:
                row[metric + '_improvement_pct'] = (
                    float(100 * (baseline[metric + '_mean'] - data[metric + '_mean']) /
                          baseline[metric + '_mean']) if model == 'MLP' else 0.0)
            comparisons.append(row)
    comparison = pd.DataFrame(comparisons)
    assert len(summary) == 5 and len(comparison) == 10
    for frame in [by_horizon, by_fold, summary, comparison]:
        assert np.isfinite(frame.select_dtypes(include=[np.number])).all().all()
    for path, frame in zip(outputs, [by_horizon, by_fold, summary, comparison]):
        frame.to_csv(path, index=False, mode='x')
    assert all(hashlib.sha256(p.read_bytes()).hexdigest() == h for p, h in hashes.items())
    print('Existing MLP metric comparisons passed:', comparison_count)
    print('Maximum reconstructed MLP RMSE difference:', maximum_rmse_difference)
    print(comparison.to_string(index=False))
    print('NO MODEL TRAINING PERFORMED')


if __name__ == '__main__':
    main()
