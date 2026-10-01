"""Daily multi-output linear SVR. tsxv splits audited on calendar indices."""
from pathlib import Path
import hashlib
import inspect
import json
import warnings
import numpy as np
import pandas as pd
from sklearn.compose import TransformedTargetRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.multioutput import MultiOutputRegressor
from sklearn.svm import LinearSVR
from sklearn.exceptions import ConvergenceWarning
from statsmodels.tsa.stattools import bds
from tsxv.splitTrainValTest import split_train_val_test_groupKFold

ROOT = Path(__file__).resolve().parents[1]


def configuration():
    return json.loads((ROOT / 'experiment.json').read_text(encoding='utf-8'))


def load_daily():
    cfg = configuration()
    frame = pd.read_csv(ROOT / 'data/processed/daily_2020_2025.csv', parse_dates=['date'])
    frame['date'] = pd.to_datetime(frame.date, utc=True)
    if frame.duplicated(['symbol', 'date']).any():
        raise ValueError('Duplicate daily rows')
    grid = pd.date_range(cfg['start'], cfg['end_exclusive'], freq='D', inclusive='left', tz='UTC')
    panel = frame.pivot(index='date', columns='symbol', values='close').reindex(grid)
    if set(panel.columns) != set(cfg['symbols']):
        raise ValueError('Unexpected symbols')
    values = panel.to_numpy()
    if np.isinf(values).any() or (values[np.isfinite(values)] <= 0).any():
        raise ValueError('Invalid daily close')
    return panel[cfg['symbols']]


def targets(close, window):
    return 100 * np.log(close).diff().rolling(window, min_periods=window).std(ddof=0)


def calendar_folds(panel, cfg):
    """Native 5 tsxv folds + explicit chronological filtering.

    Use common 28-day index windows, then truncate FEATURES only. Library
    outputs are indices, mapped to rolling volatility instead of raw prices.
    This makes all 16 experiments use identical eligible forecast origins.
    """
    grid = panel.index
    horizon = cfg['horizon']
    maximum = max(max(cfg['input_windows']), max(cfg['volatility_windows']) + 1)
    # All past history and seven future daily closes must exist for all assets.
    finite = np.isfinite(panel.to_numpy()).all(axis=1)
    eligible = pd.Series(finite).rolling(maximum + horizon).sum().shift(-horizon).eq(maximum + horizon).to_numpy()
    raw = split_train_val_test_groupKFold(np.arange(len(grid)), cfg['cv_calendar_input'], horizon, cfg['cv_jump'])
    tr_end = pd.Timestamp(cfg['train_end_exclusive'], tz='UTC')
    va_end = pd.Timestamp(cfg['validation_end_exclusive'], tz='UTC')
    result, audit = [], []
    for f in range(5):
        fold = {}
        for role, xi, yi in [('train', 0, 1), ('val', 2, 3), ('test', 4, 5)]:
            x, y = raw[xi][f], raw[yi][f]
            if len(x) == 0:
                raise ValueError('Empty tsxv split')
            anchors = x[:, -1].astype(int)
            assert np.array_equal(y, anchors[:, None] + np.arange(1, horizon + 1))
            first, last = grid[anchors], grid[y[:, -1]]
            mask = eligible[anchors]
            if role == 'train':
                mask &= last < tr_end
            elif role == 'val':
                mask &= (first >= tr_end) & (last < va_end)
            else:
                mask &= first >= va_end
            selected = anchors[mask]
            if len(selected) < 2:
                raise ValueError(f'Insufficient {role} samples in fold {f+1}')
            fold[role] = selected
            for anchor in selected:
                audit.append(dict(fold=f+1, split=role, anchor=str(grid[anchor]),
                    history_start=str(grid[anchor-maximum+1]), target_start=str(grid[anchor+1]),
                    target_end=str(grid[anchor+horizon]), native_samples=len(anchors),
                    retained_samples=len(selected)))
        assert fold['train'].max() + horizon < fold['val'].min()
        assert fold['val'].max() + horizon < fold['test'].min()
        result.append(fold)
    return result, pd.DataFrame(audit)


def arrays(close, vol, anchors, lag, horizon=7):
    values = close.to_numpy(float)
    X = values[anchors[:, None] + np.arange(-lag+1, 1)]
    y = vol.to_numpy(float)[anchors[:, None] + np.arange(1, horizon+1)]
    baseline = np.repeat(vol.to_numpy(float)[anchors, None], horizon, axis=1)
    if not all(np.isfinite(v).all() for v in (X, y, baseline)):
        raise ValueError('Nonfinite model input or target')
    return X, y, baseline


def estimator(C, epsilon, seed=42):
    return Pipeline([
        ('x_scale', StandardScaler()),
        ('regressor', TransformedTargetRegressor(
            regressor=MultiOutputRegressor(LinearSVR(C=C, epsilon=epsilon,
                loss='squared_epsilon_insensitive', dual=False, max_iter=50000, tol=1e-6,
                random_state=seed)), transformer=StandardScaler()))])


def fit_model(model, X, y):
    with warnings.catch_warnings():
        warnings.simplefilter('error', ConvergenceWarning)
        model.fit(X, y)
    return model


def metric_rows(y, prediction):
    error = prediction - y
    mse = (error**2).mean(axis=0)
    with np.errstate(divide='ignore', invalid='ignore'):
        ape = np.where(y != 0, 100 * np.abs(error/y), np.nan)
    mape = np.array([np.nanmean(a) if np.isfinite(a).any() else np.nan for a in ape.T])
    rows = [dict(horizon=h+1, mse=mse[h], rmse=np.sqrt(mse[h]),
        mae=np.abs(error[:, h]).mean(), mape=mape[h], mape_zero_targets=int((y[:, h] == 0).sum()),
        n=len(y), negative_predictions=int((prediction[:, h] < 0).sum())) for h in range(y.shape[1])]
    rows.append(dict(horizon=0, **{key: float(np.mean([r[key] for r in rows]))
        for key in ('mse', 'rmse', 'mae', 'mape')}, n=len(y),
        mape_zero_targets=int((y == 0).sum()), negative_predictions=int((prediction < 0).sum())))
    return rows


def residual_bds(residual, minimum=10):
    if len(residual) < minimum or np.std(residual) <= 1e-12:
        return dict(bds_stat=np.nan, bds_pvalue=np.nan, bds_status='insufficient_or_constant', n=len(residual))
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        stat, pvalue = bds(residual, max_dim=2)
    stat, pvalue = float(np.asarray(stat)), float(np.asarray(pvalue))
    ok = np.isfinite(stat) and np.isfinite(pvalue)
    return dict(bds_stat=stat if ok else np.nan, bds_pvalue=pvalue if ok else np.nan,
                bds_status='ok_small_sample' if ok else 'undefined', n=len(residual))


def provenance(cfg):
    source = inspect.getsource(split_train_val_test_groupKFold)
    return dict(configuration=cfg, tsxv_source_sha256=hashlib.sha256(source.encode()).hexdigest(),
        daily_sha256=hashlib.sha256((ROOT/'data/processed/daily_2020_2025.csv').read_bytes()).hexdigest(),
        warning='Native tsxv GroupKFold is not forward-only; chronological filters are required. Folds overlap.')
