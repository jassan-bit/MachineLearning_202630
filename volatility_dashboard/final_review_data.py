"""Read the curated final-review publication without loading or fitting models.

The public bundle mirrors result/delivery paths under ``files/``.  Nothing in
this module imports the research runners or falls back to unpublished files.
"""
from functools import lru_cache
import json
from pathlib import Path, PurePosixPath
from urllib.parse import quote

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
BUNDLE_ROOT = ROOT / 'dashboard_data' / 'final_review_20261018'
RESULTS = 'results/final_regression_20261018'
FEEDBACK = 'results/professor_feedback_20261018'
DELIVERY = 'delivery/final_regression_local_20261018'
WINDOW_REVIEW = f'{RESULTS}/window_review_original_vs_optimized_2025'
REFERENCE = 'HAR-Ridge + XGBoost'
FAMILIES = ('kNN', 'Ridge', 'Lasso', 'DecisionTree', 'RandomForest', 'XGBoost', 'SVR')
OPTIMIZERS = {'grid': 'Grid Search', 'random': 'Random Search',
              'bayesian': 'Bayesiana', 'genetic': 'Genética'}
YEARS = (2025, 2026)
WINDOWS = (7, 14, 21, 28)
METRIC_COLUMNS = ['model', 'mse', 'rmse', 'mae', 'r2']


def _relative(relative):
    path = PurePosixPath(str(relative).replace('\\', '/'))
    if path.is_absolute() or '..' in path.parts or not path.parts:
        raise ValueError('La ruta debe pertenecer al paquete público.')
    if path.parts[0] not in ('results', 'delivery') or ':' in str(path):
        raise ValueError('La ruta debe pertenecer a results o delivery.')
    return path


def artifact_path(relative):
    """Return a confined public artifact path; do not open the source tree."""
    relative = _relative(relative)
    root = (BUNDLE_ROOT / 'files').resolve()
    target = root.joinpath(*relative.parts).resolve()
    if not target.is_relative_to(root):
        raise ValueError('La ruta sale del paquete público.')
    return target


def public_url(relative):
    """Build the URL served by the app's manifest-constrained static route."""
    return '/final-review/' + quote(_relative(relative).as_posix(), safe='/')


def available(relative):
    return artifact_path(relative).is_file()


@lru_cache(maxsize=40)
def _cached_csv(path, mtime_ns, size):
    return pd.read_csv(path)


@lru_cache(maxsize=30)
def _cached_json(path, mtime_ns, size):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def csv(relative, required=True):
    path = artifact_path(relative)
    if not path.is_file() and not required:
        return pd.DataFrame()
    stat = path.stat()
    return _cached_csv(str(path), stat.st_mtime_ns, stat.st_size).copy(deep=True)


def metadata(relative, required=False):
    path = artifact_path(relative)
    if not path.is_file() and not required:
        return {}
    stat = path.stat()
    # A caller cannot modify a shared cached dictionary.
    return json.loads(json.dumps(_cached_json(str(path), stat.st_mtime_ns, stat.st_size)))


def _selection(year, optimizer='all', window=None):
    year = int(year)
    if year not in YEARS or optimizer not in ('all', *OPTIMIZERS):
        raise ValueError('Año o método de optimización no disponible.')
    if window is not None and int(window) not in WINDOWS:
        raise ValueError('Ventana objetivo no disponible.')
    return year, optimizer, int(window) if window is not None else None


def _benchmark_labels(optimizer):
    methods = OPTIMIZERS if optimizer == 'all' else (optimizer,)
    return [f'{family} [{method}]' for method in methods for family in FAMILIES]


def metrics(year, optimizer='all', window=7):
    """Original plus 7/28 rivals; macro metrics on the selected daily target."""
    year, optimizer, window = _selection(year, optimizer, window)
    frame = csv(f'{RESULTS}/evaluation/macro_by_window.csv')
    labels = [REFERENCE, *_benchmark_labels(optimizer)]
    frame = frame.loc[frame.year.eq(year) & frame.volatility_window.eq(window)
                      & frame.model.isin(labels), METRIC_COLUMNS].copy()
    if frame.model.duplicated().any() or set(frame.model) != set(labels):
        raise ValueError('El panel de métricas publicado está incompleto.')
    if not np.isfinite(frame[['mse', 'rmse', 'mae', 'r2']].to_numpy(float)).all():
        raise ValueError('El panel contiene métricas no finitas.')
    return frame.sort_values(['mse', 'model']).reset_index(drop=True)


def dm_2025(optimizer='all', window=7):
    """Filter frozen 112 comparisons, preserving every adjusted p-value."""
    _, optimizer, window = _selection(2025, optimizer, window)
    full = csv(f'{WINDOW_REVIEW}/dm_all_112.csv')
    expected = {(w, method, family) for w in WINDOWS for method in OPTIMIZERS
                for family in FAMILIES}
    keys = list(zip(full.volatility_window, full.method, full.family))
    if len(keys) != 112 or len(set(keys)) != 112 or set(keys) != expected:
        raise ValueError('La familia publicada debe contener las 112 comparaciones.')
    if not full.model_a.eq(REFERENCE).all():
        raise ValueError('La orientación DM publicada no es la esperada.')
    for size in (7, 28, 112):
        p = pd.to_numeric(full[f'p_holm{size}'], errors='coerce')
        if not p.between(0, 1).all():
            raise ValueError('El paquete contiene valores p ajustados inválidos.')
    selected = full.volatility_window.eq(window)
    if optimizer != 'all':
        selected &= full.method.eq(optimizer)
    return full.loc[selected].reset_index(drop=True)


def dm_decision(row, family_size):
    """Interpret the stored adjustment and the original-minus-rival loss."""
    p = float(row[f'p_holm{family_size}'])
    difference = float(row['diferencia_mse'])
    if not np.isfinite(p) or str(row.get('estado', 'OK')) != 'OK':
        return 'No estimable'
    if p >= .05 or difference == 0:
        return 'No rechaza H₀'
    return ('Favorece al original' if difference < 0 else 'Favorece al benchmark')


def window_summary():
    """All four 2025 targets and all 28 rivals, independent of UI filters."""
    frame = csv(f'{WINDOW_REVIEW}/window_summary.csv')
    if set(frame.volatility_window) != set(WINDOWS) or len(frame) != 4:
        raise ValueError('Falta alguna ventana del resumen publicado.')
    return frame.sort_values('volatility_window').reset_index(drop=True)


def candidate_global(year):
    """Separate global HAR experiments: four tests/year, never a window test."""
    year, _, _ = _selection(year)
    metrics_frame = csv(f'{RESULTS}/evaluation/macro_metrics.csv')
    dm = csv(f'{RESULTS}/evaluation/directed_dm_all_years.csv')
    labels = [f'HAR-XG [{method}]' for method in OPTIMIZERS]
    selected = metrics_frame.loc[metrics_frame.year.eq(year)
                                 & metrics_frame.model.isin(labels), METRIC_COLUMNS]
    directed = dm.loc[dm.year.eq(year)].copy()
    if (len(selected) != 4 or len(directed) != 4
            or not directed.family_size.eq(4).all()
            or not directed.model_b.eq(REFERENCE).all()):
        raise ValueError('Falta la familia global de cuatro candidatos HAR.')
    merged = selected.merge(directed[['model_a', 'p', 'p_holm', 'diferencia_mse']],
                            left_on='model', right_on='model_a', validate='one_to_one')
    original = metrics_frame.loc[metrics_frame.year.eq(year)
                                 & metrics_frame.model.eq(REFERENCE)]
    if len(merged) != 4 or len(original) != 1:
        raise ValueError('Falta la referencia global publicada.')
    merged['mse_original'] = float(original.iloc[0].mse)
    merged['mse_reduction_pct'] = 100 * (1 - merged.mse / merged.mse_original)
    merged['decision_holm4'] = [
        'Favorece al candidato' if p < .05 and d < 0 else
        'Favorece al original' if p < .05 and d > 0 else 'No rechaza H₀'
        for p, d in zip(merged.p_holm, merged.diferencia_mse)]
    return merged.sort_values('mse').reset_index(drop=True)


def outer_metrics(optimizer='all'):
    _selection(2025, optimizer)
    frame = csv(f'{RESULTS}/outer_fold_mean_sd.csv', required=False)
    if frame.empty:
        return frame
    return frame.loc[frame.model.isin(_benchmark_labels(optimizer))].copy()


def efficiency():
    """Small repeated benchmark; this diagnostic is not a held-out ranking."""
    frame = csv(f'{RESULTS}/efficiency/standard_vs_optimized_summary.csv', required=False)
    if frame.empty:
        return frame
    return frame.loc[frame.train_rows.eq(frame.train_rows.max())].copy()
