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


def bootstrap_protocol():
    """Read the recorded global bootstrap settings, without rerunning research."""
    protocol = metadata(f'{RESULTS}/evaluation/protocol.json', required=True)
    intervals = protocol['intervals']
    settings = dict(repetitions=int(intervals['reps']),
                    block_length=int(intervals['block_length']),
                    seed=int(protocol['seed']), coverage=float(intervals['coverage']))
    if (settings['repetitions'] < 1000 or settings['block_length'] < 1
            or settings['seed'] < 0 or not 0 < settings['coverage'] < 1
            or 'stationary bootstrap percentile' not in intervals['method']):
        raise ValueError('El protocolo bootstrap global no es válido.')
    return settings


def optimizer_intervals(year, optimizer='all'):
    """Global metric intervals; the upper window filter intentionally does not apply.

    These are individual CIs of each model's MSE/RMSE/MAE, not CIs of a
    paired model difference. A common date resample was used by the study.
    """
    year, optimizer, _ = _selection(year, optimizer)
    settings = bootstrap_protocol()
    full = csv(f'{RESULTS}/evaluation/metric_intervals.csv')
    required = {'year', 'model', 'metric', 'estimate', 'ci_lower', 'ci_upper',
                'ci_method', 'nominal_coverage', 'unit'}
    if not required.issubset(full.columns):
        raise ValueError('Faltan columnas de los intervalos bootstrap globales.')
    wanted_metrics = ('mse', 'rmse', 'mae')
    labels = [REFERENCE, *_benchmark_labels(optimizer)]
    frame = full.loc[full.year.eq(year) & full.model.isin(labels)
                     & full.metric.isin(wanted_metrics)].copy()
    expected = {(model, metric) for model in labels for metric in wanted_metrics}
    keys = list(zip(frame.model, frame.metric))
    if len(keys) != len(expected) or set(keys) != expected:
        raise ValueError('El panel bootstrap global está incompleto o duplicado.')
    values = frame[['estimate', 'ci_lower', 'ci_upper', 'nominal_coverage']].to_numpy(float)
    if (not np.isfinite(values).all() or (frame.ci_lower > frame.ci_upper).any()
            or not np.allclose(frame.nominal_coverage, settings['coverage'], rtol=0, atol=1e-12)
            or not frame.ci_method.eq('paired stationary bootstrap percentile').all()):
        raise ValueError('Los límites o el método bootstrap global no son válidos.')
    units = {'mse': 'percentage_points_squared', 'rmse': 'percentage_points',
             'mae': 'percentage_points'}
    if not frame.unit.eq(frame.metric.map(units)).all():
        raise ValueError('Las unidades de los intervalos no corresponden a las métricas.')
    macro = csv(f'{RESULTS}/evaluation/macro_metrics.csv')
    control = macro.loc[macro.year.eq(year) & macro.model.isin(labels)]
    if len(control) != len(labels) or control.model.duplicated().any():
        raise ValueError('Falta el control de métricas globales del año seleccionado.')
    control = control.set_index('model')
    expected_estimates = [float(control.loc[row.model, row.metric]) for row in frame.itertuples()]
    if not np.allclose(frame.estimate, expected_estimates, rtol=1e-10, atol=1e-12):
        raise ValueError('Las estimaciones bootstrap no corresponden a las métricas globales.')
    return frame.sort_values(['metric', 'estimate', 'model']).reset_index(drop=True)


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
    effects = csv(f'{RESULTS}/evaluation/paired_effects_vs_original.csv')
    effects = effects.loc[effects.year.eq(year) & effects.model.isin(labels)]
    if (len(effects) != 4 or effects.model.duplicated().any()
            or not effects.reference.eq(REFERENCE).all()):
        raise ValueError('Faltan los cuatro intervalos emparejados de los candidatos HAR.')
    merged = merged.merge(effects[['model', 'mse_gain', 'gain_ci_lower', 'gain_ci_upper']],
                          on='model', validate='one_to_one')
    if (not np.isfinite(merged[['mse_gain', 'gain_ci_lower', 'gain_ci_upper']].to_numpy(float)).all()
            or (merged.gain_ci_lower > merged.gain_ci_upper).any()
            or not np.allclose(merged.mse_gain, merged.mse_original - merged.mse,
                               rtol=1e-10, atol=1e-12)):
        raise ValueError('Los intervalos HAR no respetan la orientación original − candidato.')
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
