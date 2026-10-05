"""Read-only access to saved forecasts, causal features and fitted artifacts."""
from functools import lru_cache, wraps
from threading import RLock
from pathlib import Path
import hashlib
import json
import sys

import joblib
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT/'src') not in sys.path:
    sys.path.insert(0, str(ROOT/'src'))
from optimize_minute_svr import load_panel, calendars, minute_features, features, predict
from volatility_experiment import targets
from improve_classical_forecast import har_features, predict_bundle
from .metrics import metric_table

DATA = ROOT/'data/processed/minute_2023_2025'
FAMILIES = {
    'k-NN': 'optimized_minute_knn_2023_2025',
    'Ridge': 'optimized_minute_ridge_2023_2025',
    'Lasso': 'optimized_minute_lasso_2023_2025',
    'Random Forest': 'optimized_minute_randomforest_2023_2025',
    'XGBoost': 'optimized_minute_xgboost_2023_2025',
    'SVR Lineal': 'optimized_minute_2023_2025',
    'HAR-Ridge + XGBoost': 'improved_classical_2023_2025',
}
FEATURES = ['retorno_diario_relativo','retorno_cuadrado_relativo','volatilidad_minuto_relativa',
            'retorno_absoluto_minuto_relativo','volatilidad_negativa_relativa','max_retorno_minuto_relativo']
KEYS = ['symbol','volatility_window','origin','horizon']


def read_json(path):
    return json.loads(path.read_text(encoding='utf-8'))


def file_digest(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def serialized_cache(function):
    # lru_cache alone can run the same cold audit in both Gunicorn threads.
    lock = RLock()
    cached = lru_cache(maxsize=1)(function)

    @wraps(function)
    def wrapped(*args, **kwargs):
        with lock:
            return cached(*args, **kwargs)

    def clear():
        with lock:
            cached.cache_clear()

    wrapped.cache_clear = clear
    wrapped.cache_info = cached.cache_info
    return wrapped


@lru_cache(maxsize=1)
def dataset():
    return load_panel()


@lru_cache(maxsize=1)
def temporal_calendar():
    return calendars(dataset())


@serialized_cache
def repository():
    frames, audit = [], []
    panel = dataset()
    folds, eligible, native_audit = temporal_calendar()
    if not all(tr.max()+7 < va.min() for tr,va in folds):
        raise ValueError('Cortes temporales sin separación de etiquetas.')
    train = np.flatnonzero(eligible & (panel.index+pd.Timedelta(days=7)<pd.Timestamp('2025-01-01',tz='UTC')))
    test = np.flatnonzero(eligible & (panel.index>=pd.Timestamp('2025-01-01',tz='UTC')))
    expected_keys = pd.MultiIndex.from_product([panel.columns,[7,14,21,28],
        [str(value) for value in panel.index[test]],range(1,8)],names=KEYS).sort_values()
    manifest = read_json(ROOT/'results/minute_2023_2025/data_manifest.json')
    for name, digest in manifest.items():
        if file_digest(DATA/name) != digest:
            raise ValueError(f'Dataset modificado: {name}; revisar procedencia antes de comparar.')
    reference = None
    for label, family in FAMILIES.items():
        out = ROOT/'results'/family
        record = dict(model=label, source=str(out.relative_to(ROOT)), n_test=0, n_predictions=0,
                      fecha_inicio_test='', fecha_fin_test='', target='std de retornos diarios ×100, ddof=0',
                      estado_comparable='Excluido', reason='')
        try:
            verification = read_json(out/'verification.json')
            if not (verification.get('passed') or (label == 'HAR-Ridge + XGBoost' and verification.get('status') == 'passed')):
                raise ValueError('La verificacion guardada no paso.')
            selection = out/'selected_inputs.csv'
            selection_digest = verification.get('selection_sha256')
            if selection_digest is None and (out/'status.json').exists():
                selection_digest = read_json(out/'status.json').get('selection_sha256')
            if selection_digest != hashlib.sha256(selection.read_bytes()).hexdigest():
                raise ValueError('La seleccion no coincide con su verificacion.')
            frame = pd.read_csv(out/'predictions.csv.gz')
            if label == 'k-NN':
                frame = frame.rename(columns={'knn':'forecast'})
            elif label == 'SVR Lineal':
                frame = frame.rename(columns={'svr':'forecast'})
            if frame.duplicated(KEYS).any():
                raise ValueError('Predicciones duplicadas.')
            if not np.isfinite(frame[['actual','forecast']]).all().all():
                raise ValueError('Predicciones u objetivos no finitos.')
            indexed = frame.set_index(KEYS).sort_index()
            if not indexed.index.equals(expected_keys):
                raise ValueError('El test no contiene exactamente las observaciones elegibles del calendario común.')
            if reference is not None:
                if not indexed.index.equals(reference.index):
                    raise ValueError('Fechas, activos, ventanas u horizontes diferentes del test de referencia.')
                if not np.array_equal(indexed.actual.to_numpy(), reference.actual.to_numpy()):
                    raise ValueError('La variable objetivo no coincide exactamente.')
            if set(frame.horizon) != set(range(1,8)) or set(frame.volatility_window) != {7,14,21,28}:
                raise ValueError('Horizontes o ventanas inesperados.')
            saved_calendar = pd.read_csv(out/'native_cv_audit.csv')
            pd.testing.assert_frame_equal(saved_calendar,native_audit,check_dtype=False)
            for (symbol, window), group in frame.groupby(['symbol','volatility_window']):
                origin = pd.to_datetime(group.origin, utc=True)
                expected = pd.Series(targets(panel[symbol], window).to_numpy(), index=panel.index)
                for h, rows in group.groupby('horizon'):
                    y = expected.shift(-int(h)).reindex(pd.to_datetime(rows.origin, utc=True))
                    np.testing.assert_allclose(y, rows.actual, rtol=1e-10)
                artifact = artifact_for(label, symbol, window)
                if pd.Timestamp(artifact['fitted_through']) >= pd.Timestamp('2025-01-01', tz='UTC'):
                    raise ValueError('El entrenamiento final incluye etiquetas de test.')
                lag = input_window(label,symbol,window)
                X, y, base = arrays(symbol,lag,window)
                models = [artifact['model']] if label != 'HAR-Ridge + XGBoost' else []
                if label == 'HAR-Ridge + XGBoost':
                    H, _, _ = har_features(panel[symbol], minute_summary(symbol), window)
                    np.testing.assert_allclose(artifact['ridge_model'].named_steps['standardscaler'].mean_, H[train].mean(axis=0))
                for model in models:
                    if hasattr(model,'named_steps') and 'standardscaler' in model.named_steps:
                        np.testing.assert_allclose(model.named_steps['standardscaler'].mean_,X[train].mean(axis=0))
                with threadpool_limits(limits=1):
                    forecast = (predict_bundle(artifact, panel[symbol], minute_summary(symbol), test)
                                if label == 'HAR-Ridge + XGBoost' else predict(artifact['model'],X[test],base[test]))
                np.testing.assert_allclose(group.pivot(index='origin',columns='horizon',values='forecast'),forecast,rtol=1e-10,atol=1e-10)
            if reference is None:
                reference = indexed[['actual']]
            frame['origin'] = pd.to_datetime(frame.origin, utc=True)
            frame['model'] = label
            frames.append(frame[KEYS+['actual','forecast','model']])
            record.update(n_test=frame.origin.nunique(), n_predictions=len(frame),
                          fecha_inicio_test=str(frame.origin.min().date()), fecha_fin_test=str(frame.origin.max().date()),
                          estado_comparable='Comparable', reason='Objetivos y claves identicos; verificacion y seleccion comprobadas.')
        except (OSError, ValueError, KeyError, AssertionError) as exc:
            record['reason'] = str(exc)
        audit.append(record)
    for name in ['minute_2023_2025', 'cv_comparison']:
        audit.append(dict(model=f'Referencia historica: {name}', source=f'results/{name}', n_test=None,
                          n_predictions=None, fecha_inicio_test='', fecha_fin_test='', target='Protocolo historico',
                          estado_comparable='Excluido', reason='Splits o ajustes diferentes; fuera del ranking principal.'))
    if not frames:
        raise ValueError('No hay modelos comparables disponibles.')
    predictions = pd.concat(frames, ignore_index=True)
    return predictions, metric_table(predictions), pd.DataFrame(audit)


@lru_cache(maxsize=4)
def artifact_for(model, symbol, window):
    return joblib.load(ROOT/'results'/FAMILIES[model]/'models'/f'{symbol}_v{window}.joblib')


def input_window(model, symbol, window):
    artifact = artifact_for(model, symbol, window)
    return int(artifact['input_window'])


@lru_cache(maxsize=4)
def minute_summary(symbol):
    # Keep only daily summaries; do not repeatedly allocate minute returns
    # for every model, input lag and volatility window.
    return minute_features(np.load(DATA/f'{symbol}.npy', mmap_mode='r'))


@lru_cache(maxsize=8)
def arrays(symbol, lag, window):
    minute = minute_summary(symbol)
    return features(dataset()[symbol], minute, lag, window)


def feature_names(lag):
    return [f'{name}_lag_{offset}' for offset in range(lag) for name in FEATURES] + [f'decaimiento_conocido_h{h}' for h in range(1,8)]


@lru_cache(maxsize=32)
def eda_frame(symbol, window, horizon):
    close = dataset()[symbol]
    X, y, base = arrays(symbol, 7, window)
    frame = pd.DataFrame(X[:, :6], index=close.index, columns=FEATURES)
    frame['close'] = close
    frame['retorno_log_pct'] = 100*np.log(close).diff()
    frame['volatilidad_historica'] = targets(close, window)
    frame['volatilidad_futura'] = y[:, horizon-1]
    frame['decaimiento_conocido'] = X[:, -7+horizon-1]
    return frame


def hyperparameters(model, symbol, window):
    artifact = artifact_for(model, symbol, window)
    if model == 'HAR-Ridge + XGBoost':
        return dict(input_window=artifact['input_window'], alpha=artifact['alpha'],
                    ridge_weight=artifact['ridge_weight'], xgboost_weight=1-artifact['ridge_weight'],
                    escalamiento='StandardScaler solo train en HAR-Ridge; XGBoost sin escalador',
                    xgboost=hyperparameters('XGBoost', symbol, window))
    fitted = artifact['model']
    if hasattr(fitted, 'named_steps'):
        fitted = list(fitted.named_steps.values())[-1]
    if model == 'SVR Lineal':
        fitted = fitted.regressor_.estimators_[0]
    keep = ['n_neighbors','weights','p','metric','alpha','n_estimators','max_depth','min_samples_leaf',
            'min_samples_split','max_features','learning_rate','subsample','colsample_bytree','C','epsilon','loss','tol','max_iter']
    params = {k:v for k,v in fitted.get_params().items() if k in keep}
    return dict(input_window=input_window(model,symbol,window), n_features=6*input_window(model,symbol,window)+7,
                escalamiento='StandardScaler solo train' if model not in ['Random Forest','XGBoost'] else 'Sin escalador', **params)


@lru_cache(maxsize=64)
def importance(model, symbol, window, horizon):
    artifact = artifact_for(model, symbol, window)
    if model == 'HAR-Ridge + XGBoost':
        names = []
        for span in [1,3,7,14,28]:
            names.extend([f'{name}_media_{span}d' for name in
                          ['retorno','retorno_rms','riesgo_negativo']+FEATURES[2:]])
        names += ['log_volatilidad_actual','variabilidad_retornos_cuadrados_28d']
        names += [f'decaimiento_conocido_h{h}' for h in range(1,8)]
        values = artifact['ridge_model'].named_steps['ridge'].coef_[horizon-1]
        frame = pd.DataFrame({'feature':names, 'importance':values})
        return frame.reindex(frame.importance.abs().sort_values(ascending=False).index).head(15), 'Coeficientes del componente HAR-Ridge estandarizado; no representan la importancia del conjunto combinado'
    lag = input_window(model,symbol,window)
    names = feature_names(lag)
    if model in ['Ridge','Lasso']:
        net = artifact['model'].named_steps[model.lower()]
        values = net.coef_[horizon-1]
        method = 'Coeficientes por feature estandarizada; salida: correccion relativa'
    elif model == 'SVR Lineal':
        target = artifact['model'].named_steps['transformedtargetregressor']
        values = target.regressor_.estimators_[horizon-1].coef_*target.transformer_.scale_[horizon-1]
        method = 'Coeficientes por feature estandarizada, deshaciendo escalado del target relativo'
    elif model in ['Random Forest','XGBoost']:
        values = artifact['model'].feature_importances_
        method = 'Reduccion de impureza (agregada sobre salidas)' if model == 'Random Forest' else 'Ganancia normalizada (agregada sobre salidas)'
    else:
        # Descriptive permutation on a fixed, spaced subset of the existing
        # test, never used for selection or fitting. Only seven feature groups.
        X, y, base = arrays(symbol,lag,window)
        origins = repository()[0].query('model == @model and symbol == @symbol and volatility_window == @window').origin.unique()
        anchors = dataset().index.get_indexer(origins)
        anchors = anchors[np.linspace(0,len(anchors)-1,min(120,len(anchors)),dtype=int)]
        original = X[anchors].copy()
        def forecast(values):
            with threadpool_limits(limits=1):
                return predict(artifact['model'],values,base[anchors])
        baseline = np.sqrt(np.mean((y[anchors,horizon-1]-forecast(original)[:,horizon-1])**2))
        names = FEATURES+['decaimiento_conocido_todos_horizontes']
        groups = [list(range(j,6*lag,6)) for j in range(6)]+[list(range(6*lag,6*lag+7))]
        values = []
        rng = np.random.default_rng(42)
        for indices in groups:
            deltas = []
            for _ in range(3):
                changed = original.copy()
                changed[:,indices] = original[rng.permutation(len(original))][:,indices]
                deltas.append(np.sqrt(np.mean((y[anchors,horizon-1]-forecast(changed)[:,horizon-1])**2))-baseline)
            values.append(float(np.mean(deltas)))
        method = f'Permutation por grupos de lags: aumento RMSE, {len(anchors)} fechas espaciadas de test, 3 repeticiones; diagnostico descriptivo'
    frame = pd.DataFrame({'feature':names, 'importance':values})
    return frame.reindex(frame.importance.abs().sort_values(ascending=False).index).head(15), method


def tuning(model,symbol,window):
    path = ROOT/'results'/FAMILIES[model]/'search.csv'
    if not path.exists():
        return pd.DataFrame()
    frame = pd.read_csv(path)
    return frame[(frame.symbol==symbol)&(frame.volatility_window==window)]
