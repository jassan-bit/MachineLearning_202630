"""Bind a completed offline model audit to the files served by the dashboard."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import pandas as pd

KEYS = ['symbol', 'volatility_window', 'origin', 'horizon']
SYMBOLS = ['BNBUSDT', 'BTCUSDT', 'ETHUSDT', 'XRPUSDT']
WINDOWS = [7, 14, 21, 28]


def file_digest(path):
    digest = hashlib.sha256()
    if Path(path).suffix == '.py':
        # Python treats CRLF and LF identically; preserve the same fingerprint
        # when Windows working files and Linux checkouts use different endings.
        digest.update(Path(path).read_bytes().replace(b'\r\n', b'\n'))
        return digest.hexdigest()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def forecast_digest(frame):
    """Fingerprint parsed keys and floats as well as their source file."""
    ordered = frame[KEYS + ['actual', 'forecast']].copy()
    ordered['origin'] = pd.to_datetime(ordered.origin, utc=True)
    ordered = ordered.sort_values(KEYS)
    digest = hashlib.sha256()
    digest.update('\0'.join(ordered.symbol).encode('utf-8'))
    for column in ['volatility_window', 'horizon']:
        digest.update(ordered[column].to_numpy(dtype='<i8').tobytes())
    digest.update(ordered.origin.astype('int64').to_numpy(dtype='<i8').tobytes())
    digest.update(ordered[['actual', 'forecast']].to_numpy(dtype='<f8').tobytes())
    return digest.hexdigest()


def audit_paths(root, families):
    root = Path(root)
    paths = [
        'volatility_dashboard/data_loader.py',
        'volatility_dashboard/calendar.py',
        'volatility_dashboard/runtime_audit.py',
        'volatility_dashboard/metrics.py',
        'volatility_dashboard/diebold_mariano.py',
        'src/improve_classical_forecast.py',
        'src/optimize_minute_svr.py',
        'src/minute_experiment.py',
        'src/volatility_experiment.py',
        'results/minute_2023_2025/data_manifest.json',
    ]
    manifest = json.loads((root / paths[-1]).read_text(encoding='utf-8'))
    paths.extend('data/processed/minute_2023_2025/' + name for name in manifest)
    for family in families.values():
        prefix = 'results/' + family + '/'
        paths.extend(prefix + name for name in [
            'verification.json', 'selected_inputs.csv', 'predictions.csv.gz',
            'native_cv_audit.csv', 'calendar.csv',
        ])
        paths.extend(prefix + 'models/' + symbol + '_v' + str(window) + '.joblib'
                     for symbol in SYMBOLS for window in WINDOWS)
        for name in ['status.json', 'configuration.json', 'macro_metrics.csv',
                     'selected_metrics.csv', 'search.csv']:
            if (root / prefix / name).is_file():
                paths.append(prefix + name)
    return sorted(set(paths))


def build_audit(root, families, predictions, audit):
    comparable = set(audit.loc[audit.estado_comparable == 'Comparable', 'model'])
    if comparable != set(families) or set(predictions.model) != set(families):
        raise ValueError('La auditoría completa debe verificar los siete modelos originales.')
    paths = audit_paths(root, families)
    return dict(
        format_version=1,
        status='passed',
        verified_at=datetime.now(timezone.utc).isoformat(),
        model_families=dict(families),
        verified_artifacts=len(families) * len(SYMBOLS) * len(WINDOWS),
        files={relative: file_digest(Path(root) / relative) for relative in paths},
        forecasts={label: dict(rows=len(frame), origins=frame.origin.nunique(),
                              sha256=forecast_digest(frame))
                   for label, frame in predictions.groupby('model', sort=False)},
    )


def read_audit(root, path, families):
    audit = json.loads(Path(path).read_text(encoding='utf-8'))
    if (audit.get('format_version') != 1 or audit.get('status') != 'passed'
            or audit.get('model_families') != dict(families)
            or set(audit.get('forecasts', {})) != set(families)):
        raise ValueError('Auditoría web incompatible; ejecutar src/build_dashboard_audit.py.')
    paths = audit_paths(root, families)
    if set(audit.get('files', {})) != set(paths):
        raise ValueError('Cambió el inventario auditado; ejecutar src/build_dashboard_audit.py.')
    for relative in paths:
        if file_digest(Path(root) / relative) != audit['files'][relative]:
            raise ValueError(f'Archivo distinto del auditado: {relative}; regenerar la auditoría web.')
    return audit


def check_forecast(frame, label, audit):
    expected = audit['forecasts'][label]
    if (len(frame) != expected['rows'] or frame.origin.nunique() != expected['origins']
            or forecast_digest(frame) != expected['sha256']):
        raise ValueError('Las predicciones cargadas difieren de la auditoría completa.')
