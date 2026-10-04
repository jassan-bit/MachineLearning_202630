"""Freeze a protocol before downloading a new period; evaluate saved SVRs once.

Historical inputs/models are read-only. The new data are separately cached and
the original 2025 evaluation is never overwritten or used for a new selection.
"""
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
import argparse
import hashlib
import importlib.metadata
import json
import zipfile

import joblib
import numpy as np
import pandas as pd
import requests
from threadpoolctl import threadpool_limits

from minute_experiment import ROOT, DATA, load_panel
from optimize_minute_svr import features, minute_features, metrics, predict

OUT = ROOT / 'results/current_delivery_audit/holdout'
RAW = ROOT / 'data/raw/frozen_holdout_2026'
NEW_DATA = ROOT / 'data/processed/frozen_holdout_2026'
MODEL_DIR = ROOT / 'results/optimized_minute_2023_2025'
SYMBOLS = ['BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'XRPUSDT']
BASE_URL = 'https://data.binance.vision/data/spot/monthly/klines'


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def freeze():
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / 'protocol.json'
    if path.exists():
        raise RuntimeError('Protocol already frozen; do not overwrite it')
    if any(RAW.glob('**/*.zip')) or NEW_DATA.exists():
        raise RuntimeError('New period already cached; cannot claim pre-download freeze')
    assets = list((MODEL_DIR / 'models').glob('*.joblib'))
    assets += [MODEL_DIR / 'selected_inputs.csv']
    assets += list(DATA.glob('*.npy')) + [DATA / 'daily_target_closes.csv']
    assets += [Path(__file__), ROOT / 'src/optimize_minute_svr.py',
               ROOT / 'src/minute_experiment.py', ROOT / 'src/volatility_experiment.py']
    protocol = dict(
        frozen_at_utc=datetime.now(timezone.utc).isoformat(),
        period_start='2026-01-01', period_end_exclusive='2026-09-01',
        last_origin='2026-08-24', symbols=SYMBOLS,
        selection='Use the 16 existing saved SVRs and 2024-selected inputs without refitting or tuning',
        target='Daily log-return rolling population standard deviation, percent, windows 7/14/21/28',
        horizons=list(range(1, 8)),
        exclusions='Same common 36-day calendar with all finite daily closes, including all seven future objectives; no imputation',
        metrics=['r2', 'rmse', 'mae', 'mse', 'mape'],
        aggregation='Mean over seven horizons per asset/window; mean over the 16 configurations',
        baseline='Known current volatility repeated for all seven horizons',
        primary='Paired comparison of macro mean horizon RMSE against persistence',
        inference='Descriptive additional evaluation, no selection or retraining based on these results',
        prior_use='No 2026 observations found in the audited project artifacts; external prior use is not established',
        file_sha256={str(p.relative_to(ROOT)).replace('\\', '/'): sha(p) for p in assets},
        versions={name: importlib.metadata.version(name) for name in
                  ['numpy', 'pandas', 'scikit-learn', 'requests', 'joblib']},
    )
    path.write_text(json.dumps(protocol, ensure_ascii=False, indent=2), encoding='utf-8')
    (OUT / 'protocol.sha256').write_text(sha(path) + '  protocol.json\n', encoding='ascii')
    print('Frozen protocol:', sha(path), flush=True)


def verify_protocol():
    path = OUT / 'protocol.json'
    expected = (OUT / 'protocol.sha256').read_text().split()[0]
    if sha(path) != expected:
        raise ValueError('Frozen protocol changed')
    protocol = json.loads(path.read_text(encoding='utf-8'))
    for relative, digest in protocol['file_sha256'].items():
        if sha(ROOT / relative) != digest:
            raise ValueError(f'Frozen artifact changed: {relative}')
    return protocol, expected


def fetch_month(task):
    symbol, month = task
    folder = RAW / symbol
    folder.mkdir(parents=True, exist_ok=True)
    name = f'{symbol}-1m-{month}.zip'
    path = folder / name
    checksum = path.with_suffix('.zip.CHECKSUM')
    url = f'{BASE_URL}/{symbol}/1m/{name}'
    if not checksum.exists():
        response = requests.get(url + '.CHECKSUM', timeout=(15, 45))
        response.raise_for_status()
        checksum.write_text(response.text, encoding='utf-8')
    expected = checksum.read_text().split()[0]
    if not path.exists():
        response = requests.get(url, timeout=(15, 120))
        response.raise_for_status()
        if hashlib.sha256(response.content).hexdigest() != expected:
            raise ValueError(f'Official checksum mismatch: {url}')
        path.write_bytes(response.content)
    if sha(path) != expected:
        raise ValueError(f'Cached archive changed: {path}')
    print('Downloaded and verified', symbol, month, flush=True)
    return dict(symbol=symbol, month=month, url=url, sha256=expected,
                retrieved_at_utc=datetime.now(timezone.utc).isoformat(), bytes=path.stat().st_size)


def download():
    protocol, digest = verify_protocol()
    months = pd.period_range(protocol['period_start'][:7], '2026-08', freq='M').astype(str)
    tasks = [(symbol, month) for symbol in SYMBOLS for month in months]
    with ThreadPoolExecutor(max_workers=4) as pool:
        records = list(pool.map(fetch_month, tasks))
    pd.DataFrame(records).to_csv(OUT / 'sources.csv', index=False)
    print('Verified archives:', len(records), 'protocol:', digest, flush=True)


def evaluate():
    protocol, digest = verify_protocol()
    if (OUT / 'summary.json').exists():
        raise RuntimeError('Evaluation already completed; inspect results instead of repeating selection')
    NEW_DATA.mkdir(parents=True, exist_ok=True)
    start = pd.Timestamp(protocol['period_start'], tz='UTC')
    end = pd.Timestamp(protocol['period_end_exclusive'], tz='UTC')
    new_dates = pd.date_range(start, end, freq='D', inclusive='left')
    original = load_panel()
    panel = original.reindex(original.index.append(new_dates))
    choices = pd.read_csv(MODEL_DIR / 'selected_inputs.csv')
    audit, predictions, result_rows = [], [], []
    combined = {}
    for symbol in SYMBOLS:
        values = np.full(len(new_dates) * 1440, np.nan)
        for month in pd.period_range('2026-01', '2026-08', freq='M').astype(str):
            path = RAW / symbol / f'{symbol}-1m-{month}.zip'
            expected = path.with_suffix('.zip.CHECKSUM').read_text().split()[0]
            if sha(path) != expected:
                raise ValueError(f'Invalid source: {path}')
            with zipfile.ZipFile(path) as archive:
                if len(archive.namelist()) != 1:
                    raise ValueError('Unexpected source archive structure')
                with archive.open(archive.namelist()[0]) as stream:
                    frame = pd.read_csv(stream, header=None, usecols=[0, 4, 6])
            raw = frame[0].to_numpy(dtype='int64')
            unit = 'us' if raw[0] > 10**14 else 'ms'
            dates = pd.DatetimeIndex(pd.to_datetime(raw, unit=unit, utc=True))
            month_start = pd.Timestamp(month + '-01', tz='UTC')
            prices = frame[4].to_numpy(float)
            close_times = pd.DatetimeIndex(pd.to_datetime(frame[6], unit=unit, utc=True))
            if dates.has_duplicates or not dates.is_monotonic_increasing:
                raise ValueError('Duplicate/unordered timestamps')
            if not ((dates >= month_start) & (dates < month_start + pd.offsets.MonthBegin(1))).all():
                raise ValueError('Dates outside source month')
            if not (dates == dates.floor('min')).all():
                raise ValueError('Unaligned timestamps')
            if not np.isfinite(prices).all() or (prices <= 0).any():
                raise ValueError('Invalid close prices')
            good = close_times == dates + pd.Timedelta(minutes=1) - pd.Timedelta(1, unit=unit)
            positions = np.asarray((dates - start) // pd.Timedelta(minutes=1), dtype=int)
            values[positions[good]] = prices[good]
            audit.append(dict(symbol=symbol, month=month, rows=len(frame),
                              invalid_close_times=int((~good).sum()), timestamp_unit=unit))
        days = values.reshape(-1, 1440)
        complete = np.isfinite(days).all(axis=1)
        panel.loc[new_dates, symbol] = np.where(complete, days[:, -1], np.nan)
        np.save(NEW_DATA / f'{symbol}.npy', values)
        combined[symbol] = np.concatenate([np.load(DATA / f'{symbol}.npy'), values])
    panel.loc[new_dates].to_csv(NEW_DATA / 'daily_target_closes.csv')
    panel.loc[new_dates].to_csv(OUT / 'daily_target_closes.csv')
    pd.DataFrame(audit).to_csv(OUT / 'quality.csv', index=False)
    finite = np.isfinite(panel.to_numpy()).all(axis=1)
    eligible = pd.Series(finite).rolling(36).sum().shift(-7).eq(36).to_numpy()
    test = np.flatnonzero(eligible & (panel.index >= start) & (panel.index < end))
    if len(test) == 0 or (panel.index[test[-1]] + pd.Timedelta(days=7) >= end):
        raise ValueError('Invalid evaluation calendar')
    for symbol in SYMBOLS:
        minute = minute_features(combined[symbol])
        for choice in choices[choices.symbol.eq(symbol)].itertuples(index=False):
            window = int(choice.volatility_window)
            X, y, base = features(panel[symbol], minute, int(choice.input_window), window)
            artifact = joblib.load(MODEL_DIR / 'models' / f'{symbol}_v{window}.joblib')
            if not np.isfinite(X[test]).all() or not np.isfinite(y[test]).all():
                raise ValueError('Nonfinite eligible features/targets')
            forecast = predict(artifact['model'], X[test], base[test])
            baseline = np.repeat(base[test, None], 7, axis=1)
            for model, output in [('SVR_optimized', forecast), ('Persistence', baseline)]:
                result_rows.append(dict(symbol=symbol, volatility_window=window,
                                        model=model, n_origins=len(test), **metrics(y[test], output)))
            for row, anchor in enumerate(test):
                for h in range(7):
                    predictions.append(dict(symbol=symbol, volatility_window=window,
                        origin=str(panel.index[anchor]), horizon=h + 1,
                        target_date=str(panel.index[anchor + h + 1]),
                        actual=y[anchor, h], svr=forecast[row, h], persistence=baseline[row, h]))
    results = pd.DataFrame(result_rows)
    results.to_csv(OUT / 'metrics.csv', index=False)
    cols = ['r2', 'rmse', 'mae', 'mse', 'mape']
    macro = results.groupby('model')[cols].mean()
    macro.to_csv(OUT / 'macro_metrics.csv')
    forecast_frame = pd.DataFrame(predictions)
    forecast_frame.to_csv(OUT / 'predictions.csv.gz', index=False, compression={'method': 'gzip', 'mtime': 0})
    # Recompute metrics from exported prediction matrices, independently of loop.
    for (symbol, window), group in forecast_frame.groupby(['symbol', 'volatility_window']):
        actual = group.pivot(index='origin', columns='horizon', values='actual').to_numpy()
        for label, column in [('SVR_optimized', 'svr'), ('Persistence', 'persistence')]:
            output = group.pivot(index='origin', columns='horizon', values=column).to_numpy()
            expected = metrics(actual, output)
            row = results[(results.symbol == symbol) & (results.volatility_window == window) & (results.model == label)].iloc[0]
            np.testing.assert_allclose(row[cols].to_numpy(float), [expected[c] for c in cols], rtol=1e-12)
    verify_protocol()
    summary = dict(protocol_sha256=digest, evaluated_at_utc=datetime.now(timezone.utc).isoformat(),
        first_origin=str(panel.index[test[0]]), last_origin=str(panel.index[test[-1]]),
        origins=len(test), forecast_rows=len(forecast_frame), configurations=16,
        original_models_unchanged=True, exported_metrics_verified=True,
        macro_metrics=macro.reset_index().to_dict(orient='records'),
        missing_minutes={s: int(np.isnan(combined[s][-len(new_dates) * 1440:]).sum()) for s in SYMBOLS})
    (OUT / 'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=['freeze', 'download', 'evaluate'])
    action = parser.parse_args().action
    with threadpool_limits(limits=1):
        {'freeze': freeze, 'download': download, 'evaluate': evaluate}[action]()
