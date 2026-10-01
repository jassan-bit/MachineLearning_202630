"""Official Binance 1m archives, checksummed, resumable; daily UTC closes."""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import argparse
import hashlib
import json
import time
import zipfile
import numpy as np
import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
SYMBOLS = ('BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'XRPUSDT')
BASE = 'https://data.binance.vision/data/spot/monthly/klines'


def get(url):
    for attempt in range(5):
        try:
            r = requests.get(url, timeout=(15, 120))
            r.raise_for_status()
            return r
        except requests.RequestException:
            if attempt == 4:
                raise
            time.sleep(2 ** attempt)


def aggregate(frame, symbol, month):
    """Keep calendar gaps; incomplete days are explicitly ineligible."""
    raw = pd.to_numeric(frame.iloc[:, 0], errors='raise').astype('int64')
    unit = 'us' if raw.iloc[0] > 10**14 else 'ms'
    dates = pd.DatetimeIndex(pd.to_datetime(raw, unit=unit, utc=True))
    start = pd.Timestamp(month + '-01', tz='UTC')
    end = start + pd.offsets.MonthBegin(1)
    if dates.has_duplicates or not dates.is_monotonic_increasing:
        raise ValueError('Duplicate or unordered minute timestamps')
    if not ((dates >= start) & (dates < end)).all() or not (dates == dates.floor('min')).all():
        raise ValueError('Invalid minute timestamps')
    close = pd.to_numeric(frame.iloc[:, 4], errors='raise').to_numpy(float)
    if not np.isfinite(close).all() or (close <= 0).any():
        raise ValueError('Invalid close')
    close_time = pd.DatetimeIndex(pd.to_datetime(frame.iloc[:, 6], unit=unit, utc=True))
    expected_end = dates + pd.Timedelta(minutes=1) - pd.Timedelta(1, unit=unit)
    valid_time = close_time == expected_end
    s = pd.Series(close, index=dates)
    days = pd.date_range(start, end, freq='D', inclusive='left')
    out = pd.DataFrame({'close': s.resample('D').last(), 'minutes': s.resample('D').count()}).reindex(days)
    out['minutes'] = out.minutes.fillna(0).astype(int)
    out['invalid_close_times'] = pd.Series(~valid_time, index=dates).resample('D').sum().reindex(days, fill_value=0).astype(int)
    out['complete'] = out.minutes.eq(1440) & out.invalid_close_times.eq(0)
    out.loc[~out.complete, 'close'] = np.nan
    out['symbol'] = symbol
    out.index.name = 'date'
    return out.reset_index(), unit


def process(task):
    symbol, month = task
    rawdir = ROOT / 'data/raw/minute_2020_2025' / symbol
    rawdir.mkdir(parents=True, exist_ok=True)
    name = f'{symbol}-1m-{month}.zip'
    path = rawdir / name
    url = f'{BASE}/{symbol}/1m/{name}'
    checkpath = path.with_suffix('.zip.CHECKSUM')
    if not checkpath.exists():
        checkpath.write_text(get(url + '.CHECKSUM').text, encoding='utf-8')
    expected = checkpath.read_text().split()[0]
    if not path.exists():
        part = path.with_suffix('.part')
        part.write_bytes(get(url).content)
        if hashlib.sha256(part.read_bytes()).hexdigest() != expected:
            raise ValueError(f'Checksum mismatch: {url}')
        part.replace(path)
    sha = hashlib.sha256(path.read_bytes()).hexdigest()
    if sha != expected:
        raise ValueError(f'Cached checksum mismatch: {path}')
    with zipfile.ZipFile(path) as z:
        names = z.namelist()
        if len(names) != 1:
            raise ValueError('Unexpected archive layout')
        with z.open(names[0]) as f:
            frame = pd.read_csv(f, header=None)
    daily, unit = aggregate(frame, symbol, month)
    print(f'{symbol} {month}: {len(frame)} minutes, {int(daily.complete.sum())} complete days', flush=True)
    return daily, dict(symbol=symbol, month=month, url=url, sha256=sha, timestamp_unit=unit,
                       minutes=len(frame), incomplete_days=int((~daily.complete).sum()),
                       invalid_close_times=int(daily.invalid_close_times.sum()))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--workers', type=int, default=4)
    args = parser.parse_args()
    months = pd.period_range('2020-01', '2025-12', freq='M').astype(str)
    tasks = [(s, m) for s in SYMBOLS for m in months]
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        records = list(pool.map(process, tasks))
    out = ROOT / 'data/processed/daily_2020_2025.csv'
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.concat([r[0] for r in records]).sort_values(['symbol', 'date']).to_csv(out, index=False)
    result = ROOT / 'results'
    result.mkdir(exist_ok=True)
    pd.DataFrame([r[1] for r in records]).to_csv(result / 'minute_sources.csv', index=False)
    (result / 'data_manifest.json').write_text(json.dumps(dict(source='Binance public 1m archives',
        start='2020-01-01', end_exclusive='2026-01-01', symbols=SYMBOLS,
        daily_sha256=hashlib.sha256(out.read_bytes()).hexdigest(), archives=len(tasks)), indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
