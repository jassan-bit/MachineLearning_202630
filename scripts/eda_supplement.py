"""Read-only supplementary EDA. Prints JSON; never trains or writes datasets."""
import json
import platform
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import scipy
import statsmodels
from statsmodels.tsa.stattools import adfuller, kpss, pacf


def analyze():
    root = Path(__file__).resolve().parents[1]
    raw = pd.read_csv(root / 'data/processed/crypto_binance_master_1h.csv')
    quality = {'original_rows': len(raw), 'original_columns': len(raw.columns),
               'duplicate_rows': int(raw.duplicated().sum()),
               'duplicate_keys': int(raw.duplicated(['symbol', 'open_time']).sum()),
               'original_nan': int(raw.isna().sum().sum())}
    raw['open_time'] = pd.to_datetime(raw.open_time, format='mixed', utc=True)
    raw = raw.sort_values(['symbol', 'open_time']).drop_duplicates(['symbol', 'open_time'])
    ranges = raw.groupby('symbol').open_time.agg(['min', 'max'])
    frame = raw.loc[raw.open_time.between(ranges['min'].max(), ranges['max'].min())].copy()
    frame['delta_h'] = frame.groupby('symbol').open_time.diff().dt.total_seconds() / 3600
    frame['new_segment'] = frame.delta_h.notna() & frame.delta_h.ne(1)
    frame['segment_id'] = frame.groupby('symbol').new_segment.cumsum().astype(int)
    keys = ['symbol', 'segment_id']
    frame['log_return'] = np.log(frame.close / frame.groupby(keys).close.shift(1))
    frame['volatility'] = frame.groupby(keys).log_return.transform(
        lambda x: x.rolling(30, min_periods=30).std(ddof=1))
    clean = frame.dropna(subset=['log_return', 'volatility']).copy()
    assert len(raw) == 289094 and len(frame) == 267710 and len(clean) == 266060
    assert clean.groupby('symbol').size().eq(53212).all()
    assert np.isfinite(clean[['log_return', 'volatility']]).all().all()
    quality.update(common_rows=len(frame), useful_rows=len(clean),
                   final_columns=len(clean.columns), gaps=int(frame.new_segment.sum()))
    result = {'versions': {'python': platform.python_version(), 'numpy': np.__version__,
                          'pandas': pd.__version__, 'scipy': scipy.__version__,
                          'statsmodels': statsmodels.__version__},
              'quality': quality, 'hourly': {}, 'weekday': {}, 'segments': [],
              'stationarity': [], 'pacf': []}
    for symbol, asset in clean.groupby('symbol', sort=True):
        result['hourly'][symbol] = asset.groupby(asset.open_time.dt.hour).volatility.median().tolist()
        result['weekday'][symbol] = asset.groupby(asset.open_time.dt.dayofweek).volatility.median().tolist()
        # ADF/KPSS and standard PACF require regular sampling: never concatenate gaps.
        sizes = asset.groupby('segment_id').size()
        segment = asset.loc[asset.segment_id.eq(sizes.idxmax())].sort_values('open_time')
        assert segment.open_time.diff().dropna().eq(pd.Timedelta(hours=1)).all()
        result['segments'].append({'symbol': symbol, 'segment_id': int(sizes.idxmax()),
                                   'n': len(segment), 'start': str(segment.open_time.min()),
                                   'end': str(segment.open_time.max())})
        for variable in ['log_return', 'volatility']:
            values = segment[variable].to_numpy(dtype=float)
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter('always')
                adf = adfuller(values, maxlag=48, regression='c', autolag=None)
                kp = kpss(values, regression='c', nlags='auto')
            result['stationarity'].append({
                'symbol': symbol, 'variable': variable,
                'adf_statistic': float(adf[0]), 'adf_pvalue': float(adf[1]),
                'adf_lags': int(adf[2]), 'adf_nobs': int(adf[3]),
                'kpss_statistic': float(kp[0]), 'kpss_pvalue': float(kp[1]),
                'kpss_lags': int(kp[2]),
                'warnings': [str(w.message).strip() for w in caught]})
            result['pacf'].append({'symbol': symbol, 'variable': variable,
                                   'values': pacf(values, nlags=48, method='ywm').tolist()})
    return result


if __name__ == '__main__':
    print(json.dumps(analyze(), ensure_ascii=False, indent=2, allow_nan=False))
