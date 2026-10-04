"""Read-only, reproducible quality audit of the delivery's 2023--2025 data.

Run: .venv-repro/Scripts/python.exe src/audit_current_dataset.py
No downloads, data repairs, model fits, or changes to original results.
"""
from pathlib import Path
import hashlib
import json
import zipfile

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'results/current_delivery_audit/quality'
FIG = ROOT / 'book/figures'
DATA = ROOT / 'data/processed/minute_2023_2025'
SYMBOLS = ['BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'XRPUSDT']
START = pd.Timestamp('2023-01-01', tz='UTC')
END = pd.Timestamp('2026-01-01', tz='UTC')
MINUTES = pd.date_range(START, END, freq='min', inclusive='left')
DAYS = pd.date_range(START, END, freq='D', inclusive='left')
FIELDS = ['open_time', 'open', 'high', 'low', 'close', 'volume',
          'close_time', 'quote_volume', 'trades', 'taker_base', 'taker_quote', 'ignore']


def sha256(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def duplicate_counts(frame, unit):
    """Near duplicates must share a UTC minute, never just a repeated price.

    Exact = all 12 original fields identical. Near = different original row,
    same floored open minute, times <= 1 second apart, equal trade count, and
    all OHLC/volume numeric fields rtol=1e-6, atol=1e-8. Counts are extra rows
    relative to the first row in that minute. Conflicting same-minute rows
    remain separately reported and are not silently resolved.
    """
    stamp = pd.DatetimeIndex(pd.to_datetime(frame.open_time, unit=unit, utc=True))
    exact = int(frame.duplicated().sum())
    keys = stamp.floor('min')
    repeats = keys.duplicated(keep=False)
    near = 0
    if repeats.any():
        candidates = frame.loc[repeats].drop_duplicates().copy()
        candidates['_key'] = keys[repeats][~frame.loc[repeats].duplicated()]
        for _, group in candidates.groupby('_key'):
            reference = group.iloc[0]
            for _, row in group.iloc[1:].iterrows():
                value_fields = ['open', 'high', 'low', 'close', 'volume',
                                'quote_volume', 'taker_base', 'taker_quote']
                same_values = np.allclose(row[value_fields].to_numpy(float),
                    reference[value_fields].to_numpy(float), rtol=1e-6, atol=1e-8)
                factor = 1000 if unit == 'ms' else 1000000
                same_times = all(abs(row[k]-reference[k]) <= factor
                                 for k in ['open_time', 'close_time'])
                near += int(same_values and same_times and row.trades == reference.trades)
    repeated_keys = int(keys.duplicated().sum())
    return dict(exact_duplicate_extra_rows=exact, near_duplicate_extra_rows=near,
                duplicate_minute_extra_rows=repeated_keys,
                conflicting_minute_extra_rows=max(0, repeated_keys-exact-near))


def effective_size(values, max_lag):
    """ACF-based descriptive ESS of a mean, retaining calendar gaps.

    Stop at first nonpositive ACF or max_lag; assume weak stationarity.
    Not a bound or count of independent training examples.
    """
    x = np.asarray(values, float)
    finite = np.isfinite(x)
    n = int(finite.sum())
    if n < 3:
        return dict(n=n, n_effective=np.nan, integrated_time=np.nan, used_lags=0)
    centered = x - np.nanmean(x)
    variance = np.nanmean(centered*centered)
    if variance <= 1e-20:
        return dict(n=n, n_effective=np.nan, integrated_time=np.nan, used_lags=0)
    positive = []
    for lag in range(1, min(max_lag, len(x)-1)+1):
        valid = finite[lag:] & finite[:-lag]
        if valid.sum() < 3:
            break
        rho = float(np.mean(centered[lag:][valid]*centered[:-lag][valid])/variance)
        if rho <= 0:
            break
        positive.append(rho)
    tau = max(1., 1+2*sum(positive))
    return dict(n=n, n_effective=float(np.clip(n/tau, 1, n)),
                integrated_time=tau, used_lags=len(positive))


def robust_limits(values, multiplier=6.):
    finite = np.asarray(values, float)
    finite = finite[np.isfinite(finite)]
    center = float(np.median(finite))
    scale = float(1.4826*np.median(np.abs(finite-center)))
    if scale <= 0:
        raise ValueError('Degenerate MAD: cannot apply the declared outlier rule')
    return center, scale, center-multiplier*scale, center+multiplier*scale


def audit_sources():
    sources = pd.read_csv(ROOT/'results/minute_2023_2025/sources.csv')
    if len(sources) != 144 or sources.duplicated(['symbol', 'month']).any():
        raise ValueError('Expected 144 unique source files')
    records, exclusions, missing, field_counts, totals = [], [], [], [], []
    monthly_missing = pd.DataFrame(0, index=pd.period_range('2023-01', '2025-12', freq='M').astype(str), columns=SYMBOLS)
    for symbol in SYMBOLS:
        reconstructed = np.full(len(MINUTES), np.nan)
        physical_presence = np.zeros(len(MINUTES), dtype=bool)
        for source in sources[sources.symbol.eq(symbol)].itertuples():
            path = ROOT/Path(source.source.replace('\\', '/'))
            actual_hash = sha256(path)
            checksum = path.with_suffix('.zip.CHECKSUM').read_text().split()[0]
            if actual_hash != source.sha256 or actual_hash != checksum:
                raise ValueError(f'Source checksum failed: {path}')
            with zipfile.ZipFile(path) as archive:
                if len(archive.namelist()) != 1:
                    raise ValueError(f'Unexpected zip members: {path}')
                with archive.open(archive.namelist()[0]) as stream:
                    frame = pd.read_csv(stream, header=None, names=FIELDS)
            numeric = frame.to_numpy(dtype=float)
            unit = 'us' if frame.open_time.iloc[0] > 10**14 else 'ms'
            stamp = pd.DatetimeIndex(pd.to_datetime(frame.open_time, unit=unit, utc=True))
            closing = pd.DatetimeIndex(pd.to_datetime(frame.close_time, unit=unit, utc=True))
            month_start = pd.Timestamp(source.month+'-01', tz='UTC')
            month_end = month_start+pd.offsets.MonthBegin(1)
            in_month = np.asarray((stamp >= month_start) & (stamp < month_end))
            aligned = np.asarray(stamp == stamp.floor('min'))
            valid_close_time = np.asarray(closing == stamp+pd.Timedelta(minutes=1)-pd.Timedelta(1, unit=unit))
            valid_price = np.isfinite(frame.close.to_numpy()) & (frame.close.to_numpy() > 0)
            dups = duplicate_counts(frame, unit)
            price = frame[['open', 'high', 'low', 'close']].to_numpy(float)
            volume = frame[['volume', 'quote_volume', 'taker_base', 'taker_quote']].to_numpy(float)
            bad_ohlc = ((frame.low > frame.high) | (frame.open < frame.low) |
                        (frame.open > frame.high) | (frame.close < frame.low) | (frame.close > frame.high)).to_numpy()
            bad_taker = ((frame.taker_base > frame.volume+1e-8) |
                         (frame.taker_quote > frame.quote_volume+1e-8)).to_numpy()
            bad_trade = (~np.isfinite(frame.trades) | (frame.trades < 0) | (frame.trades % 1 != 0)).to_numpy()
            valid = valid_price & valid_close_time & aligned & in_month & ~stamp.duplicated()
            record = dict(symbol=symbol, month=source.month, source=path.relative_to(ROOT).as_posix(),
                sha256=actual_hash, checksum_matches=True, sources_hash_matches=True,
                timestamp_unit=unit, rows_before=len(frame), rows_after=int(valid.sum()),
                excluded_rows=int((~valid).sum()), **dups,
                nonfinite_numeric_cells=int((~np.isfinite(numeric)).sum()),
                nonpositive_or_nonfinite_close_rows=int((~valid_price).sum()),
                nonpositive_or_nonfinite_ohlc_cells=int(((price <= 0) | ~np.isfinite(price)).sum()),
                negative_or_nonfinite_volume_cells=int(((volume < 0) | ~np.isfinite(volume)).sum()),
                inconsistent_ohlc_rows=int(bad_ohlc.sum()), taker_exceeds_total_rows=int(bad_taker.sum()),
                invalid_trade_count_rows=int(bad_trade.sum()), nonaligned_open_rows=int((~aligned).sum()),
                outside_month_rows=int((~in_month).sum()), unordered_open_transitions=int(np.sum(np.diff(stamp.asi8) < 0)),
                invalid_close_time_rows=int((~valid_close_time).sum()))
            records.append(record)
            for j, name in enumerate(FIELDS):
                field_counts.append(dict(symbol=symbol, month=source.month, field=name,
                    raw_nulls=int(frame[name].isna().sum()), nonfinite=int((~np.isfinite(numeric[:, j])).sum())))
            position = np.asarray((stamp-START)//pd.Timedelta(minutes=1), dtype=int)
            physical_presence[position[in_month & aligned]] = True
            reconstructed[position[valid]] = frame.close.to_numpy()[valid]
            for rowno in np.flatnonzero(~valid):
                exclusions.append(dict(symbol=symbol, month=source.month, csv_row_1based=int(rowno+1),
                    open_utc=str(stamp[rowno]), close_utc=str(closing[rowno]),
                    reason='invalid_close_time' if not valid_close_time[rowno] else 'invalid_key_or_price'))
            if record['nonpositive_or_nonfinite_close_rows'] or record['duplicate_minute_extra_rows'] or record['outside_month_rows'] or record['nonaligned_open_rows']:
                raise ValueError(f'Unresolved raw key/price problem: {record}')
        current = np.load(DATA/f'{symbol}.npy', mmap_mode='r')
        np.testing.assert_array_equal(reconstructed, current)
        day_complete = np.isfinite(reconstructed.reshape(-1, 1440)).all(axis=1)
        raw_gap = ~physical_presence
        nulls = ~np.isfinite(current)
        for group in np.split(np.flatnonzero(nulls), np.flatnonzero(np.diff(np.flatnonzero(nulls)) > 1)+1):
            if not len(group):
                continue
            missing.append(dict(symbol=symbol, start_utc=str(MINUTES[group[0]]),
                end_utc=str(MINUTES[group[-1]]), minutes=len(group),
                raw_absent_minutes=int(raw_gap[group].sum()),
                present_but_excluded_minutes=int((~raw_gap[group]).sum())))
        month_keys = MINUTES[nulls].strftime('%Y-%m')
        for month, count in pd.Series(month_keys).value_counts().items():
            monthly_missing.loc[month, symbol] = int(count)
        totals.append(dict(symbol=symbol, expected_minutes=len(current),
            raw_present_minutes=int(physical_presence.sum()), raw_missing_minutes=int(raw_gap.sum()),
            excluded_present_minutes=int((physical_presence & nulls).sum()), valid_minutes=int(np.isfinite(current).sum()),
            processed_missing_minutes=int(nulls.sum()), expected_days=len(DAYS), complete_days=int(day_complete.sum()),
            incomplete_days=int((~day_complete).sum()), reconstructed_matches_processed=True))
        print(f'{symbol}: 36 verified raw archives; {np.isfinite(current).sum():,} valid closes', flush=True)
    pd.DataFrame(records).to_csv(OUT/'raw_file_audit.csv', index=False)
    pd.DataFrame(exclusions).to_csv(OUT/'excluded_raw_rows.csv', index=False)
    pd.DataFrame(field_counts).to_csv(OUT/'raw_field_missingness.csv', index=False)
    pd.DataFrame(missing).to_csv(OUT/'missing_spans.csv', index=False)
    totals = pd.DataFrame(totals)
    totals.to_csv(OUT/'quality_by_asset.csv', index=False)
    monthly_missing.index.name = 'month'
    monthly_missing.to_csv(OUT/'minute_missingness_matrix.csv')
    fig, ax = plt.subplots(figsize=(9, 5), constrained_layout=True)
    picture = ax.imshow(monthly_missing.T.to_numpy(), aspect='auto', cmap='YlOrRd', vmin=0, vmax=81)
    ax.set_yticks(range(4), SYMBOLS)
    ax.set_xticks(range(0, 36, 3), monthly_missing.index[::3], rotation=45)
    ax.set_title('Minutos faltantes por mes y activo (calendario UTC)')
    ax.set_xlabel('Todos los huecos: marzo de 2023, 81 minutos por activo')
    fig.colorbar(picture, ax=ax, label='Minutos sin cierre válido')
    fig.savefig(FIG/'current_quality_missingness.png', dpi=160)
    plt.close(fig)
    return totals


def audit_daily():
    panel = pd.read_csv(DATA/'daily_target_closes.csv', index_col=0)
    panel.index = pd.to_datetime(panel.index, utc=True)
    np.testing.assert_array_equal(panel.index, DAYS)
    for symbol in SYMBOLS:
        values = np.load(DATA/f'{symbol}.npy').reshape(-1, 1440)
        expected = np.where(np.isfinite(values).all(axis=1), values[:, -1], np.nan)
        np.testing.assert_allclose(panel[symbol], expected, equal_nan=True, rtol=1e-12)
    derived = pd.DataFrame(index=DAYS)
    ranges, ess, outlier, sensitivity, origins = [], [], [], [], []
    development = DAYS < pd.Timestamp('2025-01-01', tz='UTC')
    initial_history = DAYS < pd.Timestamp('2024-01-01', tz='UTC')
    for symbol in SYMBOLS:
        returns = 100*np.log(panel[symbol]).diff()
        minute_values = np.load(DATA/f'{symbol}.npy')
        minute_ret = 100*np.diff(np.log(minute_values), prepend=np.nan).reshape(-1, 1440)
        derived[f'{symbol}:close'] = panel[symbol]
        derived[f'{symbol}:daily_return'] = returns
        for label, value in [('minute_rv', np.sqrt(np.sum(minute_ret**2, axis=1))),
                             ('minute_absolute', np.sum(np.abs(minute_ret), axis=1)/np.sqrt(1440)),
                             ('minute_negative', np.sqrt(np.sum(np.minimum(minute_ret, 0)**2, axis=1))),
                             ('minute_max_absolute', np.max(np.abs(minute_ret), axis=1))]:
            derived[f'{symbol}:{label}'] = value
        center, scale, lower, upper = robust_limits(returns[initial_history])
        flagged = (returns < lower) | (returns > upper)
        clipped = returns.clip(lower, upper)
        for period, mask in [('training_history_2023', initial_history),
                             ('validation_2024', development & ~initial_history)]:
            original = returns[mask].dropna()
            changed = clipped[mask].dropna()
            outlier.append(dict(symbol=symbol, frequency='daily', period=period, n=len(original), center_2023=center,
                mad_sigma_2023=scale, lower_limit_2023=lower, upper_limit_2023=upper,
                multiplier=6., flagged_count=int(flagged[mask].sum()),
                flagged_percentage=float(100*flagged[mask].sum()/len(original))))
            sensitivity.append(dict(symbol=symbol, period=period, statistic='daily_return_std', window=0,
                original=float(original.std(ddof=0)), winsorized=float(changed.std(ddof=0)),
                percent_change=float(100*(changed.std(ddof=0)/original.std(ddof=0)-1))))
        # Raw-minute returns use their own 2023 scale, rather than daily limits.
        minute_series = minute_ret.reshape(-1)
        minute_training = MINUTES < pd.Timestamp('2024-01-01', tz='UTC')
        minute_validation = (MINUTES >= pd.Timestamp('2024-01-01', tz='UTC')) & (MINUTES < pd.Timestamp('2025-01-01', tz='UTC'))
        mcenter, mscale, mlower, mupper = robust_limits(minute_series[minute_training])
        for period, mask in [('training_history_2023', minute_training), ('validation_2024', minute_validation)]:
            sample = minute_series[mask]
            finite = np.isfinite(sample)
            flags = (sample < mlower) | (sample > mupper)
            outlier.append(dict(symbol=symbol, frequency='minute', period=period,
                n=int(finite.sum()), center_2023=mcenter, mad_sigma_2023=mscale,
                lower_limit_2023=mlower, upper_limit_2023=mupper, multiplier=6.,
                flagged_count=int(flags.sum()), flagged_percentage=float(100*flags.sum()/finite.sum())))
        clipped_minute = np.clip(minute_ret, mlower, mupper)
        original_rv = np.sqrt(np.sum(minute_ret**2, axis=1))
        changed_rv = np.sqrt(np.sum(clipped_minute**2, axis=1))
        for period, mask in [('training_history_2023', initial_history), ('validation_2024', development & ~initial_history)]:
            original_mean = float(np.nanmean(original_rv[mask]))
            changed_mean = float(np.nanmean(changed_rv[mask]))
            sensitivity.append(dict(symbol=symbol, period=period, statistic='mean_minute_realized_variation', window=0,
                original=original_mean, winsorized=changed_mean,
                percent_change=float(100*(changed_mean/original_mean-1))))
        for date in returns.index[flagged & development]:
            origins.append(dict(symbol=symbol, date=str(date), daily_return=float(returns.loc[date]),
                lower_limit_2023=lower, upper_limit_2023=upper))
        for window in [7, 14, 21, 28]:
            target = returns.rolling(window, min_periods=window).std(ddof=0)
            derived[f'{symbol}:target_w{window}'] = target
            transformed = clipped.rolling(window, min_periods=window).std(ddof=0)
            for period, mask in [('all_2023_2025_descriptive', np.ones(len(DAYS), dtype=bool)),
                                 ('development_2023_2024', development),
                                 ('retrospective_2025', ~development)]:
                sample = target[mask].dropna()
                ranges.append(dict(symbol=symbol, window=window, period=period, n=len(sample),
                    first_target_date=str(sample.index.min()), last_target_date=str(sample.index.max()),
                    minimum=float(sample.min()), maximum=float(sample.max()),
                    min_date=str(sample.idxmin()), max_date=str(sample.idxmax()),
                    zero_targets=int(sample.eq(0).sum())))
            # ESS is intentionally development-only and preserves missing days.
            for cap in [30, 60, 90]:
                ess.append(dict(symbol=symbol, window=window, period='development_2023_2024',
                    maximum_lag=cap, **effective_size(target[development], cap)))
            for period, mask in [('training_history_2023', initial_history),
                                 ('validation_2024', development & ~initial_history)]:
                original = target[mask].dropna()
                changed = transformed[mask].dropna()
                sensitivity.append(dict(symbol=symbol, period=period, statistic='mean_rolling_volatility',
                    window=window, original=float(original.mean()), winsorized=float(changed.mean()),
                    percent_change=float(100*(changed.mean()/original.mean()-1))))
    null_matrix = derived.isna().astype(int)
    null_matrix.index.name = 'date'
    null_matrix.to_csv(OUT/'daily_missingness_matrix.csv')
    nulls = pd.DataFrame(dict(variable=derived.columns, total_rows=len(derived),
        missing_rows=derived.isna().sum().to_numpy(), finite_rows=np.isfinite(derived).sum().to_numpy()))
    nulls.to_csv(OUT/'derived_missingness_counts.csv', index=False)
    pd.DataFrame(ranges).to_csv(OUT/'target_ranges.csv', index=False)
    pd.DataFrame(ess).to_csv(OUT/'effective_sample_size.csv', index=False)
    pd.DataFrame(outlier).to_csv(OUT/'outlier_counts.csv', index=False)
    pd.DataFrame(origins).to_csv(OUT/'outlier_dates.csv', index=False)
    pd.DataFrame(sensitivity).to_csv(OUT/'outlier_sensitivity.csv', index=False)
    fig, axes = plt.subplots(2, 1, figsize=(12, 7), constrained_layout=True)
    axes[0].imshow(null_matrix.T.to_numpy(), aspect='auto', cmap='Greys', vmin=0, vmax=1,
                   interpolation='nearest')
    axes[0].set_title('Matriz de ausencia de cierres, características y objetivos diarios')
    axes[0].set_yticks(np.arange(0, len(derived.columns), 10), SYMBOLS)
    axes[0].set_xticks([0, 365, 731], ['2023-01-01', '2024-01-01', '2025-01-01'])
    focus = (DAYS >= pd.Timestamp('2023-03-15', tz='UTC')) & (DAYS <= pd.Timestamp('2023-04-30', tz='UTC'))
    subset = null_matrix.loc[focus, [k for k in null_matrix.columns if k.startswith('BTCUSDT:')]]
    axes[1].imshow(subset.T.to_numpy(), aspect='auto', cmap='Greys', vmin=0, vmax=1,
                   interpolation='nearest')
    axes[1].set_title('Detalle BTC; las cuatro monedas tienen el mismo patrón de ausencia')
    axes[1].set_yticks(range(len(subset.columns)), [k.split(':')[1] for k in subset.columns], fontsize=8)
    ticks = np.arange(0, len(subset), 5)
    axes[1].set_xticks(ticks, subset.index[ticks].strftime('%m-%d'), rotation=45)
    axes[1].set_xlabel('Negro = faltante; blanco = disponible. UTC')
    fig.savefig(FIG/'current_quality_daily_missingness.png', dpi=160)
    plt.close(fig)
    sensitivity_frame = pd.DataFrame(sensitivity)
    fig, ax = plt.subplots(figsize=(9, 5), constrained_layout=True)
    sample = sensitivity_frame.query("period == 'validation_2024' and statistic == 'mean_rolling_volatility'")
    for symbol, group in sample.groupby('symbol', sort=False):
        ax.plot(group.window, group.percent_change, marker='o', label=symbol)
    ax.axhline(0, color='black', linewidth=.8)
    ax.set_xticks([7, 14, 21, 28])
    ax.set_xlabel('Ventana del objetivo (días)')
    ax.set_ylabel('Cambio en volatilidad media de 2024 (%)')
    ax.set_title('Sensibilidad descriptiva: winsorización con límites MAD fijados en 2023')
    ax.legend()
    fig.savefig(FIG/'current_quality_outlier_sensitivity.png', dpi=160)
    plt.close(fig)
    return derived


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(exist_ok=True)
    manifest = json.loads((ROOT/'results/minute_2023_2025/data_manifest.json').read_text())
    hashes = {name: sha256(DATA/name) for name in manifest}
    if hashes != manifest:
        raise ValueError('Processed data manifest mismatch')
    totals = audit_sources()
    audit_daily()
    metadata = dict(audit='2023--2025 delivery quality; no mutations of inputs or model artifacts',
        period_start=str(START), period_end_exclusive=str(END), symbols=SYMBOLS,
        source_files_verified=144, processed_hashes=hashes,
        source_manifest_sha256=sha256(ROOT/'results/minute_2023_2025/sources.csv'),
        script_sha256=sha256(Path(__file__)),
        rules=dict(near_duplicate='Same asset/minute; all OHLC and volumes rtol=1e-6 atol=1e-8; equal trades; times within 1s; exclude exact rows',
                   outlier='Minute and daily returns, each with its own median +/- 6*1.4826*MAD; thresholds from 2023 only; descriptive counts and sensitivity in 2023/2024 only',
                   sensitivity='Winsorize temporary copies of returns; compare daily std/mean rolling volatility and mean intraday realized variation; no model fit or target replacement',
                   effective_n='n/(1+2 sum positive ACF until first nonpositive lag or cap 30/60/90); calendar gaps retained; development only; weak-stationarity approximation for mean, not independent-training count'),
        missingness_mechanism='MCAR/MAR/MNAR not identifiable from this synchronized single interruption; operational pattern does not prove a stochastic missingness mechanism',
        guarantees=dict(raw_reconstruction_matches_processed=True, original_data_unchanged=True,
                        no_2025_outlier_tuning=True), totals=totals.to_dict('records'))
    (OUT/'methodology.json').write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding='utf-8')
    artifact_hashes = {p.name: sha256(p) for p in sorted(OUT.glob('*')) if p.is_file() and p.name != 'artifact_hashes.json'}
    (OUT/'artifact_hashes.json').write_text(json.dumps(artifact_hashes, indent=2), encoding='utf-8')
    print(json.dumps(dict(source_files=144, processed_manifest_matches=True, output=str(OUT)), ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
