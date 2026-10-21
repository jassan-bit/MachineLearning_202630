"""Development-only MI, VIF and intraday diagnostics for the published inputs.

This is a retrospective audit, not a feature selector or model experiment.
Only the 2023--2024 prefix of each memory-mapped minute array is accessed.
Run: .venv-repro/Scripts/python.exe src/professor_feedback_diagnostics.py
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import platform
import time
import warnings

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
import sklearn
from sklearn.feature_selection import mutual_info_regression
from threadpoolctl import threadpool_limits

from improve_classical_forecast import har_features
from optimize_minute_svr import features, minute_features

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'data/processed/minute_2023_2025'
OUT = ROOT / 'results/professor_feedback_20261018/diagnostics'
SYMBOLS = ['BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'XRPUSDT']
START = pd.Timestamp('2023-01-01', tz='UTC')
END = pd.Timestamp('2025-01-01', tz='UTC')
TRAIN_END = pd.Timestamp('2024-01-01', tz='UTC')
SPANS = [1, 3, 7, 14, 28]
WINDOWS = [7, 14, 21, 28]
DAILY_NAMES = ['daily_return_over_base', 'daily_return_squared_over_base_squared',
               'minute_realized_over_base', 'minute_absolute_over_base',
               'minute_downside_over_base', 'minute_max_abs_over_base']


def har_names():
    """Exactly the 44 columns emitted by improve_classical_forecast.har_features."""
    fields = ['daily_return_mean', 'daily_return_rms', 'daily_downside_rms',
              'minute_realized_mean', 'minute_absolute_mean',
              'minute_downside_mean', 'minute_max_abs_mean']
    return ([f'{field}_{span}d_over_base' for span in SPANS for field in fields]
            + ['log_current_volatility', 'daily_squared_return_std_28d_over_base_squared']
            + [f'known_decay_h{h}' for h in range(1, 8)])


def development_prefix(path, days):
    """Read/hash the permitted prefix, never the held-out values in the file."""
    mapped = np.load(path, mmap_mode='r', allow_pickle=False)
    count = days * 1440
    if mapped.ndim != 1 or len(mapped) < count:
        raise ValueError(f'Unexpected source shape: {path}')
    values = np.array(mapped[:count], dtype=np.float64, copy=True)
    if np.any(np.isfinite(values) & (values <= 0)):
        raise ValueError('Minute prices must be positive or missing')
    return values, dict(path=str(path.relative_to(ROOT)), accessed_values=count,
                       accessed_start_utc=str(START), accessed_end_exclusive_utc=str(END),
                       development_float64_bytes_sha256=hashlib.sha256(values.tobytes()).hexdigest(),
                       held_out_values_accessed=False)


def daily_closes(values, index):
    """A daily close is valid only if every minute of its calendar day exists."""
    blocks = np.asarray(values).reshape(len(index), 1440)
    complete = np.isfinite(blocks).all(axis=1)
    return pd.Series(np.where(complete, blocks[:, -1], np.nan), index=index)


def shared_eligible(panel):
    """Same conservative 28-day history + 7 labels calendar as existing models."""
    finite = np.isfinite(panel.to_numpy()).all(axis=1)
    return pd.Series(finite).rolling(36).sum().shift(-7).eq(36).to_numpy()


def nonconstant_columns(X):
    X = np.asarray(X, dtype=float)
    if X.ndim != 2 or not np.isfinite(X).all():
        raise ValueError('A finite two-dimensional design is required')
    return np.ptp(X, axis=0) > 1e-12 * np.maximum(1., np.max(np.abs(X), axis=0))


def vif_diagnostics(X, names):
    """OLS auxiliary regressions with intercept; constants are explicitly undefined.

    Centering/standardizing is local to this diagnostic sample. A pseudoinverse
    of the correlation matrix is deliberately not used to hide dependencies.
    """
    X = np.asarray(X, dtype=float)
    if len(names) != X.shape[1] or len(X) < 3:
        raise ValueError('Feature labels or sample size do not match the design')
    usable = nonconstant_columns(X)
    Z = (X[:, usable] - X[:, usable].mean(axis=0)) / X[:, usable].std(axis=0, ddof=0)
    singular = np.linalg.svd(Z, compute_uv=False)
    tolerance = max(Z.shape) * np.finfo(float).eps * singular[0] if len(singular) else 0.
    rank = int(np.sum(singular > tolerance))
    summary = dict(n_rows=len(X), n_columns=X.shape[1], n_nonconstant=int(usable.sum()),
                   n_constant=int((~usable).sum()), centered_rank=rank,
                   condition_number=float(singular[0] / singular[-1]) if rank == Z.shape[1] and len(singular) else np.inf,
                   svd_tolerance=tolerance)
    rows = []
    position = 0
    for j, name in enumerate(names):
        if not usable[j]:
            rows.append(dict(feature=name, vif=np.nan, auxiliary_r2=np.nan,
                             status='constant_undefined', high_vif_over_10=False))
            continue
        response = Z[:, position]
        predictors = np.delete(Z, position, axis=1)
        coefficients = np.linalg.lstsq(predictors, response, rcond=None)[0]
        residual = response - predictors @ coefficients
        unexplained = float(residual @ residual / (response @ response))
        dependent = unexplained <= 1e-12
        vif = np.inf if dependent else 1. / unexplained
        rows.append(dict(feature=name, vif=vif, auxiliary_r2=1. - unexplained,
                         status='linear_dependency' if dependent else 'finite',
                         high_vif_over_10=bool(vif > 10)))
        position += 1
    return rows, summary


def mutual_information(X, targets, names, neighbors=5):
    """Marginal continuous kNN MI in nats; no significance or selection claims."""
    X, targets = np.asarray(X, float), np.asarray(targets, float)
    if targets.ndim == 1:
        targets = targets[:, None]
    usable = nonconstant_columns(X)
    if len(targets) != len(X) or not np.isfinite(targets).all():
        raise ValueError('Targets must be finite and aligned')
    Z = (X[:, usable] - X[:, usable].mean(axis=0)) / X[:, usable].std(axis=0, ddof=0)
    rows = []
    for h in range(targets.shape[1]):
        response = targets[:, h]
        scores = np.zeros(X.shape[1])
        constant_y = np.ptp(response) <= 1e-12 * max(1., np.max(np.abs(response)))
        if not constant_y and usable.any():
            standardized_y = (response - response.mean()) / response.std(ddof=0)
            scores[usable] = mutual_info_regression(Z, standardized_y,
                discrete_features=False, n_neighbors=neighbors, random_state=42)
        for j, name in enumerate(names):
            rows.append(dict(feature=name, horizon=h+1, mi_nats=float(scores[j]),
                             n_neighbors=neighbors, n_rows=len(X),
                             constant_feature=bool(not usable[j]), constant_target=constant_y))
    return rows


def calendar_block_indices(length, repeats=499, block=7, seed=42):
    """Circular blocks of original calendar positions, including missing dates."""
    if length < 1 or block < 1:
        raise ValueError('Positive calendar length/block are required')
    rng = np.random.default_rng(seed)
    starts = rng.integers(0, length, size=(repeats, (length+block-1)//block))
    return ((starts[:, :, None] + np.arange(block)) % length).reshape(repeats, -1)[:, :length]


def intraday_profiles(values, index, repeats=499):
    """UTC profiles retain missing slots and normalize each COMPLETE day's r^2.

    Minute r_t is the change from the previous close, assigned to the UTC
    opening slot of its terminal one-minute candle. No return crosses a gap.
    """
    returns = (100 * np.diff(np.log(values), prepend=np.nan)).reshape(len(index), 1440)
    finite = np.isfinite(returns)
    complete = finite.all(axis=1)
    squared = returns * returns
    day_average = np.mean(squared, axis=1)
    normalizable = complete & (day_average > 0)
    normalized = np.full_like(squared, np.nan)
    normalized[normalizable] = squared[normalizable] / day_average[normalizable, None]
    rows, hours, summaries = [], [], []
    for period, mask in [('development_2023_2024', np.ones(len(index), dtype=bool)),
                         ('train_2023', index < TRAIN_END), ('validation_2024', index >= TRAIN_END)]:
        r = returns[mask]
        a = np.abs(r)
        n = normalized[mask]
        count = np.isfinite(r).sum(axis=0)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', RuntimeWarning)
            abs_mean, abs_median = np.nanmean(a, axis=0), np.nanmedian(a, axis=0)
            r2_mean, normalized_mean = np.nanmean(r*r, axis=0), np.nanmean(n, axis=0)
        for minute in range(1440):
            rows.append(dict(period=period, minute_of_day_utc=minute, hour_utc=minute//60,
                n_finite_returns=int(count[minute]), n_complete_days=int(np.isfinite(n[:, minute]).sum()),
                mean_abs_return_pct=abs_mean[minute], median_abs_return_pct=abs_median[minute],
                mean_squared_return_pct_squared=r2_mean[minute],
                mean_daily_normalized_squared_return=normalized_mean[minute]))
        # Incomplete days remain nan rows at their original calendar positions.
        daily_hour = n.reshape(len(n), 24, 60).mean(axis=2)
        positions = calendar_block_indices(len(n), repeats=repeats)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', RuntimeWarning)
            boot = np.nanmean(daily_hour[positions], axis=1)
            low, high = np.nanquantile(boot, [.025, .975], axis=0)
            hourly = np.nanmean(daily_hour, axis=0)
        for hour in range(24):
            hours.append(dict(period=period, hour_utc=hour,
                mean_daily_normalized_squared_return=hourly[hour],
                ci95_calendar_block_lower=low[hour], ci95_calendar_block_upper=high[hour],
                n_calendar_days=len(n), n_complete_days=int(np.isfinite(daily_hour[:, hour]).sum()),
                bootstrap_repeats=repeats, calendar_block_days=7))
        summaries.append(dict(period=period, n_calendar_days=int(mask.sum()),
            n_expected_minute_returns=int(r.size), n_finite_minute_returns=int(np.isfinite(r).sum()),
            n_complete_return_days=int(complete[mask].sum()), n_normalizable_days=int(normalizable[mask].sum()),
            hourly_peak_utc=int(np.nanargmax(hourly)), hourly_trough_utc=int(np.nanargmin(hourly)),
            peak_normalized_squared_return=float(np.nanmax(hourly)),
            trough_normalized_squared_return=float(np.nanmin(hourly)),
            hourly_peak_to_trough_ratio=float(np.nanmax(hourly)/np.nanmin(hourly))))
    return rows, hours, summaries


def write_frame(rows, path):
    frame = rows if isinstance(rows, pd.DataFrame) else pd.DataFrame(rows)
    frame.to_csv(path, index=False, float_format='%.12g')
    return frame


def plot_profiles(hours, path):
    fig, axes = plt.subplots(2, 2, figsize=(12, 7), sharex=True, sharey=True)
    colors = {'development_2023_2024': '#17324D', 'train_2023': '#087F8C', 'validation_2024': '#BD811F'}
    labels = {'development_2023_2024': '2023–2024', 'train_2023': '2023', 'validation_2024': '2024'}
    for ax, symbol in zip(axes.flat, SYMBOLS):
        for period, color in colors.items():
            subset = hours.loc[(hours.symbol == symbol) & (hours.period == period)]
            x = subset.hour_utc.to_numpy()
            ax.plot(x, subset.mean_daily_normalized_squared_return, color=color, label=labels[period])
            if period == 'development_2023_2024':
                ax.fill_between(x, subset.ci95_calendar_block_lower.to_numpy(),
                                subset.ci95_calendar_block_upper.to_numpy(), color=color, alpha=.13)
        ax.axhline(1., color='gray', linestyle='--', linewidth=.8)
        ax.set(title=symbol, xlabel='Hora UTC', ylabel='r² / media diaria de r²', xticks=range(0, 24, 3))
        ax.grid(alpha=.15)
    axes[0, 0].legend(fontsize=8)
    fig.suptitle('Perfil intradía de desarrollo: días completos, sin interpolar\nIC individuales por bloques de siete días de calendario')
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def prepare_output_directory(output):
    """Keep the audit local and never replace an existing experiment's results."""
    output = Path(output).resolve()
    result_root = (ROOT / 'results').resolve()
    if not output.is_relative_to(result_root) or output == result_root:
        raise ValueError('Diagnostic output must be a new directory within this workspace/results')
    if (output / 'metadata.json').exists() or (output.exists() and any(output.iterdir())):
        raise FileExistsError(f'Refusing to overwrite existing diagnostic output: {output}')
    output.mkdir(parents=True, exist_ok=True)
    return output


def run(output=OUT, bootstrap_repeats=499):
    started = time.monotonic()
    if bootstrap_repeats < 1:
        raise ValueError('At least one bootstrap replicate is required')
    output = prepare_output_directory(output)
    index = pd.date_range(START, END, freq='D', inclusive='left')
    panel, minutes, manifests, intraday, hourly, seasonality = {}, {}, [], [], [], []
    for symbol in SYMBOLS:
        values, manifest = development_prefix(DATA / f'{symbol}.npy', len(index))
        manifests.append(dict(symbol=symbol, **manifest))
        panel[symbol] = daily_closes(values, index)
        minutes[symbol] = minute_features(values)
        minute_rows, hour_rows, summaries = intraday_profiles(values, index, bootstrap_repeats)
        intraday.extend(dict(symbol=symbol, **row) for row in minute_rows)
        hourly.extend(dict(symbol=symbol, **row) for row in hour_rows)
        seasonality.extend(dict(symbol=symbol, **row) for row in summaries)
        print(f'{symbol}: minute profiles audited on development prefix only', flush=True)
    panel = pd.DataFrame(panel)
    eligible = shared_eligible(panel)
    periods = [('train_2023', eligible & (index + pd.Timedelta(days=7) < TRAIN_END)),
               ('development_2023_2024', eligible & (index + pd.Timedelta(days=7) < END))]
    mi, vif, designs, calendar = [], [], [], []
    for period, mask in periods:
        for position in np.flatnonzero(mask):
            calendar.append(dict(period=period, origin=str(index[position]),
                                 target_end=str(index[position]+pd.Timedelta(days=7))))
        for symbol in SYMBOLS:
            for window in WINDOWS:
                H, y, base = har_features(panel[symbol], minutes[symbol], window)
                daily, ycheck, basecheck = features(panel[symbol], minutes[symbol], 1, window)
                np.testing.assert_allclose(y, ycheck, equal_nan=True)
                np.testing.assert_allclose(base, basecheck, equal_nan=True)
                for family, X, names in [('HAR44', H, har_names()), ('daily6', daily[:, :6], DAILY_NAMES)]:
                    design = X[mask]
                    context = dict(period=period, symbol=symbol, volatility_window=window, feature_family=family)
                    rows, summary = vif_diagnostics(design, names)
                    vif.extend(dict(**context, n_rows=len(design), **row) for row in rows)
                    designs.append(dict(**context, **summary))
                    for target_kind, target in [('absolute_volatility', y[mask]),
                                                ('relative_correction', y[mask]/base[mask, None]-1.)]:
                        rows = mutual_information(design, target, names)
                        mi.extend(dict(**context, target_kind=target_kind, **row) for row in rows)
            print(f'{period}/{symbol}: MI and VIF completed', flush=True)
    frames = {}
    for name, rows in [('mutual_information', mi), ('variance_inflation', vif), ('design_audit', designs),
                       ('diagnostic_calendar', calendar), ('intraday_minute_profile', intraday),
                       ('intraday_hour_profile', hourly), ('intraday_summary', seasonality)]:
        frames[name] = write_frame(rows, output / f'{name}.csv')
    plot_profiles(frames['intraday_hour_profile'], output / 'intraday_development.png')
    source_paths = [Path(__file__), ROOT/'src/improve_classical_forecast.py', ROOT/'src/optimize_minute_svr.py',
                    ROOT/'src/minute_experiment.py', ROOT/'src/volatility_experiment.py']
    metadata = dict(status='complete', diagnostic_only=True, model_or_feature_selection_performed=False,
        test_values_read=False, held_out_test_restored=False, github_or_render_updates_authorized=False,
        development_start=str(START), development_end_exclusive=str(END),
        training_target_end_exclusive=str(TRAIN_END), feature_dimensions=dict(HAR44=44, daily6=6),
        targets='Seven absolute daily volatility labels and the actual fitted relative corrections y/base - 1',
        origin_rule='Shared finite 36-calendar-day blocks; 28 days of history through origin and all seven labels before the cutoff',
        counts={name:len(frame) for name,frame in frames.items()},
        input_prefixes=manifests,
        source_sha256={str(path.relative_to(ROOT)):hashlib.sha256(path.read_bytes()).hexdigest() for path in source_paths},
        output_sha256={path.name:hashlib.sha256(path.read_bytes()).hexdigest() for path in sorted(output.iterdir()) if path.suffix in ('.csv','.png')},
        methods=dict(mi='sklearn continuous nearest-neighbor mutual_info_regression; k=5; seed=42; nats; sample-local standardization; negative estimates clipped to zero by the estimator',
                     vif='Auxiliary centered OLS for every nonconstant column against all other nonconstant columns; constants undefined; SSE/SST <= 1e-12 reported as linear dependency',
                     intraday='Raw valid minute-return profiles and complete-day normalized squared returns; UTC terminal candle opening slots; no gap crossing or interpolation',
                     uncertainty='Hourly individual percentile intervals from 499 circular bootstrap replicates of seven original calendar days; seed=42; missing calendar positions preserved'),
        limitations=[
            'This diagnostic was designed after the previous model and inspected 2025 results; development-only recomputation cannot make historical EDA prospective.',
            'Marginal MI is a descriptive estimate, not conditional importance, a p-value, or proof of incremental prediction. Serial dependence and overlapping labels affect effective information.',
            'Positive MI can reflect the current volatility normalization, shared returns, persistence, or estimator bias; absolute and fitted relative targets are shown separately.',
            'VIF describes unpenalized linear redundancy. A threshold of 10 is an audit flag, not an instruction to remove predictors or a failure of Ridge.',
            'Intraday intervals are individual and descriptive, without simultaneous coverage or a formal test of stable seasonality; days excluded for gaps can change the represented sample.',
            '2023 and 2024 profiles are within development and do not establish stability in future years.',
            'No trading performance or model improvement is demonstrated by these diagnostics alone.'],
        versions=dict(python=platform.python_version(), numpy=np.__version__, pandas=pd.__version__,
                      scipy=scipy.__version__, sklearn=sklearn.__version__), seconds=time.monotonic()-started)
    metadata['methods']['uncertainty'] = metadata['methods']['uncertainty'].replace('499',str(bootstrap_repeats))
    (output/'metadata.json').write_text(json.dumps(metadata, indent=2, allow_nan=False), encoding='utf-8')
    print(json.dumps(dict(status='complete', counts=metadata['counts'], seconds=metadata['seconds'])), flush=True)
    return metadata


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=OUT)
    parser.add_argument('--bootstrap-repeats', type=int, default=499)
    arguments = parser.parse_args()
    with threadpool_limits(limits=1):
        run(arguments.output, arguments.bootstrap_repeats)
