"""Reproduce descriptive EDA of the CURRENT 2023--2025 minute dataset.

No model is fitted or changed. Associations use development labels ending
before 2025; PCA/scaling/distance thresholds use 2023 only. 2025 appears only
in descriptive distribution drift. Run with --check for deterministic checks.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import warnings
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp, kurtosis, skew
from sklearn.covariance import LedoitWolf
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.seasonal import STL
from statsmodels.tsa.stattools import adfuller, kpss, pacf
from threadpoolctl import threadpool_limits

from minute_experiment import DATA, ROOT, load_panel
from optimize_minute_svr import calendars, features, minute_features
from volatility_experiment import targets

OUT = ROOT / 'results/current_delivery_audit/eda'
FIG = ROOT / 'book/figures'
TRAIN_END = pd.Timestamp('2024-01-01', tz='UTC')
DEV_END = pd.Timestamp('2025-01-01', tz='UTC')
FAMILIES = ['daily_return', 'daily_return_squared', 'minute_realized',
            'minute_absolute', 'minute_downside', 'minute_max_abs']
SHORT = ['r/b', 'r²/b²', 'RV/b', 'abs/b', 'down/b', 'max/b']


def names(lag):
    return [f'{family}_lag{k}_normalized' for k in range(lag) for family in FAMILIES] + [
        f'known_decay_h{h}' for h in range(1, 8)]


def development_masks(index, eligible):
    """All seven target days, rather than just origin, obey each cutoff."""
    last_label = index + pd.Timedelta(days=7)
    return eligible & (last_label < TRAIN_END), eligible & (last_label < DEV_END)


def longest_segment(series):
    """Preserve daily calendar; never join observations across a missing day."""
    if not isinstance(series.index, pd.DatetimeIndex):
        raise ValueError('Daily DatetimeIndex required')
    if not (series.index.to_series().diff().dropna() == pd.Timedelta(days=1)).all():
        raise ValueError('Input must retain the daily calendar')
    good = np.isfinite(series.to_numpy(float))
    boundaries = np.flatnonzero(np.diff(np.r_[False, good, False]))
    if not len(boundaries):
        return series.iloc[:0]
    spans = boundaries.reshape(-1, 2)
    a, b = max(spans, key=lambda pair: pair[1] - pair[0])
    return series.iloc[a:b]


def describe(values):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    q = np.quantile(values, [.01, .05, .25, .5, .75, .95, .99])
    iqr = q[4] - q[2]
    constant = numerically_constant(values)
    return dict(n=len(values), mean=float(values.mean()), std=float(values.std(ddof=1)),
                min=float(values.min()), max=float(values.max()),
                **{f'p{p:02d}': float(v) for p, v in zip([1, 5, 25, 50, 75, 95, 99], q)},
                skewness=np.nan if constant else float(skew(values, bias=False)),
                excess_kurtosis=np.nan if constant else float(kurtosis(values, fisher=True, bias=False)),
                numerically_constant=constant,
                tukey_outliers=0 if constant else int(((values < q[2]-1.5*iqr) | (values > q[4]+1.5*iqr)).sum()))


def numerically_constant(values):
    """Ignore roundoff, e.g. the identically-zero decay at window=7/h=7."""
    values = np.asarray(values, dtype=float)
    finite = values[np.isfinite(values)]
    return bool(np.ptp(finite) <= 1e-12*max(1., np.max(np.abs(finite))))


def write_csv(rows, name):
    frame = rows if isinstance(rows, pd.DataFrame) else pd.DataFrame(rows)
    frame.to_csv(OUT / name, index=False, float_format='%.10g',
                 compression='gzip' if name.endswith('.gz') else None)
    return frame


def save_fig(fig, name):
    fig.savefig(FIG / f'current_eda_{name}.png', dpi=150, bbox_inches='tight')
    plt.close(fig)


def temporal_audit(symbol, series_map, stationarity, pacfs, decompositions,
                   weekday_rows, seasonality):
    for variable, whole in series_map.items():
        dev = whole.loc[whole.index < DEV_END]
        segment = longest_segment(dev)
        values = segment.to_numpy()
        regression = 'ct' if variable == 'log_close' else 'c'
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            adf = adfuller(values, regression=regression, autolag='AIC')
            kp = kpss(values, regression=regression, nlags='auto')
        stationarity.append(dict(symbol=symbol, variable=variable, n=len(values),
            first=str(segment.index[0]), last=str(segment.index[-1]), regression=regression,
            adf_stat=adf[0], adf_p=adf[1], adf_lags=adf[2], kpss_stat=kp[0],
            kpss_p=kp[1], kpss_lags=kp[2], warnings=' | '.join(str(w.message) for w in caught)))
        coef = pacf(values, nlags=30, method='ywm')
        for lag, value in enumerate(coef):
            pacfs.append(dict(symbol=symbol, variable=variable, lag=lag, pacf=value,
                              n=len(values), first=str(segment.index[0]), last=str(segment.index[-1])))
        if variable in ['daily_return', 'volatility_w14']:
            fit = STL(segment, period=7, robust=True).fit()
            denom = np.var(fit.resid + fit.seasonal, ddof=1)
            strength = max(0, 1-np.var(fit.resid, ddof=1)/denom) if denom else 0
            # Magnitude only: weekday groups are serially dependent.
            groups = dev.dropna().groupby(dev.dropna().index.dayofweek)
            mean = dev.mean()
            between = sum(len(g)*(g.mean()-mean)**2 for _, g in groups)
            total = ((dev.dropna()-mean)**2).sum()
            seasonality.append(dict(symbol=symbol, variable=variable, period=7,
                stl_seasonal_strength=strength, weekday_eta_squared=between/total,
                n_stl=len(segment), first=str(segment.index[0]), last=str(segment.index[-1])))
            for day, group in groups:
                weekday_rows.append(dict(symbol=symbol, variable=variable, weekday=int(day),
                    n=len(group), mean=group.mean(), median=group.median(), std=group.std()))
            for date, observed, trend, seasonal, resid in zip(segment.index, values,
                    fit.trend, fit.seasonal, fit.resid):
                decompositions.append(dict(symbol=symbol, variable=variable, date=str(date),
                    observed=observed, trend=trend, seasonal=seasonal, residual=resid))


def association_audit(symbol, window, lag, X, y, base, mask, rows, redundancy):
    feature_names = names(lag) + ['current_volatility']
    predictors = pd.DataFrame(np.column_stack([X[mask], base[mask]]), columns=feature_names)
    constants = {name: numerically_constant(predictors[name]) for name in feature_names}
    # Undefined association for constant columns, rather than correlating roundoff.
    predictors.loc[:, [name for name, constant in constants.items() if constant]] = np.nan
    responses = {'future_volatility': y[mask], 'relative_future_correction': y[mask]/base[mask, None]-1}
    for response, values in responses.items():
        target_names = [f'h{h}' for h in range(1, 8)]
        combined = pd.concat([predictors, pd.DataFrame(values, columns=target_names)], axis=1)
        for method in ['pearson', 'spearman']:
            matrix = combined.corr(method=method).loc[feature_names, target_names]
            for feature in feature_names:
                for h, name in enumerate(target_names, 1):
                    r = matrix.loc[feature, name]
                    rows.append(dict(symbol=symbol, volatility_window=window, input_window=lag,
                        predictor=feature, target=response, horizon=h, method=method,
                        n=int(mask.sum()), correlation=r,
                        status='numerically_constant' if constants[feature] else 'descriptive',
                        r_squared=r*r if method == 'pearson' else np.nan))
    for method in ['pearson', 'spearman']:
        matrix = predictors.iloc[:, :-1].corr(method=method).to_numpy()
        a, b = np.triu_indices(matrix.shape[0], 1)
        for i, j in zip(a, b):
            if np.isfinite(matrix[i, j]) and abs(matrix[i, j]) >= .9:
                redundancy.append(dict(symbol=symbol, volatility_window=window, input_window=lag,
                    method=method, feature_a=feature_names[i], feature_b=feature_names[j],
                    correlation=matrix[i, j]))


def multivariate_audit(symbol, window, lag, index, X, train, dev, summary,
                       components, distances, loading_rows, plot_data):
    scaler = StandardScaler().fit(X[train])
    Z_train = scaler.transform(X[train])
    Z_dev = scaler.transform(X[dev])
    pca = PCA(svd_solver='full').fit(Z_train)
    cumulative = np.cumsum(pca.explained_variance_ratio_)
    k95 = int(np.searchsorted(cumulative, .95)+1)
    scores_train = pca.transform(Z_train)
    scores_dev = pca.transform(Z_dev)
    covariance = LedoitWolf().fit(Z_train)
    md_train = covariance.mahalanobis(Z_train)
    md_dev = covariance.mahalanobis(Z_dev)
    threshold = float(np.quantile(md_train, .99))
    summary.append(dict(symbol=symbol, volatility_window=window, input_window=lag,
        n_features=X.shape[1], n_train=int(train.sum()), n_development=int(dev.sum()),
        fit_origin_first=str(index[train][0]), fit_origin_last=str(index[train][-1]),
        fit_label_last=str(index[train][-1]+pd.Timedelta(days=7)),
        components_90=int(np.searchsorted(cumulative, .9)+1), components_95=k95,
        pc1_variance=pca.explained_variance_ratio_[0],
        pc1_pc2_variance=cumulative[1], shrinkage=covariance.shrinkage_,
        distance_train_p99=threshold, development_above_p99=int((md_dev>threshold).sum()),
        validation_2024_above_p99=int(((md_dev>threshold)&(index[dev]>=TRAIN_END)).sum())))
    for pc, (variance, total) in enumerate(zip(pca.explained_variance_ratio_, cumulative), 1):
        components.append(dict(symbol=symbol, volatility_window=window, input_window=lag,
                               component=pc, variance_ratio=variance, cumulative_variance=total))
    for pc in [0, 1]:
        for feature, value in zip(names(lag), pca.components_[pc]):
            loading_rows.append(dict(symbol=symbol, volatility_window=window, input_window=lag,
                                     component=pc+1, feature=feature, loading=value))
    for row, origin, distance in zip(Z_dev, index[dev], md_dev):
        most = int(np.argmax(np.abs(row)))
        distances.append(dict(symbol=symbol, volatility_window=window, input_window=lag,
            origin=str(origin), year=origin.year, mahalanobis_squared=distance,
            threshold_train_p99=threshold, flagged=bool(distance>threshold),
            largest_standardized_feature=names(lag)[most], largest_abs_z=abs(row[most])))
    if window == 14:
        plot_data[symbol] = (pca, scores_dev, index[dev], md_dev>threshold)


def check():
    """Verify chronology, true future alignment and missing-calendar handling."""
    dates = pd.date_range('2023-01-01', '2025-12-31', tz='UTC')
    train, dev = development_masks(dates, np.ones(len(dates), dtype=bool))
    assert dates[train][-1] == pd.Timestamp('2023-12-24', tz='UTC')
    assert dates[dev][-1] == pd.Timestamp('2024-12-24', tz='UTC')
    daily = pd.Series(np.arange(10, dtype=float), index=dates[:10])
    daily.iloc[4] = np.nan
    chosen = longest_segment(daily)
    assert chosen.index[0] == dates[5] and len(chosen) == 5
    try:
        longest_segment(daily.dropna())
    except ValueError:
        pass
    else:
        raise AssertionError('Compressed calendar was accepted')
    rng = np.random.default_rng(23)
    degenerate = describe(np.linspace(-2e-16, 2e-16, 20))
    assert degenerate['numerically_constant'] and np.isnan(degenerate['skewness'])
    assert degenerate['tukey_outliers'] == 0
    prices = 100*np.exp(np.cumsum(rng.normal(0, .0001, 90*1440)))
    close = pd.Series(prices.reshape(-1, 1440)[:, -1])
    minute = minute_features(prices)
    X, y, base = features(close, minute, 14, 14)
    vol = targets(close, 14).to_numpy()
    for h in range(1, 8):
        np.testing.assert_allclose(y[28:70, h-1], vol[28+h:70+h])
    assert len(names(14)) == X.shape[1] == 91
    changed = prices.copy()
    changed[61*1440:] *= np.exp(np.arange(len(changed)-61*1440)*.00001)
    changed_close = pd.Series(changed.reshape(-1, 1440)[:, -1])
    X2, y2, base2 = features(changed_close, minute_features(changed), 14, 14)
    np.testing.assert_allclose(X[28:61], X2[28:61])
    np.testing.assert_allclose(base[28:61], base2[28:61])
    assert not np.allclose(y[60], y2[60])
    print('PASS: target alignment, target-end cutoffs, feature causality, daily gaps, constant-column handling', flush=True)


def run():
    OUT.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(exist_ok=True)
    panel = load_panel()
    _, eligible, _ = calendars(panel)
    train, dev = development_masks(panel.index, eligible)
    choices_path = ROOT / 'results/optimized_minute_2023_2025/selected_inputs.csv'
    choices = pd.read_csv(choices_path)
    univariate, associations, redundancy = [], [], []
    pca_summary, components, distances, loadings = [], [], [], []
    stationarity, pacfs, decompositions, weekdays, seasonality, drift = [], [], [], [], [], []
    raw_summary, coverage = [], []
    box_data, scatter_data, pca_data = {}, {}, {}
    for symbol in panel.columns:
        close = panel[symbol]
        minute = minute_features(np.load(DATA / f'{symbol}.npy'))
        returns = 100*np.log(close).diff()
        raw = pd.DataFrame(np.column_stack([returns, returns**2, minute]),
                           index=panel.index, columns=FAMILIES)
        temporal = {'log_close': np.log(close), 'daily_return': returns,
                    'daily_return_squared': returns**2}
        for w in [7, 14, 21, 28]:
            raw[f'volatility_w{w}'] = targets(close, w)
            temporal[f'volatility_w{w}'] = raw[f'volatility_w{w}']
        for variable in raw.columns:
            for year in [2023, 2024, 2025]:
                values = raw.loc[raw.index.year == year, variable].dropna()
                raw_summary.append(dict(symbol=symbol, variable=variable, year=year,
                                        **describe(values)))
            for a, b in [(2023, 2024), (2024, 2025)]:
                before = raw.loc[raw.index.year == a, variable].dropna().to_numpy()
                after = raw.loc[raw.index.year == b, variable].dropna().to_numpy()
                pooled = np.sqrt((before.var(ddof=1)+after.var(ddof=1))/2)
                drift.append(dict(symbol=symbol, variable=variable, year_a=a, year_b=b,
                    n_a=len(before), n_b=len(after), median_a=np.median(before), median_b=np.median(after),
                    standardized_mean_difference=(after.mean()-before.mean())/pooled,
                    median_ratio=np.median(after)/np.median(before) if np.median(before) else np.nan,
                    ks_distance=ks_2samp(before, after).statistic))
        temporal_audit(symbol, temporal, stationarity, pacfs, decompositions, weekdays, seasonality)
        for selected in choices.loc[choices.symbol == symbol].itertuples(index=False):
            w, lag = int(selected.volatility_window), int(selected.input_window)
            X, y, base = features(close, minute, lag, w)
            assert np.isfinite(X[dev]).all() and np.isfinite(y[dev]).all()
            assert panel.index[dev][-1]+pd.Timedelta(days=7) < DEV_END
            assert panel.index[train][-1]+pd.Timedelta(days=7) < TRAIN_END
            for j, name in enumerate(names(lag)):
                univariate.append(dict(symbol=symbol, volatility_window=w, input_window=lag,
                                       kind='predictor', variable=name, **describe(X[dev, j])))
            for h in range(1, 8):
                for kind, values in [('future_volatility', y[dev, h-1]),
                    ('relative_future_correction', y[dev, h-1]/base[dev]-1)]:
                    univariate.append(dict(symbol=symbol, volatility_window=w, input_window=lag,
                                           kind=kind, variable=f'h{h}', **describe(values)))
            univariate.append(dict(symbol=symbol, volatility_window=w, input_window=lag,
                kind='baseline', variable='current_volatility', **describe(base[dev])))
            association_audit(symbol, w, lag, X, y, base, dev, associations, redundancy)
            multivariate_audit(symbol, w, lag, panel.index, X, train, dev, pca_summary,
                              components, distances, loadings, pca_data)
            coverage.append(dict(symbol=symbol, volatility_window=w, input_window=lag,
                features=X.shape[1], n_train_2023=int(train.sum()), n_development=int(dev.sum()),
                origin_first=str(panel.index[dev][0]), origin_last=str(panel.index[dev][-1]),
                label_last=str(panel.index[dev][-1]+pd.Timedelta(days=7))))
            if w == 14:
                box_data[symbol] = (X[dev, :6], y[dev, 0], y[dev, 6], base[dev])
                scatter_data[symbol] = (X[dev, 2], X[dev, -1], y[dev, 6]/base[dev]-1)
        print(f'Computed CURRENT development EDA: {symbol}', flush=True)
    uni = write_csv(univariate, 'univariate.csv.gz')
    assoc = write_csv(associations, 'future_associations.csv.gz')
    write_csv(redundancy, 'redundant_pairs.csv.gz')
    ps = write_csv(pca_summary, 'pca_summary.csv')
    write_csv(components, 'pca_variance.csv')
    write_csv(distances, 'multivariate_distances.csv.gz')
    write_csv(loadings, 'pca_loadings.csv')
    st = write_csv(stationarity, 'stationarity.csv')
    pa = write_csv(pacfs, 'pacf.csv')
    decomp = write_csv(decompositions, 'stl_components.csv.gz')
    write_csv(weekdays, 'weekday_summary.csv')
    se = write_csv(seasonality, 'seasonality.csv')
    dr = write_csv(drift, 'period_drift.csv')
    write_csv(raw_summary, 'raw_daily_summary_by_year.csv')
    write_csv(coverage, 'coverage.csv')
    figures(panel, box_data, scatter_data, pca_data, assoc, pa, decomp, dr)
    evidence = dict(dataset='data/processed/minute_2023_2025', source_frequency='1 minute UTC',
        statistical_frequency='daily', symbols=list(panel.columns),
        development_origin_first=str(panel.index[dev][0]), development_origin_last=str(panel.index[dev][-1]),
        development_last_label=str(panel.index[dev][-1]+pd.Timedelta(days=7)), n_development=int(dev.sum()),
        training_origin_last=str(panel.index[train][-1]), training_last_label=str(panel.index[train][-1]+pd.Timedelta(days=7)),
        n_training=int(train.sum()), preprocessing='Exact optimize_minute_svr.features/minute_features; no imputation',
        pca='PCA and StandardScaler fit only on 2023. 2024 projected; no feature selection applied',
        extremes='LedoitWolf squared Mahalanobis, training empirical p99; descriptive flags, no removals',
        temporal='Longest contiguous DEVELOPMENT segment, no joining across missing days; weekly robust STL',
        drift='2023 vs 2024 and 2024 vs 2025 DESCRIPTIVE ONLY, no p-values or tuning',
        effects='Pearson r and r^2, Spearman rho; no IID significance claims',
        selected_inputs_sha256=hashlib.sha256(choices_path.read_bytes()).hexdigest(),
        data_sha256={p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(DATA.glob('*')) if p.is_file()},
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        n_univariate=len(uni), n_associations=len(assoc), n_pca=len(ps),
        notes=['2025 was explored historically: this report cannot restore a pristine test.',
               'P-values unadjusted; ADF/KPSS conflicting results do not establish stationarity.',
               'Overlapping rolling targets mechanically induce persistence and serial dependence.',
               'Numerically constant columns (range <= 1e-12 max(1, absolute maximum)) have undefined shape and association; no roundoff outliers.',
               'STL is a descriptive two-sided decomposition, never a forecasting feature.'])
    (OUT / 'metadata.json').write_text(json.dumps(evidence, indent=2, ensure_ascii=False), encoding='utf-8')
    print(json.dumps({k: evidence[k] for k in ['n_training', 'n_development', 'n_univariate', 'n_associations']}), flush=True)


def figures(panel, boxes, scatter, pca_data, assoc, pacfs, decompositions, drift):
    symbols = list(panel.columns)
    fig, axes = plt.subplots(4, 2, figsize=(13, 13))
    for row, symbol in enumerate(symbols):
        X, y1, y7, base = boxes[symbol]
        axes[row, 0].boxplot(X, tick_labels=SHORT, showfliers=True)
        axes[row, 0].set_title(f'{symbol}: predictores actuales normalizados, w=14')
        axes[row, 1].boxplot([base, y1, y7], tick_labels=['actual', 'futuro h1', 'futuro h7'])
        axes[row, 1].set_title(f'{symbol}: volatilidad diaria, puntos porcentuales')
        for ax in axes[row]:
            ax.grid(axis='y', alpha=.25)
    fig.suptitle('Distribuciones en desarrollo 2023–2024; puntos extremos conservados')
    fig.tight_layout()
    save_fig(fig, 'boxplots')
    for method in ['pearson', 'spearman']:
        fig, axes = plt.subplots(4, 4, figsize=(18, 17))
        for row, symbol in enumerate(symbols):
            for col, w in enumerate([7, 14, 21, 28]):
                part = assoc[(assoc.symbol == symbol)&(assoc.volatility_window == w)&
                             (assoc.method == method)&(assoc.target == 'relative_future_correction')]
                nlag = int(part.input_window.iloc[0])
                chosen = names(nlag)[:6] + names(nlag)[-7:]
                matrix = part.pivot(index='predictor', columns='horizon', values='correlation').loc[chosen]
                ax = axes[row, col]
                im = ax.imshow(matrix, cmap='RdBu_r', vmin=-1, vmax=1, aspect='auto')
                ax.set_xticks(range(7), [f'h{h}' for h in range(1, 8)])
                ax.set_yticks(range(13), SHORT+[f'salida h{h}' for h in range(1, 8)], fontsize=8)
                ax.set_title(f'{symbol}, w={w}')
        fig.subplots_adjust(top=.93, right=.9, hspace=.3, wspace=.55)
        fig.colorbar(im, ax=axes, fraction=.015, pad=.02, label='correlación')
        fig.suptitle(f'{method.capitalize()}: predictores al origen vs corrección futura σ(t+h)/σ(t)−1\n'
                     'Desarrollo 2023–2024; CSV incluye todos los rezagos y volatilidad futura absoluta')
        save_fig(fig, f'{method}_future')
    fig, axes = plt.subplots(4, 2, figsize=(13, 13))
    for row, symbol in enumerate(symbols):
        realized, decay, residual = scatter[symbol]
        for ax, predictor, label in zip(axes[row], [realized, decay], ['RV/minuto normalizada', 'salida conocida h7']):
            ax.scatter(predictor, residual, s=8, alpha=.35)
            ax.set(xlabel=label, ylabel='corrección futura h7', title=f'{symbol}, w=14')
            ax.grid(alpha=.2)
    fig.suptitle('Relaciones por origen; extremos conservados, sin ajuste de regresión')
    fig.tight_layout()
    save_fig(fig, 'scatter')
    fig, axes = plt.subplots(4, 2, figsize=(13, 13))
    for row, symbol in enumerate(symbols):
        pca, scores, dates, flagged = pca_data[symbol]
        ax = axes[row, 0]
        cum = np.cumsum(pca.explained_variance_ratio_)
        ax.plot(np.arange(1, len(cum)+1), cum)
        ax.axhline(.95, color='grey', ls='--')
        ax.set(xlabel='componentes', ylabel='varianza acumulada', title=f'{symbol}, w=14; ajuste 2023')
        ax = axes[row, 1]
        for year, color in [(2023, '#0072B2'), (2024, '#E69F00')]:
            mask = dates.year == year
            ax.scatter(scores[mask, 0], scores[mask, 1], s=9, alpha=.5, color=color, label=str(year))
        ax.scatter(scores[flagged, 0], scores[flagged, 1], facecolors='none', edgecolors='red', s=45, label='distancia > p99 2023')
        ax.set(xlabel='PC1', ylabel='PC2', title='Proyección; extremos según todas las variables')
        ax.legend(fontsize=8)
    fig.tight_layout()
    save_fig(fig, 'pca')
    fig, axes = plt.subplots(4, 3, figsize=(16, 13))
    for row, symbol in enumerate(symbols):
        part = decompositions[(decompositions.symbol == symbol)&(decompositions.variable == 'volatility_w14')]
        dates = pd.to_datetime(part.date, utc=True)
        axes[row, 0].plot(dates, part.observed, lw=.6, label='σ14')
        axes[row, 0].plot(dates, part.trend, lw=1.3, label='tendencia STL')
        axes[row, 0].set_title(f'{symbol}, mayor tramo continuo de desarrollo')
        axes[row, 0].legend(fontsize=8)
        axes[row, 1].plot(dates, part.seasonal, lw=.6)
        axes[row, 1].set_title('Componente semanal STL (periodo=7)')
        for variable, label in [('daily_return', 'r'), ('daily_return_squared', 'r²'), ('volatility_w14', 'σ14')]:
            pa = pacfs[(pacfs.symbol == symbol)&(pacfs.variable == variable)&(pacfs.lag > 0)]
            axes[row, 2].plot(pa.lag, pa.pacf, marker='.', lw=.8, label=label)
        axes[row, 2].axhline(0, color='grey', lw=.8)
        axes[row, 2].set_title('PACF (Yule–Walker, 30 rezagos)')
        axes[row, 2].legend(fontsize=8)
        for ax in axes[row]:
            ax.grid(alpha=.2)
            ax.tick_params(axis='x', rotation=25)
    fig.tight_layout()
    save_fig(fig, 'temporal')
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    chosen = ['daily_return', 'minute_realized', 'minute_downside', 'volatility_w7', 'volatility_w14', 'volatility_w28']
    for ax, year in zip(axes, [2023, 2024]):
        matrix = drift[drift.year_a == year].pivot(index='symbol', columns='variable', values='ks_distance').loc[symbols, chosen]
        im = ax.imshow(matrix, cmap='YlOrRd', vmin=0, vmax=1)
        ax.set_xticks(range(len(chosen)), ['r', 'RV', 'down', 'σ7', 'σ14', 'σ28'])
        ax.set_yticks(range(4), symbols)
        ax.set_title(f'Distancia KS descriptiva: {year} vs {year+1}')
        for i in range(4):
            for j in range(len(chosen)):
                ax.text(j, i, f'{matrix.iloc[i,j]:.2f}', ha='center', va='center')
    fig.colorbar(im, ax=axes, fraction=.025, label='distancia entre distribuciones')
    fig.suptitle('2025 solo se describe; no interviene en correlaciones, PCA ni selección')
    save_fig(fig, 'drift')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check', action='store_true')
    args = parser.parse_args()
    with threadpool_limits(limits=1):
        if args.check:
            check()
        else:
            check()
            run()
