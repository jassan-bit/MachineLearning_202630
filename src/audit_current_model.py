"""Diagnostics of the frozen delivery SVR; never changes its fitted artifacts.

Run with .venv-repro/Scripts/python.exe src/audit_current_model.py.
All additional fits are learning-curve diagnostics on 2023--2024 only.
"""
from pathlib import Path
import argparse
import hashlib
from importlib.metadata import version
import json
import zipfile

import joblib
import numpy as np
import pandas as pd
from scipy.stats import chi2
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from statsmodels.api import OLS, add_constant
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.stats.multitest import multipletests
from statsmodels.stats.stattools import jarque_bera
from statsmodels.tsa.stattools import acf, pacf
from threadpoolctl import threadpool_limits

from minute_experiment import ROOT, DATA, load_panel, estimator
from optimize_minute_svr import features, minute_features, calendars, metrics, predict

SOURCE = ROOT / 'results/optimized_minute_2023_2025'
OUT = ROOT / 'results/current_delivery_audit/model'
FIGURES = ROOT / 'book/figures'
METRICS = ['r2', 'rmse', 'mae', 'mse', 'mape']
BLOCKS = [35, 56, 70]
SEED = 20261004


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def block_indices(n, length, repetitions, seed):
    """Circular moving blocks; ONE date draw shared by all series and models."""
    if not 1 <= length <= n or repetitions < 1:
        raise ValueError('Invalid bootstrap dimensions')
    rng = np.random.default_rng(seed)
    starts = rng.integers(0, n, size=(repetitions, (n+length-1)//length))
    return ((starts[:, :, None]+np.arange(length)) % n).reshape(repetitions, -1)[:, :n]


def bootstrap_metric_arrays(y, predictions, indices):
    """Metrics calculated per horizon, then averaged within each configuration.

    y and predictions: (date, configuration, horizon). No concatenated-series R2.
    A common multiplicity matrix preserves paired/cross-asset dependencies.
    """
    n, configurations, horizons = y.shape
    counts = np.zeros((len(indices), n), dtype=float)
    rows = np.repeat(np.arange(len(indices)), n)
    np.add.at(counts, (rows, indices.ravel()), 1 / n)
    yf = y.reshape(n, -1)
    mean = counts @ yf
    variance = counts @ (yf*yf) - mean*mean
    if (variance <= 0).any():
        raise ValueError('A resampled target is constant')
    values = []
    for p in predictions:
        e = (p-y).reshape(n, -1)
        mse = counts @ (e*e)
        series = [1-mse/variance, np.sqrt(mse), counts @ np.abs(e), mse,
                  counts @ (100*np.abs(e)/yf)]
        values.append(np.stack([s.reshape(len(indices), configurations, horizons).mean(axis=2)
                                for s in series], axis=2))
    return values


def reconstruct_coefficients(model):
    """Undo BOTH fitted scalers to express the relative correction z in X units."""
    xs = model.steps[0][1]
    target = model.steps[1][1]
    linear = target.regressor_.estimators_
    coef = np.column_stack([m.coef_ for m in linear])
    intercept = np.array([m.intercept_[0] for m in linear])
    ys = target.transformer_
    weights = coef*ys.scale_[None, :]/xs.scale_[:, None]
    offset = intercept*ys.scale_+ys.mean_-xs.mean_ @ weights
    # One training SD of each ORIGINAL engineered X changes z by this amount.
    standardized = weights*xs.scale_[:, None]
    return weights, offset, standardized


def feature_names(lag):
    groups = ['daily_return', 'daily_return_squared', 'minute_realized',
              'minute_absolute', 'minute_downside', 'minute_maximum']
    rows = []
    for offset in range(lag):
        for group in groups:
            rows.append(dict(feature=f'{group}_lag{offset}', group=group,
                             lag_days=offset, decay_horizon=0))
    for h in range(1, 8):
        rows.append(dict(feature=f'known_window_decay_h{h}', group='known_window_decay',
                         lag_days=-1, decay_horizon=h))
    return rows


def recent_training_subset(train, fraction, target_horizon=7, validation=None):
    """Nested recent histories share their last origin and NEVER borrow validation."""
    if not 0 < fraction <= 1:
        raise ValueError('Invalid learning fraction')
    chosen = train[-max(2, int(np.ceil(len(train)*fraction))):]
    if validation is not None and chosen.max()+target_horizon >= validation.min():
        raise ValueError('Training labels overlap validation')
    return chosen


def hac_moment_normality(residual, maxlags=35):
    """Joint skew/excess-kurtosis Wald diagnostic with delta-method HAC covariance.

    A two-moment check is not an omnibus proof of a Gaussian distribution.
    Estimation of the mean/variance is included in the influence functions.
    """
    x = (residual-residual.mean()) / residual.std(ddof=0)
    skew = np.mean(x**3)
    kurt = np.mean(x**4)
    influence = np.column_stack([
        x**3-3*x-skew-1.5*skew*(x*x-1),
        x**4-4*skew*x-kurt-2*kurt*(x*x-1)])
    influence -= influence.mean(axis=0)
    n = len(x)
    longrun = influence.T @ influence / n
    for lag in range(1, min(maxlags, n-1)+1):
        gamma = influence[lag:].T @ influence[:-lag] / n
        longrun += (1-lag/(maxlags+1))*(gamma+gamma.T)
    moments = np.array([skew, kurt-3])
    statistic = float(n*moments @ np.linalg.pinv(longrun) @ moments)
    return statistic, float(chi2.sf(statistic, 2))


def collect_predictions(source=SOURCE):
    frame = pd.read_csv(source/'predictions.csv.gz')
    frame['origin'] = pd.to_datetime(frame.origin, utc=True)
    keys = sorted(set(zip(frame.symbol, frame.volatility_window)))
    dates = pd.DatetimeIndex(sorted(frame.origin.unique()))
    if not np.all(np.diff(dates.asi8) == pd.Timedelta(days=1).value):
        raise ValueError('Bootstrap calendar is not consecutive daily origins')
    matrices = {name: [] for name in ['actual', 'svr', 'persistence']}
    for symbol, window in keys:
        g = frame[(frame.symbol == symbol) & (frame.volatility_window == window)]
        if g.duplicated(['origin', 'horizon']).any():
            raise ValueError('Duplicate forecasts')
        for name in matrices:
            m = g.pivot(index='origin', columns='horizon', values=name).reindex(dates)
            if list(m.columns) != list(range(1, 8)) or not np.isfinite(m.to_numpy()).all():
                raise ValueError('Forecast grids differ')
            matrices[name].append(m.to_numpy())
    arrays = {name: np.stack(items, axis=1) for name, items in matrices.items()}
    return keys, dates, arrays


def bootstrap(keys, dates, arrays, repetitions):
    y, svr, persistence = [arrays[name] for name in ['actual', 'svr', 'persistence']]
    points = bootstrap_metric_arrays(y, [svr, persistence], np.arange(len(dates))[None, :])
    # Exact agreement with the delivery: aggregation is not changed by this audit.
    original = pd.read_csv(SOURCE/'selected_metrics.csv')
    for model, point in zip(['SVR_optimized', 'Persistence'], points):
        selected = original[original.model == model].set_index(['symbol', 'volatility_window'])
        np.testing.assert_allclose(point[0], selected.loc[keys, METRICS].to_numpy(), rtol=1e-10, atol=1e-10)
    loss_difference = ((svr-y)**2-(persistence-y)**2).mean(axis=2)
    point_loss = loss_difference.mean(axis=0)
    interval_rows, loss_rows = [], []
    indices_export = {}
    for length in BLOCKS:
        indices = block_indices(len(dates), length, repetitions, SEED+length)
        indices_export[f'length_{length}'] = indices.astype(np.int16)
        draws = bootstrap_metric_arrays(y, [svr, persistence], indices)
        for config_index, (symbol, window) in enumerate(keys + [('GLOBAL_MACRO', 0)]):
            scope = 'macro' if symbol == 'GLOBAL_MACRO' else 'configuration'
            sample = [d.mean(axis=1) if scope == 'macro' else d[:, config_index, :] for d in draws]
            point = [p[0].mean(axis=0) if scope == 'macro' else p[0, config_index] for p in points]
            for label, values, observed in zip(['SVR_optimized', 'Persistence', 'SVR_minus_Persistence'],
                    [sample[0], sample[1], sample[0]-sample[1]],
                    [point[0], point[1], point[0]-point[1]]):
                lower, upper = np.quantile(values, [.025, .975], axis=0)
                for j, metric in enumerate(METRICS):
                    interval_rows.append(dict(scope=scope, symbol=symbol, volatility_window=window,
                        block_days=length, repetitions=repetitions, model=label, metric=metric,
                        point=observed[j], ci95_low=lower[j], ci95_high=upper[j]))
        loss_draws = np.stack([loss_difference[index].mean(axis=0) for index in indices])
        for j, (symbol, window) in enumerate(keys + [('GLOBAL_MACRO', 0)]):
            observed = point_loss.mean() if symbol == 'GLOBAL_MACRO' else point_loss[j]
            sample = loss_draws.mean(axis=1) if symbol == 'GLOBAL_MACRO' else loss_draws[:, j]
            lower, upper = np.quantile(sample, [.025, .975])
            # Recenter under H0, preserving serial blocks and paired forecasts.
            pvalue = (1+np.sum(np.abs(sample-observed) >= abs(observed)))/(repetitions+1)
            loss_rows.append(dict(symbol=symbol, volatility_window=window, block_days=length,
                delta_mse_svr_minus_persistence=observed, ci95_low=lower, ci95_high=upper,
                pvalue_centered_two_sided=pvalue))
        print(f'Bootstrap blocks={length}, repetitions={repetitions}', flush=True)
    intervals = pd.DataFrame(interval_rows)
    losses = pd.DataFrame(loss_rows)
    losses['pvalue_holm_16'] = np.nan
    for length in BLOCKS:
        chosen = (losses.block_days == length) & (losses.symbol != 'GLOBAL_MACRO')
        losses.loc[chosen, 'pvalue_holm_16'] = multipletests(
            losses.loc[chosen, 'pvalue_centered_two_sided'], method='holm')[1]
    intervals.to_csv(OUT/'metric_confidence_intervals.csv', index=False)
    losses.to_csv(OUT/'paired_loss_bootstrap.csv', index=False)
    np.savez_compressed(OUT/'shared_bootstrap_indices.npz', **indices_export)
    return intervals, losses


def residual_diagnostics(keys, dates, arrays):
    rows, correlation_rows, ljung_rows = [], [], []
    for j, (symbol, window) in enumerate(keys):
        for h in range(7):
            y = arrays['actual'][:, j, h]
            fitted = arrays['svr'][:, j, h]
            base = arrays['persistence'][:, j, h]
            e = y-fitted
            jb, jb_p, skew, kurt = jarque_bera(e)
            moment, moment_p = hac_moment_normality(e)
            # Conditional SECOND moment: nonzero conditional bias can also drive it.
            pz = (fitted-fitted.mean())/fitted.std()
            bz = (base-base.mean())/base.std()
            regressors = add_constant(np.column_stack([pz, pz*pz, bz]))
            second_moment = OLS(e*e, regressors).fit(cov_type='HAC',
                cov_kwds={'maxlags': 35, 'use_correction': True})
            contrast = np.eye(4)[1:]
            wald = second_moment.wald_test(contrast, scalar=True, use_f=False)
            meta = dict(symbol=symbol, volatility_window=window, horizon=h+1)
            ac = acf(e, nlags=56, fft=False)
            pc = pacf(e, nlags=56, method='ywm')
            rows.append(dict(**meta, n=len(e), residual_mean=e.mean(), residual_sd=e.std(),
                skewness=skew, kurtosis=kurt, jb_statistic=jb, jb_pvalue_nominal=jb_p,
                normality_moments_hac_statistic=moment, normality_moments_hac_pvalue=moment_p,
                second_moment_hac_wald=float(wald.statistic), second_moment_hac_pvalue=float(wald.pvalue),
                hac_maxlags=35, second_moment_r2=second_moment.rsquared, acf_lag1=ac[1], acf_lag7=ac[7],
                acf_lag35=ac[35], negative_clipped_predictions=int((fitted == 0).sum())))
            for lag in range(1, 57):
                correlation_rows.append(dict(**meta, lag=lag, acf=ac[lag], pacf=pc[lag]))
            for kind, residual in [('residual', e), ('squared_centered_residual', (e-e.mean())**2)]:
                lb = acorr_ljungbox(residual, lags=[7, 14, 28, 35, 56], model_df=0, return_df=True)
                for lag, value in lb.iterrows():
                    ljung_rows.append(dict(**meta, series=kind, lag=int(lag),
                        lb_statistic=value.lb_stat, pvalue_nominal=value.lb_pvalue))
    diagnostic = pd.DataFrame(rows)
    for column in ['jb_pvalue_nominal', 'normality_moments_hac_pvalue', 'second_moment_hac_pvalue']:
        diagnostic[column+'_holm_112'] = multipletests(diagnostic[column], method='holm')[1]
    ljung = pd.DataFrame(ljung_rows)
    ljung['pvalue_holm_560'] = np.nan
    for kind in ljung.series.unique():
        mask = ljung.series == kind
        ljung.loc[mask, 'pvalue_holm_560'] = multipletests(ljung.loc[mask, 'pvalue_nominal'], method='holm')[1]
    correlations = pd.DataFrame(correlation_rows)
    diagnostic.to_csv(OUT/'residual_diagnostics.csv', index=False)
    correlations.to_csv(OUT/'residual_acf_pacf.csv', index=False)
    ljung.to_csv(OUT/'residual_ljung_box.csv', index=False)
    return diagnostic, correlations, ljung


def coefficients_and_learning(keys, arrays, dates):
    panel = load_panel()
    folds, eligible, _ = calendars(panel)
    training_dates = panel.index + pd.Timedelta(days=7)
    cutoff = pd.Timestamp('2025-01-01', tz='UTC')
    if any(training_dates[np.r_[tr, va]].max() >= cutoff for tr, va in folds):
        raise ValueError('Learning curve used 2025 labels')
    coeff_rows, intercept_rows, curve_rows, prediction_checks = [], [], [], []
    for symbol in panel.columns:
        minute = minute_features(np.load(DATA/f'{symbol}.npy'))
        for window in [7, 14, 21, 28]:
            artifact = joblib.load(SOURCE/'models'/f'{symbol}_v{window}.joblib')
            model = artifact['model']
            lag = artifact['input_window']
            X, y, base = features(panel[symbol], minute, lag, window)
            origins = panel.index.get_indexer(dates)
            if (origins < 0).any():
                raise ValueError('Missing prediction origin')
            w, b, standardized = reconstruct_coefficients(model)
            z = X[origins] @ w + b
            np.testing.assert_allclose(z, model.predict(X[origins]), rtol=1e-10, atol=1e-10)
            index = keys.index((symbol, window))
            calculated = np.maximum(base[origins, None]*(1+z), 0)
            np.testing.assert_allclose(calculated, arrays['svr'][:, index], rtol=1e-10, atol=1e-10)
            prediction_checks.append(dict(symbol=symbol, volatility_window=window,
                max_prediction_reconstruction_error=float(np.abs(calculated-arrays['svr'][:, index]).max())))
            for i, feature in enumerate(feature_names(lag)):
                for h in range(7):
                    coeff_rows.append(dict(symbol=symbol, volatility_window=window, input_window=lag,
                        horizon=h+1, **feature, weight_relative_correction=w[i, h],
                        effect_one_training_sd_relative_correction=standardized[i, h],
                        fitted_x_mean=model.steps[0][1].mean_[i], fitted_x_sd=model.steps[0][1].scale_[i]))
            for h in range(7):
                intercept_rows.append(dict(symbol=symbol, volatility_window=window,
                    horizon=h+1, intercept_relative_correction=b[h]))
            for fold, (train, validation) in enumerate(folds, 1):
                for fraction in [.25, .5, .75, 1.0]:
                    tr = recent_training_subset(train, fraction, validation=validation)
                    curve_model = make_pipeline(StandardScaler(), estimator(artifact['C'], artifact['epsilon']))
                    curve_model.fit(X[tr], y[tr]/base[tr, None]-1)
                    record = dict(symbol=symbol, volatility_window=window, input_window=lag,
                        C=artifact['C'], epsilon=artifact['epsilon'], fold=fold, training_fraction=fraction,
                        n_train=len(tr), n_validation=len(validation),
                        train_first=str(panel.index[tr.min()]), train_last=str(panel.index[tr.max()]),
                        train_target_end=str(panel.index[tr.max()+7]),
                        validation_first=str(panel.index[validation.min()]),
                        validation_target_end=str(panel.index[validation.max()+7]))
                    for prefix, selected in [('train', tr), ('validation', validation)]:
                        record.update({prefix+'_'+name: value for name, value in
                            metrics(y[selected], predict(curve_model, X[selected], base[selected])).items()})
                    record['persistence_validation_rmse'] = metrics(y[validation],
                        np.repeat(base[validation, None], 7, axis=1))['rmse']
                    curve_rows.append(record)
            print(f'Coefficients/learning {symbol} window={window}', flush=True)
    coeff = pd.DataFrame(coeff_rows)
    coeff.to_csv(OUT/'coefficients_engineered_units.csv', index=False)
    pd.DataFrame(intercept_rows).to_csv(OUT/'intercepts_engineered_units.csv', index=False)
    grouped = coeff.assign(abs_effect=coeff.effect_one_training_sd_relative_correction.abs()).groupby(
        ['symbol', 'volatility_window', 'horizon', 'group'], as_index=False).agg(
            sum_absolute_sd_effect=('abs_effect', 'sum'), mean_absolute_sd_effect=('abs_effect', 'mean'),
            mean_signed_sd_effect=('effect_one_training_sd_relative_correction', 'mean'),
            n_features=('feature', 'size'))
    grouped.to_csv(OUT/'coefficient_groups.csv', index=False)
    curve = pd.DataFrame(curve_rows)
    curve.to_csv(OUT/'learning_curve.csv', index=False)
    summary = curve.groupby('training_fraction', as_index=False).agg(
        mean_n_train=('n_train', 'mean'), train_rmse=('train_rmse', 'mean'),
        validation_rmse=('validation_rmse', 'mean'), persistence_validation_rmse=('persistence_validation_rmse', 'mean'),
        n_fits=('fold', 'size'))
    summary.to_csv(OUT/'learning_curve_summary.csv', index=False)
    # An additional chronological view uses the full history in each expanding fold.
    # Its cumulative OOF validation sample grows (60 ... 359), but periods change too.
    forward = curve[curve.training_fraction == 1].groupby('fold', as_index=False).agg(
        n_train=('n_train', 'first'), n_validation=('n_validation', 'first'),
        train_rmse=('train_rmse', 'mean'), validation_rmse=('validation_rmse', 'mean'))
    forward['cumulative_validation_origins'] = forward.n_validation.cumsum()
    forward.to_csv(OUT/'learning_forward_calendar.csv', index=False)
    pd.DataFrame(prediction_checks).to_csv(OUT/'coefficient_prediction_verification.csv', index=False)
    return grouped, summary, curve


def plot_results(intervals, correlations, groups, learning):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size': 10, 'figure.dpi': 140, 'axes.spines.top': False,
                         'axes.spines.right': False})
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    for ax, metric in zip(axes, ['r2', 'rmse']):
        selected = intervals[(intervals.scope == 'macro') & (intervals.block_days == 35)
                             & (intervals.metric == metric) & (intervals.model != 'SVR_minus_Persistence')]
        for j, (_, row) in enumerate(selected.iterrows()):
            ax.errorbar(j, row.point, yerr=[[row.point-row.ci95_low], [row.ci95_high-row.point]],
                        fmt='o', capsize=8, color=['#235789', '#ed7d31'][j], markersize=7)
        ax.set_xticks(range(2), ['SVR lineal', 'Persistencia'])
        ax.set_ylabel('R² macro' if metric == 'r2' else 'RMSE macro (puntos porcentuales)')
        ax.set_title('IC 95 %; bloques compartidos de 35 días')
        ax.grid(axis='y', alpha=.2)
    fig.savefig(FIGURES/'current_model_confidence_intervals.png')
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
    for ax, name in zip(axes, ['acf', 'pacf']):
        m = correlations.pivot(index=['symbol', 'volatility_window', 'horizon'], columns='lag', values=name)
        im = ax.imshow(m.to_numpy(), aspect='auto', cmap='RdBu_r', vmin=-1, vmax=1,
                       extent=[.5, 56.5, len(m)-.5, -.5])
        ax.set_yticks(np.arange(3, len(m), 7),
                      [f'{s[:3]} w={w}' for s, w, _ in list(m.index)[3::7]])
        ax.set_xlabel('Rezago diario')
        ax.set_title(name.upper()+' de 112 series de residuos')
        ax.axvline(35, color='black', alpha=.4, linewidth=.8)
    fig.colorbar(im, ax=axes, label='Correlación', shrink=.8)
    fig.savefig(FIGURES/'current_model_residual_acf_pacf.png')
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(8, 4), constrained_layout=True)
    ax.plot(learning.mean_n_train, learning.train_rmse, 'o-', label='Entrenamiento')
    ax.plot(learning.mean_n_train, learning.validation_rmse, 'o-', label='Validación temporal 2024')
    ax.axhline(learning.persistence_validation_rmse.iloc[0], linestyle=':', color='#777', label='Persistencia (validación)')
    ax.set_xlabel('Orígenes de entrenamiento medios por ajuste')
    ax.set_ylabel('RMSE medio (puntos porcentuales)')
    ax.set_title('Configuraciones fijas; 16 modelos × 6 cortes; validación común')
    ax.legend()
    ax.grid(alpha=.2)
    fig.savefig(FIGURES/'current_model_learning_curve.png')
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(10, 4), constrained_layout=True)
    m = groups.groupby(['horizon', 'group']).mean_absolute_sd_effect.mean().unstack()
    for name in m:
        ax.plot(m.index, m[name], 'o-', label=name.replace('minute_', 'minuto_').replace('_', ' '))
    ax.set_xlabel('Horizonte (días)')
    ax.set_ylabel('|Cambio en corrección relativa| por 1 DE de entrada')
    ax.set_title('Media por característica y entre 16 configuraciones; sin interpretación causal')
    ax.legend(ncol=2, fontsize=8)
    ax.grid(alpha=.2)
    fig.savefig(FIGURES/'current_model_coefficients.png')
    plt.close(fig)


def run(repetitions=2000):
    OUT.mkdir(parents=True, exist_ok=True)
    files = [SOURCE/'predictions.csv.gz', SOURCE/'selected_metrics.csv', SOURCE/'selected_inputs.csv',
             *sorted((SOURCE/'models').glob('*.joblib'))]
    before = {str(path.relative_to(ROOT)): sha(path) for path in files}
    keys, dates, arrays = collect_predictions()
    intervals, losses = bootstrap(keys, dates, arrays, repetitions)
    diagnostic, correlations, ljung = residual_diagnostics(keys, dates, arrays)
    groups, summary, curve = coefficients_and_learning(keys, arrays, dates)
    plot_results(intervals, correlations, groups, summary)
    after = {str(path.relative_to(ROOT)): sha(path) for path in files}
    if before != after:
        raise ValueError('Frozen delivery artifact changed')
    config = dict(status='complete', seed=SEED, repetitions=repetitions, block_days=BLOCKS,
        bootstrap='Circular moving blocks on daily origins shared by all 112 series and both models; percentile intervals; recomputed per-series R2/RMSE then macro averages',
        paired_test='Mean squared loss difference per origin, centered block bootstrap under zero mean; two-sided; Holm over 16 configurations separately for each length',
        n_origins=len(dates), first_origin=str(dates.min()), last_origin=str(dates.max()),
        n_configurations=len(keys), n_residual_series=len(diagnostic), hac_maxlags=35,
        normality='JB nominal, complemented by skewness/excess-kurtosis Wald using delta-method Bartlett HAC influence-function covariance; Holm family of 112',
        heteroscedasticity='Squared residual regressed on standardized fitted volatility, its square and current volatility; joint slopes Wald with Bartlett HAC(35); conditional second-moment diagnostic, potentially affected by conditional mean bias; Holm family of 112',
        correlation='ACF and Yule-Walker PACF 1..56; Ljung-Box lags 7/14/28/35/56, model_df=0 (2025 model fixed, not estimated on its residuals); nominal p, Holm family of 560 per residual/squared-residual series type',
        learning='Fixed configurations selected for delivery; six original 2024 forward cuts; nested most recent 25/50/75/100 percent training rows; same validation per cut; 384 additional diagnostic fits; no 2025 labels in fit/validation',
        learning_limitation='2024 also selected these configurations, so curve is descriptive/conditional rather than new unbiased validation; calendar periods and regime also affect error',
        inference_limitation='2025 retrospective; fixed fitted models; uncertainty conditional on observed period and approximate local stationarity, not selection or refitting uncertainty; 358 days provide few long blocks',
        original_files_sha256=before, originals_unchanged=True,
        input_data_sha256={p.name: sha(p) for p in sorted(DATA.glob('*.npy'))},
        source_sha256=sha(Path(__file__)), versions={name: version(name) for name in
            ['numpy', 'pandas', 'scipy', 'scikit-learn', 'statsmodels', 'joblib']})
    (OUT/'provenance.json').write_text(json.dumps(config, indent=2, ensure_ascii=False), encoding='utf-8')
    (OUT/'README.txt').write_text(
        'Frozen SVR delivery audit, computed on 2026-10-04.\n'
        'Reproduce: .venv-repro/Scripts/python.exe src/audit_current_model.py\n'
        'Source predictions: results/optimized_minute_2023_2025/predictions.csv.gz\n'
        'No saved models, source data, original metrics or forecasts are changed.\n'
        'metric_confidence_intervals.csv: per-configuration and macro percentile 95% intervals.\n'
        'shared_bootstrap_indices.npz: one date draw per replication shared by both models and all series.\n'
        'paired_loss_bootstrap.csv: SVR minus persistence MSE, recentered two-sided bootstrap, Holm over 16 configurations.\n'
        'residual_diagnostics.csv: 112 separate series, JB nominal and dependence-adjusted HAC diagnostics.\n'
        'residual_acf_pacf.csv / residual_ljung_box.csv: no concatenation of series; Holm families stated in provenance.\n'
        'learning_curve.csv: 384 auxiliary 2023--2024 fits, fixed delivery configuration, same validation per cut.\n'
        'coefficients_engineered_units.csv: inverse-scaled coefficients and changes in relative correction per training SD.\n'
        'Use provenance.json for assumptions, limitations, parameters and SHA-256 hashes.\n'
        '2025 was already explored. These are conditional retrospective diagnostics, not a pristine test or causal analysis.\n',
        encoding='utf-8')
    with zipfile.ZipFile(OUT/'model_audit_evidence.zip', 'w', compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(OUT.iterdir()):
            if path.is_file() and path.name != 'model_audit_evidence.zip':
                archive.write(path, path.name)
        archive.write(Path(__file__), 'audit_current_model.py')
        archive.write(ROOT/'tests/test_current_model_audit.py', 'test_current_model_audit.py')
        for path in sorted(FIGURES.glob('current_model_*.png')):
            archive.write(path, 'figures/'+path.name)
    print('Frozen originals unchanged; audit complete', flush=True)
    print(summary.to_string(index=False), flush=True)


def additional_period_bootstrap(repetitions=2000):
    """Read only the additional frozen forecast export; never reevaluate models."""
    folder = ROOT/'results/current_delivery_audit/holdout'
    prediction_hash = sha(folder/'predictions.csv.gz')
    keys, dates, arrays = collect_predictions(folder)
    y, svr, persistence = [arrays[name] for name in ['actual', 'svr', 'persistence']]
    points = bootstrap_metric_arrays(y, [svr, persistence], np.arange(len(dates))[None, :])
    original = pd.read_csv(folder/'macro_metrics.csv').set_index('model')
    for model, point in zip(['SVR_optimized', 'Persistence'], points):
        np.testing.assert_allclose(point[0].mean(axis=0), original.loc[model, METRICS].to_numpy(),
                                   atol=1e-10, rtol=1e-10)
    rows = []
    for length in BLOCKS:
        indices = block_indices(len(dates), length, repetitions, SEED+length)
        draws = [v.mean(axis=1) for v in bootstrap_metric_arrays(y, [svr, persistence], indices)]
        point = [v[0].mean(axis=0) for v in points]
        for label, sample, observed in zip(['SVR_optimized', 'Persistence', 'SVR_minus_Persistence'],
                [draws[0], draws[1], draws[0]-draws[1]], [point[0], point[1], point[0]-point[1]]):
            lower, upper = np.quantile(sample, [.025, .975], axis=0)
            for j, metric in enumerate(METRICS):
                rows.append(dict(model=label, metric=metric, block_days=length, repetitions=repetitions,
                    point=observed[j], ci95_low=lower[j], ci95_high=upper[j]))
    pd.DataFrame(rows).to_csv(folder/'metric_confidence_intervals.csv', index=False)
    if sha(folder/'predictions.csv.gz') != prediction_hash:
        raise ValueError('Additional frozen forecasts changed')
    (folder/'bootstrap_provenance.json').write_text(json.dumps(dict(seed=SEED, repetitions=repetitions,
        block_days=BLOCKS, forecast_sha256=prediction_hash, source_sha256=sha(Path(__file__)),
        first_origin=str(dates.min()), last_origin=str(dates.max()), n_origins=len(dates),
        method='Same paired circular date block draws shared by 112 series; recompute horizon metrics and macro means; percentile intervals',
        limitation='236 origins contain few long blocks; uncertainty conditional on fixed models and observed period, approximate local stationarity; no refit/reselection; not individual forecast intervals'),
        indent=2, ensure_ascii=False), encoding='utf-8')
    print(pd.DataFrame(rows).to_string(index=False), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--repetitions', type=int, default=2000)
    parser.add_argument('--additional-bootstrap', action='store_true')
    args = parser.parse_args()
    with threadpool_limits(limits=1):
        if args.additional_bootstrap:
            additional_period_bootstrap(args.repetitions)
        else:
            run(args.repetitions)
