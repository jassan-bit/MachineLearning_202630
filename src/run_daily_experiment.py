"""Run all 64 window/asset configurations; train/val/test never shuffled."""
from pathlib import Path
import argparse
import json
import time
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import acf
from volatility_experiment import (ROOT, configuration, load_daily, targets, calendar_folds,
    arrays, estimator, fit_model, metric_rows, residual_bds, provenance)


def eda(panel, cfg, out, figs):
    # EDA is restricted to TRAIN, before validation or test are inspected.
    train = panel.loc[panel.index < pd.Timestamp(cfg['train_end_exclusive'], tz='UTC')]
    summaries, correlations = [], []
    for symbol in cfg['symbols']:
        close = train[symbol]
        returns = np.log(close).diff()
        for name, series in [('close', close), ('log_return', returns)]:
            summaries.append(dict(symbol=symbol, variable=name, **series.describe().to_dict(),
                missing=int(series.isna().sum()), skew=series.skew(), kurtosis=series.kurt()))
        fig, axes = plt.subplots(3, 2, figsize=(13, 10))
        axes[0, 0].plot(close.index, close); axes[0, 0].set_title('Cierre diario (USDT)')
        axes[0, 1].plot(returns.index, returns); axes[0, 1].set_title('Retornos logarítmicos diarios')
        axes[1, 0].hist(returns.dropna(), bins=60); axes[1, 0].set_title('Distribución de retornos')
        # ACF on longest contiguous stretch, without collapsing gaps.
        groups = returns.notna().ne(returns.notna().shift()).cumsum()
        segments = [g for _, g in returns.groupby(groups) if g.notna().all()]
        segment = max(segments, key=len)
        for ax, series, name in [(axes[1, 1], segment, 'retornos'), (axes[2, 0], segment**2, 'retornos al cuadrado')]:
            values = acf(series, nlags=min(40, len(series)//4), fft=True)
            ax.bar(np.arange(len(values)), values); ax.set_title(f'ACF: {name}')
            ax.set_xlabel('Rezago en días')
            for lag, value in enumerate(values):
                correlations.append(dict(symbol=symbol, variable=name, lag=lag, acf=value,
                    segment_start=str(segment.index[0]), segment_end=str(segment.index[-1]), n=len(segment)))
        for w in cfg['volatility_windows']:
            vol = targets(close, w)
            axes[2, 1].plot(vol.index, vol, label=f'{w} días', alpha=.7)
            summaries.append(dict(symbol=symbol, variable=f'volatility_{w}', **vol.describe().to_dict(),
                missing=int(vol.isna().sum()), skew=vol.skew(), kurtosis=vol.kurt()))
        axes[2, 1].set_title('Volatilidad (%) sin anualizar'); axes[2, 1].legend()
        fig.suptitle(f'{symbol} — EDA exclusivo de TRAIN (2020–2022)')
        fig.tight_layout(); fig.savefig(figs/f'eda_{symbol}.png', dpi=130); plt.close(fig)
    pd.DataFrame(summaries).to_csv(out/'eda_summary.csv', index=False)
    pd.DataFrame(correlations).to_csv(out/'eda_acf.csv', index=False)


def run():
    started = time.perf_counter()
    cfg = configuration(); panel = load_daily()
    out = ROOT/'results'; figs = ROOT/'notebooks/figs'; models = out/'models'
    for folder in (out, figs, models): folder.mkdir(parents=True, exist_ok=True)
    (out/'run_status.json').write_text(json.dumps({'status': 'running', 'protocol': provenance(cfg)}, indent=2))
    folds, audit = calendar_folds(panel, cfg)
    audit.to_csv(out/'fold_calendar.csv', index=False)
    eda(panel, cfg, out, figs)
    metrics, predictions, diagnostics, searches, selections = [], [], [], [], []
    for symbol in cfg['symbols']:
      close = panel[symbol]
      for w in cfg['volatility_windows']:
        vol = targets(close, w)
        for lag in cfg['input_windows']:
          key = dict(symbol=symbol, volatility_window=w, input_window=lag)
          datasets = [{role: arrays(close, vol, anchors, lag) for role, anchors in fold.items()} for fold in folds]
          candidates = []
          for C in cfg['C']:
            for eps in cfg['epsilon']:
              fitted, scores = [], []
              for f, ds in enumerate(datasets):
                m = fit_model(estimator(C, eps, cfg['seed']), *ds['train'][:2])
                score = np.sqrt(np.mean((m.predict(ds['val'][0])-ds['val'][1])**2, axis=0)).mean()
                fitted.append(m); scores.append(score)
                searches.append(dict(**key, C=C, epsilon=eps, fold=f+1, val_rmse=score))
              candidates.append((float(np.mean(scores)), C, eps, fitted))
          score, C, eps, fitted = min(candidates, key=lambda item: (item[0], item[1], item[2]))
          selections.append(dict(**key, C=C, epsilon=eps, val_rmse=score))
          for f, (ds, model) in enumerate(zip(datasets, fitted), 1):
            joblib.dump(dict(model=model, **key, horizon=7, C=C, epsilon=eps,
                fitted_through=str(panel.index[folds[f-1]['train'].max()+7])),
                models/f'{symbol}_v{w}_l{lag}_f{f}.joblib')
            for role, (X, y, baseline) in ds.items():
              pred = model.predict(X)
              for name, p in [('SVR', pred), ('Persistence', baseline)]:
                for row in metric_rows(y, p):
                  metrics.append(dict(**key, fold=f, split=role, model=name, **row))
                if role == 'test':
                  diagnostics.append(dict(**key, fold=f, model=name,
                      **residual_bds(y[:, 0]-p[:, 0], cfg['bds_min_observations'])))
              for i, anchor in enumerate(folds[f-1][role]):
                for h in range(7):
                  predictions.append(dict(**key, fold=f, split=role,
                      origin=str(panel.index[anchor]), target_date=str(panel.index[anchor+h+1]), horizon=h+1,
                      actual=y[i, h], svr=pred[i, h], persistence=baseline[i, h], residual=y[i, h]-pred[i, h]))
          print(f'{symbol} volatility={w} input={lag}: validation RMSE={score:.6f}', flush=True)
          # Incremental results are recoverable; status remains running until all checks finish.
          for filename, records in [('metrics', metrics), ('bds', diagnostics), ('search', searches), ('selection', selections)]:
            pd.DataFrame(records).to_csv(out/f'{filename}.csv', index=False)
    pd.DataFrame(predictions).to_csv(out/'predictions.csv.gz', index=False, compression='gzip')
    selection = pd.DataFrame(selections)
    # Input-window selection uses validation only, separately for each target definition.
    chosen = selection.loc[selection.groupby(['symbol', 'volatility_window']).val_rmse.idxmin()].copy()
    registry = []
    for row in chosen.to_dict('records'):
        symbol, w, lag = row['symbol'], int(row['volatility_window']), int(row['input_window'])
        close = panel[symbol]; vol = targets(close, w)
        anchors = np.arange(max(28, lag-1, w), len(panel)-7)
        good = []
        for a in anchors:
            if np.isfinite(close.iloc[a-28:a+8]).all(): good.append(a)
        X, y, _ = arrays(close, vol, np.asarray(good), lag)
        final = fit_model(estimator(row['C'], row['epsilon']), X, y)
        filename = f'{symbol}_v{w}_deployment.joblib'
        artifact = dict(model=final, symbol=symbol, volatility_window=w, input_window=lag, horizon=7,
            unit='percentage_points', annualized=False, ddof=0, fitted_through='2025-12-31',
            evaluation='deployment refit on all eligible 2020–2025; metrics refer to fold models')
        joblib.dump(artifact, models/filename)
        registry.append(dict(**row, artifact=filename))
    pd.DataFrame(registry).to_csv(out/'model_registry.csv', index=False)
    make_plots(pd.DataFrame(metrics), pd.DataFrame(predictions), figs)
    pd.DataFrame(metrics).query("split == 'test'").groupby(
        ['symbol', 'volatility_window', 'input_window', 'model', 'horizon'])[['mape','mae','rmse','mse']].agg(['mean','std']).to_csv(out/'test_summary.csv')
    (out/'run_status.json').write_text(json.dumps(dict(status='complete', seconds=time.perf_counter()-started,
        configurations=len(selections), folds=5, protocol=provenance(cfg)), indent=2), encoding='utf-8')


def make_plots(metrics, predictions, figs):
    for (symbol, w), group in metrics.query("split == 'test'").groupby(['symbol', 'volatility_window']):
        fig, ax = plt.subplots(1, 2, figsize=(12, 4))
        for lag, g in group.query("model == 'SVR'").groupby('input_window'):
            g.query('horizon == 0').plot(x='fold', y='rmse', ax=ax[0], marker='o', label=f'{lag} cierres')
            g.query('horizon > 0').groupby('horizon').rmse.mean().plot(ax=ax[1], marker='o', label=f'{lag} cierres')
        ax[0].set_title('RMSE por fold'); ax[1].set_title('RMSE medio por horizonte')
        for a in ax: a.set_ylabel('Puntos porcentuales'); a.legend()
        fig.suptitle(f'{symbol} — volatilidad móvil de {w} días'); fig.tight_layout()
        fig.savefig(figs/f'rmse_{symbol}_v{w}.png', dpi=130); plt.close(fig)
        for lag in (7, 14, 21, 28):
            ranked = group.query("model == 'SVR' and horizon == 0 and input_window == @lag").sort_values(['rmse','fold'])
            selected = [int(ranked.iloc[i].fold) for i in (0, len(ranked)//2, -1)]
            fig, axes = plt.subplots(3, 1, figsize=(13, 9))
            for ax, f, label in zip(axes, selected, ['mejor', 'mediano', 'peor']):
                sub = predictions.query('symbol == @symbol and volatility_window == @w and input_window == @lag and fold == @f and horizon == 1')
                for role, color in [('train', '#2364aa'), ('val', '#e58f16'), ('test', '#25814e')]:
                    g = sub[sub.split == role].sort_values('target_date')
                    dates = pd.to_datetime(g.target_date, utc=True)
                    ax.plot(dates, g.actual, color=color, marker='.', label=f'{role} real')
                    ax.plot(dates, g.svr, color=color, linestyle='--', label=f'{role} predicho')
                ax.set_title(f'Fold {f}: {label} por RMSE test, visualización descriptiva h=1')
                ax.set_ylabel('Volatilidad (%)'); ax.legend(ncol=3, fontsize=8)
            fig.suptitle(f'{symbol} — volatilidad {w} días, entrada {lag} días; orígenes muestreados')
            fig.tight_layout(); fig.savefig(figs/f'pred_{symbol}_v{w}_l{lag}.png', dpi=110); plt.close(fig)


if __name__ == '__main__':
    run()
