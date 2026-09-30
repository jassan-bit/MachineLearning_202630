"""Diagnósticos de las predicciones guardadas; no ajusta modelos ni abre TEST."""
from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import jarque_bera, probplot

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs/tables'
FIG = ROOT / 'book/_static/figures'
MODELS = ['svr', 'persistence']


def calendar_correlations(series, maxlag=168):
    """ACF centrada, denominador común; huecos excluidos y n de pares registrado."""
    if not isinstance(series.index, pd.DatetimeIndex) or series.index.has_duplicates:
        raise ValueError('Se requiere un índice temporal único')
    grid = pd.date_range(series.index.min(), series.index.max(), freq='h')
    full = series.reindex(grid)
    centered = full-full.mean()
    denominator = (centered**2).sum()
    return [(k, (centered*centered.shift(k)).sum()/denominator if denominator > 0 else np.nan,
             int((full.notna() & full.shift(k).notna()).sum()))
            for k in range(1, maxlag + 1)]


def block_bootstrap(oof, block_hours, reps=499, seed=42):
    """Bootstrap circular pareado por calendario, separado dentro de cada fold."""
    if block_hours < 1 or reps < 2:
        raise ValueError('Longitud y réplicas inválidas')
    rng = np.random.default_rng(seed)
    draws = np.zeros((reps, 2))
    point = np.zeros(2)
    groups = list(oof.groupby('fold'))
    for _, g in groups:
        idx = pd.date_range(g.time.min(), g.time.max(), freq='h')
        n = len(idx)
        if block_hours > n:
            raise ValueError('El bloque excede el fold')
        q, rem = divmod(n, block_hours)
        starts = rng.integers(n, size=(reps, q))
        tail = rng.integers(n, size=reps)
        for j, name in enumerate(MODELS):
            arr = g.assign(error=(g.y-g[name])**2).pivot(
                index='time', columns='symbol', values='error').reindex(idx).to_numpy()
            mask = np.isfinite(arr)
            filled = np.nan_to_num(arr)
            ext = np.concatenate([filled, filled[:block_hours]])
            em = np.concatenate([mask, mask[:block_hours]])
            zeros = np.zeros((1, arr.shape[1]))
            cs = np.vstack([zeros, ext.cumsum(axis=0)])
            cc = np.vstack([zeros, em.cumsum(axis=0)])
            sums = (cs[starts+block_hours]-cs[starts]).sum(axis=1)
            counts = (cc[starts+block_hours]-cc[starts]).sum(axis=1)
            if rem:
                sums += cs[tail+rem]-cs[tail]
                counts += cc[tail+rem]-cc[tail]
            if (counts == 0).any():
                raise ValueError('Réplica sin observaciones: no se descarta silenciosamente')
            draws[:, j] += np.sqrt(sums/counts).mean(axis=1)/len(groups)
            point[j] += np.sqrt(np.nanmean(arr, axis=0)).mean()/len(groups)
    rows = []
    for j, name in enumerate(MODELS):
        rows.append(dict(block_hours=block_hours, metric='macro_rmse', model=name,
                         estimate=point[j], low=np.quantile(draws[:, j], .025),
                         high=np.quantile(draws[:, j], .975)))
    delta = draws[:, 0]-draws[:, 1]
    rows.append(dict(block_hours=block_hours, metric='delta_macro_rmse',
                     model='svr_minus_persistence', estimate=point[0]-point[1],
                     low=np.quantile(delta, .025), high=np.quantile(delta, .975)))
    return pd.DataFrame(rows)


def main():
    tracked = [OUT/'base_validation_predictions.csv', OUT/'base_model_protocol.json',
               OUT/'base_folds.csv', ROOT/'data/splits/development_80.csv']
    hashes = {p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() for p in tracked}
    meta = json.loads((OUT/'base_metadata.json').read_text(encoding='utf-8'))
    assert hashes['data/splits/development_80.csv'] == meta['development_sha256']
    assert hashes['outputs/tables/base_model_protocol.json'] == meta['protocol_sha256']
    oof = pd.read_csv(tracked[0], parse_dates=['time'])
    assert not oof.duplicated(['symbol', 'fold', 'time']).any()
    assert np.isfinite(oof[['y', *MODELS]]).all().all()
    metrics = pd.read_csv(OUT/'base_metrics_by_fold.csv')
    rows, acfrows = [], []
    for symbol, part in oof.groupby('symbol'):
        folds = sorted(part.fold.unique())
        fig, axes = plt.subplots(len(folds), 4, figsize=(17, 3.3*len(folds)),
                                 constrained_layout=True)
        for i, fold in enumerate(folds):
            g = part.loc[part.fold.eq(fold)].sort_values('time').set_index('time')
            for name, color in zip(MODELS, ['tab:blue', 'tab:orange']):
                residual = g.y-g[name]
                reference = metrics.query('symbol == @symbol and fold == @fold and model == @name').iloc[0]
                assert len(g) == reference.n
                assert np.isclose(np.sqrt(np.mean(residual**2)), reference.rmse)
                values = calendar_correlations(residual)
                corr = {k: rho for k, rho, _ in values}
                acfrows.extend(dict(symbol=symbol, fold=int(fold), model=name,
                                    lag_hours=k, correlation=rho, pairs=n)
                               for k, rho, n in values)
                middle = g.index.min()+(g.index.max()-g.index.min())/2
                first, second = residual.loc[residual.index <= middle], residual.loc[residual.index > middle]
                jb = jarque_bera(residual)
                rows.append(dict(symbol=symbol, fold=int(fold), model=name, n=len(g),
                    mean_residual=residual.mean(), std_residual=residual.std(),
                    skew=residual.skew(), excess_kurtosis=residual.kurt(),
                    jb_stat=jb.statistic, jb_p_iid=jb.pvalue,
                    abs_residual_fitted_spearman=residual.abs().corr(g[name], method='spearman'),
                    variance_second_over_first=second.var()/first.var(),
                    acf_1h=corr[1], acf_24h=corr[24], acf_168h=corr[168]))
                full = residual.reindex(pd.date_range(g.index.min(), g.index.max(), freq='h'))
                axes[i, 0].plot(full.index, full, color=color, lw=.35, alpha=.65, label=name)
                (normal, ordered), (slope, intercept, _) = probplot(residual, dist='norm')
                axes[i, 1].plot(normal, ordered, '.', color=color, ms=1, alpha=.35)
                axes[i, 1].plot(normal, slope*normal+intercept, color=color, lw=.8)
                axes[i, 2].scatter(g[name], residual, s=1, color=color, alpha=.12)
                axes[i, 3].plot(range(1, 169), [v[1] for v in values], color=color, label=name)
            axes[i, 0].set_title(f'Fold {fold}: residuo (pp)'); axes[i, 0].legend(fontsize=7)
            axes[i, 0].tick_params(axis='x', rotation=35, labelsize=7)
            axes[i, 1].set_title('Q-Q normal: marginal, descriptivo')
            axes[i, 2].set_title('Residuo frente a predicción (pp)')
            axes[i, 3].set_title('Correlación residual por rezago horario')
            axes[i, 3].axhline(0, color='gray', lw=.6)
        fig.suptitle(f'{symbol}: ambos modelos, cinco folds de DEVELOPMENT')
        fig.savefig(FIG/f'residuals_all_{symbol}.png', dpi=120)
        plt.close(fig)
        print('Residuos completados:', symbol, flush=True)
    pd.DataFrame(rows).to_csv(OUT/'base_residual_diagnostics_all.csv', index=False)
    pd.DataFrame(acfrows).to_csv(OUT/'base_residual_correlations_all.csv', index=False)
    sensitivity = pd.concat([block_bootstrap(oof, b) for b in [24, 168, 336]], ignore_index=True)
    original = pd.read_csv(OUT/'base_confidence_intervals.csv')
    weekly = sensitivity.loc[sensitivity.block_hours.eq(168)].reset_index(drop=True)
    assert (weekly.model == original.model).all()
    np.testing.assert_allclose(weekly[['estimate', 'low', 'high']],
                               original[['estimate', 'low', 'high']], rtol=1e-10, atol=1e-12)
    sensitivity.to_csv(OUT/'base_bootstrap_sensitivity.csv', index=False)
    for p in tracked:
        assert hashes[p.relative_to(ROOT).as_posix()] == hashlib.sha256(p.read_bytes()).hexdigest()
    metadata = dict(source_hashes=hashes, test_read=False, model_refit=False,
        residual_definition='observed_minus_predicted', diagnostics=len(rows),
        blocks_hours=[24, 168, 336], bootstrap_reps=499, seed=42,
        inference='conditional_on_selected_configuration_not_independent_test',
        correlation='centered_autocovariance_common_denominator_hourly_grid_missing_pairs_excluded',
        variance_halves='calendar_midpoint')
    (OUT/'extended_diagnostics_metadata.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')
    print(sensitivity.to_string(index=False), flush=True)


if __name__ == '__main__':
    main()
