"""Análisis unidimensional de las 12 columnas originales; solo DEVELOPMENT."""
from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'data/splits/development_80.csv'
SYMBOLS = ['BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'XRPUSDT', 'SOLUSDT']
FIELDS = ['open', 'high', 'low', 'close', 'volume', 'quote_asset_volume',
          'number_of_trades', 'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume']
LABELS = {
    'open': ('Precio de apertura', 'USDT por unidad del activo'),
    'high': ('Precio máximo', 'USDT por unidad del activo'),
    'low': ('Precio mínimo', 'USDT por unidad del activo'),
    'close': ('Precio de cierre', 'USDT por unidad del activo'),
    'volume': ('Volumen base', 'Unidades del activo base'),
    'quote_asset_volume': ('Volumen cotizado', 'USDT'),
    'number_of_trades': ('Número de operaciones', 'Operaciones'),
    'taker_buy_base_asset_volume': ('Volumen comprador taker base', 'Unidades del activo base'),
    'taker_buy_quote_asset_volume': ('Volumen comprador taker cotizado', 'USDT')}


def main():
    fingerprint = hashlib.sha256(SOURCE.read_bytes()).hexdigest()
    df = pd.read_csv(SOURCE)
    for field in ['open_time', 'close_time']:
        df[field] = pd.to_datetime(df[field], format='ISO8601', utc=True)
    assert len(df) == 214165 and set(df.symbol) == set(SYMBOLS)
    assert df.open_time.between(pd.Timestamp('2020-08-11 06:00', tz='UTC'), pd.Timestamp('2025-07-01 18:00', tz='UTC')).all()
    assert np.isfinite(df[FIELDS].to_numpy()).all()
    tables = ROOT / 'outputs/tables'
    figures = ROOT / 'book/_static/figures'
    tables.mkdir(parents=True, exist_ok=True); figures.mkdir(parents=True, exist_ok=True)
    summary, temporal, types = [], [], []
    for field in df.columns:
        kind = 'Temporal' if field.endswith('_time') else ('Categórica nominal' if field == 'symbol' else ('Numérica discreta' if field == 'number_of_trades' else 'Numérica continua'))
        types.append(dict(variable=field, type=kind, cardinality=int(df[field].nunique()), missing=int(df[field].isna().sum())))
    for symbol in SYMBOLS:
        g = df.loc[df.symbol.eq(symbol)]
        for field in FIELDS:
            x = g[field]
            p = x.quantile([.01,.05,.25,.5,.75,.95,.99], interpolation='linear')
            iqr = p.loc[.75]-p.loc[.25]
            lower, upper = p.loc[.25]-1.5*iqr, p.loc[.75]+1.5*iqr
            below, above = int((x<lower).sum()), int((x>upper).sum())
            summary.append(dict(symbol=symbol, variable=field, n=len(x), mean=x.mean(), median=x.median(),
                                std=x.std(ddof=1), min=x.min(), p01=p.loc[.01], p05=p.loc[.05],
                                q1=p.loc[.25], q3=p.loc[.75], p95=p.loc[.95], p99=p.loc[.99], max=x.max(),
                                skew=x.skew(), excess_kurtosis=x.kurt(), zero=int(x.eq(0).sum()),
                                iqr=iqr, lower=lower, upper=upper, below=below, above=above,
                                flagged=below+above, percent=100*(below+above)/len(x)))
        for field in ['open_time', 'close_time']:
            temporal.append(dict(symbol=symbol, variable=field, n=len(g), cardinality=g[field].nunique(),
                                 min=g[field].min().isoformat(), max=g[field].max().isoformat()))
    stats = pd.DataFrame(summary)
    stats.to_csv(tables/'univariate_summary.csv', index=False)
    pd.DataFrame(types).to_csv(tables/'univariate_types.csv', index=False)
    pd.DataFrame(temporal).to_csv(tables/'univariate_temporal.csv', index=False)
    freq = df.symbol.value_counts().reindex(SYMBOLS).rename_axis('symbol').reset_index(name='frequency')
    freq['percent'] = 100*freq.frequency/len(df)
    freq.to_csv(tables/'univariate_categories.csv', index=False)
    # Verificar concordancia con la auditoría IQR previa, sin reescribirla.
    previous_path = tables/'development_outliers_iqr.csv'
    if previous_path.exists():
        previous = pd.read_csv(previous_path)
        merged = stats.merge(previous, on=['symbol','variable'], suffixes=('_new','_old'))
        assert len(merged)==45 and merged.flagged_new.eq(merged.flagged_old).all()
        assert np.allclose(merged.lower_new, merged.lower_old) and np.allclose(merged.upper_new, merged.upper_old)
    plt.rcParams.update({'font.size':10})
    for field in FIELDS:
        fig, axes = plt.subplots(5,2,figsize=(13,14),constrained_layout=True)
        for i,symbol in enumerate(SYMBOLS):
            x=df.loc[df.symbol.eq(symbol),field]
            row=stats.loc[stats.symbol.eq(symbol)&stats.variable.eq(field)].iloc[0]
            axes[i,0].hist(x,bins=60,log=True,color='#2563a6')
            axes[i,0].set_ylabel('Frecuencia (escala log)')
            axes[i,0].axvline(row['median'],color='#c2410c',linestyle='--',label='Mediana')
            axes[i,0].legend(fontsize=8)
            axes[i,1].boxplot(x,vert=False,whis=1.5,flierprops={'marker':'.','markersize':2,'alpha':.25})
            axes[i,1].set_yticks([])
            for ax in axes[i]:
                ax.set_title(symbol); ax.set_xlabel(LABELS[field][1]); ax.grid(alpha=.2)
        fig.suptitle(f'{field} · histogramas y boxplots · DEVELOPMENT',fontsize=15)
        fig.savefig(figures/f'univariate_{field}.png',dpi=140);plt.close(fig)
    fig,ax=plt.subplots(figsize=(9,4),constrained_layout=True)
    bars=ax.bar(freq.symbol,freq.frequency,color='#2563a6')
    ax.bar_label(bars,labels=[f'{n:,} (20%)' for n in freq.frequency])
    ax.set_ylim(0,52000);ax.set_ylabel('Observaciones');ax.set_title('Frecuencias de symbol · DEVELOPMENT')
    fig.savefig(figures/'univariate_symbol.png',dpi=150);plt.close(fig)
    # Distribución temporal de registros: los meses de los extremos son parciales.
    monthly = df.assign(month=df.open_time.dt.strftime('%Y-%m')).groupby(['month','symbol']).size().unstack().reindex(columns=SYMBOLS)
    monthly.to_csv(tables/'univariate_monthly_coverage.csv')
    fig,ax=plt.subplots(figsize=(14,4),constrained_layout=True)
    ax.bar(monthly.index,monthly[SYMBOLS[0]],color='#2563a6')
    ticks=np.arange(0,len(monthly),6);ax.set_xticks(ticks,monthly.index[ticks],rotation=35,ha='right')
    ax.set_ylabel('Velas por activo');ax.set_title('Distribución mensual de open_time · idéntica en los cinco activos')
    assert monthly.eq(monthly.iloc[:,0],axis=0).all().all()
    fig.savefig(figures/'univariate_time_coverage.png',dpi=150);plt.close(fig)
    assert hashlib.sha256(SOURCE.read_bytes()).hexdigest()==fingerprint
    (tables/'univariate_metadata.json').write_text(json.dumps({'source':'data/splits/development_80.csv',
        'sha256':fingerprint,'rows':len(df),'std_ddof':1,'quantiles':'linear',
        'kurtosis':'Exceso de Fisher corregido por sesgo; referencia normal 0',
        'normality_tests':'No aplicados: normalidad marginal no requerida; dependencia temporal no compatible con interpretación iid convencional.',
        'treatment':'Conservar registros sin transformar ni imputar',
        'target_recalculated':False,'versions':{'pandas':pd.__version__,'numpy':np.__version__,'matplotlib':matplotlib.__version__}},ensure_ascii=False,indent=2),encoding='utf-8')
    print(stats[['symbol','variable','mean','median','skew','excess_kurtosis','flagged','percent']].to_string(index=False))
    print('\nTipos y cardinalidades:\n',pd.DataFrame(types).to_string(index=False))
    return stats


if __name__ == '__main__':
    main()
