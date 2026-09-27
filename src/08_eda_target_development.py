"""EDA 2.1: volatilidad futura a 24h; solo DEVELOPMENT, sin modelos."""
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
TABLES = ROOT / 'outputs/tables'
FIGURES = ROOT / 'book/_static/figures'
SYMBOLS = ['BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'XRPUSDT', 'SOLUSDT']
FIELDS = ['open', 'high', 'low', 'close', 'volume', 'quote_asset_volume',
          'number_of_trades', 'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume']


def forward_count(mask):
    """Número de cierres válidos desde t hasta t+24, ambos inclusive."""
    return mask.astype(int).rolling(25, min_periods=25).sum().shift(-24)


def main():
    fingerprint = hashlib.sha256(SOURCE.read_bytes()).hexdigest()
    df = pd.read_csv(SOURCE)
    for col in ['open_time', 'close_time']:
        df[col] = pd.to_datetime(df[col], format='ISO8601', utc=True)
    assert len(df) == 214165 and set(df.symbol) == set(SYMBOLS)
    assert not df.duplicated(['symbol', 'open_time']).any()
    start, end = pd.Timestamp('2020-08-11 06:00', tz='UTC'), pd.Timestamp('2025-07-01 18:00', tz='UTC')
    assert df.open_time.between(start, end).all()
    assert (df.close > 0).all() and np.isfinite(df[FIELDS].to_numpy()).all()
    calendar = pd.date_range(start, end, freq='h', name='open_time')
    groups, summaries, counts, correlations, extremes, sensitivity = {}, [], [], [], [], []
    output = []
    for symbol in SYMBOLS:
        g = df.loc[df.symbol.eq(symbol)].set_index('open_time').sort_index().reindex(calendar)
        observed = g.close.notna()
        conventional = g.close_time.eq(g.index + pd.Timedelta(hours=1)-pd.Timedelta(milliseconds=1))
        valid_close = observed & conventional
        # Retornos en la cuadrícula horaria: los huecos y cierres irregulares dan NaN.
        log_close = np.log(g.close.where(valid_close))
        r = log_close.diff()
        y = 100*np.sqrt(r.pow(2).rolling(24, min_periods=24).sum().shift(-24))
        y = y.where(forward_count(valid_close).eq(25))
        edge = pd.Series(g.index+pd.Timedelta(hours=24) > end, index=g.index)
        holes = ~edge & forward_count(observed).lt(25)
        irregular = ~edge & ~holes & forward_count(valid_close).lt(25)
        usable = observed & y.notna()
        assert int(observed.sum()) == int((observed & edge).sum() + (observed & holes).sum() + (observed & irregular).sum() + usable.sum())
        # Verificación directa de cada ventana válida: ni desalineación ni futuro fuera de DEVELOPMENT.
        windows = np.lib.stride_tricks.sliding_window_view(g.close.to_numpy(), 25)
        expected = 100*np.sqrt(np.square(np.diff(np.log(windows), axis=1)).sum(axis=1))
        selected = usable.iloc[:-24].to_numpy()
        assert np.allclose(y.iloc[:-24].to_numpy()[selected], expected[selected], rtol=1e-11, atol=1e-11)
        assert not usable.iloc[-24:].any()
        assert (g.index[usable]+pd.Timedelta(hours=24) <= end).all()
        g['rv_future_24h_pct'] = y
        g['return_1h_pct'] = 100*r
        g['rv_past_24h_pct'] = 100*np.sqrt(r.pow(2).rolling(24, min_periods=24).sum())
        groups[symbol] = g
        x = y.dropna()
        q1, q3 = x.quantile([.25,.75])
        lower, upper = q1-1.5*(q3-q1), q3+1.5*(q3-q1)
        flag = (x<lower) | (x>upper)
        summaries.append(dict(symbol=symbol, n=len(x), min=x.min(), q1=q1, median=x.median(),
                              mean=x.mean(), q3=q3, p95=x.quantile(.95), p99=x.quantile(.99), max=x.max(),
                              std=x.std(), skew=x.skew(), excess_kurtosis=x.kurt(),
                              zero=int(x.eq(0).sum()), lower=lower, upper=upper,
                              iqr_flagged=int(flag.sum()), iqr_percent=100*flag.mean(),
                              log_skew=np.log(x[x>0]).skew(), log_excess_kurtosis=np.log(x[x>0]).kurt(),
                              acf_1h=y.corr(y.shift(1)), acf_24h=y.corr(y.shift(24))))
        counts.append(dict(symbol=symbol, observed=int(observed.sum()), valid=int(usable.sum()),
                           boundary=int((observed & edge).sum()), missing_window=int((observed & holes).sum()),
                           irregular_window=int((observed & irregular).sum())))
        # Sensibilidad descriptiva: aceptar cierres irregulares, manteniendo huecos y borde.
        loose = 100*np.sqrt(np.log(g.close).diff().pow(2).rolling(24, min_periods=24).sum().shift(-24))
        sensitivity.append(dict(symbol=symbol, strict_n=len(x), relaxed_n=int(loose.notna().sum()),
                                strict_mean=x.mean(), relaxed_mean=loose.mean(), strict_p99=x.quantile(.99),
                                relaxed_p99=loose.quantile(.99), strict_max=x.max(), relaxed_max=loose.max()))
        for field in FIELDS + ['return_1h_pct', 'rv_past_24h_pct']:
            pair = g[[field, 'rv_future_24h_pct']].dropna()
            correlations.append(dict(symbol=symbol, predictor=field, n=len(pair),
                                      pearson=pair.iloc[:,0].corr(pair.iloc[:,1]),
                                      spearman=pair.iloc[:,0].corr(pair.iloc[:,1], method='spearman')))
        peak = x.idxmax()
        extremes.append(dict(symbol=symbol, anchor_open_time=peak.isoformat(),
                              prediction_time=(peak+pd.Timedelta(hours=1)).isoformat(),
                              last_future_open_time=(peak+pd.Timedelta(hours=24)).isoformat(), max=x.max()))
        rows = g.loc[observed, ['rv_future_24h_pct','rv_past_24h_pct','return_1h_pct']].copy()
        rows['symbol'] = symbol
        rows['reason'] = np.select([edge.loc[observed], holes.loc[observed], irregular.loc[observed]],
                                   ['boundary','missing_window','irregular_window'], default='valid')
        output.append(rows.reset_index())
    TABLES.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)
    tables = {'summary':summaries, 'coverage':counts, 'correlations':correlations,
              'extremes':extremes, 'sensitivity':sensitivity}
    for name, rows in tables.items():
        pd.DataFrame(rows).to_csv(TABLES / f'eda_target_{name}.csv', index=False)
    pd.concat(output, ignore_index=True).to_csv(TABLES / 'eda_target_development_derived.csv', index=False)
    summary = pd.DataFrame(summaries).set_index('symbol')
    plt.rcParams.update({'font.size':10})
    fig, axes = plt.subplots(5, 3, figsize=(15,16), constrained_layout=True)
    for i,symbol in enumerate(SYMBOLS):
        x = groups[symbol].rv_future_24h_pct.dropna()
        axes[i,0].hist(x, bins=70, color='#2563a6', log=True)
        axes[i,0].set_xlabel('Volatilidad futura 24h (%)')
        axes[i,0].set_ylabel('Frecuencia (log)')
        axes[i,1].boxplot(x, vert=False, whis=1.5, flierprops={'marker':'.','markersize':2,'alpha':.3})
        axes[i,1].set_yticks([])
        axes[i,1].set_xlabel('Volatilidad futura 24h (%)')
        axes[i,2].hist(np.log(x[x>0]), bins=70, color='#168575')
        axes[i,2].set_xlabel('ln(volatilidad en %) · diagnóstico')
        for ax in axes[i]: ax.set_title(symbol); ax.grid(alpha=.2)
    fig.suptitle('Variable objetivo · distribución, boxplot y logaritmo · DEVELOPMENT', fontsize=15)
    fig.savefig(FIGURES/'eda_target_distribution.png', dpi=150); plt.close(fig)
    fig, axes = plt.subplots(5, 1, figsize=(14,15), constrained_layout=True)
    for ax,symbol in zip(axes,SYMBOLS):
        y = groups[symbol].rv_future_24h_pct
        ax.plot(y.index, y, linewidth=.45, color='#2563a6')
        ax.plot(y.resample('MS').median(), linewidth=1.8, color='#c2410c', label='Mediana mensual')
        ax.set_title(symbol); ax.set_ylabel('Volatilidad 24h (%)'); ax.grid(alpha=.2); ax.legend()
    fig.suptitle('Evolución temporal · fecha de apertura de la vela de referencia (UTC)', fontsize=15)
    fig.savefig(FIGURES/'eda_target_time.png', dpi=150); plt.close(fig)
    corr = pd.DataFrame(correlations)
    fig,axes = plt.subplots(1,2,figsize=(14,8),constrained_layout=True)
    for ax,kind in zip(axes,['pearson','spearman']):
        pivot = corr.pivot(index='predictor',columns='symbol',values=kind).reindex(index=FIELDS+['return_1h_pct','rv_past_24h_pct'],columns=SYMBOLS)
        im=ax.imshow(pivot,vmin=-1,vmax=1,cmap='RdBu_r',aspect='auto')
        ax.set_xticks(range(5),SYMBOLS,rotation=40,ha='right')
        ax.set_yticks(range(len(pivot)),pivot.index)
        for row in range(len(pivot)):
            for col in range(5):
                val=pivot.iloc[row,col]
                ax.text(col,row,f'{val:.2f}',ha='center',va='center',color='white' if abs(val)>.6 else 'black',fontsize=9)
        ax.set_title(kind.capitalize())
    fig.colorbar(im,ax=axes,label='Correlación descriptiva con la volatilidad futura')
    fig.savefig(FIGURES/'eda_target_correlations.png',dpi=150);plt.close(fig)
    # Relaciones conjuntas en escala original: todos los pares, sin muestreo.
    fig,axes=plt.subplots(5,2,figsize=(13,15),constrained_layout=True)
    for i,symbol in enumerate(SYMBOLS):
        g=groups[symbol]
        for j,field in enumerate(['rv_past_24h_pct','quote_asset_volume']):
            pair=g[[field,'rv_future_24h_pct']].dropna()
            axes[i,j].hexbin(pair[field],pair.rv_future_24h_pct,gridsize=45,bins='log',mincnt=1,cmap='Blues')
            axes[i,j].set_title(symbol)
            axes[i,j].set_xlabel('Volatilidad pasada 24h (%)' if j==0 else 'Volumen actual (USDT)')
            axes[i,j].set_ylabel('Volatilidad futura 24h (%)')
    fig.suptitle('Densidad conjunta · color más oscuro: mayor número de pares',fontsize=15)
    fig.savefig(FIGURES/'eda_target_relations.png',dpi=150);plt.close(fig)
    assert fingerprint == hashlib.sha256(SOURCE.read_bytes()).hexdigest()
    (TABLES/'eda_target_metadata.json').write_text(json.dumps({
        'source':'data/splits/development_80.csv','sha256':fingerprint,
        'target':'100 * sqrt(sum(log(C[t+j]/C[t+j-1])**2 for j in 1..24))',
        'anchor':'open_time de la vela t; predicción tras su cierre',
        'conventional_close':'open_time + 1h - 1ms',
        'window':'25 cierres horarios consecutivos convencionales; sin imputación',
        'annualized':False,'models_executed':False,'test_read':False,
        'versions':{'pandas':pd.__version__,'numpy':np.__version__,'matplotlib':matplotlib.__version__}
    },ensure_ascii=False,indent=2),encoding='utf-8')
    print('COBERTURA\n',pd.DataFrame(counts).to_string(index=False))
    print('\nRESUMEN\n',summary.to_string())
    print('\nSENSIBILIDAD\n',pd.DataFrame(sensitivity).to_string(index=False))
    print('\nEXTREMOS\n',pd.DataFrame(extremes).to_string(index=False))
    return tables


if __name__ == '__main__':
    main()
