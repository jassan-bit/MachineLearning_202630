"""Componente temporal de close, solo DEVELOPMENT; no entrenamiento predictivo."""
from pathlib import Path
import hashlib,json,warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import statsmodels
from statsmodels.tsa.stattools import adfuller,kpss,acf,pacf
from statsmodels.tsa.seasonal import STL
ROOT = Path(__file__).resolve().parents[1]
def longest(s):
    valid=s.notna(); groups=(valid!=valid.shift()).cumsum()
    candidates=[v for _,v in s[valid].groupby(groups[valid])]
    return max(candidates,key=len)
def main():
    source=ROOT/'data/splits/development_80.csv';sha=hashlib.sha256(source.read_bytes()).hexdigest()
    df=pd.read_csv(source,usecols=['symbol','open_time','close_time','close'])
    for col in ['open_time','close_time']:df[col]=pd.to_datetime(df[col],utc=True,format='ISO8601')
    assert df.open_time.notna().all() and df.close_time.notna().all()
    assert df.open_time.eq(df.open_time.dt.floor('h')).all()
    grid=pd.date_range(df.open_time.min(),df.open_time.max(),freq='h')
    out=ROOT/'outputs/tables';figdir=ROOT/'book/_static/figures'
    audits=[];tests=[];cross=[];drift=[];season=[];series=[];acfrows=[]
    for symbol,g in df.groupby('symbol'):
        duplicates=int(g.duplicated('open_time').sum());assert duplicates==0
        g=g.set_index('open_time').reindex(grid)
        regular=g.close_time.eq(grid+pd.Timedelta(hours=1)-pd.Timedelta(milliseconds=1))
        c=g.close.where(regular);r=np.log(c).diff()
        target=100*r.rolling(24,min_periods=24).std(ddof=0).shift(-24)
        daily=c.resample('D').last().where(c.resample('D').count().eq(24))
        seg=longest(daily);logseg=np.log(seg)
        audits.append(dict(symbol=symbol,observed=int(g.close.notna().sum()),expected=len(grid),missing=int(g.close.isna().sum()),irregular_closes=int((g.close.notna()&~regular).sum()),duplicates=duplicates,coverage_pct=100*g.close.notna().mean(),daily_segment_start=str(seg.index.min()),daily_segment_end=str(seg.index.max()),daily_segment_n=len(seg)))
        for name,s in [('log_close',logseg),('daily_log_return',logseg.diff().dropna())]:
            a=adfuller(s,maxlag=14,regression='c',autolag='AIC')
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter('always');k=kpss(s,regression='c',nlags='auto')
            tests.append(dict(symbol=symbol,series=name,n=len(s),adf_stat=a[0],adf_p=a[1],adf_lags=a[2],kpss_stat=k[0],kpss_p=k[1],kpss_lags=k[2],kpss_warning=' | '.join(str(v.message) for v in w)))
        stl=STL(logseg,period=7,robust=True).fit()
        strength=max(0,1-np.var(stl.resid)/np.var(stl.seasonal+stl.resid))
        season.append(dict(symbol=symbol,weekly_strength=strength,period_days=7,n=len(seg)))
        for lag in [0,1,6,12,24,48,168]:
            pairs=pd.DataFrame({'x':c.shift(lag),'y':target}).dropna()
            cross.append(dict(symbol=symbol,lag_hours=lag,n=len(pairs),pearson=pairs.x.corr(pairs.y),spearman=pairs.x.corr(pairs.y,method='spearman')))
        for year in sorted(set(grid.year)):
            for name,s in [('close',c),('target_pct',target)]:
                v=s[s.index.year==year].dropna()
                drift.append(dict(symbol=symbol,year=year,variable=name,n=len(v),mean=v.mean(),std=v.std(),q10=v.quantile(.1),median=v.median(),q90=v.quantile(.9)))
        series.append(pd.DataFrame({'symbol':symbol,'time':daily.index,'daily_close':daily.to_numpy()}))
        fig,ax=plt.subplots(4,3,figsize=(18,18),constrained_layout=True)
        a=ax.flat
        a[0].plot(c.index,c,lw=.35);a[0].set_title('Serie horaria · close (USDT)')
        a[1].plot(daily.index,daily,lw=.5,label='Diario, 24 cierres válidos')
        for freq,label in [('W-SUN','Media semanal'),('MS','Media mensual')]:
            v=c.resample(freq).mean();a[1].plot(v.index,v,label=label,lw=1)
        a[1].legend(fontsize=7);a[1].set_title('Agregaciones descriptivas · USDT')
        for axis,key,labels,title in [(a[2],c.index.hour,range(24),'Hora UTC'),(a[3],c.index.dayofweek,range(7),'Día: 0=lunes'),(a[4],c.index.month,range(1,13),'Mes')]:
            axis.boxplot([c[key==k].dropna() for k in labels],tick_labels=list(labels),showfliers=False)
            axis.set_title('close por '+title+' · sin puntos extremos');axis.tick_params(axis='x',labelsize=7)
        a[5].plot(logseg.index,logseg,label='log close',lw=.5);a[5].plot(logseg.index,stl.trend,label='Tendencia STL');a[5].legend();a[5].set_title('STL semanal · tramo diario continuo')
        a[6].plot(logseg.index,stl.seasonal,label='Estacional',lw=.6);a[6].plot(logseg.index,stl.resid,label='Residuo',lw=.5,alpha=.6);a[6].legend();a[6].set_title('STL · unidades logarítmicas')
        a[7].plot(daily.index,daily.rolling(30,min_periods=30).mean(),label='Media 30 días')
        a[7].set_ylabel('USDT');other=a[7].twinx();other.plot(daily.index,daily.rolling(30,min_periods=30).var(),color='orange',label='Varianza');other.set_ylabel('Varianza USDT²');a[7].set_title('Media (azul) y varianza (naranja), hacia atrás')
        for s,label in [(logseg,'log close'),(logseg.diff().dropna(),'retorno diario')]:
            ac=acf(s,nlags=30,fft=True);pc=pacf(s,nlags=30,method='ywm')
            a[8].plot(range(31),ac,label=label);a[9].plot(range(31),pc,label=label)
            acfrows.extend(dict(symbol=symbol,series=label,lag_days=j,acf=ac[j],pacf=pc[j]) for j in range(31))
        a[8].set_title('ACF · rezagos diarios');a[9].set_title('PACF · rezagos diarios');a[8].legend();a[9].legend()
        own=pd.DataFrame(cross).query('symbol == @symbol');a[10].plot(own.lag_hours,own.pearson,marker='o',label='Pearson');a[10].plot(own.lag_hours,own.spearman,marker='o',label='Spearman');a[10].legend();a[10].set_title('close(s-k) vs volatilidad futura(s) · k horas')
        years=sorted(set(grid.year));a[11].boxplot([target[target.index.year==y].dropna() for y in years],tick_labels=years,showfliers=False);a[11].set_title('Objetivo por año · % · años extremos parciales')
        for axis in a:axis.grid(alpha=.15);axis.tick_params(axis='x',rotation=35)
        fig.suptitle(symbol+' · componente temporal · DEVELOPMENT',fontsize=17)
        fig.savefig(figdir/f'temporal_{symbol}.png',dpi=120);plt.close(fig)
    for name,rows in [('audit',audits),('stationarity',tests),('cross_lags',cross),('drift',drift),('seasonality',season),('acf',acfrows)]:pd.DataFrame(rows).to_csv(out/f'temporal_{name}.csv',index=False)
    pd.concat(series).to_csv(out/'temporal_daily.csv',index=False)
    assert sha==hashlib.sha256(source.read_bytes()).hexdigest()
    (out/'temporal_metadata.json').write_text(json.dumps({'source':str(source.relative_to(ROOT)),'sha256':sha,'start':str(grid.min()),'end':str(grid.max()),'test_read':False,'statsmodels':statsmodels.__version__,'target_ddof':0,'imputation':False},indent=2),encoding='utf-8')
    print(pd.DataFrame(audits).to_string(index=False));print(pd.DataFrame(tests).drop(columns='kpss_warning').to_string(index=False));print(pd.DataFrame(season).to_string(index=False))
if __name__=='__main__':main()
