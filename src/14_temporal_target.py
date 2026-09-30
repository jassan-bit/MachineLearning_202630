"""EDA horario del objetivo definitivo; retrospectivo, nunca una feature."""
from pathlib import Path
import hashlib
import json
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import acf, pacf, adfuller, kpss
from statsmodels.tsa.seasonal import STL

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'outputs/tables'
FIG = ROOT/'book/_static/figures'


def longest(series):
    valid = series.notna()
    groups = valid.ne(valid.shift()).cumsum()
    return max((g for _, g in series.loc[valid].groupby(groups.loc[valid])), key=len)


def main():
    source = ROOT/'data/splits/development_80.csv'
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    df = pd.read_csv(source, usecols=['symbol','open_time','close_time','close'])
    for col in ['open_time','close_time']:
        df[col] = pd.to_datetime(df[col], utc=True, format='ISO8601')
    grid = pd.date_range(df.open_time.min(), df.open_time.max(), freq='h')
    tests, correlations, periods, cycles, decompositions, audits = [], [], [], [], [], []
    for symbol, g in df.groupby('symbol'):
        assert not g.open_time.duplicated().any()
        g = g.set_index('open_time').reindex(grid)
        c = g.close.where(g.close_time.eq(grid+pd.Timedelta(hours=1)-pd.Timedelta(milliseconds=1)) & g.close.gt(0))
        returns = 100*np.log(c).diff()
        y = returns.rolling(24, min_periods=24).std(ddof=0).shift(-24)
        segment = longest(y)
        assert segment.index.to_series().diff().dropna().eq(pd.Timedelta(hours=1)).all()
        audits.append(dict(symbol=symbol, total_valid=int(y.notna().sum()),
                           segment_n=len(segment), segment_start=str(segment.index.min()),
                           segment_end=str(segment.index.max())))
        a = adfuller(segment, maxlag=48, regression='c', autolag='AIC', result_object=False)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            k = kpss(segment, regression='c', nlags='auto', result_object=False)
        tests.append(dict(symbol=symbol, n=len(segment), adf_stat=a[0], adf_p=a[1], adf_lags=a[2],
                          kpss_stat=k[0], kpss_p=k[1], kpss_lags=k[2],
                          kpss_warning=' | '.join(str(w.message) for w in caught)))
        auto = acf(segment, nlags=168, fft=True)
        partial = pacf(segment, nlags=168, method='ldbiased')
        absseg, sqseg = longest(returns.abs()), longest(returns**2)
        absacf, sqacf = acf(absseg, nlags=168, fft=True), acf(sqseg, nlags=168, fft=True)
        for lag in range(1,169):
            correlations.append(dict(symbol=symbol, lag_hours=lag, target_acf=auto[lag],
                target_pacf=partial[lag], abs_return_acf=absacf[lag], squared_return_acf=sqacf[lag],
                return_segment_start=str(absseg.index.min()), return_segment_n=len(absseg)))
        issued = y.copy()
        issued.index = issued.index+pd.Timedelta(hours=1)
        for name, keys in [('hour',issued.index.hour),('weekday',issued.index.dayofweek),('month',issued.index.month)]:
            for group, values in issued.groupby(keys):
                v = values.dropna()
                cycles.append(dict(symbol=symbol, cycle=name, group=int(group), n=len(v),
                                   median=v.median(), mean=v.mean(), std=v.std(), q1=v.quantile(.25), q3=v.quantile(.75)))
        for quarter, values in issued.groupby(issued.index.strftime('%Y')+'Q'+issued.index.quarter.astype(str)):
            v = values.dropna()
            periods.append(dict(symbol=symbol, quarter=quarter, n=len(v), median=v.median(),
                                mean=v.mean(), std=v.std(), p05=v.quantile(.05), p95=v.quantile(.95)))
        fig, axes = plt.subplots(4,3,figsize=(17,16), constrained_layout=True)
        axes[0,0].plot(y.index,y,lw=.3); axes[0,0].set_title('Objetivo horario completo (%)')
        for rule, label in [('D','Día'),('W','Semana'),('MS','Mes')]:
            agg = issued.resample(rule).mean()
            axes[0,1].plot(agg.index,agg,lw=.7,label=label)
            for t, value in agg.items():
                decompositions.append(dict(symbol=symbol, kind='aggregate_'+rule, time=str(t), value=value))
        axes[0,1].set_title('Promedio del objetivo por periodo'); axes[0,1].legend(fontsize=7)
        axes[0,2].plot(y.index,y.rolling(168,min_periods=168).mean(),lw=.6,color='tab:blue')
        axes[0,2].set_ylabel('Media (pp)',color='tab:blue')
        variance_axis=axes[0,2].twinx()
        variance_axis.plot(y.index,y.rolling(168,min_periods=168).var(ddof=0),lw=.6,color='tab:orange')
        variance_axis.set_ylabel('Varianza (pp²)',color='tab:orange')
        axes[0,2].set_title('Momentos móviles de y: 168 horas')
        for axis, keys, title in zip(axes[1], [issued.index.hour,issued.index.dayofweek,issued.index.month], ['Hora UTC de emisión','Día semanal: lunes=0','Mes de emisión']):
            grouped = list(issued.groupby(keys))
            axis.boxplot([v.dropna().values for _,v in grouped], tick_labels=[str(i) for i,_ in grouped], showfliers=False)
            axis.set_title(title+' · y (%)')
        axes[2,0].plot(range(169),auto,label='ACF');axes[2,0].plot(range(169),partial,label='PACF')
        axes[2,0].set_title('Objetivo: ACF/PACF horarias');axes[2,0].legend(fontsize=7)
        axes[2,1].plot(range(169),absacf,label='|Retorno|');axes[2,1].plot(range(169),sqacf,label='Retorno²')
        axes[2,1].set_title('ACF de retornos transformados');axes[2,1].legend(fontsize=7)
        q = pd.DataFrame(periods).query('symbol == @symbol')
        axes[2,2].plot(q.quarter,q['median'],marker='.',label='Mediana');axes[2,2].plot(q.quarter,q.p95,label='P95')
        axes[2,2].tick_params(axis='x',rotation=90,labelsize=6);axes[2,2].set_title('Distribución trimestral del objetivo');axes[2,2].legend(fontsize=7)
        # Descomposición retrospectiva: no entra en el pipeline predictivo.
        stl = STL(segment,period=24,robust=True,seasonal_jump=3,trend_jump=3,low_pass_jump=3).fit()
        for axis, component, name in zip(axes[3],[stl.trend,stl.seasonal,stl.resid],['Tendencia STL','Estacionalidad STL: periodo 24h','Residuo STL']):
            axis.plot(segment.index,component,lw=.4);axis.set_title(name)
            for t,value in component.items():
                decompositions.append(dict(symbol=symbol,kind=name,time=str(t),value=value))
        for row in [0,3]:
            for axis in axes[row]:
                axis.tick_params(axis='x',rotation=30,labelsize=7)
        fig.suptitle(symbol+' · objetivo futuro 24h · EDA retrospectivo en DEVELOPMENT')
        fig.savefig(FIG/f'temporal_target_{symbol}.png',dpi=120);plt.close(fig)
        print('Temporal del objetivo completado:',symbol,flush=True)
    for name,rows in [('stationarity',tests),('correlations',correlations),('quarters',periods),('cycles',cycles),('components',decompositions),('coverage',audits)]:
        pd.DataFrame(rows).to_csv(OUT/f'temporal_target_{name}.csv',index=False)
    assert digest == hashlib.sha256(source.read_bytes()).hexdigest()
    (OUT/'temporal_target_metadata.json').write_text(json.dumps(dict(development_sha256=digest,
        test_read=False, ddof=0, window_returns=24, forecast_horizon_hours=24,
        segment_selection='longest_uninterrupted_valid_target_run', acf_pacf_lags=168,
        adf_maxlag=48, adf_autolag='AIC', kpss_lags='auto', regression='c',
        stl_period_hours=24, stl_jump=3, future_target_is_feature=False,
        calendar='nominal_issue_time_anchor_plus_1h_UTC'),indent=2),encoding='utf-8')


if __name__ == '__main__':
    main()
