"""Cambio de media retrospectivo y eventos; solo DEVELOPMENT, sin reentrenar SVR."""
from pathlib import Path
import hashlib, json, runpy
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from statsmodels.stats.multitest import multipletests

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs/tables'


def split_mean(values, minimum=90):
    x=np.asarray(values,dtype=float)
    if x.ndim!=1 or not np.isfinite(x).all() or len(x)<2*minimum:
        raise ValueError('Se requiere serie finita y dos segmentos suficientes')
    centered=x-x.mean(); cumulative=np.r_[0.,centered.cumsum()]
    candidates=np.arange(minimum,len(x)-minimum+1)
    gain=cumulative[candidates]**2*(1/candidates+1/(len(x)-candidates))
    index=int(candidates[np.argmax(gain)])
    total=np.sum(centered**2)
    return index,float(gain.max()/total) if total>0 else 0.


def circular_sample(x, length, rng):
    x=np.asarray(x); n=len(x)
    starts=rng.integers(n,size=int(np.ceil(n/length)))
    return x[((starts[:,None]+np.arange(length))%n).ravel()[:n]]


def main():
    cfg=json.loads((OUT/'change_events_protocol.json').read_text(encoding='utf-8'))
    source=ROOT/'data/splits/development_80.csv'
    digest=hashlib.sha256(source.read_bytes()).hexdigest()
    df=pd.read_csv(source,usecols=['symbol','open_time','close_time','close'])
    for col in ['open_time','close_time']:df[col]=pd.to_datetime(df[col],utc=True,format='ISO8601')
    grid=pd.date_range(df.open_time.min(),df.open_time.max(),freq='h')
    longest=runpy.run_path(str(ROOT/'src/14_temporal_target.py'))['longest']
    rows,sensitivity,events,dailies=[],[],[],[]
    for symbol,g in df.groupby('symbol'):
        assert not g.open_time.duplicated().any()
        g=g.set_index('open_time').reindex(grid)
        c=g.close.where(g.close_time.eq(grid+pd.Timedelta(hours=1)-pd.Timedelta(milliseconds=1)) & g.close.gt(0))
        y=100*np.log(c).diff().rolling(24,min_periods=24).std(ddof=0).shift(-24)
        daily=y.loc[y.index.hour==23].copy()
        daily.index=daily.index+pd.Timedelta(hours=1)
        assert daily.index.to_series().diff().dropna().eq(pd.Timedelta(days=1)).all()
        segment=longest(daily); x=segment.to_numpy(); n=len(x)
        dailies.append(pd.DataFrame({'day':daily.index,'symbol':symbol,'volatility':daily.values}))
        k,stat=split_mean(x,cfg['minimum_segment_days'])
        left,right=x[:k].mean(),x[k:].mean()
        for minimum in cfg['minimum_segment_sensitivity_days']:
            s,gain=split_mean(x,minimum)
            sensitivity.append(dict(symbol=symbol,minimum_days=minimum,candidate_day=str(segment.index[s]),gain_ratio=gain))
        for block in cfg['bootstrap_blocks_days']:
            rng=np.random.default_rng(cfg['seed'])
            null=[]; locations=[]
            for _ in range(cfg['bootstrap_reps']):
                # H0 aproximada: media constante, dependencia local por bloques.
                _,gain=split_mean(circular_sample(x-x.mean(),block,rng),cfg['minimum_segment_days'])
                null.append(gain)
                simulated=np.r_[left+circular_sample(x[:k]-left,block,rng),
                                right+circular_sample(x[k:]-right,block,rng)]
                location,_=split_mean(simulated,cfg['minimum_segment_days'])
                locations.append(location)
            lo,hi=np.quantile(locations,[.025,.975],method='nearest').astype(int)
            rows.append(dict(symbol=symbol,n=n,segment_start=str(segment.index.min()),segment_end=str(segment.index.max()),
                candidate_day=str(segment.index[k]),mean_before=left,mean_after=right,delta=right-left,
                gain_ratio=stat,block_days=block,p_raw=(1+np.sum(np.array(null)>=stat))/(len(null)+1),
                conditional_low=str(segment.index[lo]),conditional_high=str(segment.index[hi])))
        fig,axes=plt.subplots(2,2,figsize=(15,8),constrained_layout=True)
        axes[0,0].plot(segment.index,x,lw=.65)
        axes[0,0].hlines([left,right],[segment.index[0],segment.index[k]], [segment.index[k-1],segment.index[-1]],color='tab:red')
        axes[0,0].axvline(segment.index[k],color='black',ls='--');axes[0,0].set_title('Candidato de cambio de media; tramo diario continuo')
        for event,axis in zip(cfg['events'],[axes[0,1],axes[1,0],axes[1,1]]):
            date=pd.Timestamp(event['date'],tz='UTC'); width=cfg['event_window_days']
            before=daily.reindex(pd.date_range(date-pd.Timedelta(days=width),periods=width,freq='D'))
            after=daily.reindex(pd.date_range(date+pd.Timedelta(days=1),periods=width,freq='D'))
            events.append(dict(symbol=symbol,event=event['name'],date=event['date'],source=event['source'],
                n_before=int(before.notna().sum()),n_after=int(after.notna().sum()),mean_before=before.mean(),
                mean_after=after.mean(),median_before=before.median(),median_after=after.median(),
                delta=after.mean()-before.mean()))
            around=daily.reindex(pd.date_range(date-pd.Timedelta(days=width),date+pd.Timedelta(days=width),freq='D'))
            axis.plot(np.arange(-width,width+1),around,marker='.',lw=.8)
            axis.axvline(0,color='black',ls='--');axis.set_title(event['name'],fontsize=9)
            axis.set_xlabel('Días respecto de la fecha documentada')
        for axis in axes.flat:axis.set_ylabel('Volatilidad (%)')
        axes[0,0].tick_params(axis='x',rotation=25,labelsize=8)
        fig.suptitle(symbol+' · diagnósticos retrospectivos; no atribución causal')
        fig.savefig(ROOT/f'book/_static/figures/change_events_{symbol}.png',dpi=120);plt.close(fig)
        print('Cambios y eventos:',symbol,flush=True)
    result=pd.DataFrame(rows)
    for block,group in result.groupby('block_days'):
        result.loc[group.index,'p_holm']=multipletests(group.p_raw,method='holm')[1]
    result.to_csv(OUT/'change_points.csv',index=False)
    pd.DataFrame(sensitivity).to_csv(OUT/'change_points_sensitivity.csv',index=False)
    pd.DataFrame(events).to_csv(OUT/'event_windows.csv',index=False)
    pd.concat(dailies,ignore_index=True).to_csv(OUT/'daily_nonoverlapping_volatility.csv',index=False)
    assert digest==hashlib.sha256(source.read_bytes()).hexdigest()
    (OUT/'change_events_metadata.json').write_text(json.dumps(dict(development_sha256=digest,test_read=False,
        protocol_sha256=hashlib.sha256((OUT/'change_events_protocol.json').read_bytes()).hexdigest(),
        model_changed=False,events=cfg['events'],bootstrap_reps=cfg['bootstrap_reps'],seed=cfg['seed']),indent=2,ensure_ascii=False),encoding='utf-8')


if __name__=='__main__':main()
