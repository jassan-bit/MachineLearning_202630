"""EDA bidimensional: close y derivados; volatilidad centrada ddof=0; solo DEVELOPMENT."""
from pathlib import Path
import hashlib
import itertools
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import sklearn
from sklearn.feature_selection import mutual_info_regression

ROOT = Path(__file__).resolve().parents[1]
SYMBOLS = ['BTCUSDT','ETHUSDT','BNBUSDT','XRPUSDT','SOLUSDT']
FEATURES = ['close','close_lag_1h','close_lag_24h','return_1h_pct','abs_return_1h_pct','sigma_past_24h_pct']
TARGET = 'sigma_future_24h_pct'


def block_means(values, length, reps=1999, seed=42):
    """Bootstrap circular pareado sobre calendario completo, sin comprimir huecos."""
    n = len(values)
    good = np.isfinite(values).all(axis=1)
    filled = np.where(good[:,None], values, 0.)
    extended = np.concatenate([filled, filled[:length]], axis=0)
    mask = np.concatenate([good.astype(int), good[:length].astype(int)])
    cs = np.vstack([np.zeros((1,values.shape[1])), extended.cumsum(axis=0)])
    cc = np.r_[0,mask.cumsum()]
    sums = cs[length:length+n]-cs[:n]
    counts = cc[length:length+n]-cc[:n]
    rng = np.random.default_rng(seed)
    full, remainder = divmod(n,length)
    starts = rng.integers(n,size=(reps,full))
    total = sums[starts].sum(axis=1); denominator = counts[starts].sum(axis=1)
    if remainder:
        tail = rng.integers(n,size=reps)
        total += cs[tail+remainder]-cs[tail]
        denominator += cc[tail+remainder]-cc[tail]
    assert (denominator>0).all()
    return total/denominator[:,None]


def main():
    source=ROOT/'data/splits/development_80.csv'
    fingerprint=hashlib.sha256(source.read_bytes()).hexdigest()
    df=pd.read_csv(source,usecols=['symbol','open_time','close_time','close'])
    for field in ['open_time','close_time']: df[field]=pd.to_datetime(df[field],format='ISO8601',utc=True)
    start=pd.Timestamp('2020-08-11 06:00',tz='UTC');end=pd.Timestamp('2025-07-01 18:00',tz='UTC')
    assert len(df)==214165 and set(df.symbol)==set(SYMBOLS) and df.open_time.between(start,end).all()
    assert not df.duplicated(['symbol','open_time']).any()
    assert np.isfinite(df.close).all() and df.close.gt(0).all()
    grid=pd.date_range(start,end,freq='h',name='anchor_open_time')
    tables=ROOT/'outputs/tables';figures=ROOT/'book/_static/figures'
    tables.mkdir(parents=True,exist_ok=True);figures.mkdir(parents=True,exist_ok=True)
    groups={}; associations=[];vifs=[];coverage=[]; matrices=[]
    for symbol in SYMBOLS:
        g=df.loc[df.symbol.eq(symbol)].set_index('open_time').reindex(grid)
        valid=g.close.notna() & g.close_time.eq(grid+pd.Timedelta(hours=1)-pd.Timedelta(milliseconds=1))
        c=g.close.where(valid);r=np.log(c).diff()
        x=pd.DataFrame({'close':c,'close_lag_1h':c.shift(1),'close_lag_24h':c.shift(24),
                        'return_1h_pct':100*r,'abs_return_1h_pct':100*r.abs(),
                        'sigma_past_24h_pct':100*r.rolling(24,min_periods=24).std(ddof=0),
                        TARGET:100*r.rolling(24,min_periods=24).std(ddof=0).shift(-24)},index=grid)
        forward=valid.astype(int).rolling(25).sum().shift(-24).eq(25)
        x[TARGET]=x[TARGET].where(forward)
        # Comprobación independiente de cada objetivo contra los 24 retornos futuros.
        windows=np.lib.stride_tricks.sliding_window_view(c.to_numpy(),25)
        direct=100*np.std(np.diff(np.log(windows),axis=1),axis=1,ddof=0)
        good=x[TARGET].iloc[:-24].notna().to_numpy()
        assert np.allclose(x[TARGET].iloc[:-24].to_numpy()[good],direct[good],rtol=1e-9,atol=1e-10)
        assert x[TARGET].iloc[-24:].isna().all()
        common=x.dropna();groups[symbol]=x
        coverage.append(dict(symbol=symbol,observed=int(g.close.notna().sum()),valid_target=int(x[TARGET].notna().sum()),complete_pairs=len(common)))
        # Escala auxiliar para estimación MI/VIF, sin crear un preprocesador de modelado.
        z=(common[FEATURES]-common[FEATURES].mean())/common[FEATURES].std(ddof=0)
        mi=mutual_info_regression(z,common[TARGET],n_neighbors=5,random_state=42)
        for j,field in enumerate(FEATURES):
            associations.append(dict(symbol=symbol,predictor=field,n=len(common),
                pearson=common[field].corr(common[TARGET]),spearman=common[field].corr(common[TARGET],method='spearman'),mi_nats=mi[j]))
            others=np.column_stack([np.ones(len(z)),z.drop(columns=field)])
            residual=z[field].to_numpy()-others@np.linalg.lstsq(others,z[field],rcond=None)[0]
            ratio=np.dot(residual,residual)/np.dot(z[field],z[field])
            vifs.append(dict(symbol=symbol,predictor=field,n=len(common),vif=1/ratio if ratio>1e-14 else np.inf))
        for method in ['pearson','spearman']:
            corr=common.corr(method=method)
            for a in corr:
                for b in corr: matrices.append(dict(symbol=symbol,method=method,row=a,column=b,value=corr.loc[a,b]))
        fig,axes=plt.subplots(2,3,figsize=(14,8),constrained_layout=True)
        for ax,field in zip(axes.flat,FEATURES):
            im=ax.hexbin(common[field],common[TARGET],gridsize=40,bins='log',mincnt=1,cmap='Blues')
            ax.set_xlabel(field);ax.set_ylabel('Volatilidad futura (%)');fig.colorbar(im,ax=ax,label='Pares (log)')
        fig.suptitle(f'{symbol} · close y derivados frente al objetivo · DEVELOPMENT')
        fig.savefig(figures/f'bivariate_{symbol}.png',dpi=130);plt.close(fig)
    assoc=pd.DataFrame(associations);vif=pd.DataFrame(vifs)
    assoc.to_csv(tables/'bivariate_associations.csv',index=False)
    vif.to_csv(tables/'bivariate_vif.csv',index=False)
    pd.DataFrame(coverage).to_csv(tables/'bivariate_coverage.csv',index=False)
    pd.DataFrame(matrices).to_csv(tables/'bivariate_correlations.csv',index=False)
    aligned=pd.DataFrame({s:groups[s][TARGET] for s in SYMBOLS})
    common=aligned.dropna(); common.to_csv(tables/'bivariate_target_aligned.csv')
    descriptions=common.describe(percentiles=[.25,.5,.75]).T
    descriptions.to_csv(tables/'bivariate_group_summary.csv')
    comparisons=[]
    for length in [24,168,336]:
        means=block_means(aligned.to_numpy(),length)
        rows=[]
        for i,j in itertools.combinations(range(5),2):
            d=common.iloc[:,i]-common.iloc[:,j];estimate=d.mean()
            boot=means[:,i]-means[:,j]
            p=(1+np.count_nonzero(np.abs(boot-estimate)>=abs(estimate)))/(len(boot)+1)
            rows.append(dict(block_hours=length,asset_a=SYMBOLS[i],asset_b=SYMBOLS[j],n=len(d),
                difference_pp=estimate,paired_effect=estimate/d.std(ddof=1),ci_low=np.quantile(boot,.025),
                ci_high=np.quantile(boot,.975),p_raw=p))
        # Holm en la familia de diez comparaciones de cada longitud de bloque.
        order=np.argsort([a['p_raw'] for a in rows]);adjusted=np.maximum.accumulate([(10-k)*rows[idx]['p_raw'] for k,idx in enumerate(order)])
        for idx,p in zip(order,adjusted):rows[idx]['p_holm']=min(1.,p)
        comparisons.extend(rows)
    pd.DataFrame(comparisons).to_csv(tables/'bivariate_group_comparisons.csv',index=False)
    fig,ax=plt.subplots(figsize=(10,5),constrained_layout=True)
    ax.boxplot([common[s] for s in SYMBOLS],tick_labels=SYMBOLS,flierprops={'marker':'.','markersize':2,'alpha':.2})
    ax.set_ylabel('Volatilidad futura 24h (%) · desviación ddof=0')
    ax.set_title(f'Activo y volatilidad · {len(common):,} timestamps comunes');ax.grid(alpha=.2)
    fig.savefig(figures/'bivariate_groups.png',dpi=150);plt.close(fig)
    for method in ['pearson','spearman']:
        fig,axes=plt.subplots(2,3,figsize=(17,11),constrained_layout=True)
        for ax,symbol in zip(axes.flat,SYMBOLS):
            corr=groups[symbol].dropna().corr(method=method)
            im=ax.imshow(corr,vmin=-1,vmax=1,cmap='RdBu_r')
            labels=['close','lag 1h','lag 24h','retorno','|retorno|','sigma pasada','sigma futura']
            ax.set_xticks(range(7),labels,rotation=65,ha='right');ax.set_yticks(range(7),labels);ax.set_title(symbol)
            for i in range(7):
                for j in range(7):ax.text(j,i,f'{corr.iloc[i,j]:.2f}',ha='center',va='center',fontsize=7,color='white' if abs(corr.iloc[i,j])>.6 else 'black')
        axes.flat[-1].axis('off');fig.colorbar(im,ax=list(axes.flat),shrink=.6)
        fig.suptitle(f'Correlaciones {method} · muestra completa por activo')
        fig.savefig(figures/f'bivariate_heatmap_{method}.png',dpi=140);plt.close(fig)
    assert fingerprint==hashlib.sha256(source.read_bytes()).hexdigest()
    (tables/'bivariate_metadata.json').write_text(json.dumps({'source':'data/splits/development_80.csv','sha256':fingerprint,
        'target':'100 * std(r[t+1],...,r[t+24], ddof=0); t etiqueta la apertura de la última vela cerrada',
        'forecast_time':'t + 1 hora; datos disponibles después del cierre',
        'bootstrap':{'reps':1999,'seed':42,'main_block_hours':168,'sensitivity':[24,336],
                     'method':'Circular pareado sobre calendario; medias solo sobre filas comunes válidas'},
        'MI':{'neighbors':5,'seed':42,'units':'nats'},'target_ddof':0,'test_read':False,
        'versions':{'pandas':pd.__version__,'numpy':np.__version__,'sklearn':sklearn.__version__,'matplotlib':matplotlib.__version__}},ensure_ascii=False,indent=2),encoding='utf-8')
    print(pd.DataFrame(coverage).to_string(index=False))
    print(assoc.to_string(index=False));print(vif.to_string(index=False))
    print(pd.DataFrame(comparisons).query('block_hours == 168').to_string(index=False))
    return assoc


if __name__=='__main__':main()
