"""SVR lineal vs persistencia; selección y diagnóstico en DEVELOPMENT."""
from pathlib import Path
import json,hashlib,time,warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import jarque_bera,probplot,spearmanr
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVR
from sklearn.model_selection import TimeSeriesSplit
from sklearn.exceptions import ConvergenceWarning
import joblib,sklearn
ROOT = Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs/tables';FIG=ROOT/'book/_static/figures'
def metrics(y,p):
    return dict(rmse=float(np.sqrt(np.mean((y-p)**2))),mape=float(100*np.mean(abs((y-p)/y))) if np.all(y!=0) else np.nan,r2=float(1-np.sum((y-p)**2)/np.sum((y-y.mean())**2)),mae=float(np.mean(abs(y-p))))
def model(C,epsilon):
    return Pipeline([('scale',StandardScaler()),('svr',LinearSVR(C=C,epsilon=epsilon,loss='squared_epsilon_insensitive',dual=False,tol=1e-5,max_iter=20000,random_state=42))])
def fit(m,X,y):
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter('always');m.fit(X,y)
    if any(issubclass(v.category,ConvergenceWarning) for v in w):raise RuntimeError('SVR no convergió')
    return m

def prepare(df,grid,L=168):
    frames={}
    for symbol,g in df.groupby('symbol'):
        assert not g.open_time.duplicated().any()
        g=g.set_index('open_time').reindex(grid)
        valid=g.close_time.eq(grid+pd.Timedelta(hours=1)-pd.Timedelta(milliseconds=1))
        c=g.close.where(valid&np.isfinite(g.close)&g.close.gt(0));r=np.log(c).diff()
        X=pd.DataFrame(np.column_stack([c.shift(k).to_numpy() for k in range(L)]),index=grid,columns=[f'lag_{k}' for k in range(L)])
        X['y']=100*r.rolling(24,min_periods=24).std(ddof=0).shift(-24)
        X['persistence']=100*r.rolling(24,min_periods=24).std(ddof=0)
        frames[symbol]=X
    common=np.logical_and.reduce([v.notna().all(axis=1).to_numpy() for v in frames.values()])
    return {s:v.loc[common] for s,v in frames.items()}

def contained_block(data, start, end, L=168, horizon=24):
    """Confinar entradas, referencia pasada y etiqueta al calendario del bloque.

    Fechas referidas a apertura de vela. La predicción se emite tras cerrar
    el ancla; el objetivo termina tras cerrar la vela ancla+horizon.
    """
    history_hours = max(L-1, 24)  # Persistence necesita 25 cierres.
    first = data.index-pd.Timedelta(hours=history_hours)
    last = data.index+pd.Timedelta(hours=horizon)
    block = data.loc[(first >= start) & (last <= end)]
    if block.empty:
        raise ValueError(f'Bloque sin ventanas completas: {start} a {end}')
    assert (block.index-pd.Timedelta(hours=history_hours) >= start).all()
    assert (block.index+pd.Timedelta(hours=horizon) <= end).all()
    return block

def main():
    started=time.perf_counter();source=ROOT/'data/splits/development_80.csv';sha=hashlib.sha256(source.read_bytes()).hexdigest()
    cfg=json.loads((OUT/'base_model_protocol.json').read_text())
    assert cfg['horizon_hours']==cfg['volatility_returns']==24 and cfg['ddof']==0
    assert cfg['boundary_policy']=='strict_block_containment'
    protocol_sha=hashlib.sha256((OUT/'base_model_protocol.json').read_bytes()).hexdigest()
    df=pd.read_csv(source,usecols=['symbol','open_time','close_time','close'])
    for col in ['open_time','close_time']:df[col]=pd.to_datetime(df[col],utc=True,format='ISO8601')
    grid=pd.date_range(df.open_time.min(),df.open_time.max(),freq='h')
    frames=prepare(df,grid,cfg['window_closes']);symbols=sorted(frames);cols=[f'lag_{k}' for k in range(cfg['window_closes'])]
    folds=list(TimeSeriesSplit(n_splits=cfg['folds']).split(grid));runs=[];preds={};audit=[]
    history_hours=max(cfg['window_closes']-1,24)
    for C in cfg['C']:
      for eps in cfg['epsilon']:
        key=(C,eps);preds[key]=[]
        for f,(tr,va) in enumerate(folds,1):
          start,end=grid[va[0]],grid[va[-1]]
          for symbol in symbols:
            data=frames[symbol]
            train=contained_block(data,grid[tr[0]],grid[tr[-1]],cfg['window_closes'])
            val=contained_block(data,start,end,cfg['window_closes'])
            assert train.index.max()+pd.Timedelta(hours=24)<start
            assert val.index.min()-pd.Timedelta(hours=history_hours)>=start
            m=model(C,eps);tick=time.perf_counter();fit(m,train[cols],train.y);elapsed=time.perf_counter()-tick
            p=m.predict(val[cols]);assert np.isfinite(p).all()
            runs.append(dict(C=C,epsilon=eps,fold=f,symbol=symbol,n_train=len(train),n_val=len(val),fit_seconds=elapsed,n_iter=int(m.named_steps['svr'].n_iter_),**metrics(val.y.to_numpy(),p)))
            preds[key].append(pd.DataFrame({'time':val.index,'symbol':symbol,'fold':f,'y':val.y,'svr':p,'persistence':val.persistence,
                'history_start':val.index-pd.Timedelta(hours=history_hours),
                'target_end':val.index+pd.Timedelta(hours=24),
                'prediction_time':val.index+pd.Timedelta(hours=1),
                'target_available_time':val.index+pd.Timedelta(hours=25)}))
            if key==(cfg['C'][0],cfg['epsilon'][0]):
                candidates=data.loc[(data.index>=start)&(data.index<=end)]
                left=candidates.index-pd.Timedelta(hours=history_hours)<start
                right=candidates.index+pd.Timedelta(hours=24)>end
                audit.append(dict(fold=f,symbol=symbol,
                    train_block_start=str(grid[tr[0]]),train_block_end=str(grid[tr[-1]]),
                    validation_block_start=str(start),validation_block_end=str(end),
                    train_start=str(train.index.min()),train_end=str(train.index.max()),
                    train_history_start=str(train.index.min()-pd.Timedelta(hours=history_hours)),
                    label_end=str(train.index.max()+pd.Timedelta(hours=24)),
                    validation_start=str(val.index.min()),validation_end=str(val.index.max()),
                    validation_history_start=str(val.index.min()-pd.Timedelta(hours=history_hours)),
                    validation_label_end=str(val.index.max()+pd.Timedelta(hours=24)),
                    candidate_validation_rows=len(candidates),excluded_history=int(left.sum()),
                    excluded_target=int((right&~left).sum()),n_train=len(train),n_val=len(val)))
        print('Configuración completada',C,eps,flush=True)
    search=pd.DataFrame(runs);scores=search.groupby(['C','epsilon']).rmse.mean().sort_values();best=tuple(scores.index[0])
    search.to_csv(OUT/'base_search.csv',index=False);scores.rename('mean_rmse').reset_index().to_csv(OUT/'base_selection.csv',index=False)
    pd.DataFrame(audit).to_csv(OUT/'base_folds.csv',index=False)
    oof=pd.concat(preds[best],ignore_index=True);oof.to_csv(OUT/'base_validation_predictions.csv',index=False)
    rows=[];diagnostics=[];coeff=[];learn=[];cis=[]
    for symbol in symbols:
        part=oof.loc[oof.symbol.eq(symbol)].sort_values('time')
        for f,g in part.groupby('fold'):
            for name in ['svr','persistence']:rows.append(dict(symbol=symbol,fold=f,model=name,n=len(g),negative_predictions=int((g[name]<0).sum()),**metrics(g.y.to_numpy(),g[name].to_numpy())))
        # Curva con prefijos cronológicos del entrenamiento del último fold, validación fija.
        tr,va=folds[-1];start,end=grid[va[0]],grid[va[-1]];data=frames[symbol]
        val=contained_block(data,start,end,cfg['window_closes'])
        for fraction in [.25,.5,1.]:
            prefix_end=grid[tr[max(1,int(len(tr)*fraction))-1]]
            sub=contained_block(data,grid[tr[0]],prefix_end,cfg['window_closes'])
            m=fit(model(*best),sub[cols],sub.y)
            learn.append(dict(symbol=symbol,fraction=fraction,n_train=len(sub),
                train_block_start=str(grid[tr[0]]),train_block_end=str(prefix_end),
                history_start=str(sub.index.min()-pd.Timedelta(hours=history_hours)),
                target_end=str(sub.index.max()+pd.Timedelta(hours=24)),
                train_rmse=metrics(sub.y.to_numpy(),m.predict(sub[cols]))['rmse'],validation_rmse=metrics(val.y.to_numpy(),m.predict(val[cols]))['rmse']))
        # Diagnóstico gráfico en el último fold; autocorrelación con rezagos de calendario.
        g=part.loc[part.fold.eq(5)].set_index('time');res=g.y-g.svr
        full=res.reindex(pd.date_range(g.index.min(),g.index.max(),freq='h'))
        correlations=[full.corr(full.shift(k)) for k in range(1,169)]
        jb=jarque_bera(res);rho=spearmanr(abs(res),g.svr).statistic
        diagnostics.append(dict(symbol=symbol,fold=5,mean_residual=res.mean(),std_residual=res.std(),jb_stat=jb.statistic,jb_p_iid=jb.pvalue,abs_residual_fitted_spearman=rho,variance_second_over_first=res.iloc[len(res)//2:].var()/res.iloc[:len(res)//2].var(),acf_1h=correlations[0],acf_24h=correlations[23],acf_168h=correlations[167]))
        fig,axes=plt.subplots(2,3,figsize=(15,8),constrained_layout=True)
        axes[0,0].plot(g.index,g.y,lw=.5,label='Real');axes[0,0].plot(g.index,g.svr,lw=.5,label='SVR');axes[0,0].plot(g.index,g.persistence,lw=.4,alpha=.6,label='Persistencia');axes[0,0].legend();axes[0,0].set_title('Validación fold 5 · volatilidad (%)')
        axes[0,1].plot(g.index,res,lw=.4);axes[0,1].set_title('Residuo = real − SVR')
        probplot(res,dist='norm',plot=axes[0,2]);axes[0,2].set_title('Q-Q normal · descriptivo')
        axes[1,0].scatter(g.svr,res,s=2,alpha=.15);axes[1,0].set_xlabel('Predicción');axes[1,0].set_ylabel('Residuo');axes[1,0].set_title('Dispersión y heterocedasticidad')
        axes[1,1].plot(range(1,169),correlations);axes[1,1].set_title('Autocorrelación residual · horas')
        lr=pd.DataFrame(learn).query('symbol == @symbol');axes[1,2].plot(lr.n_train,lr.train_rmse,marker='o',label='Entrenamiento');axes[1,2].plot(lr.n_train,lr.validation_rmse,marker='o',label='Validación fija');axes[1,2].legend();axes[1,2].set_title('Curva de aprendizaje cronológica · RMSE')
        for axis in axes[0,:2]:axis.tick_params(axis='x',rotation=45,labelsize=7)
        fig.suptitle(symbol+' · SVR lineal vs persistencia · DEVELOPMENT');fig.savefig(FIG/f'base_{symbol}.png',dpi=130);plt.close(fig)
        # Ajuste final DEVELOPMENT sin consultar TEST.
        final_data=contained_block(data,grid.min(),grid.max(),cfg['window_closes'])
        final=fit(model(*best),final_data[cols],final_data.y)
        folder=ROOT/'outputs/models';folder.mkdir(exist_ok=True)
        joblib.dump(final,folder/f'linear_svr_{symbol}.joblib')
        b=final.named_steps['svr'].coef_;sc=final.named_steps['scale']
        coeff.extend(dict(symbol=symbol,lag_hours=k,standardized_coefficient=float(b[k]),coefficient_per_USDT=float(b[k]/sc.scale_[k])) for k in range(len(cols)))
    pd.DataFrame(rows).to_csv(OUT/'base_metrics_by_fold.csv',index=False)
    pd.DataFrame(rows).groupby('model')[['rmse','mape','r2','mae']].agg(['mean','std']).to_csv(OUT/'base_metrics_summary.csv')
    pd.DataFrame(diagnostics).to_csv(OUT/'base_residual_diagnostics.csv',index=False)
    pd.DataFrame(coeff).to_csv(OUT/'base_coefficients.csv',index=False);pd.DataFrame(learn).to_csv(OUT/'base_learning_curve.csv',index=False)
    # Bootstrap de bloques por fold, pareado entre modelos y activos; métrica macro RMSE.
    rng=np.random.default_rng(42);reps=cfg['bootstrap_reps'];rep_metrics=np.zeros((reps,2));point=np.zeros(2)
    for f,g in oof.groupby('fold'):
        idx=pd.date_range(g.time.min(),g.time.max(),freq='h');blocks=[]
        for name in ['svr','persistence']:
            sq=g.assign(error=(g.y-g[name])**2).pivot(index='time',columns='symbol',values='error').reindex(idx)
            arr=sq.to_numpy();mask=np.isfinite(arr);filled=np.nan_to_num(arr);B=168;n=len(arr)
            ext=np.concatenate([filled,filled[:B]]);em=np.concatenate([mask,mask[:B]])
            cs=np.vstack([np.zeros((1,5)),ext.cumsum(axis=0)]);cc=np.vstack([np.zeros((1,5)),em.cumsum(axis=0)])
            blocks.append((cs,cc,n,B));point[['svr','persistence'].index(name)]+=np.sqrt(np.nanmean(arr,axis=0)).mean()/5
        n=blocks[0][2];q,rem=divmod(n,168);starts=rng.integers(n,size=(reps,q));tail=rng.integers(n,size=reps)
        for j,(cs,cc,n,B) in enumerate(blocks):
            sums=(cs[starts+B]-cs[starts]).sum(axis=1);counts=(cc[starts+B]-cc[starts]).sum(axis=1)
            if rem:sums+=cs[tail+rem]-cs[tail];counts+=cc[tail+rem]-cc[tail]
            rep_metrics[:,j]+=np.sqrt(sums/counts).mean(axis=1)/5
    for j,name in enumerate(['svr','persistence']):cis.append(dict(metric='macro_rmse',model=name,estimate=point[j],low=np.quantile(rep_metrics[:,j],.025),high=np.quantile(rep_metrics[:,j],.975)))
    diff=rep_metrics[:,0]-rep_metrics[:,1];cis.append(dict(metric='delta_macro_rmse',model='svr_minus_persistence',estimate=point[0]-point[1],low=np.quantile(diff,.025),high=np.quantile(diff,.975)))
    pd.DataFrame(cis).to_csv(OUT/'base_confidence_intervals.csv',index=False)
    assert sha==hashlib.sha256(source.read_bytes()).hexdigest()
    assert protocol_sha==hashlib.sha256((OUT/'base_model_protocol.json').read_bytes()).hexdigest()
    meta={'selected_C':best[0],'selected_epsilon':best[1],'development_sha256':sha,
        'protocol_sha256':protocol_sha,'boundary_policy':cfg['boundary_policy'],
        'test_read':False,'eligible_common_per_asset':len(frames[symbols[0]]),'validation_rows_per_asset':len(oof)//5,'seconds':time.perf_counter()-started,'sklearn':sklearn.__version__,'selection_bias':'CI conditional on selected configuration; not independent final test','fits':150+15+5}
    (OUT/'base_metadata.json').write_text(json.dumps(meta,indent=2),encoding='utf-8')
    print(meta,flush=True);print(pd.DataFrame(cis).to_string(index=False),flush=True)
if __name__=='__main__':main()
