"""PCA, redundancia y anomalías de 168 rezagos; solo TRAIN de cada fold."""
from pathlib import Path
import importlib.util
import hashlib
import json
import time
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
import sklearn

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs/tables'
FIG=ROOT/'book/_static/figures'


def diagnose(X, trees=200, max_samples=256, quantile=.99, seed=42):
    if not len(X.columns) or any(not c.startswith('lag_') for c in X.columns):
        raise ValueError('Solo se admiten las columnas de rezagos; sin objetivo ni metadatos.')
    if not np.isfinite(X.to_numpy()).all() or (X.std(ddof=0)==0).any():
        raise ValueError('Se requieren columnas finitas y con varianza positiva.')
    scaler=StandardScaler()
    z=scaler.fit_transform(X)
    pca=PCA(svd_solver='full').fit(z)
    ratio=pca.explained_variance_ratio_
    singular=pca.singular_values_
    tol=max(z.shape)*np.finfo(z.dtype).eps*singular[0]
    rank=int((singular>tol).sum())
    corr=np.corrcoef(z,rowvar=False)
    off=corr[np.triu_indices(z.shape[1],k=1)]
    positive=ratio[ratio>0]
    stats=dict(n=len(X),p=X.shape[1],n_over_p=len(X)/X.shape[1],
        pc1_ratio=ratio[0],pc2_ratio=ratio[1],
        k90=int(np.searchsorted(ratio.cumsum(),.90)+1),
        k95=int(np.searchsorted(ratio.cumsum(),.95)+1),
        k99=int(np.searchsorted(ratio.cumsum(),.99)+1),
        effective_rank=float(np.exp(-np.sum(positive*np.log(positive)))),
        numerical_rank=rank,condition_number=float(singular[0]/singular[-1]) if rank==X.shape[1] else np.inf,
        median_abs_correlation=float(np.median(abs(off))),
        min_correlation=float(off.min()),max_correlation=float(off.max()),
        fraction_abs_correlation_gt_099=float(np.mean(abs(off)>.99)))
    forest=IsolationForest(n_estimators=trees,max_samples=min(max_samples,len(X)),
                           contamination='auto',random_state=seed,n_jobs=1)
    forest.fit(z)
    scores=-forest.score_samples(z)
    threshold=float(np.quantile(scores,quantile))
    flags=scores>threshold
    flagged_times=X.index[flags]
    episodes=int((flagged_times.to_series().diff()!=pd.Timedelta(hours=1)).sum())
    stats.update(anomaly_threshold=threshold,flagged=int(flags.sum()),
                 flagged_pct=100*flags.mean(),flagged_runs=episodes)
    assert np.isclose(ratio.sum(),1)
    assert np.allclose(scaler.mean_,X.mean())
    assert np.allclose(pca.inverse_transform(pca.transform(z)),z,atol=1e-9)
    return stats,scaler,pca,corr,scores,flags


def main():
    started=time.perf_counter()
    cfg=json.loads((OUT/'multivariate_protocol.json').read_text(encoding='utf-8'))
    hashes={name:hashlib.sha256(path.read_bytes()).hexdigest() for name,path in {
        'development':ROOT/'data/splits/development_80.csv',
        'base_folds':OUT/'base_folds.csv',
        'base_predictions':OUT/'base_validation_predictions.csv',
        'base_protocol':OUT/'base_model_protocol.json',
        'protocol':OUT/'multivariate_protocol.json'}.items()}
    audit=pd.read_csv(OUT/'base_folds.csv')
    for c in ['train_block_start','train_block_end','train_start','train_end','validation_block_start']:
        audit[c]=pd.to_datetime(audit[c],utc=True)
    df=pd.read_csv(ROOT/'data/splits/development_80.csv',usecols=['symbol','open_time','close_time','close'])
    for c in ['open_time','close_time']:
        df[c]=pd.to_datetime(df[c],utc=True,format='ISO8601')
    # Solo el mayor TRAIN participa en la construcción analítica; no se calcula
    # ningún estadístico con el bloque de validación final.
    df=df.loc[df.open_time<=audit.train_block_end.max()]
    grid=pd.date_range(df.open_time.min(),df.open_time.max(),freq='h')
    spec=importlib.util.spec_from_file_location('base_model',ROOT/'src/18_base_model.py')
    base=importlib.util.module_from_spec(spec);spec.loader.exec_module(base)
    frames=base.prepare(df,grid,L=cfg['window_closes'])
    columns=[f'lag_{k}' for k in range(cfg['window_closes'])]
    summaries=[];spectra=[];loadings=[];scales=[];score_rows=[];episodes=[];years=[];matrices={}
    for symbol in sorted(frames):
        for row in audit.loc[audit.symbol.eq(symbol)].itertuples():
            train=base.contained_block(frames[symbol],row.train_block_start,row.train_block_end,cfg['window_closes'])
            assert len(train)==row.n_train
            assert train.index.min()==row.train_start and train.index.max()==row.train_end
            assert train.index.max()+pd.Timedelta(hours=24)<row.validation_block_start
            X=train[columns]
            stats,scaler,pca,corr,scores,flags=diagnose(X,cfg['isolation_trees'],cfg['isolation_max_samples'],cfg['anomaly_quantile'],cfg['seed'])
            summaries.append(dict(symbol=symbol,fold=row.fold,train_start=str(X.index.min()),train_end=str(X.index.max()),**stats))
            ratio=pca.explained_variance_ratio_
            spectra.extend(dict(symbol=symbol,fold=row.fold,component=j+1,variance_ratio=ratio[j],cumulative_ratio=ratio[:j+1].sum(),singular_value=pca.singular_values_[j]) for j in range(len(ratio)))
            scales.extend(dict(symbol=symbol,fold=row.fold,lag=k,mean=scaler.mean_[k],scale=scaler.scale_[k]) for k in range(len(columns)))
            matrices[f'{symbol}_fold_{row.fold}']=corr
            if row.fold==cfg['plot_fold']:
                scores_frame=pd.DataFrame(dict(symbol=symbol,time=X.index,score=scores,flagged=flags,threshold=stats['anomaly_threshold']))
                score_rows.append(scores_frame)
                flagged=scores_frame.loc[flags].copy()
                flagged['run']=(flagged.time.diff()!=pd.Timedelta(hours=1)).cumsum()
                for _,g in flagged.groupby('run'):
                    peak=g.loc[g.score.idxmax()]
                    episodes.append(dict(symbol=symbol,start=g.time.min(),end=g.time.max(),n_anchors=len(g),peak_time=peak.time,max_score=peak.score))
                for year,g in scores_frame.groupby(scores_frame.time.dt.year):
                    years.append(dict(symbol=symbol,year=year,n=len(g),flagged=int(g.flagged.sum()),flagged_pct=100*g.flagged.mean()))
                coords=pca.transform(scaler.transform(X))[:,:2]
                # Fijar solo para el dibujo el signo de PC1; no cambia varianza.
                for j in range(2):
                    sign=1 if pca.components_[j].sum()>=0 else -1
                    loadings.extend(dict(symbol=symbol,component=j+1,lag_hours=k,weight=sign*pca.components_[j,k]) for k in range(len(columns)))
                    coords[:,j]*=sign
                fig,axs=plt.subplots(2,3,figsize=(16,9),constrained_layout=True)
                a=axs.flat
                im=a[0].imshow(corr,vmin=min(.90,float(corr.min())),vmax=1,cmap='viridis',origin='lower');fig.colorbar(im,ax=a[0],shrink=.75,label='Correlación (escala ampliada)')
                a[0].set(title='Pearson entre los 168 rezagos',xlabel='Rezago (h)',ylabel='Rezago (h)')
                a[1].plot(np.arange(1,169),ratio.cumsum());a[1].set_xscale('log');a[1].axhline(.95,color='orange',ls='--',label='95%');a[1].axhline(.99,color='gray',ls=':',label='99%');a[1].legend()
                a[1].set(title='PCA: varianza acumulada (detalle)',xlabel='Número de componentes (log)',ylabel='Fracción de varianza',ylim=(min(.945,float(ratio[0])-.005),1.001))
                subset=np.unique(np.linspace(0,len(X)-1,min(cfg['plot_max_points'],len(X)),dtype=int))
                scatter=a[2].scatter(coords[subset,0],coords[subset,1],c=X.index.year[subset],s=4,alpha=.4,cmap='viridis')
                fig.colorbar(scatter,ax=a[2],shrink=.75,label='Año UTC',ticks=sorted(set(X.index.year)));a[2].set(title='Proyección PC1–PC2 por año',xlabel='PC1',ylabel='PC2')
                own=pd.DataFrame(loadings).query('symbol == @symbol')
                for component,g in own.groupby('component'):a[3].plot(g.lag_hours,g.weight,label=f'PC{component}')
                a[3].legend();a[3].set(title='Pesos de los componentes',xlabel='Rezago (h)',ylabel='Peso (signo convencional)')
                # Reindexar para no dibujar segmentos que unan huecos.
                series=pd.Series(scores,index=X.index).reindex(pd.date_range(X.index.min(),X.index.max(),freq='h'))
                a[4].plot(series.index,series,lw=.4);a[4].axhline(stats['anomaly_threshold'],color='red',ls='--')
                a[4].scatter(X.index[flags],scores[flags],s=4,color='red');a[4].set(title='Atipicidad de ventanas en TRAIN',ylabel='−score_samples');a[4].tick_params(axis='x',rotation=30)
                yearly=pd.DataFrame(years).query('symbol == @symbol');a[5].bar(yearly.year,yearly.flagged_pct)
                a[5].set(title='Ventanas señaladas por año',xlabel='Año (extremos parciales)',ylabel='% de anclas')
                fig.suptitle(f'{symbol} · 168 rezagos · entrenamiento fold 5 · análisis descriptivo')
                fig.savefig(FIG/f'multivariate_{symbol}.png',dpi=130);plt.close(fig)
            print(f'Completado {symbol} fold {row.fold}: PC1={100*stats["pc1_ratio"]:.3f}%, k95={stats["k95"]}',flush=True)
    for name,rows in [('summary',summaries),('spectrum',spectra),('loadings',loadings),('scalers',scales),('episodes',episodes),('yearly_flags',years)]:
        pd.DataFrame(rows).to_csv(OUT/f'multivariate_{name}.csv',index=False)
    pd.concat(score_rows,ignore_index=True).to_csv(OUT/'multivariate_scores_fold5.csv',index=False)
    np.savez_compressed(OUT/'multivariate_correlations.npz',**matrices)
    assert hashes['development']==hashlib.sha256((ROOT/'data/splits/development_80.csv').read_bytes()).hexdigest()
    assert hashes['base_predictions']==hashlib.sha256((OUT/'base_validation_predictions.csv').read_bytes()).hexdigest()
    assert hashes['protocol']==hashlib.sha256((OUT/'multivariate_protocol.json').read_bytes()).hexdigest()
    meta=dict(hashes=hashes,fit_scope='TRAIN separately for each of 5 folds and 5 assets',
        feature_columns=columns,target_used_for_fit=False,label_used_only_for_common_eligibility=True,
        test_read=False,validation_used_for_fit=False,rows_removed=False,svr_retrained=False,
        folds_nested=True,anomaly_threshold_scope='TRAIN empirical quantile, not calibrated false positive rate',
        seconds=time.perf_counter()-started,versions=dict(numpy=np.__version__,pandas=pd.__version__,sklearn=sklearn.__version__))
    (OUT/'multivariate_metadata.json').write_text(json.dumps(meta,indent=2,ensure_ascii=False),encoding='utf-8')
    print(pd.DataFrame(summaries).query('fold == 5').to_string(index=False),flush=True)


if __name__=='__main__':
    main()
