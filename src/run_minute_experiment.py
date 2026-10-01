"""2023–2025 minute-input SVR study, daily origins and daily-volatility targets."""
import hashlib
import json
import time
import gc
import os
import joblib
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
from minute_experiment import ROOT,OUT,DATA,configuration,load_panel,windows,RowSpace,estimator,predict_artifact
from volatility_experiment import targets,calendar_folds,fit_model,residual_bds
from compare_cv_methods import adapted_folds,FUNCTIONS,scores


def save_checkpoint(path,state):
    temporary=path.with_name('checkpoint.tmp.joblib')
    joblib.dump(state,temporary,compress=1)
    temporary.replace(path)


def run():
    started=time.perf_counter(); cfg=configuration(); panel=load_panel()
    OUT.mkdir(parents=True,exist_ok=True); (OUT/'models').mkdir(exist_ok=True)
    (OUT/'status.json').write_text(json.dumps(dict(status='running',configuration=cfg)),encoding='utf-8')
    methods={}; audits={}
    methods['GroupKFold'],group_audit=calendar_folds(panel,cfg)
    for name,function in FUNCTIONS.items(): methods[name],audits[name]=adapted_folds(panel,cfg,function)
    calendar=[]
    for name,folds in methods.items():
        for f,fold in enumerate(folds,1):
            for role,anchors in fold.items():
                print(f'{name} fold {f} {role}: {len(anchors)}',flush=True)
                for a in anchors:
                    calendar.append(dict(method=name,fold=f,split=role,origin=str(panel.index[a]),
                        target_end=str(panel.index[a+7])))
    pd.DataFrame(calendar).to_csv(OUT/'calendar.csv',index=False)
    identical=all(np.array_equal(methods['KFold'][f][r],methods['ForwardChaining'][f][r])
        for f in range(5) for r in ['train','val','test'])
    assert identical, 'Revisit caching/interpretation if native methods change'
    all_anchors=np.unique(np.concatenate([a for folds in methods.values() for fold in folds for a in fold.values()]))
    index={int(a):i for i,a in enumerate(all_anchors)}
    patterns={tuple(fold['train']) for folds in methods.values() for fold in folds}
    predictions=[]; selections=[]; searches=[]; diagnostics=[]; map_audit=[]; best_primary={}
    if os.environ.get('MINUTE_RESUME')=='1' and (OUT/'checkpoint.joblib').exists():
        checkpoint=joblib.load(OUT/'checkpoint.joblib')
        assert checkpoint['configuration']==cfg,'Checkpoint configuration mismatch'
        predictions=checkpoint['predictions']; selections=checkpoint['selections']; searches=checkpoint['searches']
        diagnostics=checkpoint['diagnostics']; map_audit=checkpoint['map_audit']; best_primary=checkpoint['best_primary']
    completed={(r['symbol'],r['input_window']) for r in selections}
    selection_cache=pd.DataFrame()
    if os.environ.get('MINUTE_SELECTION_CACHE'):
        selection_cache=pd.read_csv(os.environ['MINUTE_SELECTION_CACHE'])
    for symbol in cfg['symbols']:
        minutes=np.load(DATA/f'{symbol}.npy',mmap_mode='r')
        close=panel[symbol]
        for lag in cfg['input_windows']:
            if (symbol,lag) in completed: continue
            mappings={}
            for pattern in sorted(patterns):
                rowspace=RowSpace().fit(windows(minutes,lag,np.array(pattern)))
                projected=np.empty((len(all_anchors),len(rowspace.eigen)))
                for start in range(0,len(all_anchors),64):
                    projected[start:start+64]=rowspace.transform(windows(minutes,lag,all_anchors[start:start+64]))
                mappings[pattern]=(rowspace,projected)
                map_audit.append(dict(symbol=symbol,input_window=lag,n_features=lag*1440,n_train=len(pattern),
                    rank=len(rowspace.eigen),gram_relative_error=rowspace.gram_relative_error))
            for w in cfg['volatility_windows']:
                vol=targets(close,w).to_numpy(float)
                cache={}
                for method,folds in methods.items():
                    key=dict(method=method,symbol=symbol,volatility_window=w,input_window=lag)
                    sets=[]
                    for fold in folds:
                        rowspace,projected=mappings[tuple(fold['train'])]
                        ds={}
                        for role,a in fold.items():
                            X=projected[[index[int(i)] for i in a]]
                            y=vol[a[:,None]+np.arange(1,8)]
                            baseline=np.repeat(vol[a,None],7,axis=1)
                            assert np.isfinite(y).all() and np.isfinite(baseline).all()
                            ds[role]=(X,y,baseline)
                        sets.append(ds)
                    candidates=[]
                    known=selection_cache
                    if len(known):
                        for field,value in key.items(): known=known[known[field]==value]
                    parameters=[(C,eps) for C in cfg['C'] for eps in cfg['epsilon']]
                    if len(known):
                        assert len(known)==1
                        parameters=[(float(known.iloc[0].C),float(known.iloc[0].epsilon))]
                    for C,eps in parameters:
                        models=[]; validation=[]
                        for f,(fold,ds) in enumerate(zip(folds,sets),1):
                            signature=(tuple(fold['train']),C,eps)
                            if signature not in cache:
                                cache[signature]=fit_model(estimator(C,eps,cfg['seed']),*ds['train'][:2])
                            model=cache[signature]
                            value=scores(ds['val'][1],model.predict(ds['val'][0]))['rmse']
                            validation.append(value); models.append(model)
                            searches.append(dict(**key,C=C,epsilon=eps,fold=f,val_rmse=value))
                        candidates.append((float(np.mean(validation)),C,eps,models))
                    score,C,eps,models=min(candidates,key=lambda c:(c[0],c[1],c[2]))
                    if len(known): np.testing.assert_allclose(score,known.iloc[0].val_rmse,rtol=1e-5,atol=1e-6)
                    selections.append(dict(**key,C=C,epsilon=eps,val_rmse=score,n_features=lag*1440,
                        selection_source='prior_completed_grid' if len(known) else 'full_grid'))
                    for f,(fold,ds,model) in enumerate(zip(folds,sets,models),1):
                        # All train and validation scores; detailed forecasts for test only.
                        X,y,baseline=ds['test']; pred=model.predict(X)
                        for label,p in [('SVR',pred),('Persistence',baseline)]:
                            diagnostics.append(dict(**key,fold=f,model=label,
                                **residual_bds(y[:,0]-p[:,0],minimum=6)))
                        for i,a in enumerate(fold['test']):
                            for h in range(7):
                                predictions.append(dict(**key,fold=f,origin=str(panel.index[a]),
                                    target_date=str(panel.index[a+h+1]),horizon=h+1,actual=y[i,h],
                                    svr=pred[i,h],persistence=baseline[i,h]))
                    if method==cfg['primary_method'] and score<best_primary.get((symbol,w),np.inf):
                        rowspace,_=mappings[tuple(folds[0]['train'])]
                        artifact=rowspace.export(models[0])
                        probe=folds[0]['test'][:5]
                        Xraw=windows(minutes,lag,probe)
                        np.testing.assert_allclose(predict_artifact(artifact,Xraw),
                            models[0].predict(rowspace.transform(Xraw)),rtol=1e-6,atol=1e-6)
                        artifact.update(**key,C=C,epsilon=eps,val_rmse=score,horizon=7,input_frequency='1min',
                            forecast_origin_frequency='1D',target_return_frequency='1D',
                            fitted_through=str(panel.index[folds[0]['train'].max()+7]),
                            unit='percentage_points',ddof=0,annualized=False)
                        joblib.dump(artifact,OUT/'models'/f'{symbol}_v{w}.joblib')
                        best_primary[(symbol,w)]=score
                    print(f'{method} {symbol} input={lag}d/{lag*1440}min vol={w}d: val RMSE={score:.6f}',flush=True)
            pd.DataFrame(selections).to_csv(OUT/'selection_all_inputs.csv',index=False)
            save_checkpoint(OUT/'checkpoint.joblib',dict(configuration=cfg,predictions=predictions,
                selections=selections,searches=searches,diagnostics=diagnostics,map_audit=map_audit,best_primary=best_primary))
            del mappings; gc.collect()
    prediction=pd.DataFrame(predictions)
    prediction.to_csv(OUT/'predictions.csv.gz',index=False,compression='gzip')
    selection=pd.DataFrame(selections)
    chosen=selection.loc[selection.groupby(['method','symbol','volatility_window']).val_rmse.idxmin()].copy()
    chosen.to_csv(OUT/'selected_inputs.csv',index=False)
    chosen[chosen.method==cfg['primary_method']].to_csv(OUT/'model_registry.csv',index=False)
    pd.DataFrame(searches).to_csv(OUT/'search.csv',index=False)
    pd.DataFrame(diagnostics).to_csv(OUT/'bds.csv',index=False)
    pd.DataFrame(map_audit).to_csv(OUT/'linear_algebra_audit.csv',index=False)
    common=set.intersection(*[set(str(panel.index[a]) for fold in fs for a in fold['test']) for fs in methods.values()])
    rows=[]; horizon_rows=[]
    for scope in ['common_test','all_available_test']:
        frame=prediction[prediction.origin.isin(common)] if scope=='common_test' else prediction
        for keys,g in frame.groupby(['method','symbol','volatility_window','input_window']):
            meta=dict(zip(['method','symbol','volatility_window','input_window'],keys))
            assert not g.duplicated(['origin','horizon']).any()
            y=g.pivot(index='origin',columns='horizon',values='actual').to_numpy()
            for label,column in [('SVR','svr'),('Persistence','persistence')]:
                p=g.pivot(index='origin',columns='horizon',values=column).to_numpy()
                rows.append(dict(**meta,scope=scope,model=label,n_origins=len(y),**scores(y,p)))
                for h in range(7):
                    horizon_rows.append(dict(**meta,scope=scope,model=label,horizon=h+1,n_origins=len(y),
                        **scores(y[:,h:h+1],p[:,h:h+1])))
    allmetrics=pd.DataFrame(rows)
    selected=allmetrics.merge(chosen[['method','symbol','volatility_window','input_window']],
        on=['method','symbol','volatility_window','input_window'],validate='many_to_one')
    columns=['r2','rmse','mae','mse','mape']
    macro=selected.groupby(['scope','method','symbol','model'],as_index=False)[columns].mean()
    global_=selected.groupby(['scope','method','model'],as_index=False)[columns].mean().assign(symbol='GLOBAL_MACRO')
    pd.concat([macro,global_],ignore_index=True).to_csv(OUT/'macro_metrics.csv',index=False)
    allmetrics.to_csv(OUT/'all_metrics.csv',index=False)
    selected.to_csv(OUT/'selected_metrics.csv',index=False)
    pd.DataFrame(horizon_rows).to_csv(OUT/'horizon_metrics.csv',index=False)
    for _,g in selected.query("scope == 'common_test' and model == 'Persistence'").groupby(['symbol','volatility_window']):
        np.testing.assert_allclose(g[columns],np.tile(g[columns].iloc[0],(len(g),1)),rtol=1e-10,atol=1e-10)
    record=dict(status='complete',configuration=cfg,seconds=time.perf_counter()-started,
        configurations=len(selection),common_test_origins=len(common),native_audits=audits,
        kfold_forward_identical=identical,primary_method=cfg['primary_method'],
        note='Minute inputs; daily sampling of forecast origins; daily-return volatility targets. Retrospective 2025 test.',
        source_config_sha256=hashlib.sha256((ROOT/'experiment_minute.json').read_bytes()).hexdigest())
    (OUT/'status.json').write_text(json.dumps(record,indent=2),encoding='utf-8')
    print(global_.to_string(index=False),flush=True)


if __name__=='__main__':
    with threadpool_limits(limits=4): run()
