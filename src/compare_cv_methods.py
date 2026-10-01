"""Compare native tsxv K-Fold / Forward Chaining with saved Group K-Fold.

The native small cuts are pooled by cut_id % 5, then calendar-filtered into
the fixed 2020–2022 / 2023 / 2024–2025 periods. This is an explicit adapter,
not a claim that the native functions return five large chronological folds.
The main comparison is on the exact 42 old Group K-Fold test origins.
"""
import gc
import hashlib
import inspect
import json
import time
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score
from tsxv.splitTrainValTest import split_train_val_test_kFold, split_train_val_test_forwardChaining
from volatility_experiment import ROOT, configuration, load_daily, targets, arrays, estimator, fit_model, metric_rows

OUT = ROOT/'results/cv_comparison'
FUNCTIONS = {'KFold':split_train_val_test_kFold, 'ForwardChaining':split_train_val_test_forwardChaining}


def adapted_folds(panel, cfg, function):
    grid = panel.index; h = cfg['horizon']; L = cfg['cv_calendar_input']
    maximum = max(max(cfg['input_windows']), max(cfg['volatility_windows'])+1)
    finite = np.isfinite(panel.to_numpy()).all(axis=1)
    eligible = pd.Series(finite).rolling(maximum+h).sum().shift(-h).eq(maximum+h).to_numpy()
    tr_end = pd.Timestamp(cfg['train_end_exclusive'],tz='UTC')
    va_end = pd.Timestamp(cfg['validation_end_exclusive'],tz='UTC')
    # int32 halves memory compared with the default int64 calendar.
    raw = function(np.arange(len(grid),dtype=np.int32), L, h, cfg['cv_jump'])
    native_count = len(raw[0])
    pooled = [{r:set() for r in ['train','val','test']} for _ in range(5)]
    future_count = 0
    for native_id in raw[0]:
        val_anchor = int(raw[2][native_id][0,-1])
        future_count += int(np.count_nonzero(raw[1][native_id][:,-1] >= val_anchor))
        for role, xi, yi in [('train',0,1),('val',2,3),('test',4,5)]:
            x, y = raw[xi][native_id], raw[yi][native_id]
            if not len(x): continue
            anchors = x[:,-1].astype(int)
            np.testing.assert_array_equal(y, anchors[:,None]+np.arange(1,h+1))
            first, last = grid[anchors], grid[y[:,-1]]
            mask = eligible[anchors].copy()
            if role=='train': mask &= last < tr_end
            elif role=='val': mask &= (first>=tr_end) & (last<va_end)
            else: mask &= first>=va_end
            pooled[native_id % 5][role].update(anchors[mask].tolist())
    del raw; gc.collect()
    result = [{role:np.array(sorted(values),dtype=int) for role,values in f.items()} for f in pooled]
    for f in result:
        assert all(len(a)>1 for a in f.values())
        assert f['train'].max()+h < f['val'].min()
        assert f['val'].max()+h < f['test'].min()
    assert len(set(np.concatenate([f['test'] for f in result]))) == sum(len(f['test']) for f in result)
    audit = dict(native_cuts=native_count, native_future_training_label_occurrences=future_count,
        function=function.__name__, source_sha256=hashlib.sha256(inspect.getsource(function).encode()).hexdigest())
    return result,audit


def scores(y,p):
    row = metric_rows(y,p)[-1]
    return {k:row[k] for k in ['mape','mae','rmse','mse']} | {
        'r2':float(r2_score(y,p,multioutput='uniform_average',force_finite=False))}


def summarize(predictions, chosen, common):
    rows=[]
    for scope in ['common_test','all_available_test']:
        frame=predictions[predictions.origin.isin(common)] if scope=='common_test' else predictions
        for keys,g in frame.groupby(['method','symbol','volatility_window','input_window']):
            meta=dict(zip(['method','symbol','volatility_window','input_window'],keys))
            y=g.pivot(index='origin',columns='horizon',values='actual')
            assert not y.isna().any().any()
            for name,col in [('SVR','svr'),('Persistence','persistence')]:
                p=g.pivot(index='origin',columns='horizon',values=col).reindex_like(y)
                rows.append(dict(**meta,scope=scope,model=name,n_origins=len(y),**scores(y.to_numpy(),p.to_numpy())))
    all_metrics=pd.DataFrame(rows)
    selected=all_metrics.merge(chosen[['method','symbol','volatility_window','input_window']],
        on=['method','symbol','volatility_window','input_window'],validate='many_to_one')
    columns=['r2','rmse','mae','mse','mape']
    macro=selected.groupby(['scope','method','symbol','model'],as_index=False)[columns].mean()
    global_=selected.groupby(['scope','method','model'],as_index=False)[columns].mean().assign(symbol='GLOBAL_MACRO')
    pd.concat([macro,global_],ignore_index=True).to_csv(OUT/'macro_metrics.csv',index=False)
    all_metrics.to_csv(OUT/'all_metrics.csv',index=False)
    selected.to_csv(OUT/'selected_metrics.csv',index=False)
    return selected,macro,global_


def main():
    started=time.perf_counter(); cfg=configuration(); panel=load_daily()
    OUT.mkdir(parents=True,exist_ok=True); (OUT/'models').mkdir(exist_ok=True)
    (OUT/'status.json').write_text(json.dumps(dict(status='running',configuration=cfg)),encoding='utf-8')
    methods={}; audits={}; calendar=[]
    for name,function in FUNCTIONS.items():
        print(f'Constructing native {name} cuts...',flush=True)
        methods[name],audits[name]=adapted_folds(panel,cfg,function)
        for f,fold in enumerate(methods[name],1):
            for role,anchors in fold.items():
                print(f'{name} fold {f} {role}: {len(anchors)} origins',flush=True)
                for a in anchors:
                    calendar.append(dict(method=name,fold=f,split=role,origin=str(panel.index[a]),
                        target_end=str(panel.index[a+7])))
    pd.DataFrame(calendar).to_csv(OUT/'calendar.csv',index=False)
    identical=all(np.array_equal(methods['KFold'][f][r],methods['ForwardChaining'][f][r])
        for f in range(5) for r in ['train','val','test'])
    old=pd.read_csv(ROOT/'results/predictions.csv.gz'); old=old[old.split=='test'].copy()
    old['method']='GroupKFold'
    common=set(old.origin.unique())
    for name,folds in methods.items():
        available=set(str(panel.index[a]) for f in folds for a in f['test'])
        if not common.issubset(available): raise ValueError(f'{name} misses reference test origins')
    predictions=[]; selections=[]; search=[]; fold_metrics=[]
    for symbol in cfg['symbols']:
      close=panel[symbol]
      for w in cfg['volatility_windows']:
        vol=targets(close,w)
        for lag in cfg['input_windows']:
          cache={}
          for method,folds in methods.items():
            key=dict(method=method,symbol=symbol,volatility_window=w,input_window=lag)
            datasets=[{r:arrays(close,vol,a,lag) for r,a in fold.items()} for fold in folds]
            options=[]
            for C in cfg['C']:
              for eps in cfg['epsilon']:
                fitted=[]; val_scores=[]
                for f,(ds,fold) in enumerate(zip(datasets,folds),1):
                    cachekey=(C,eps,tuple(fold['train']))
                    if cachekey not in cache:
                        cache[cachekey]=fit_model(estimator(C,eps,cfg['seed']),*ds['train'][:2])
                    model=cache[cachekey]
                    value=scores(ds['val'][1],model.predict(ds['val'][0]))['rmse']
                    fitted.append(model); val_scores.append(value)
                    search.append(dict(**key,C=C,epsilon=eps,fold=f,val_rmse=value))
                options.append((float(np.mean(val_scores)),C,eps,fitted))
            score,C,eps,fitted=min(options,key=lambda v:(v[0],v[1],v[2]))
            selections.append(dict(**key,C=C,epsilon=eps,val_rmse=score))
            for f,(ds,fold,model) in enumerate(zip(datasets,folds,fitted),1):
                joblib.dump(dict(model=model,**key,C=C,epsilon=eps),OUT/'models'/f'{method}_{symbol}_v{w}_l{lag}_f{f}.joblib')
                for role,(X,y,baseline) in ds.items():
                    pred=model.predict(X)
                    for name,p in [('SVR',pred),('Persistence',baseline)]:
                        fold_metrics.append(dict(**key,fold=f,split=role,model=name,n_origins=len(y),**scores(y,p)))
                    if role!='test': continue
                    for i,a in enumerate(fold[role]):
                        for h in range(7):
                            predictions.append(dict(**key,fold=f,origin=str(panel.index[a]),
                                target_date=str(panel.index[a+h+1]),horizon=h+1,actual=y[i,h],svr=pred[i,h],persistence=baseline[i,h]))
            print(f'{method} {symbol} v{w} l{lag}: validation RMSE {score:.6f}',flush=True)
          pd.DataFrame(selections).to_csv(OUT/'selection_all_inputs.csv',index=False)
    selection=pd.DataFrame(selections)
    chosen=selection.loc[selection.groupby(['method','symbol','volatility_window']).val_rmse.idxmin()].copy()
    old_selection=pd.read_csv(ROOT/'results/model_registry.csv').assign(method='GroupKFold')
    chosen=pd.concat([chosen,old_selection],ignore_index=True)
    chosen.to_csv(OUT/'selected_inputs.csv',index=False)
    predictions=pd.concat([pd.DataFrame(predictions),old],ignore_index=True)
    predictions.to_csv(OUT/'predictions.csv.gz',index=False,compression='gzip')
    pd.DataFrame(search).to_csv(OUT/'hyperparameter_search.csv',index=False)
    pd.DataFrame(fold_metrics).to_csv(OUT/'fold_metrics.csv',index=False)
    selected,macro,global_=summarize(predictions,chosen,common)
    for _,g in selected.query("scope == 'common_test' and model == 'Persistence'").groupby(['symbol','volatility_window']):
        for metric in ['r2','rmse','mae','mse','mape']:
            np.testing.assert_allclose(g[metric],g[metric].iloc[0],rtol=1e-12,atol=1e-12)
    record=dict(status='complete',seconds=time.perf_counter()-started,configuration=cfg,
        native_audits=audits,identical_after_temporal_filters=identical,common_test_origins=len(common),
        new_configurations=len(selection),reference_method='saved GroupKFold',
        protocol='Pool native cut IDs modulo 5; union and deduplicate origins per role; filter by fixed calendar periods.',
        selection='Validation RMSE only, first C/epsilon per input, then input per method/asset/target.',
        interpretation='Retrospective comparison on already inspected test. Not a new independent holdout.',
        daily_sha256=hashlib.sha256((ROOT/'data/processed/daily_2020_2025.csv').read_bytes()).hexdigest())
    (OUT/'status.json').write_text(json.dumps(record,indent=2),encoding='utf-8')
    print(global_.query("scope == 'common_test'").to_string(index=False),flush=True)
    print(f'Identical KFold/Forward calendars after filtering: {identical}',flush=True)


if __name__=='__main__': main()
