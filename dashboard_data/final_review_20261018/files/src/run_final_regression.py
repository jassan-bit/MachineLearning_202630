"""Run the local regression matrix with genuinely nested purged temporal CV.

Seven benchmark families x four searches x four assets x four target windows,
plus four optimized HAR-XG procedures. No production model or site is changed.
All selection finishes and every model is hashed before test forecasts begin.
Use --benchmark first; --resume continues only matching, completed task files.
"""
from __future__ import annotations
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path
import argparse
import json
import os
import sys
import time

for _key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[_key] = '1'

import joblib
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / 'src'))
from final_regression_core import (FinalRegressor, FAMILIES, METHODS, HYBRID,
    purged_inner_folds, search_space, model_label, lag_features, sample_weights)
from final_regression_search import run_search
from export_classical_holdout_baselines import (load_combined_panel, calendar, DATA, NEW_DATA, sha)
from professor_feedback_har_baseline import protected_files
from optimize_minute_svr import minute_features, features, metrics
from improve_classical_forecast import har_features
from volatility_dashboard.calendar import saved_calendar

OUT = ROOT / 'results/final_regression_20261018'
ORIGINAL = ROOT / 'results/improved_classical_2023_2025'
BOUNDARY = pd.Timestamp('2025-01-01', tz='UTC')
SEED = 42


def write_json(path, obj):
    Path(path).write_text(json.dumps(obj, indent=2, ensure_ascii=False, allow_nan=False)+'\n', encoding='utf-8')


def load_states():
    """Build causal matrices; future targets are never supplied to a selector."""
    original, panel = load_combined_panel()
    folds, old_eligible, _ = saved_calendar(original, ORIGINAL)
    eligible = calendar(panel)
    train = np.flatnonzero(eligible & (panel.index + pd.Timedelta(days=7) < BOUNDARY))
    tests = {2025: np.flatnonzero(old_eligible & (original.index >= BOUNDARY)),
             2026: np.flatnonzero(eligible & (panel.index >= pd.Timestamp('2026-01-01', tz='UTC')))}
    if (len(train),len(tests[2025]),len(tests[2026])) != (660,358,236):
        raise ValueError('Original training and evaluation calendars changed.')
    states = []
    required = np.concatenate([train, *tests.values()])
    for symbol in panel.columns:
        values = np.concatenate([np.load(DATA / f'{symbol}.npy'), np.load(NEW_DATA / f'{symbol}.npy')])
        minute = minute_features(values)
        for window in (7,14,21,28):
            X,y,base = features(panel[symbol], minute, 28, window)
            H,ycheck,_ = har_features(panel[symbol], minute, window)
            np.testing.assert_allclose(y,ycheck,equal_nan=True)
            A = np.column_stack([base,X,H])
            if A.shape[1] != 220 or not np.isfinite(A[required]).all() or not np.isfinite(y[required]).all():
                raise ValueError('Causal model schema or original eligible support changed.')
            states.append(dict(symbol=symbol, volatility_window=window, X=A, y=y))
    return panel,folds,train,tests,states


class InnerObjective:
    """Pooled original-unit MSE inside one outer train, with HAR fit reuse.

    Each search gets its own cache. Reusing a component across weights is an
    exact computational optimization, not cross-search timing assistance.
    """
    def __init__(self, state, family, folds, seed):
        self.state,self.family,self.folds,self.seed = state,family,folds,seed
        self.components = {}
        self.fits = 0

    def __call__(self, params):
        X,y = self.state['X'],self.state['y']
        total,count = 0.,0
        for fold,(tr,va) in enumerate(self.folds):
            if tr[-1]+7 >= va[0]:
                raise ValueError('Inner labels reach validation.')
            if self.family != HYBRID:
                model = FinalRegressor(self.family, params, self.seed)
                model.fit(X[tr], y[tr]); self.fits += 1
                p = model.predict(X[va])
            else:
                # The HAR component is independent of the lag/tree parameters;
                # the booster is independent of alpha/combination weight.
                har_key = (fold, float(params['har_alpha']), float(params['weight_power']))
                xgb_key = (fold, params['input_window'], params['max_depth'],
                           params['learning_rate'],params['n_estimators'],params['weight_power'])
                helper = FinalRegressor(HYBRID, params, self.seed)
                weights = sample_weights(X[tr,0], params['weight_power'])
                if har_key not in self.components:
                    from sklearn.pipeline import make_pipeline
                    from sklearn.preprocessing import StandardScaler
                    from sklearn.linear_model import Ridge
                    mask = np.arange(44) != 2
                    har = make_pipeline(StandardScaler(), Ridge(alpha=params['har_alpha'],solver='cholesky'))
                    har.fit(X[tr,176:][:,mask],y[tr]/X[tr,[0]][:,None]-1.,ridge__sample_weight=weights)
                    self.components[har_key] = np.maximum(X[va,[0]][:,None]*(1.+har.predict(X[va,176:][:,mask])),0.)
                    self.fits += 1
                if xgb_key not in self.components:
                    booster = helper._xgb(params)
                    booster.fit(lag_features(X[tr], params['input_window']),
                        y[tr]/X[tr,[0]][:,None]-1.,sample_weight=weights)
                    self.components[xgb_key] = np.maximum(X[va,[0]][:,None]*(1.+booster.predict(
                        lag_features(X[va],params['input_window']))),0.)
                    self.fits += 1
                weight = params['ridge_weight']
                p = weight*self.components[har_key]+(1.-weight)*self.components[xgb_key]
            total += float(np.sum((y[va]-p)**2)); count += len(va)*7
        return total/count


def prediction_frame(label,state,panel,origins,p,split,outer_fold=0):
    """Long original-unit forecasts with explicit dates and evaluation role."""
    return pd.DataFrame(dict(model=label,symbol=state['symbol'],volatility_window=state['volatility_window'],
        origin=np.repeat(panel.index[origins].astype(str),7),horizon=np.tile(np.arange(1,8),len(origins)),
        target_date=panel.index[(origins[:,None]+np.arange(1,8)).ravel()].astype(str),
        actual=state['y'][origins].ravel(),forecast=p.ravel(),split=split,outer_fold=outer_fold))


def task(state,family,folds,train,panel,out,budget,seed=SEED):
    """Complete six outer searches and final train-only selection for one task."""
    folder = out/'tasks'/f'{state["symbol"]}_w{state["volatility_window"]}_{family}_s{seed}'
    folder.mkdir(parents=True,exist_ok=True)
    if (folder/'complete.json').exists():
        return json.loads((folder/'complete.json').read_text(encoding='utf-8'))
    logs,outer_rows,choices,nested_predictions = [],[],[],[]
    calendar_rows=[]
    stages = [(i,tr,va) for i,(tr,va) in enumerate(folds,1)] + [(0,train,None)]
    for outer_fold,outer_train,outer_test in stages:
        inner = purged_inner_folds(outer_train,n_splits=3 if outer_fold == 0 else 2)
        for inner_fold,(tr,va) in enumerate(inner,1):
            calendar_rows.append(dict(outer_fold=outer_fold,inner_fold=inner_fold,
                n_train=len(tr),n_validation=len(va),train_last=str(panel.index[tr[-1]]),
                train_target_end=str(panel.index[tr[-1]+7]),validation_first=str(panel.index[va[0]]),
                validation_last=str(panel.index[va[-1]]),validation_target_end=str(panel.index[va[-1]+7]),
                outer_train_target_end=str(panel.index[outer_train[-1]+7]),strict_purge=True,
                inner_validation_inside_outer_training=bool(np.isin(va,outer_train).all())))
        for method in METHODS:
            start = time.perf_counter()
            objective = InnerObjective(state,family,inner,seed)
            result = run_search(search_space(family),objective,method,budget=budget,seed=seed)
            label = model_label(family,method)
            context = dict(model=label,family=family,method=method,symbol=state['symbol'],
                volatility_window=state['volatility_window'],seed=seed,outer_fold=outer_fold)
            for row in result['trials']:
                logs.append(dict(**context,trial=row['trial_number'],value=row['value'],
                    best_so_far=row['best_so_far'],seconds=row['seconds'],
                    elapsed_seconds=row['elapsed_seconds'],params=json.dumps(row['params'],sort_keys=True)))
            write_json(folder/f'search_{outer_fold}_{method}.json',result)
            params = result['best_params']
            fit_start=time.perf_counter()
            fitted=FinalRegressor(family,params,seed).fit(state['X'][outer_train],state['y'][outer_train])
            fit_seconds=time.perf_counter()-fit_start
            row=dict(**context,params=json.dumps(params,sort_keys=True),inner_mse=result['best_value'],
                n_train=len(outer_train),n_inner=len(inner),unique_evaluations=len(result['trials']),
                fitted_inner_models=objective.fits,fit_seconds=fit_seconds,
                search_seconds=result['details'].get('total_seconds',time.perf_counter()-start),
                stage_seconds=time.perf_counter()-start,
                train_target_end=str(panel.index[outer_train[-1]+7]))
            if outer_test is not None:
                predict_start=time.perf_counter(); p=fitted.predict(state['X'][outer_test])
                row.update(n_test=len(outer_test),predict_seconds=time.perf_counter()-predict_start,
                           **metrics(state['y'][outer_test],p))
                outer_rows.append(row)
                nested_predictions.append(prediction_frame(label,state,panel,outer_test,p,'outer_test',outer_fold))
            else:
                name=f'{state["symbol"]}_w{state["volatility_window"]}_{family}_{method}_s{seed}.joblib'
                path=out/'models'/name
                joblib.dump(fitted,path)
                choices.append(dict(**row,model_path=path.relative_to(out).as_posix(),model_sha256=sha(path)))
    pd.DataFrame(logs).to_csv(folder/'trials.csv',index=False)
    pd.DataFrame(outer_rows).to_csv(folder/'outer_metrics.csv',index=False)
    pd.DataFrame(choices).to_csv(folder/'selected_inputs.csv',index=False)
    pd.DataFrame(calendar_rows).to_csv(folder/'nested_calendar.csv',index=False)
    pd.concat(nested_predictions,ignore_index=True).to_csv(folder/'outer_predictions.csv.gz',index=False,
        compression=dict(method='gzip',mtime=0))
    complete=dict(symbol=state['symbol'],volatility_window=state['volatility_window'],family=family,
        seed=seed,outer_searches=len(folds)*4,final_models=4,completed=True)
    write_json(folder/'complete.json',complete)
    return complete


def source_hashes():
    paths=[ROOT/'src'/name for name in ('run_final_regression.py','final_regression_core.py',
        'final_regression_search.py','optimize_minute_svr.py','improve_classical_forecast.py',
        'export_classical_holdout_baselines.py')]
    return {p.relative_to(ROOT).as_posix():sha(p) for p in paths}


def benchmark(out):
    """Measure actual fits on development before setting computational budget."""
    panel,folds,train,tests,states=load_states()
    state=next(s for s in states if s['symbol']=='BTCUSDT' and s['volatility_window']==21)
    tr,va=folds[-1]; rows=[]
    for family in (*FAMILIES,HYBRID):
        p={k:v[0] for k,v in search_space(family).items()}
        start=time.perf_counter(); fitted=FinalRegressor(family,p).fit(state['X'][tr],state['y'][tr])
        fit_seconds=time.perf_counter()-start
        start=time.perf_counter(); forecast=fitted.predict(state['X'][va]); predict_seconds=time.perf_counter()-start
        rows.append(dict(family=family,fit_seconds=fit_seconds,predict_seconds=predict_seconds,
            n_train=len(tr),n_prediction=len(va),schema_width=220,input_window=p['input_window']))
    out.mkdir(parents=True,exist_ok=True)
    pd.DataFrame(rows).to_csv(out/'cost_pilot.csv',index=False)
    print(pd.DataFrame(rows).to_string(index=False),flush=True)


def run(out=OUT,budget=8,workers=3,resume=False):
    """Freeze a predeclared bounded study, finish matrix, then export test data."""
    import optuna, deap
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    out=Path(out).resolve()
    if not out.is_relative_to(ROOT/'results') or out==ROOT/'results':
        raise ValueError('New experiment must remain under workspace/results.')
    out.mkdir(parents=True,exist_ok=True)
    if (out/'selection_freeze.json').exists():
        raise FileExistsError('Study already frozen: inspect or run a new output directory.')
    hashes=source_hashes()
    if (out/'protocol.json').exists():
        if not resume:
            raise FileExistsError('Use --resume for the same frozen protocol.')
        protocol=json.loads((out/'protocol.json').read_text(encoding='utf-8'))
        if protocol['training_source_sha256']!=hashes or protocol['budget_per_search']!=budget:
            raise ValueError('Cannot resume after changing training code or budget.')
    else:
        before={p.relative_to(ROOT).as_posix():sha(p) for p in protected_files()}
        write_json(out/'protected_before.json',before)
        protocol=dict(created_at_utc=datetime.now(timezone.utc).isoformat(),publication='LOCAL ONLY',
            github_or_render_updates_authorized=False,reference='HAR-Ridge + XGBoost',
            regression_families=list(FAMILIES),methods=list(METHODS),additional_family=HYBRID,
            assets=['BTCUSDT','ETHUSDT','BNBUSDT','XRPUSDT'],target_windows=[7,14,21,28],horizons=list(range(1,8)),
            benchmark_combinations=448,additional_hybrid_combinations=64,final_models=512,
            outer='Six original purged2024 blocks; ALL optimization repeated independently inside EACH outer train',
            inner='Two expanding chronological purged blocks per outer training support; three for final660-origin refit',
            purge='Maximum seven-target training label ends STRICTLY before validation origin; exact calendar positions retained',
            primary_objective='Pooled natural-unit MSE over all seven horizons inside inner folds',
            target='Unchanged rolling population std of daily percent log returns, ddof0',
            budget_per_search=budget,search_domains={f:search_space(f) for f in (*FAMILIES,HYBRID)},
            grid_limitation='Budget-truncated deterministic Cartesian Grid, NOT exhaustive; sorted parameter order declared by search module. Cannot infer universal optimizer superiority.',
            adaptive_search_limitation='Small explicit compute budget; TPE startup and GA population consume substantial fraction. No convergence claim.',
            efficient_hybrid='Train-only exact component reuse across combination weights; private cache per search, not shared between optimizers',
            fitting='Relative corrections y/base-1; optional base² train-normalized weighting aligns level SSE; kNN has no training sample weights',
            har_features='Original HAR44 with exact span1 downside redundant column removed a priori; no MI-driven selection',
            estimator_seed=SEED,search_seed=SEED,seed_sensitivity='Separate diagnostics, no choice based on2025/2026',
            primary_candidates=[model_label(HYBRID,m) for m in METHODS],
            primary_year=2026,secondary_year=2025,
            inferential_rule='Lower global2026 MSE AND DM-HLN7/HAC33 Holm4 favorable against original for a predeclared hybrid candidate; show MCS/SPA/effect CIs and2025 direction, never automatic publication',
            prior_inspection='Both2025 andJan-Aug2026 previously inspected; retrospective evidence only. Nested outer estimation evaluates each fixed procedure, but selecting among procedures by outer mean is an additional selection step.',
            baseline_outer_policy='Original globally2024-selected reference is NOT used as independent outerCV control; only its frozen2025/26 forecasts enter retrospective comparison',
            training_source_sha256=hashes,workers=workers,numerical_threads_per_worker=1,
            versions={n:version(n) for n in ('numpy','pandas','scikit-learn','xgboost','optuna','deap','shap','lime')})
        write_json(out/'protocol.json',protocol)
    before=json.loads((out/'protected_before.json').read_text(encoding='utf-8'))
    if any(sha(ROOT/p)!=h for p,h in before.items()):
        raise ValueError('Protected original source/data/model changed.')
    (out/'models').mkdir(exist_ok=True)
    panel,folds,train,tests,states=load_states()
    started=time.perf_counter()
    jobs=[(state,family) for state in states for family in (*FAMILIES,HYBRID)]
    write_json(out/'status.json',dict(status='running',completed_tasks=0,total_tasks=len(jobs),publication='LOCAL ONLY'))
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures=[pool.submit(task,s,f,folds,train,panel,out,budget) for s,f in jobs]
        for count,future in enumerate(as_completed(futures),1):
            done=future.result()
            write_json(out/'status.json',dict(status='running',completed_tasks=count,total_tasks=len(jobs),
                elapsed_seconds=time.perf_counter()-started,last=done,publication='LOCAL ONLY'))
            print(f'{count}/{len(jobs)} tasks completed: {done["symbol"]} w{done["volatility_window"]} {done["family"]}',flush=True)
    for name in ('trials','outer_metrics','selected_inputs','nested_calendar'):
        frames=[pd.read_csv(p) for p in sorted((out/'tasks').glob(f'*/{name}.csv'))]
        pd.concat(frames,ignore_index=True).to_csv(out/('master_trials.csv' if name=='trials' else name+'.csv'),index=False)
    choices=pd.read_csv(out/'selected_inputs.csv')
    if len(choices)!=512 or set(choices.family)!=set((*FAMILIES,HYBRID)):
        raise ValueError('Final regression matrix incomplete.')
    frozen={r.model_path:sha(out/r.model_path) for r in choices.itertuples()}
    write_json(out/'selection_freeze.json',dict(frozen_at_utc=datetime.now(timezone.utc).isoformat(),
        protocol_sha256=sha(out/'protocol.json'),selected_inputs_sha256=sha(out/'selected_inputs.csv'),
        model_sha256=frozen,test_forecasts_or_metrics_computed=False))
    print('All512 models frozen before test forecasts.',flush=True)
    state_map={(s['symbol'],s['volatility_window']):s for s in states}
    for year,origins in tests.items():
        rows=[]
        for row in choices.itertuples():
            state=state_map[(row.symbol,row.volatility_window)]
            model=joblib.load(out/row.model_path)
            p=model.predict(state['X'][origins])
            rows.append(prediction_frame(row.model,state,panel,origins,p,'test'))
        for state in states:
            rows.append(prediction_frame('Persistencia',state,panel,origins,
                np.repeat(state['X'][origins,[0]][:,None],7,axis=1),'test'))
        reference=pd.read_csv(ORIGINAL/'predictions.csv.gz') if year==2025 else pd.read_csv(
            ROOT/'results/har_spca_research_20261005/baseline_2026/predictions.csv.gz').query(
                'model == "HAR-Ridge + XGBoost"')
        reference['model']='HAR-Ridge + XGBoost';reference['split']='test';reference['outer_fold']=0
        if 'target_date' not in reference:
            reference['target_date']=(pd.to_datetime(reference.origin,utc=True)+pd.to_timedelta(reference.horizon,unit='D')).astype(str)
        rows.append(reference[rows[0].columns])
        pd.concat(rows,ignore_index=True).to_csv(out/f'predictions_{year}.csv.gz',index=False,
            compression=dict(method='gzip',mtime=0))
        print(f'{year} forecasts exported, models34, origins{len(origins)}',flush=True)
    if any(sha(ROOT/p)!=h for p,h in before.items()) or source_hashes()!=hashes:
        raise ValueError('Protected inputs or training sources changed.')
    write_json(out/'status.json',dict(status='complete',completed_tasks=128,total_tasks=128,
        elapsed_seconds=time.perf_counter()-started,models=512,
        nested_cv_outer_folds=6,primary_seed=SEED,test_forecasts_exported=True,
        protected_original_files_unchanged=len(before),publication='LOCAL ONLY'))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,default=OUT)
    parser.add_argument('--budget',type=int,default=8)
    parser.add_argument('--workers',type=int,default=3)
    parser.add_argument('--benchmark',action='store_true')
    parser.add_argument('--resume',action='store_true')
    args=parser.parse_args()
    with threadpool_limits(limits=1):
        if args.benchmark: benchmark(args.out)
        else: run(args.out,args.budget,args.workers,args.resume)
