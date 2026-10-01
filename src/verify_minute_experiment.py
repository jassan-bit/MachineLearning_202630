"""Independently check completed forecasts, source hashes, splits and exported models."""
import hashlib
import json
import sys
import joblib
import numpy as np
import pandas as pd
from minute_experiment import ROOT,OUT,DATA,configuration,load_panel,windows,predict_artifact
from volatility_experiment import targets
from compare_cv_methods import scores


def verify():
    cfg=configuration(); status=json.loads((OUT/'status.json').read_text())
    assert status['status']=='complete' and status['configuration']==cfg
    assert status['source_config_sha256']==hashlib.sha256((ROOT/'experiment_minute.json').read_bytes()).hexdigest()
    for name,digest in json.loads((OUT/'data_manifest.json').read_text()).items():
        assert hashlib.sha256((DATA/name).read_bytes()).hexdigest()==digest,name
    calendar=pd.read_csv(OUT/'calendar.csv')
    for _,group in calendar.groupby(['method','fold']):
        train=group[group.split=='train']; val=group[group.split=='val']; test=group[group.split=='test']
        assert pd.to_datetime(train.target_end,utc=True).max()<pd.Timestamp('2024-01-01',tz='UTC')
        assert pd.to_datetime(val.origin,utc=True).min()>=pd.Timestamp('2024-01-01',tz='UTC')
        assert pd.to_datetime(val.target_end,utc=True).max()<pd.Timestamp('2025-01-01',tz='UTC')
        assert pd.to_datetime(test.origin,utc=True).min()>=pd.Timestamp('2025-01-01',tz='UTC')
    prediction=pd.read_csv(OUT/'predictions.csv.gz'); metrics=pd.read_csv(OUT/'all_metrics.csv')
    selection=pd.read_csv(OUT/'selected_inputs.csv'); allchoices=pd.read_csv(OUT/'selection_all_inputs.csv')
    assert len(allchoices)==192 and len(selection)==48
    if (OUT/'selection_cache.csv').exists():
        provenance=json.loads((OUT/'selection_cache_provenance.json').read_text())
        assert provenance['configuration']==cfg
        assert provenance['selection_cache_sha256']==hashlib.sha256((OUT/'selection_cache.csv').read_bytes()).hexdigest()
        cached=pd.read_csv(OUT/'selection_cache.csv')
        compared=cached.merge(allchoices,on=['method','symbol','volatility_window','input_window'],suffixes=('_cached','_new'),validate='one_to_one')
        assert len(compared)==len(cached)
        for field in ['C','epsilon','val_rmse']:
            np.testing.assert_allclose(compared[field+'_cached'],compared[field+'_new'],rtol=1e-5,atol=1e-6)
    minimum=allchoices.loc[allchoices.groupby(['method','symbol','volatility_window']).val_rmse.idxmin()]
    keys=['method','symbol','volatility_window','input_window']
    pd.testing.assert_frame_equal(selection[keys].reset_index(drop=True),minimum[keys].reset_index(drop=True))
    panel=load_panel()
    for (method,symbol,w,lag),group in prediction.groupby(keys):
        assert not group.duplicated(['origin','horizon']).any()
        vol=targets(panel[symbol],w)
        origins=pd.DatetimeIndex(pd.to_datetime(group.origin,utc=True))
        dates=pd.DatetimeIndex(pd.to_datetime(group.target_date,utc=True))
        np.testing.assert_allclose(group.actual,vol.loc[dates],rtol=1e-10)
        np.testing.assert_allclose(group.persistence,vol.loc[origins],rtol=1e-10)
        y=group.pivot(index='origin',columns='horizon',values='actual').to_numpy()
        for model,col in [('SVR','svr'),('Persistence','persistence')]:
            p=group.pivot(index='origin',columns='horizon',values=col).to_numpy()
            expected=scores(y,p)
            row=metrics.query('method == @method and symbol == @symbol and volatility_window == @w and input_window == @lag and model == @model and scope == "all_available_test"').iloc[0]
            for key,value in expected.items(): np.testing.assert_allclose(row[key],value,rtol=1e-10,atol=1e-10)
    chosen_metrics=metrics.merge(selection[keys],on=keys,validate='many_to_one')
    macro=pd.read_csv(OUT/'macro_metrics.csv')
    columns=['r2','rmse','mae','mse','mape']
    for row in macro.itertuples():
        frame=chosen_metrics.query('scope == @row.scope and method == @row.method and model == @row.model')
        if row.symbol!='GLOBAL_MACRO': frame=frame[frame.symbol==row.symbol]
        assert len(frame)==(16 if row.symbol=='GLOBAL_MACRO' else 4)
        np.testing.assert_allclose(frame[columns].mean(),[getattr(row,c) for c in columns],rtol=1e-10,atol=1e-10)
    sys.path.insert(0,str(ROOT))
    from fastapi.testclient import TestClient
    from app.api import app
    client=TestClient(app)
    assert client.get('/health').json()['input_frequency']=='1min'
    assert len(client.get('/models').json())==16
    for row in selection.query("method == 'ForwardChaining'").itertuples():
        artifact=joblib.load(OUT/'models'/f'{row.symbol}_v{row.volatility_window}.joblib')
        assert artifact['n_features']==row.input_window*1440
        group=prediction.query('method == "ForwardChaining" and symbol == @row.symbol and volatility_window == @row.volatility_window and input_window == @row.input_window')
        table=group.pivot(index='origin',columns='horizon',values='svr').iloc[:3]
        anchors=panel.index.get_indexer(pd.to_datetime(table.index,utc=True))
        X=windows(np.load(DATA/f'{row.symbol}.npy',mmap_mode='r'),row.input_window,anchors)
        np.testing.assert_allclose(predict_artifact(artifact,X),table.to_numpy(),rtol=1e-6,atol=1e-6)
        response=client.post('/predict',json=dict(symbol=row.symbol,volatility_window=row.volatility_window,lags=X[0].tolist()))
        assert response.status_code==200,response.text
        np.testing.assert_allclose(response.json()['prediction'],table.iloc[0].to_numpy(),rtol=1e-6,atol=1e-6)
    record=dict(status='passed',configurations=192,exported_models=16,checked_prediction_rows=len(prediction),
                checks=['data_hashes','chronological_boundaries','validation_selection','targets','persistence','metrics','minute_exports','api'])
    (OUT/'verification.json').write_text(json.dumps(record,indent=2),encoding='utf-8')
    print(json.dumps(record,indent=2))


if __name__=='__main__': verify()
