"""Pipeline de escalado y verificación temporal, sin estimador predictivo."""
from pathlib import Path
import hashlib,json
import numpy as np
import pandas as pd
import sklearn
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
ROOT = Path(__file__).resolve().parents[1]
def make_preprocessor():
    return Pipeline([('scale',StandardScaler())])
def prepare_close(g,grid,window):
    if not isinstance(window,int) or window<1:raise ValueError('window debe ser entero positivo')
    g=g.set_index('open_time').reindex(grid)
    valid=g.close_time.eq(grid+pd.Timedelta(hours=1)-pd.Timedelta(milliseconds=1))
    c=g.close.where(valid & np.isfinite(g.close) & g.close.gt(0))
    X=pd.DataFrame({f'close_lag_{lag}h':c.shift(lag) for lag in range(window)},index=grid)
    r=np.log(c).diff();y=100*r.rolling(24,min_periods=24).std(ddof=0).shift(-24)
    eligible=X.notna().all(axis=1)&y.notna()
    return X.loc[eligible],y.loc[eligible]
def main():
    source=ROOT/'data/splits/development_80.csv';sha=hashlib.sha256(source.read_bytes()).hexdigest()
    df=pd.read_csv(source,usecols=['symbol','open_time','close_time','close'])
    for field in ['open_time','close_time']:df[field]=pd.to_datetime(df[field],utc=True,format='ISO8601')
    assert not df.duplicated(['symbol','open_time']).any()
    times=pd.DatetimeIndex(sorted(df.open_time.unique()));cut=times[int(.8*len(times))]
    grid=pd.date_range(times.min(),times.max(),freq='h');rows=[]
    for symbol,g in df.groupby('symbol'):
        X,y=prepare_close(g,grid,window=1)
        train=X.index+pd.Timedelta(hours=24)<cut;val=X.index>=cut
        assert X.index[train].max()+pd.Timedelta(hours=24)<X.index[val].min()
        pipe=make_preprocessor();ztrain=pipe.fit_transform(X.loc[train])
        scaler=pipe.named_steps['scale'];mu=scaler.mean_.copy();var=scaler.var_.copy()
        zval=pipe.transform(X.loc[val])
        assert np.isfinite(ztrain).all() and np.isfinite(zval).all()
        assert np.allclose(mu,X.loc[train].mean().to_numpy())
        assert np.array_equal(mu,scaler.mean_) and np.array_equal(var,scaler.var_)
        assert np.allclose(ztrain.mean(axis=0),0,atol=1e-10)
        assert np.allclose(pipe.inverse_transform(zval),X.loc[val])
        assert np.allclose(pipe.transform(X.loc[val]*2), (X.loc[val]*2-mu)/scaler.scale_)
        rows.append(dict(symbol=symbol,train_rows=int(train.sum()),validation_rows=int(val.sum()),purged_rows=int((~train&~val).sum()),training_mean=float(mu[0]),training_scale=float(scaler.scale_[0]),train_z_mean=float(ztrain.mean()),validation_z_mean=float(zval.mean())))
    assert sha==hashlib.sha256(source.read_bytes()).hexdigest()
    out=ROOT/'outputs/tables';result=pd.DataFrame(rows);result.to_csv(out/'preprocessing_close_audit.csv',index=False)
    (out/'preprocessing_metadata.json').write_text(json.dumps({'source':'data/splits/development_80.csv','sha256':sha,'validation_anchor_start':str(cut),'demonstration_window':1,'final_window_selected':False,'imputation':False,'outliers_removed':False,'predictive_model_trained':False,'test_read':False,'sklearn':sklearn.__version__},indent=2),encoding='utf-8')
    print(result.to_string(index=False))
if __name__=='__main__':main()
