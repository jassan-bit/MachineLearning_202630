"""Auditoría temporal y diagnóstico univariado; solo DEVELOPMENT."""
from pathlib import Path
import hashlib, json
import numpy as np
import pandas as pd
ROOT = Path(__file__).resolve().parents[1]
def main():
    source=ROOT/'data/splits/development_80.csv'
    sha=hashlib.sha256(source.read_bytes()).hexdigest()
    df=pd.read_csv(source)
    df['open_time']=pd.to_datetime(df.open_time,utc=True)
    df['close_time']=pd.to_datetime(df.close_time,utc=True,format='ISO8601')
    times=pd.DatetimeIndex(sorted(df.open_time.unique()))
    boundary=times[int(len(times)*.8)]
    grid=pd.date_range(times.min(),times.max(),freq='h')
    rows=[]
    for symbol,g in df.groupby('symbol'):
        g=g.set_index('open_time').reindex(grid)
        c=g.close.where(g.close_time.eq(grid+pd.Timedelta(hours=1)-pd.Timedelta(milliseconds=1)))
        r=np.log(c).diff()
        y=100*r.rolling(24,min_periods=24).std(ddof=0).shift(-24)
        sample=pd.DataFrame({'close':c,'target':y}).dropna()
        # Última vela objetivo debe ser anterior al primer ancla de validación.
        train=sample.loc[sample.index+pd.Timedelta(hours=24)<boundary]
        val=sample.loc[sample.index>=boundary]
        purged=((sample.index<boundary)&(sample.index+pd.Timedelta(hours=24)>=boundary)).sum()
        assert train.index.max()+pd.Timedelta(hours=24)<val.index.min()
        assert sample.index.max()+pd.Timedelta(hours=24)<=times.max()
        mean=train.close.mean();scale=train.close.std(ddof=0)
        x=np.column_stack([np.ones(len(train)),(train.close-mean)/scale])
        beta=np.linalg.lstsq(x,train.target,rcond=None)[0]
        pred=beta[0]+beta[1]*(val.close.to_numpy()-mean)/scale
        real=val.target.to_numpy();base=np.full(len(real),train.target.mean())
        def rmse(p): return float(np.sqrt(np.mean((real-p)**2)))
        r2=1-np.sum((real-pred)**2)/np.sum((real-real.mean())**2)
        rows.append(dict(symbol=symbol,n_train=len(train),n_validation=len(val),purged=int(purged),rmse_close=rmse(pred),mae_close=float(np.mean(abs(real-pred))),r2_close=float(r2),rmse_train_mean=rmse(base)))
    summary=pd.DataFrame(rows)
    split=pd.read_csv(ROOT/'outputs/tables/chronological_split_audit.csv').set_index('split')
    assert pd.Timestamp(split.loc['DEVELOPMENT','end_time'])<pd.Timestamp(split.loc['TEST','start_time'])
    assert times.max()==pd.Timestamp(split.loc['DEVELOPMENT','end_time'])
    assert sha==hashlib.sha256(source.read_bytes()).hexdigest()
    meta={'development_sha256':sha,'validation_anchor_start':str(boundary),'development_end':str(times.max()),'duplicate_rows_development':int(df.duplicated().sum()),'duplicate_keys_development':int(df.duplicated(['symbol','open_time']).sum()),'test_read':False,'cross_partition_evidence':'Existing split metadata only; no fresh TEST scan','diagnostic':'OLS close only, intercept, chronological 80/20 within DEVELOPMENT, purge by target end, training-only scale','target':'100 * future rolling std of 24 hourly log returns, ddof=0'}
    out=ROOT/'outputs/tables'
    summary.to_csv(out/'leakage_close_diagnostic.csv',index=False)
    (out/'leakage_audit_metadata.json').write_text(json.dumps(meta,indent=2,ensure_ascii=False),encoding='utf-8')
    print(summary.to_string(index=False));print(json.dumps(meta,indent=2,ensure_ascii=False))
    return summary,meta
if __name__=='__main__':
    summary, metadata = main()

