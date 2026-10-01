"""Minute features and mathematically equivalent row-space linear SVR."""
from pathlib import Path
import hashlib
import json
import os
import warnings
import zipfile
import numpy as np
import pandas as pd
from sklearn.compose import TransformedTargetRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVR
from sklearn.exceptions import ConvergenceWarning

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results/minute_2023_2025'
DATA=ROOT/'data/processed/minute_2023_2025'


def configuration():
    return json.loads((ROOT/'experiment_minute.json').read_text(encoding='utf-8'))


def prepare_data():
    cfg=configuration(); DATA.mkdir(parents=True,exist_ok=True); OUT.mkdir(parents=True,exist_ok=True)
    start=pd.Timestamp(cfg['start'],tz='UTC'); end=pd.Timestamp(cfg['end_exclusive'],tz='UTC')
    ndays=(end-start).days; nminutes=ndays*1440
    calendar=pd.date_range(start,end,freq='D',inclusive='left')
    panel={}; audit=[]
    for symbol in cfg['symbols']:
        values=np.full(nminutes,np.nan,dtype=np.float64)
        for month in pd.period_range('2023-01','2025-12',freq='M').astype(str):
            path=ROOT/f'data/raw/minute_2020_2025/{symbol}/{symbol}-1m-{month}.zip'
            expected=path.with_suffix('.zip.CHECKSUM').read_text().split()[0]
            sha=hashlib.sha256(path.read_bytes()).hexdigest()
            if sha!=expected: raise ValueError(f'Invalid source checksum: {path}')
            with zipfile.ZipFile(path) as z:
                with z.open(z.namelist()[0]) as stream:
                    frame=pd.read_csv(stream,header=None,usecols=[0,4,6])
            raw=frame[0].to_numpy(dtype=np.int64)
            unit='us' if raw[0]>10**14 else 'ms'
            stamp=pd.DatetimeIndex(pd.to_datetime(raw,unit=unit,utc=True))
            month_start=pd.Timestamp(month+'-01',tz='UTC')
            if stamp.has_duplicates or not stamp.is_monotonic_increasing:
                raise ValueError('Duplicate or unordered minute source')
            assert ((stamp>=month_start)&(stamp<month_start+pd.offsets.MonthBegin(1))).all()
            assert (stamp==stamp.floor('min')).all()
            closing=pd.DatetimeIndex(pd.to_datetime(frame[6],unit=unit,utc=True))
            good=closing==stamp+pd.Timedelta(minutes=1)-pd.Timedelta(1,unit=unit)
            close=frame[4].to_numpy(float)
            if not np.isfinite(close).all() or (close<=0).any(): raise ValueError('Invalid price')
            position=np.asarray((stamp-start)//pd.Timedelta(minutes=1),dtype=int)
            values[position[good]]=close[good]
            audit.append(dict(symbol=symbol,month=month,sha256=sha,source=str(path.relative_to(ROOT)),
                rows=len(frame),invalid_close_times=int((~good).sum()),timestamp_unit=unit))
        np.save(DATA/f'{symbol}.npy',values)
        days=values.reshape(ndays,1440)
        complete=np.isfinite(days).all(axis=1)
        panel[symbol]=np.where(complete,days[:,-1],np.nan)
        print(f'{symbol}: {np.isfinite(values).sum():,} valid minute closes; {complete.sum()} complete days',flush=True)
    panel=pd.DataFrame(panel,index=calendar); panel.index.name='date'
    panel.to_csv(DATA/'daily_target_closes.csv')
    pd.DataFrame(audit).to_csv(OUT/'sources.csv',index=False)
    manifest={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(DATA.glob('*')) if p.is_file()}
    (OUT/'data_manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    return panel


def load_panel():
    panel=pd.read_csv(DATA/'daily_target_closes.csv',index_col=0)
    panel.index=pd.to_datetime(panel.index,utc=True)
    return panel


def windows(minutes, lag_days, anchors):
    """One row per daily origin; EVERY minute from L calendar days included."""
    anchors=np.asarray(anchors,dtype=int)
    if (anchors<lag_days-1).any(): raise ValueError('Insufficient history')
    view=np.lib.stride_tricks.sliding_window_view(minutes,lag_days*1440)[::1440]
    result=np.asarray(view[anchors-lag_days+1],dtype=float)
    if not np.isfinite(result).all(): raise ValueError('Missing minutes in input')
    return result


class RowSpace:
    """Orthonormal basis of scaled TRAIN rows; preserves the linear model.

    Unlike PCA feature selection, all numerically nonzero training directions
    are retained. The L2-regularized optimum has no component orthogonal to
    the training rows; the exported coefficient vector uses all minute columns.
    """
    def fit(self,X):
        self.scaler=StandardScaler().fit(X)
        Z=self.scaler.transform(X)
        gram=Z@Z.T
        eigen,U=np.linalg.eigh(gram)
        threshold=max(float(eigen[-1])*1e-12,1e-12)
        keep=eigen>threshold
        self.eigen=eigen[keep]
        self.basis=Z.T@(U[:,keep]/np.sqrt(self.eigen))
        self.train=Z@self.basis
        error=np.linalg.norm(self.train@self.train.T-gram)/max(np.linalg.norm(gram),1)
        if error>1e-9: raise ValueError(f'Linear Gram reconstruction error: {error}')
        self.gram_relative_error=float(error)
        return self

    def transform(self,X):
        return self.scaler.transform(X)@self.basis

    def export(self,model):
        coefficients=np.column_stack([m.coef_ for m in model.regressor_.estimators_])
        intercept=np.array([m.intercept_[0] for m in model.regressor_.estimators_])
        ys=model.transformer_
        weights=(self.basis@coefficients)*ys.scale_[None,:]/self.scaler.scale_[:,None]
        offset=intercept*ys.scale_+ys.mean_-self.scaler.mean_@weights
        return dict(weights=weights,intercept=offset,n_features=int(len(self.scaler.mean_)),
                    rank=int(len(self.eigen)),gram_relative_error=self.gram_relative_error)


class StrictLinearSVR(LinearSVR):
    def fit(self,X,y,sample_weight=None):
        # Enforce convergence checks inside each parallel worker as well.
        with warnings.catch_warnings():
            warnings.simplefilter('error',ConvergenceWarning)
            return super().fit(X,y,sample_weight=sample_weight)


def estimator(C,epsilon,seed=42):
    return TransformedTargetRegressor(regressor=MultiOutputRegressor(StrictLinearSVR(C=C,epsilon=epsilon,
        loss='squared_epsilon_insensitive',dual=False,tol=1e-6,max_iter=50000,random_state=seed),
        n_jobs=int(os.environ.get('SVR_N_JOBS','1'))),
        transformer=StandardScaler())


def predict_artifact(artifact,X):
    X=np.asarray(X,dtype=float)
    if X.ndim!=2 or X.shape[1]!=artifact['n_features'] or not np.isfinite(X).all():
        raise ValueError('Invalid minute features')
    return X@artifact['weights']+artifact['intercept']


if __name__=='__main__': prepare_data()
