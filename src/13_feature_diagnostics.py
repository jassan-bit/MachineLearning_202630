"""OLS univariadas diagnósticas: todas las entradas y candidatos originales, sin TEST."""
from pathlib import Path
import hashlib
import json
import runpy
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'outputs/tables'
BASE = runpy.run_path(str(ROOT/'src/18_base_model.py'))
RAW = ['open','high','low','volume','quote_asset_volume','number_of_trades',
       'taker_buy_base_asset_volume','taker_buy_quote_asset_volume']


def main():
    source = ROOT/'data/splits/development_80.csv'
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    df = pd.read_csv(source)
    for col in ['open_time','close_time']:
        df[col] = pd.to_datetime(df[col], utc=True, format='ISO8601')
    grid = pd.date_range(df.open_time.min(), df.open_time.max(), freq='h')
    frames = BASE['prepare'](df,grid,168)
    folds = pd.read_csv(OUT/'base_folds.csv')
    results, availability = [], []
    for symbol, original in df.groupby('symbol'):
        x = frames[symbol].copy()
        original = original.set_index('open_time').reindex(grid)
        for col in RAW:
            x[col] = original[col].reindex(x.index)
        returns = 100*np.log(original.close.where(original.close_time.eq(grid+pd.Timedelta(hours=1)-pd.Timedelta(milliseconds=1)))).diff()
        x['return_1h_pct'] = returns.reindex(x.index)
        x['abs_return_1h_pct'] = x.return_1h_pct.abs()
        x['past_volatility_24h_pct'] = x.persistence
        columns = [c for c in x if c not in ['y','persistence']]
        assert len(columns) == 179 and np.isfinite(x[columns]).all().all()
        for row in folds.loc[folds.symbol.eq(symbol)].itertuples():
            tr = BASE['contained_block'](x,pd.Timestamp(row.train_block_start),pd.Timestamp(row.train_block_end))
            va = BASE['contained_block'](x,pd.Timestamp(row.validation_block_start),pd.Timestamp(row.validation_block_end))
            assert len(tr) == row.n_train and len(va) == row.n_val
            mu, scale = tr[columns].mean().to_numpy(), tr[columns].std(ddof=0).to_numpy()
            assert (scale>0).all()
            z = (tr[columns].to_numpy()-mu)/scale
            ymean = tr.y.mean()
            slope = np.mean(z*(tr.y.to_numpy()-ymean)[:,None],axis=0)
            predicted = ymean+(va[columns].to_numpy()-mu)/scale*slope
            errors = va.y.to_numpy()[:,None]-predicted
            rmse = np.sqrt(np.mean(errors**2,axis=0))
            r2 = 1-np.sum(errors**2,axis=0)/np.sum((va.y-va.y.mean())**2)
            baseline = np.sqrt(np.mean((va.y-ymean)**2))
            for j,col in enumerate(columns):
                results.append(dict(symbol=symbol,fold=row.fold,variable=col,n_train=len(tr),n_validation=len(va),
                    train_mean=mu[j],train_scale=scale[j],coefficient_standardized=slope[j],
                    intercept=ymean,rmse=rmse[j],r2=r2[j],train_mean_baseline_rmse=baseline,
                    high_score_alert=bool(r2[j]>=.8)))
        print('Diagnóstico univariado completado:',symbol,flush=True)
    for col in columns:
        lag = int(col.split('_')[1]) if col.startswith('lag_') else 0
        history = lag if col.startswith('lag_') else (24 if col=='past_volatility_24h_pct' else (1 if 'return' in col else 0))
        availability.append(dict(variable=col,alias='close' if col=='lag_0' else '',
            first_required_opening_offset_hours=-history,last_required_opening_offset_hours=-lag,
            available_by_anchor_plus_hours=1-lag, prediction_anchor_plus_hours=1,
            role='SVR_input' if col.startswith('lag_') else 'diagnostic_only',
            future_target_used_as_input=False))
    pd.DataFrame(results).to_csv(OUT/'leakage_all_features_scores.csv',index=False)
    pd.DataFrame(availability).to_csv(OUT/'leakage_feature_availability.csv',index=False)
    assert digest == hashlib.sha256(source.read_bytes()).hexdigest()
    metadata = dict(development_sha256=digest, test_read=False, features=179,
        evaluations=len(results), regression='independent_OLS_with_intercept_train_only_scaling',
        threshold_r2=.8, selection_performed=False, model_protocol_changed=False,
        folds_sha256=hashlib.sha256((OUT/'base_folds.csv').read_bytes()).hexdigest())
    (OUT/'leakage_all_features_metadata.json').write_text(json.dumps(metadata,indent=2),encoding='utf-8')


if __name__ == '__main__':
    main()
