"""Comprobaciones de integración de las correcciones y sus artefactos guardados."""
from pathlib import Path
import hashlib
import json
import re
import runpy
import platform
import numpy as np
import pandas as pd
import nbformat
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs/tables'


def main():
    source=ROOT/'data/splits/development_80.csv'
    digest=hashlib.sha256(source.read_bytes()).hexdigest()
    expected=json.loads((OUT/'base_metadata.json').read_text(encoding='utf-8'))['development_sha256']
    assert digest==expected
    for name in ['leakage_all_features','temporal_target','preprocessing_full']:
        meta=json.loads((OUT/f'{name}_metadata.json').read_text(encoding='utf-8'))
        assert meta['development_sha256']==digest and meta['test_read'] is False
    counts=pd.read_csv(OUT/'temporal_target_coverage.csv').set_index('symbol').total_valid.sort_index()
    original=pd.read_csv(OUT/'eda_target_summary.csv').set_index('symbol').n.sort_index()
    np.testing.assert_array_equal(counts,original)
    residuals=pd.read_csv(OUT/'base_residual_diagnostics_all.csv')
    assert len(residuals)==50 and not residuals.duplicated(['symbol','fold','model']).any()
    assert residuals.groupby(['symbol','model']).size().eq(5).all()
    scores=pd.read_csv(OUT/'leakage_all_features_scores.csv')
    assert len(scores)==4475 and not scores.duplicated(['symbol','fold','variable']).any()
    availability=pd.read_csv(OUT/'leakage_feature_availability.csv')
    assert len(availability)==179
    assert availability.available_by_anchor_plus_hours.le(availability.prediction_anchor_plus_hours).all()
    # Contraste independiente de la OLS vectorizada con sklearn, dos entradas y dos activos.
    df=pd.read_csv(source,usecols=['symbol','open_time','close_time','close'])
    for col in ['open_time','close_time']:
        df[col]=pd.to_datetime(df[col],utc=True,format='ISO8601')
    base=runpy.run_path(str(ROOT/'src/18_base_model.py'))
    frames=base['prepare'](df,pd.date_range(df.open_time.min(),df.open_time.max(),freq='h'),168)
    folds=pd.read_csv(OUT/'base_folds.csv')
    independent=0
    for symbol in ['BTCUSDT','XRPUSDT']:
        row=folds.loc[folds.symbol.eq(symbol)&folds.fold.eq(3)].iloc[0]
        tr=base['contained_block'](frames[symbol],pd.Timestamp(row.train_block_start),pd.Timestamp(row.train_block_end))
        va=base['contained_block'](frames[symbol],pd.Timestamp(row.validation_block_start),pd.Timestamp(row.validation_block_end))
        for variable in ['lag_0','lag_167']:
            model=make_pipeline(StandardScaler(),LinearRegression()).fit(tr[[variable]],tr.y)
            prediction=model.predict(va[[variable]])
            recorded=scores.query('symbol == @symbol and fold == 3 and variable == @variable').iloc[0]
            np.testing.assert_allclose([np.sqrt(mean_squared_error(va.y,prediction)),r2_score(va.y,prediction)],
                                       [recorded.rmse,recorded.r2],rtol=1e-9,atol=1e-10)
            independent+=1
    daily=pd.read_csv(OUT/'daily_nonoverlapping_volatility.csv',parse_dates=['day'])
    derived=pd.read_csv(OUT/'eda_target_development_derived.csv',parse_dates=['open_time'])
    expected_daily=derived.loc[derived.open_time.dt.hour.eq(23)].copy()
    expected_daily['day']=expected_daily.open_time+pd.Timedelta(hours=1)
    joined=daily.merge(expected_daily[['day','symbol','rv_future_24h_pct']],on=['day','symbol'],validate='one_to_one',how='outer',indicator=True)
    assert joined['_merge'].eq('both').all()
    np.testing.assert_allclose(joined.volatility,joined.rv_future_24h_pct,equal_nan=True,rtol=1e-10)
    assert len(pd.read_csv(OUT/'change_points.csv'))==15
    assert len(pd.read_csv(OUT/'event_windows.csv'))==15
    notebooks=['09_univariate_development','13_feature_diagnostics','14_temporal_target','17_preprocessing_close','18_base_model','22_change_events']
    code_cells=0
    for name in notebooks:
        nb=nbformat.read(ROOT/'notebooks'/f'{name}.ipynb',as_version=4)
        for cell in nb.cells:
            if cell.cell_type=='code':
                assert cell.execution_count is not None,(name,'sin ejecutar')
                assert not any(o.output_type=='error' for o in cell.outputs),(name,'error')
                code_cells+=1
    checked_links=0
    for path in (ROOT/'book/sections').glob('*.md'):
        text=path.read_text(encoding='utf-8')
        for target in re.findall(r'\]\(([^)]+)\)',text)+re.findall(r'^```\{figure\} ([^\n]+)',text,flags=re.M):
            target=target.strip().split('#')[0]
            if target and not re.match(r'\w+://',target):
                assert (path.parent/target).resolve().exists(),(str(path),target)
                checked_links+=1
    for path in (ROOT/'src').glob('*.py'):
        compile(path.read_text(encoding='utf-8-sig'),str(path),'exec')
    assert digest==hashlib.sha256(source.read_bytes()).hexdigest()
    result=dict(development_sha256=digest,test_read=False,python=platform.python_version(),
        independent_OLS_checks=independent,executed_notebooks=notebooks,executed_code_cells=code_cells,
        checked_local_links=checked_links,coverage_matches_target=True,residual_cases=50,
        feature_evaluations=4475,daily_labels_match_hourly_definition=True,
        clean_environment_rebuild=json.loads((OUT/'clean_reproduction_metadata.json').read_text(encoding='utf-8')).get('status')=='passed' if (OUT/'clean_reproduction_metadata.json').exists() else False)
    (OUT/'correction_validation.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    main()
