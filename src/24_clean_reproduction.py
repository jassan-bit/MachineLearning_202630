"""Reejecución computacional aislada, desde DEVELOPMENT, sin copiar TEST."""
from pathlib import Path
import hashlib, json, shutil, subprocess, sys, time
from datetime import datetime
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]


def main():
    assert sys.prefix!=sys.base_prefix,'Ejecutar con .venv-repro/Scripts/python.exe'
    assert Path(np.__file__).resolve().is_relative_to(Path(sys.prefix).resolve())
    destination=(ROOT/'outputs'/('clean_reproduction_'+datetime.now().strftime('%Y%m%d_%H%M%S_%f'))).resolve()
    assert destination.is_relative_to((ROOT/'outputs').resolve())
    destination.mkdir(exist_ok=False)
    for folder in ['src','tests','data/splits','outputs/tables','outputs/models','book/sections','book/_static/figures','notebooks','logs']:
        (destination/folder).mkdir(parents=True,exist_ok=True)
    for folder,pattern in [('src','*.py'),('src','*.cjs'),('tests','*.py'),('book/sections','*.md')]:
        for source in (ROOT/folder).glob(pattern):shutil.copy2(source,destination/folder/source.name)
    for name in ['base_model_protocol.json','multivariate_protocol.json','change_events_protocol.json','base_report_template.md','entregable1_checklist.csv']:
        shutil.copy2(ROOT/'outputs/tables'/name,destination/'outputs/tables'/name)
    shutil.copy2(ROOT/'book/entregable1_master.md',destination/'book/entregable1_master.md')
    shutil.copy2(ROOT/'data/splits/development_80.csv',destination/'data/splits/development_80.csv')
    manifest=dict(python=sys.version,interpreter=sys.executable,isolated_environment=True,
        test_copied=False,test_read=False,status='running',stages=[],
        scope='all_development_analysis_and_170_model_fits; source_download_and_original_split_not_repeated')
    manifest['workspace']=str(destination)
    manifest['development_sha256']=hashlib.sha256((destination/'data/splits/development_80.csv').read_bytes()).hexdigest()
    manifest['source_hashes']={p.relative_to(destination).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in (destination/'src').glob('*') if p.is_file()}
    manifest_path=ROOT/'outputs/tables/clean_reproduction_metadata.json'
    def save():manifest_path.write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    freeze=subprocess.check_output([sys.executable,'-m','pip','freeze'],text=True)
    (ROOT/'requirements-lock-windows-py310.txt').write_text(freeze,encoding='utf-8')
    stages=[['-m','pip','check'],['-m','unittest','discover','-s','tests','-v'],
        '04_missingness_development.cjs','05_duplicates_development.cjs','06_outliers_development.py',
        '07_consistency_development.py','08_eda_target_development.py','09_univariate_development.py',
        '11_bivariate_close.py','14_temporal_close.py','18_base_model.py','12_multivariate_close.py',
        '13_feature_diagnostics.py','14_temporal_target.py','17_validate_pipeline.py','19_extended_diagnostics.py','22_change_events.py']
    for i,stage in enumerate(stages):
        command=([sys.executable,*stage] if isinstance(stage,list) else
                 [shutil.which('node') if stage.endswith('.cjs') else sys.executable,'src/'+stage])
        started=time.perf_counter()
        log=destination/'logs'/f'{i:02d}.txt'
        print('Reproducción:',stage,flush=True)
        with log.open('w',encoding='utf-8') as handle:
            result=subprocess.run(command,cwd=destination,stdout=handle,stderr=subprocess.STDOUT)
        manifest['stages'].append(dict(command=command,exit_code=result.returncode,seconds=time.perf_counter()-started,log=str(log)))
        if result.returncode:
            manifest['status']='failed';save();raise RuntimeError('Etapa fallida; consultar '+str(log))
        save()
    comparisons={
        'base_selection.csv':['mean_rmse'],
        'base_metrics_by_fold.csv':['rmse','mape','r2'],
        'base_confidence_intervals.csv':['estimate','low','high'],
        'base_learning_curve.csv':['train_rmse','validation_rmse'],
        'eda_target_summary.csv':['mean','median','std'],
        'univariate_summary.csv':['mean','median','std'],
        'multivariate_summary.csv':['pc1_ratio','effective_rank'],
        'leakage_all_features_scores.csv':['rmse','r2'],
        'temporal_target_stationarity.csv':['adf_stat','kpss_stat'],
        'base_residual_diagnostics_all.csv':['mean_residual','acf_1h'],
        'base_bootstrap_sensitivity.csv':['estimate','low','high'],
        'change_points.csv':['gain_ratio','p_raw','p_holm'],
        'event_windows.csv':['mean_before','mean_after']}
    checks=[]
    for name,cols in comparisons.items():
        expected=pd.read_csv(ROOT/'outputs/tables'/name)[cols]
        actual=pd.read_csv(destination/'outputs/tables'/name)[cols]
        try:
            np.testing.assert_allclose(actual,expected,rtol=1e-5,atol=1e-8,equal_nan=True,err_msg=name)
        except AssertionError:
            manifest['status']='comparison_failed';manifest['failed_comparison']=name;save();raise
        checks.append(dict(file=name,rows=len(actual),columns=cols,max_abs_difference=float(np.nanmax(np.abs(actual.to_numpy()-expected.to_numpy())))))
    manifest['comparisons']=checks;manifest['status']='passed'
    assert manifest['development_sha256']==hashlib.sha256((destination/'data/splits/development_80.csv').read_bytes()).hexdigest()
    save();print('Reproducción limpia completada.',flush=True)


if __name__=='__main__':main()
