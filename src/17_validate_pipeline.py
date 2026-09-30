"""Verifica el escalado de las 168 entradas en los 25 entrenamientos reales."""
from pathlib import Path
import hashlib
import json
import runpy
import numpy as np
import pandas as pd
import nbformat

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs/tables'


def main():
    base=runpy.run_path(str(ROOT/'src/18_base_model.py'))
    source=ROOT/'data/splits/development_80.csv'
    digest=hashlib.sha256(source.read_bytes()).hexdigest()
    df=pd.read_csv(source,usecols=['symbol','open_time','close_time','close'])
    for col in ['open_time','close_time']:
        df[col]=pd.to_datetime(df[col],utc=True,format='ISO8601')
    grid=pd.date_range(df.open_time.min(),df.open_time.max(),freq='h')
    frames=base['prepare'](df,grid,168)
    folds=pd.read_csv(OUT/'base_folds.csv')
    rows=[]
    columns=[f'lag_{k}' for k in range(168)]
    for row in folds.itertuples():
        x=frames[row.symbol]
        tr=base['contained_block'](x,pd.Timestamp(row.train_block_start),pd.Timestamp(row.train_block_end))
        va=base['contained_block'](x,pd.Timestamp(row.validation_block_start),pd.Timestamp(row.validation_block_end))
        # Misma clase de scaler y configuración que el pipeline final; no ajusta SVR.
        scaler=base['model'](.01,.01).named_steps['scale']
        ztr=scaler.fit_transform(tr[columns])
        before=scaler.mean_.copy(),scaler.var_.copy(),scaler.scale_.copy()
        zva=scaler.transform(va[columns])
        altered=scaler.transform(va[columns]*2)
        assert len(tr)==row.n_train and len(va)==row.n_val
        assert np.isfinite(ztr).all() and np.isfinite(zva).all()
        np.testing.assert_allclose(scaler.mean_,tr[columns].mean(),rtol=1e-12)
        np.testing.assert_allclose(scaler.inverse_transform(zva),va[columns],rtol=1e-12)
        np.testing.assert_allclose(altered,(va[columns]*2-before[0])/before[2])
        for original,current in zip(before,[scaler.mean_,scaler.var_,scaler.scale_]):
            np.testing.assert_array_equal(original,current)
        error=float(np.abs(ztr.mean(axis=0)).max())
        assert error<1e-10
        rows.append(dict(symbol=row.symbol,fold=row.fold,features=168,n_train=len(tr),n_validation=len(va),
            max_abs_scaled_train_mean=error,mean_lag0=scaler.mean_[0],scale_lag0=scaler.scale_[0],
            validation_does_not_change_scaler=True,inverse_recovers_validation=True))
    pd.DataFrame(rows).to_csv(OUT/'preprocessing_full_audit.csv',index=False)
    assert digest==hashlib.sha256(source.read_bytes()).hexdigest()
    (OUT/'preprocessing_full_metadata.json').write_text(json.dumps(dict(development_sha256=digest,test_read=False,
        folds_sha256=hashlib.sha256((OUT/'base_folds.csv').read_bytes()).hexdigest(),checks=len(rows),features=168),indent=2),encoding='utf-8')
    print('Escalado verificado en 25 entrenamientos, 168 columnas cada uno.')


def render(write_notebook=True):
    table=runpy.run_path(str(ROOT/'src/09_render_univariate_report.py'))['table']
    meta=json.loads((OUT/'preprocessing_full_metadata.json').read_text(encoding='utf-8'))
    assert meta['development_sha256']==hashlib.sha256((ROOT/'data/splits/development_80.csv').read_bytes()).hexdigest()
    assert meta['folds_sha256']==hashlib.sha256((OUT/'base_folds.csv').read_bytes()).hexdigest()
    audit=pd.read_csv(OUT/'preprocessing_full_audit.csv')
    path=ROOT/'book/sections/02_eda.md'
    text=path.read_text(encoding='utf-8')
    text=text.replace('Cuando se entrene el regresor, se integrará como etapa final y el pipeline completo se ajustará de nuevo en cada fold.',
                      'En el modelo vigente, LinearSVR es la etapa final y el pipeline completo se ajusta de nuevo en cada fold, como se verifica en la sección 3.')
    text=text.replace('La ejecución actual verifica el preprocesamiento con una columna de cierre, sin entrenar OLS, SVR ni persistencia. Esta comprobación no fija la longitud definitiva de entrada de los modelos y no genera métricas de desempeño predictivo.',
                      'La verificación de esta sección cubre las 168 columnas y los 25 entrenamientos del protocolo vigente. Comprueba el escalado sin volver a ajustar el SVR; sus métricas predictivas están en la sección 3.')
    text=text.replace('La función `prepare_close` construye ventanas de L cierres: el último cierre observado y sus L−1 rezagos horarios. L debe especificarse explícitamente; no se selecciona en esta sección.',
                      'La función `prepare` del modelo construye ventanas de 168 cierres: el último cierre observado y sus 167 rezagos horarios. Esta longitud está fijada en el protocolo, no optimizada en esta sección.')
    text=text.replace('La comprobación ejecutada utiliza L=1 únicamente para verificar el pipeline básico. Si se amplía L, cada rezago será una columna y se ajustará su escala con el entrenamiento.',
                      'Cada uno de los 168 rezagos es una columna y su escala se ajusta con el entrenamiento del fold.')
    text=text.replace('Mantener L explícito; comparar ventanas dentro de validación, sin selección definitiva aquí.',
                      'Fijar L=168 para este experimento; cualquier comparación posterior de ventanas deberá usar validación de DEVELOPMENT.')
    before,_=text.split('### 2.9.7 Verificación ejecutada',1)
    section='### 2.9.7 Verificación ejecutada\n\nSe comprueban los 25 bloques de entrenamiento de la sección 3, con 168 entradas y confinamiento de historia y objetivo. Para cada uno se verifica que las medias del escalador coincidan con TRAIN, que transformar validación no modifique medias, varianzas ni escalas, que las salidas sean finitas y que la transformación inversa recupere los cierres. Una perturbación artificial de validación confirma que se aplican los mismos parámetros ya aprendidos. No se exige media cero en validación.\n\n'
    section+=table(['Activo','Fold','n TRAIN','n VALIDATION','Máxima media TRAIN estandarizada absoluta'],
        [[r.symbol,r.fold,r.n_train,r.n_validation,f'{r.max_abs_scaled_train_mean:.2e}'] for r in audit.itertuples()])
    section+='\n\nLos 25 controles se superan. Esto verifica propiedades concretas del preprocesamiento, no una garantía universal de ausencia de fuga. [Auditoría completa](../../outputs/tables/preprocessing_full_audit.csv).\n\n### 2.9.8 Reproducibilidad\n\nEjecutar `python src/17_validate_pipeline.py`. El script comparte preparación, fronteras y clase de escalado con el modelo; no utiliza TEST. Los metadatos conservan las huellas de DEVELOPMENT y de los folds. La comprobación histórica de una única columna en `src/17_preprocessing_close.py` queda sustituida por esta verificación del protocolo vigente. [Notebook ejecutado](../../notebooks/17_preprocessing_close.ipynb).\n'
    path.write_text(before+section,encoding='utf-8')
    if write_notebook:
        code='''from pathlib import Path
import runpy
import pandas as pd
from IPython.display import display
ROOT = next(p for p in [Path.cwd(), *Path.cwd().parents] if (p/'src/17_validate_pipeline.py').is_file())
module=runpy.run_path(str(ROOT/'src/17_validate_pipeline.py'))
RECALCULAR=False
if RECALCULAR:
    module['main']()
module['render'](False)
display(pd.read_csv(ROOT/'outputs/tables/preprocessing_full_audit.csv'))
'''
        nb=nbformat.v4.new_notebook(cells=[nbformat.v4.new_markdown_cell('# Verificación del preprocesamiento vigente'),nbformat.v4.new_code_cell(code),nbformat.v4.new_markdown_cell(section.replace('../../outputs/','../outputs/').replace('../../notebooks/','./'))])
        nb.metadata.kernelspec=dict(name='python3',display_name='Python 3',language='python')
        nbformat.write(nb,ROOT/'notebooks/17_preprocessing_close.ipynb')


if __name__=='__main__':
    main()
    render()
