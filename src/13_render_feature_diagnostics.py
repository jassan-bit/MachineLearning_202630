"""Actualiza auditoría individual con los folds estrictos vigentes."""
from pathlib import Path
import hashlib
import json
import re
import runpy
import nbformat
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'outputs/tables'
table = runpy.run_path(str(ROOT/'src/09_render_univariate_report.py'))['table']


def render(write_notebook=True):
    meta=json.loads((OUT/'leakage_all_features_metadata.json').read_text(encoding='utf-8'))
    assert meta['development_sha256']==hashlib.sha256((ROOT/'data/splits/development_80.csv').read_bytes()).hexdigest()
    assert meta['folds_sha256']==hashlib.sha256((OUT/'base_folds.csv').read_bytes()).hexdigest()
    scores=pd.read_csv(OUT/'leakage_all_features_scores.csv')
    availability=pd.read_csv(OUT/'leakage_feature_availability.csv')
    assert len(scores)==4475 and len(availability)==179
    lag=scores.loc[scores.variable.str.startswith('lag_')]
    candidate=scores.loc[~scores.variable.str.startswith('lag_')]
    text=['### 2.5.3 Diagnóstico individual de todas las entradas',
        'Se reemplaza el diagnóstico limitado a close por **179 regresiones univariadas por activo y fold**: 168 cierres rezagados (`lag_0` es close), las otras ocho columnas numéricas originales, retorno horario, retorno absoluto y volatilidad histórica de 24 horas. Son **4.475 evaluaciones** en los mismos cinco folds cronológicos de la sección 3. Cada OLS con intercepto utiliza una única columna y escalado ajustado únicamente con TRAIN; no hay búsqueda de hiperparámetros. El modelo base sigue siendo el SVR con 168 cierres: estas OLS son diagnósticas.',
        'Se exige la misma elegibilidad y el mismo confinamiento temporal del modelo: historia de 168 cierres, objetivo y referencia completos dentro de cada bloque. Las muestras de entrenamiento y validación coinciden con la auditoría `base_folds.csv`. La referencia constante usa la media de y de TRAIN. RMSE se expresa en puntos porcentuales y R² compara con la media observada del bloque validado. AUC no aplica a regresión.',
        table(['Activo','Evaluaciones de rezagos','RMSE mínimo–máximo','R² mínimo–máximo','Alertas R² ≥ 0,8'],
            [[s,len(g),f'{g.rmse.min():.4f} a {g.rmse.max():.4f}',f'{g.r2.min():.4f} a {g.r2.max():.4f}',int(g.high_score_alert.sum())] for s,g in lag.groupby('symbol')]),
        'Los rangos reúnen los 168 rezagos y cinco folds de cada activo, no son intervalos de confianza ni resultados de una selección. Los candidatos originales y derivados se resumen sobre los 25 bloques activo-fold:',
        table(['Variable','RMSE medio','R² mínimo–máximo','Alertas R² ≥ 0,8'],
            [[v,f'{g.rmse.mean():.4f}',f'{g.r2.min():.4f} a {g.r2.max():.4f}',int(g.high_score_alert.sum())] for v,g in candidate.groupby('variable')]),
        f'Se registran **{int(scores.high_score_alert.sum())} alertas** con el umbral descriptivo R² ≥ 0,8 de la guía. Un resultado alto es una señal para revisar disponibilidad y alineación, no prueba de fuga; un resultado bajo tampoco acredita su ausencia. Se conservan todas las evaluaciones, sin escoger columnas según su mejor fold ni interpretar el máximo entre miles de evaluaciones como evidencia confirmatoria.',
        '[Resultados de las 4.475 evaluaciones](../../outputs/tables/leakage_all_features_scores.csv). El archivo guarda parámetros del escalado, coeficiente, intercepto, conteos y error de la media de TRAIN. No se realizan pruebas de significación ni se usan estos resultados para modificar el modelo.',
        '**Disponibilidad.** Para cada rezago k, el cierre procede de la vela s−k y está disponible nominalmente en s−k+1 hora. Los ocho campos numéricos de la vela actual se consideran disponibles solo cuando termina esa vela, en s+1 hora. El retorno necesita los cierres s−1 y s; la volatilidad pasada necesita los cierres s−24 a s. Todos preceden o coinciden con la emisión nominal s+1 hora. Las claves symbol y fechas no son predictores. La etiqueta futura nunca aparece en las entradas.',
        '[Registro de disponibilidad de las 179 variables](../../outputs/tables/leakage_feature_availability.csv). La disponibilidad es nominal tras el cierre; los archivos históricos no registran latencia real de recepción. Un sistema operativo deberá esperar confirmación de la vela cerrada. El objetivo conserva sus propias 24 horas futuras y solo se usa como etiqueta en el ajuste o la evaluación.',
        'La volatilidad pasada comparte fórmula con el objetivo, pero utiliza otro intervalo temporal: no es fuga por ese solo hecho. Se mantienen separadas las variables disponibles, los candidatos excluidos por diseño y los datos futuros prohibidos. Este diagnóstico no certifica ausencia universal de fuga ni decide transformaciones a partir de TEST.']
    section='\n\n'.join(text)
    path=ROOT/'book/sections/02_eda.md'
    doc=path.read_text(encoding='utf-8')
    begin,rest=doc.split('### 2.5.3 ',1)
    _,tail=rest.split('### 2.5.4 ',1)
    doc=begin+section+'\n\n### 2.5.4 '+tail
    # Keep the old audit as provenance, identify the new executable evidence.
    start,end=doc.split('### 2.5.7 Reproducibilidad',1)
    _,after=end.split('## 2.6 ',1)
    doc=start+'### 2.5.7 Reproducibilidad\n\nEjecutar `python src/13_feature_diagnostics.py` y `python src/13_render_feature_diagnostics.py`. Las tablas `leakage_all_features_scores.csv` y `leakage_feature_availability.csv` y los metadatos guardan el diagnóstico vigente y sus huellas. El script anterior `src/13_leakage_audit.py` se conserva como antecedente; su diagnóstico de un solo cierre no sustituye esta evaluación de todas las entradas. [Notebook de auditoría individual](../../notebooks/13_feature_diagnostics.ipynb).\n\n## 2.6 '+after
    path.write_text(doc,encoding='utf-8')
    if write_notebook:
        code='''from pathlib import Path
import runpy
import pandas as pd
from IPython.display import display
ROOT = next(p for p in [Path.cwd(), *Path.cwd().parents] if (p/'src/13_feature_diagnostics.py').is_file())
RECALCULAR = False
if RECALCULAR:
    runpy.run_path(str(ROOT/'src/13_feature_diagnostics.py'), run_name='__main__')
runpy.run_path(str(ROOT/'src/13_render_feature_diagnostics.py'))['render'](False)
display(pd.read_csv(ROOT/'outputs/tables/leakage_feature_availability.csv'))
display(pd.read_csv(ROOT/'outputs/tables/leakage_all_features_scores.csv').groupby('symbol')[['rmse','r2']].agg(['min','mean','max']))
'''
        nb=nbformat.v4.new_notebook(cells=[nbformat.v4.new_markdown_cell('# Auditoría de disponibilidad y diagnóstico individual'),nbformat.v4.new_code_cell(code),nbformat.v4.new_markdown_cell(section.replace('../../outputs/','../outputs/'))])
        nb.metadata.kernelspec=dict(name='python3',display_name='Python 3',language='python')
        nbformat.write(nb,ROOT/'notebooks/13_feature_diagnostics.ipynb')
    print('Auditoría individual integrada.')


if __name__=='__main__':
    render()
