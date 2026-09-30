"""Sincroniza la sección 3 y su notebook con los resultados del protocolo vigente."""
from pathlib import Path
import csv
import hashlib
import json
import re
import pandas as pd
import nbformat

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'outputs/tables'


def table(headers, rows):
    return '\n'.join(['| '+' | '.join(headers)+' |',
                      '| '+' | '.join(['---']*len(headers))+' |'] +
                     ['| '+' | '.join(map(str,r))+' |' for r in rows])


def render(write_notebook=True):
    meta=json.loads((OUT/'base_metadata.json').read_text(encoding='utf-8'))
    assert meta['boundary_policy']=='strict_block_containment'
    assert meta['protocol_sha256']==hashlib.sha256((OUT/'base_model_protocol.json').read_bytes()).hexdigest()
    folds=pd.read_csv(OUT/'base_folds.csv').drop_duplicates('fold')
    scores=pd.read_csv(OUT/'base_selection.csv')
    metrics=pd.read_csv(OUT/'base_metrics_by_fold.csv')
    cis=pd.read_csv(OUT/'base_confidence_intervals.csv')
    extended=json.loads((OUT/'extended_diagnostics_metadata.json').read_text(encoding='utf-8'))
    for path, digest in extended['source_hashes'].items():
        assert hashlib.sha256((ROOT/path).read_bytes()).hexdigest()==digest, 'Reejecutar src/19_extended_diagnostics.py: '+path
    residuals=pd.read_csv(OUT/'base_residual_diagnostics_all.csv')
    sensitivity=pd.read_csv(OUT/'base_bootstrap_sensitivity.csv')
    assert len(residuals)==50 and set(residuals.model)=={'svr','persistence'}
    learning=pd.read_csv(OUT/'base_learning_curve.csv')
    coeff=pd.read_csv(OUT/'base_coefficients.csv')
    text=(OUT/'base_report_template.md').read_text(encoding='utf-8')
    stamps=lambda s: pd.Timestamp(s).strftime('%Y-%m-%d %H:%M')
    blocks={}
    blocks['FOLDS']=table(['Fold','Inicio bloque validación UTC','Fin bloque validación UTC','Primera ancla evaluada','Última ancla evaluada','n train','n val'],
        [[r.fold,stamps(r.validation_block_start),stamps(r.validation_block_end),stamps(r.validation_start),stamps(r.validation_end),r.n_train,r.n_val] for r in folds.itertuples()])
    blocks['BOUNDARY_AUDIT']=table(['Fold','Última ancla train','Última vela objetivo train','Inicio historial validación','Última vela objetivo validación','Excluidas por historia','Excluidas por horizonte'],
        [[r.fold,stamps(r.train_end),stamps(r.label_end),stamps(r.validation_history_start),stamps(r.validation_label_end),r.excluded_history,r.excluded_target] for r in folds.itertuples()])
    blocks['BOUNDARY_AUDIT']+=f'\n\nSobre las filas previamente elegibles, se excluyen {folds.excluded_history.sum():,} anclas por historial y {folds.excluded_target.sum():,} por horizonte, por activo. Se evalúan {meta["validation_rows_per_asset"]:,} anclas por activo y {int(metrics.loc[metrics.model.eq("svr"),"n"].sum()):,} predicciones por método. La pérdida inicial de historial no es una imputación ni se rellena con TRAIN.'
    blocks['SELECTION']=table(['C','epsilon (pp)','RMSE medio'],
        [[r.C,r.epsilon,f'{r.mean_rmse:.6f}'] for r in scores.itertuples()])
    blocks['SELECTION']+=f'\n\nSe seleccionan **C={meta["selected_C"]:g} y epsilon={meta["selected_epsilon"]:g}**. La ejecución de los {meta["fits"]} ajustes y diagnósticos tomó {meta["seconds"]:.1f} segundos en el entorno local; no es una comparación de coste entre algoritmos.'
    summary=metrics.groupby('model')[['rmse','mape','r2','mae']].agg(['mean','std'])
    blocks['SUMMARY']=table(['Modelo','RMSE media ± DE','MAPE media ± DE (%)','R² media ± DE','MAE media ± DE'],
        [[m]+[f'{r[k,"mean"]:.4f} ± {r[k,"std"]:.4f}' for k in ['rmse','mape','r2','mae']] for m,r in summary.iterrows()])
    assets=metrics.groupby(['symbol','model'])[['rmse','mape','r2']].mean()
    blocks['ASSETS']=table(['Activo','Modelo','RMSE','MAPE (%)','R²'],
        [[s,m,f'{r.rmse:.4f}',f'{r.mape:.2f}',f'{r.r2:.4f}'] for (s,m),r in assets.iterrows()])
    sv=summary.loc['svr',('rmse','mean')]; pe=summary.loc['persistence',('rmse','mean')]
    negatives=int(metrics.loc[metrics.model.eq('svr'),'negative_predictions'].sum())
    blocks['COMPARISON']=f'El RMSE medio es **{pe:.4f}** para Persistence y **{sv:.4f}** para SVR; la diferencia SVR menos Persistence es **{sv-pe:.4f} puntos porcentuales**. '+('El SVR no supera la referencia en este experimento.' if sv>=pe else 'El SVR obtiene menor RMSE medio en esta validación de desarrollo; no demuestra superioridad final independiente.')+f' Se registran **{negatives} predicciones negativas** del SVR. Los resultados corresponden a ventanas completamente contenidas en sus respectivos bloques.'
    blocks['CI']=table(['Métrica','Modelo o diferencia','Estimación','Límite 2,5 %','Límite 97,5 %'],
        [[r.metric,r.model,f'{r.estimate:.4f}',f'{r.low:.4f}',f'{r.high:.4f}'] for r in cis.itertuples()])
    blocks['CI']+='\n\n**Sensibilidad a la longitud del bloque.** Se mantienen las predicciones, la semilla y las 499 réplicas; no se vuelve a seleccionar el modelo. Se contrastan 24 horas (horizonte del objetivo), 168 horas (una semana, análisis principal) y 336 horas (dos semanas).\n\n'
    blocks['CI']+=table(['Bloque (h)','Modelo o diferencia','RMSE o diferencia (pp)','IC 95 %: inferior','IC 95 %: superior'],
        [[r.block_hours,r.model,f'{r.estimate:.4f}',f'{r.low:.4f}',f'{r.high:.4f}'] for r in sensitivity.itertuples()])
    delta=sensitivity.loc[sensitivity.model.eq('svr_minus_persistence')]
    blocks['CI']+='\n\n'+('En las tres longitudes, el intervalo de la diferencia queda por encima de cero: la desventaja del SVR es consistente en esta comprobación de sensibilidad.' if (delta.low>0).all() else 'La conclusión debe examinarse para cada longitud: no se escoge el bloque que favorece un modelo.')+' Estos tres escenarios no identifican una longitud óptima ni corrigen el sesgo de selección. La réplica de 168 horas reproduce los intervalos originales. [Resultados completos de sensibilidad](../../outputs/tables/base_bootstrap_sensitivity.csv).'
    def span(g,c):
        return f'{g[c].min():.4f} a {g[c].max():.4f}'
    blocks['RESIDUALS']=table(['Activo','Modelo','Media residuo: rango entre folds','ACF 1h: rango','ACF 24h: rango','ACF 168h: rango'],
        [[s,m]+[span(g,c) for c in ['mean_residual','acf_1h','acf_24h','acf_168h']]
        for (s,m),g in residuals.groupby(['symbol','model'])])
    blocks['RESIDUALS']+='\n\nLos rangos resumen cinco folds, no son intervalos de confianza. Una media residual negativa indica sobreestimación media; una positiva, subestimación. El confinamiento de ventanas no elimina la dependencia inducida por objetivos solapados dentro de un mismo bloque.\n\n'
    blocks['RESIDUALS']+=f'La media residual es negativa en {int(residuals.mean_residual.lt(0).sum())} de los 50 casos. La ACF a una hora está entre {residuals.acf_1h.min():.4f} y {residuals.acf_1h.max():.4f}; esta persistencia impide tratar los errores horarios como observaciones independientes. No se atribuye toda la dependencia a una variable omitida.\n\n'
    blocks['RESIDUALS']+=table(['Activo','Modelo','Spearman magnitud–predicción: rango','Varianza segunda/primera mitad: rango','Asimetría: rango','Exceso de curtosis: rango'],
        [[s,m]+[span(g,c) for c in ['abs_residual_fitted_spearman','variance_second_over_first','skew','excess_kurtosis']]
        for (s,m),g in residuals.groupby(['symbol','model'])])
    blocks['RESIDUALS']+='\n\n[Los 50 diagnósticos individuales](../../outputs/tables/base_residual_diagnostics_all.csv) y [correlaciones con número de pares por rezago](../../outputs/tables/base_residual_correlations_all.csv) permiten revisar la heterogeneidad sin agrupar residuos de folds diferentes. La división en mitades para el cociente de varianzas usa el punto medio del calendario de cada fold.\n\n'
    blocks['RESIDUALS']+='\n\n'.join(f'```{{figure}} ../_static/figures/residuals_all_{s}.png\n:alt: Residuos de ambos modelos en los cinco folds para {s}.\n\n{s}: cada fila corresponde a un fold; azul SVR y naranja Persistence.\n```' for s in sorted(residuals.symbol.unique()))
    blocks['LEARNING']=table(['Activo','Fracción calendario train','n train','RMSE train','RMSE validación'],
        [[r.symbol,r.fraction,r.n_train,f'{r.train_rmse:.4f}',f'{r.validation_rmse:.4f}'] for r in learning.itertuples()])
    tendencies=[]
    for s,g in learning.groupby('symbol'):
        vals=g.sort_values('fraction').validation_rmse.to_numpy()
        trend='disminuye en ambos incrementos' if all(vals[1:]<vals[:-1]) else 'no disminuye de forma monótona'
        tendencies.append(f'{s}: {trend}')
    blocks['LEARNING']+='\n\nEvolución del error de validación al ampliar el prefijo: '+'; '.join(tendencies)+'. No se concluye que más historia resuelva uniformemente el problema.'
    selected=coeff.loc[coeff.groupby('symbol').standardized_coefficient.apply(lambda s:s.abs().idxmax())]
    blocks['COEFFICIENTS']=table(['Activo','Rezago (h)','Coeficiente estandarizado','Coeficiente por USDT'],
        [[r.symbol,r.lag_hours,f'{r.standardized_coefficient:.6f}',f'{r.coefficient_per_USDT:.8f}'] for r in selected.itertuples()])
    maxr2=metrics.loc[metrics.model.eq('svr'),'r2'].max()
    blocks['HIGH_SCORE']=f'El máximo R² del SVR entre las 25 evaluaciones es {maxr2:.4f}. '+('No alcanza la alerta de 0,8–0,9 de la guía; un desempeño bajo tampoco demuestra ausencia de fuga.' if maxr2<.8 else 'Alcanza la alerta de la guía y requiere examinar especialmente fuga, dependencia y comparación con Persistence.')
    blocks['FIGURES']='\n\n'.join(f'```{{figure}} ../_static/figures/base_{s}.png\n:alt: Predicciones, residuos y curva cronológica con ventanas contenidas para {s}.\n\n{s}: validación del último fold y curva de aprendizaje con fronteras estrictas.\n```' for s in sorted(metrics.symbol.unique()))
    for name,value in blocks.items():
        assert '{{'+name+'}}' in text,name
        text=text.replace('{{'+name+'}}',value)
    assert not re.search(r'\{\{\w+\}\}',text)
    (ROOT/'book/sections/03_modelo_base.md').write_text(text,encoding='utf-8')
    checklist_path=OUT/'entregable1_checklist.csv'
    with checklist_path.open(encoding='utf-8',newline='') as handle:
        reader=csv.DictReader(handle)
        fields,checklist=reader.fieldnames,list(reader)
    for row in checklist:
        key=row['requisito'].split(' ',1)[0]
        if key in {'2.6.36','2.6.37','2.6.38','2.6.39','4.3','3.5','3.7'}:
            row.update(estado='RESPONDIDO',evidencia='Cinco folds crecientes; historial y objetivo contenidos en cada bloque; pruebas y auditoría por timestamp.',
                       archivo_resultado='outputs/tables/base_folds.csv',
                       observaciones='Solo DEVELOPMENT. Purga de horizonte 24h y formación de historial 167h; ningún cruce de fronteras en las predicciones guardadas.')
        elif row['seccion']=='3. Modelo Base' and key not in {'3.3','3.16'}:
            row.update(estado='RESPONDIDO',
                       evidencia='Sección 3 regenerada con el protocolo de confinamiento estricto y 170 ajustes.',
                       archivo_resultado='book/sections/03_modelo_base.md',
                       observaciones='Validación de desarrollo; selección y evaluación usan los mismos folds. Residuos de ambos modelos en los cinco folds; IC condicionales con sensibilidad 24/168/336 horas. TEST reservado.')
    with checklist_path.open('w',encoding='utf-8',newline='') as handle:
        writer=csv.DictWriter(handle,fieldnames=fields)
        writer.writeheader();writer.writerows(checklist)
    if write_notebook:
        intro='# SVR lineal y Persistence: ventanas contenidas\n\nEl notebook presenta resultados guardados del protocolo estricto. Cambiar `REENTRENAR = True` para repetir los 170 ajustes. El código ejecutable está en `src/18_base_model.py`, y las pruebas en `tests/test_temporal_boundaries.py`. TEST permanece reservado.'
        code='''from pathlib import Path
import runpy, json, hashlib
import pandas as pd
from IPython.display import display, Image, Markdown
ROOT = next(p for p in [Path.cwd(), *Path.cwd().parents]
            if (p/'src/18_base_model.py').is_file())
REENTRENAR = False
if REENTRENAR:
    runpy.run_path(str(ROOT/'src/18_base_model.py'), run_name='__main__')
    runpy.run_path(str(ROOT/'src/19_extended_diagnostics.py'), run_name='__main__')
report = runpy.run_path(str(ROOT/'src/18_render_base_report.py'))
report['render'](write_notebook=False)
meta=json.loads((ROOT/'outputs/tables/base_metadata.json').read_text())
assert meta['protocol_sha256']==hashlib.sha256((ROOT/'outputs/tables/base_model_protocol.json').read_bytes()).hexdigest()
assert meta['development_sha256']==hashlib.sha256((ROOT/'data/splits/development_80.csv').read_bytes()).hexdigest()
print('Resultados de',meta['fits'],'ajustes; protocolo:',meta['boundary_policy'])
display(pd.read_csv(ROOT/'outputs/tables/base_folds.csv'))
display(pd.read_csv(ROOT/'outputs/tables/base_metrics_by_fold.csv'))
display(pd.read_csv(ROOT/'outputs/tables/base_residual_diagnostics_all.csv'))
display(pd.read_csv(ROOT/'outputs/tables/base_bootstrap_sensitivity.csv'))
'''
        description=re.sub(r'```\{figure\}[^\n]*\n.*?```','',text,flags=re.S)
        description=description.replace('../../outputs/','../outputs/').replace('../../requirements.txt','../requirements.txt').replace('../../notebooks/','./')
        nb=nbformat.v4.new_notebook(cells=[nbformat.v4.new_markdown_cell(intro),nbformat.v4.new_code_cell(code),nbformat.v4.new_markdown_cell(description)])
        for s in sorted(metrics.symbol.unique()):
            nb.cells.append(nbformat.v4.new_code_cell(f"display(Image(filename=str(ROOT/'book/_static/figures/base_{s}.png')))"))
            nb.cells.append(nbformat.v4.new_code_cell(f"display(Image(filename=str(ROOT/'book/_static/figures/residuals_all_{s}.png')))"))
        nb.metadata.kernelspec=dict(name='python3',display_name='Python 3',language='python')
        nbformat.write(nb,ROOT/'notebooks/18_base_model.ipynb')
    print('Informe del modelo sincronizado con el protocolo estricto.')


if __name__=='__main__':
    render()
