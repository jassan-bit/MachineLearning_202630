"""Genera sección 2.4 y notebook desde diagnósticos calculados en TRAIN."""
from pathlib import Path
import csv
import hashlib
import json
import re
import numpy as np
import pandas as pd
import nbformat

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs/tables'


def table(headers,rows):
    escape=lambda value: str(value).replace('|',r'\|')
    return '\n'.join(['| '+' | '.join(map(escape,headers))+' |','| '+' | '.join(['---']*len(headers))+' |']+
                     ['| '+' | '.join(map(escape,r))+' |' for r in rows])


def render(write_notebook=True):
    meta=json.loads((OUT/'multivariate_metadata.json').read_text(encoding='utf-8'))
    for key,name in [('base_folds','base_folds.csv'),('base_protocol','base_model_protocol.json'),('base_predictions','base_validation_predictions.csv'),('protocol','multivariate_protocol.json')]:
        assert meta['hashes'][key]==hashlib.sha256((OUT/name).read_bytes()).hexdigest(),name
    s=pd.read_csv(OUT/'multivariate_summary.csv')
    last=s.loc[s.fold.eq(5)].copy()
    folds=pd.read_csv(OUT/'base_folds.csv').drop_duplicates('fold')
    load=pd.read_csv(OUT/'multivariate_loadings.csv')
    episodes=pd.read_csv(OUT/'multivariate_episodes.csv')
    yearly=pd.read_csv(OUT/'multivariate_yearly_flags.csv')
    text=r'''## 2.4 Análisis multivariado de los 168 rezagos

### 2.4.1 Matriz real y alcance temporal

El modelo utiliza una variable de origen (`close`), pero **168 predictores distintos**: $X_{i,s}=(C_{i,s},C_{i,s-1},\ldots,C_{i,s-167})$. Por ello, la matriz sí requiere análisis multivariado. Se estudia cada activo por separado; no se mezclan precios nominales de criptomonedas distintas ni se incluyen `symbol`, timestamps, el objetivo futuro o Persistence como columnas del análisis.

Se reproducen exactamente las anclas de TRAIN de los cinco folds de la sección 3, incluida la intersección de elegibilidad entre activos y el confinamiento de historias y etiquetas al bloque. La elegibilidad verifica que el objetivo exista, pero **su valor no se utiliza para ajustar PCA, el escalador ni el detector de anomalías**. Cada ajuste usa solamente su TRAIN; no se ajusta ni se elige nada con VALIDATION o TEST. Los TRAIN son crecientes y se solapan entre folds; sus resultados no representan 25 muestras independientes.

{{COVERAGE}}

La relación n/p describe filas por predictor, no observaciones estadísticamente independientes. Dos ventanas contiguas comparten 167 cierres. Se conservan los huecos del calendario, sin imputación ni eliminación de valores extremos. Los paneles detallados muestran TRAIN del fold 5 y las tablas de estabilidad comparan los cinco TRAIN.

### 2.4.2 Redundancia, rango y correlación

Se estandarizan las 168 columnas con `StandardScaler` ajustado de nuevo en cada TRAIN y activo. Se calcula su matriz de Pearson. Se informa la mediana del valor absoluto de las 14.028 correlaciones fuera de la diagonal y la proporción con $|r|>0.99$; ese umbral es descriptivo, no una regla de selección.

La SVD de la matriz estandarizada y centrada permite distinguir rango numérico y redundancia: el rango cuenta los valores singulares mayores que $\max(n,p)\,\epsilon_{\mathrm{mach}}\,s_{max}$, donde $\epsilon_{\mathrm{mach}}$ es la precisión de máquina. El número de condición es $\kappa=s_{max}/s_{min}$ cuando el rango es completo. Un número elevado señala direcciones con escalas de variación muy distintas; no prueba fuga ni determina automáticamente qué rezagos eliminar. El VIF de los candidatos de 2.3 se complementa aquí con el espectro de la matriz completa, en lugar de añadir 168 regresiones auxiliares. Para anomalías se utiliza Isolation Forest, que no requiere invertir la covarianza como la distancia de Mahalanobis convencional.

{{REDUNDANCY}}

Las altas asociaciones entre niveles de precio rezagados pueden reflejar persistencia, tendencia y mezcla de periodos. La estandarización no elimina esa estructura ni vuelve estacionarias las series. La redundancia ayuda a explicar por qué los coeficientes individuales del SVR deben interpretarse con cautela.

### 2.4.3 PCA y dimensión efectiva

Se ejecuta [PCA de scikit-learn](https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html) con SVD completa (`svd_solver="full"`) sobre cada matriz estandarizada de TRAIN. PCA centra las columnas; el escalado se realiza explícitamente antes. Se conservan los 168 componentes para diagnosticar el espectro, sin introducir PCA en el SVR ni seleccionar variables a partir del resultado.

Se reporta el mínimo número de componentes para alcanzar 90 %, 95 % y 99 % de la varianza, con umbrales fijados en el protocolo antes del cálculo. Además, con $q_j=\lambda_j/\sum_k\lambda_k$, la dimensión efectiva por entropía es:

$$
d_{efectiva}=\exp\left(-\sum_{j:q_j>0}q_j\ln q_j\right).
$$

Esta dimensión continua resume la concentración del espectro y no equivale al rango numérico ni a un número óptimo de predictores para pronosticar.

{{PCA}}

{{PCA_INTERPRETATION}}

La alineación absoluta de PC1 con el vector uniforme $\mathbf{1}/\sqrt{168}$ permite examinar si el primer componente representa principalmente el nivel conjunto de los cierres. Los pesos de PC1 y PC2 se grafican por rezago. El signo de un componente es arbitrario; para dibujarlo se orienta hacia suma de pesos no negativa, sin cambiar la varianza explicada.

{{LOADINGS}}

**Explicar varianza de precios no equivale a explicar volatilidad futura.** Las direcciones de baja varianza pueden contener información predictiva. No se interpreta un PC1 dominante como evidencia de que un único componente baste para el pronóstico ni como razón para cambiar ahora el modelo base.

### 2.4.4 Estabilidad descriptiva entre entrenamientos

{{STABILITY}}

Los rangos resumen los cinco TRAIN crecientes por activo. No son intervalos de confianza y no permiten una prueba de estabilidad independiente: los folds comparten historia y al ampliarlos cambian los periodos y regímenes representados. Cada PCA tiene su propia base; las coordenadas de componentes de distintos ajustes no se comparan directamente como si compartieran orientación.

### 2.4.5 Anomalías multivariadas

Se ajusta [Isolation Forest](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.IsolationForest.html) directamente sobre las **168 columnas estandarizadas**, sin reducirlas antes mediante PCA. Se fijan 200 árboles, submuestras de 256 ventanas, todas las columnas disponibles, `contamination="auto"`, semilla 42 y ejecución secuencial. El puntaje usado es $a(X)=-\operatorname{score\_samples}(X)$: valores mayores representan mayor atipicidad según el detector.

La señalización utiliza un umbral propio y explícito: percentil 99 de los puntajes del mismo TRAIN, con desigualdad estricta `score > umbral`. No se utiliza el umbral automático de `predict`. El percentil se fija como criterio descriptivo antes de observar los resultados; no es una probabilidad de error ni una tasa de falsos positivos calibrada. Señalar aproximadamente el 1 % de TRAIN es una consecuencia de esa regla, no un descubrimiento de que el 1 % de los datos sea erróneo.

{{ANOMALIES}}

Se agrupan anclas señaladas consecutivas separadas exactamente por una hora en secuencias descriptivas. Los huecos interrumpen las secuencias. Estas secuencias no identifican eventos económicos independientes: las ventanas se solapan y dos secuencias pueden compartir cierres.

{{EPISODES}}

La tabla localiza, por activo, la secuencia que contiene el puntaje máximo de TRAIN del fold 5. No atribuye causas a las fechas. Los precios de nivel extremo, las transiciones de nivel o las trayectorias poco habituales pueden generar puntuaciones altas sin ser errores de cotización. Las ventanas señaladas se conservan; no se recortan ni se reentrena el SVR para mejorar métricas después de observarlas.

{{YEARLY}}

Los porcentajes anuales usan como denominador las anclas elegibles de cada año, con extremos parciales. Se calculan con un único detector ajustado sobre todo TRAIN del fold 5. Son una descripción retrospectiva, no detección en línea disponible en cada fecha histórica ni evidencia de deterioro futuro. Los puntajes tampoco están calibrados para comparar riesgo entre activos.

### 2.4.6 Estructura, subpoblaciones y gráficos

Los activos son grupos conocidos, tratados por separado. Las proyecciones PC1–PC2 se colorean por año UTC para examinar cómo se distribuyen los periodos en la representación. Solo para esa visualización se muestran hasta 5.000 anclas uniformemente espaciadas en orden temporal; PCA, correlaciones y anomalías utilizan todas las filas elegibles de TRAIN. Los gráficos por activo tienen bases propias y no permiten comparar directamente coordenadas entre criptomonedas.

Las matrices de correlación y las curvas de varianza acumulada muestran escalas ampliadas, identificadas en sus ejes, para hacer visible la redundancia sin ocultar las diferencias entre rezagos. No parten necesariamente de cero; los valores completos se conservan en las tablas y matrices numéricas.

La distribución temporal en la proyección puede reflejar niveles de precio y cambios de escala, sin demostrar clusters o regímenes discretos. No se ejecuta clustering, UMAP ni t-SNE: el objetivo de esta sección es caracterizar la matriz real, no elegir un número de regímenes sin un criterio temporal de estabilidad. El clustering de regímenes exigiría una representación pertinente, validación de estabilidad y una pregunta adicional; no se declaran grupos descubiertos a partir de estos gráficos.

{{FIGURES}}

### 2.4.7 Consecuencias y reproducibilidad

El análisis establece la dimensión nominal, la concentración de varianza, la redundancia y la atipicidad conjunta de la entrada utilizada. Motiva estudiar, en un experimento posterior, representaciones de retornos o reducción de dimensionalidad; no prueba que mejoren el pronóstico. Cualquier uso predictivo de PCA, selección de componentes o tratamiento de anomalías deberá ajustarse dentro del entrenamiento y elegirse con validación temporal, conservando TEST reservado. Los resultados del SVR y Persistence permanecen intactos.

El protocolo previo está en `outputs/tables/multivariate_protocol.json`. El cálculo está en `src/12_multivariate_close.py`; `src/12_render_multivariate_report.py` sincroniza informe y notebook. Se guardan los 25 resúmenes, el espectro completo, parámetros de escalado, matrices de correlación, pesos de PC1/PC2, puntajes del fold 5 y secuencias señaladas en `outputs/tables/multivariate_*`. Los metadatos registran semillas, versiones, huellas y alcance de ajuste. Las pruebas verifican dimensiones conocidas, rechazo de la etiqueta como entrada y las fronteras temporales compartidas con el modelo.

Para reproducir: `python -m unittest discover -s tests -v`, `python src/12_multivariate_close.py` y `python src/12_render_multivariate_report.py`. El notebook carga los resultados verificados por defecto; `RECALCULAR = True` repite los 25 diagnósticos.

[Notebook multivariado con resultados ejecutados](../../notebooks/12_multivariate_scope_close.ipynb).

'''
    vals={}
    vals['COVERAGE']=table(['Fold','Inicio anclas TRAIN UTC','Fin anclas TRAIN UTC','n por activo','p','n/p'],
        [[r.fold,pd.Timestamp(r.train_start).strftime('%Y-%m-%d %H:%M'),pd.Timestamp(r.train_end).strftime('%Y-%m-%d %H:%M'),r.n_train,168,f'{r.n_train/168:.2f}'] for r in folds.itertuples()])
    vals['REDUNDANCY']=table(['Activo · TRAIN fold 5','Mediana |r|','Pares con |r| > 0.99 (%)','Rango numérico','Número de condición'],
        [[r.symbol,f'{r.median_abs_correlation:.5f}',f'{100*r.fraction_abs_correlation_gt_099:.2f}',r.numerical_rank,f'{r.condition_number:.1f}'] for r in last.itertuples()])
    vals['PCA']=table(['Activo · TRAIN fold 5','PC1 (%)','PC2 (%)','k90','k95','k99','Dimensión efectiva'],
        [[r.symbol,f'{100*r.pc1_ratio:.3f}',f'{100*r.pc2_ratio:.3f}',r.k90,r.k95,r.k99,f'{r.effective_rank:.3f}'] for r in last.itertuples()])
    vals['PCA_INTERPRETATION']=f'En TRAIN del fold 5, PC1 concentra entre {100*last.pc1_ratio.min():.3f}% y {100*last.pc1_ratio.max():.3f}% de la varianza estandarizada. Para alcanzar 95 % se requieren entre {last.k95.min()} y {last.k95.max()} componentes; para 99 %, entre {last.k99.min()} y {last.k99.max()}. La dimensión efectiva varía entre {last.effective_rank.min():.3f} y {last.effective_rank.max():.3f}, mientras que el rango numérico se informa separadamente.'
    vals['LOADINGS']=table(['Activo','|coseno(PC1, nivel uniforme)|'],
        [[sym,f'{abs(g.weight.sum()/np.sqrt(168)):.6f}'] for sym,g in load.loc[load.component.eq(1)].groupby('symbol')])
    cosines=[abs(g.weight.sum()/np.sqrt(168)) for _,g in load.loc[load.component.eq(1)].groupby('symbol')]
    vals['LOADINGS']+=f'\n\nLa alineación mínima observada es {min(cosines):.6f}. La cercanía a uno respalda interpretar PC1 principalmente como un movimiento conjunto de nivel en estos entrenamientos, no como una medida de volatilidad futura.'
    vals['STABILITY']=table(['Activo','PC1 mín.–máx. (%)','k95 mín.–máx.','k99 mín.–máx.','Dimensión efectiva mín.–máx.'],
        [[sym,f'{100*g.pc1_ratio.min():.3f}–{100*g.pc1_ratio.max():.3f}',f'{g.k95.min()}–{g.k95.max()}',f'{g.k99.min()}–{g.k99.max()}',f'{g.effective_rank.min():.3f}–{g.effective_rank.max():.3f}'] for sym,g in s.groupby('symbol')])
    vals['ANOMALIES']=table(['Activo · TRAIN fold 5','Umbral TRAIN','Anclas señaladas','Porcentaje','Secuencias consecutivas'],
        [[r.symbol,f'{r.anomaly_threshold:.5f}',r.flagged,f'{r.flagged_pct:.3f}',r.flagged_runs] for r in last.itertuples()])
    peaks=episodes.loc[episodes.groupby('symbol').max_score.idxmax()]
    vals['EPISODES']=table(['Activo','Inicio secuencia UTC','Fin secuencia UTC','Anclas','Puntaje máximo'],
        [[r.symbol,pd.Timestamp(r.start).strftime('%Y-%m-%d %H:%M'),pd.Timestamp(r.end).strftime('%Y-%m-%d %H:%M'),r.n_anchors,f'{r.max_score:.5f}'] for r in peaks.itertuples()])
    yearly=yearly.pivot(index='symbol',columns='year',values='flagged_pct')
    vals['YEARLY']=table(['Activo']+[f'{y} (% señalado)' for y in yearly.columns],[[sym]+[f'{v:.2f}' for v in row] for sym,row in yearly.iterrows()])
    vals['FIGURES']='\n\n'.join(f'```{{figure}} ../_static/figures/multivariate_{sym}.png\n:alt: Correlación de 168 rezagos, PCA y anomalías de {sym}, solo TRAIN fold 5.\n\n{sym}: matriz real de entrada, varianza acumulada, proyección por año, pesos y anomalías retrospectivas.\n```' for sym in sorted(last.symbol))
    for key,value in vals.items():
        assert '{{'+key+'}}' in text
        text=text.replace('{{'+key+'}}',value)
    assert not re.search(r'\{\{\w+\}\}',text)
    book=ROOT/'book/sections/02_eda.md'
    current=book.read_text(encoding='utf-8')
    current=current.replace('Los resultados describen los datos disponibles para desarrollar el modelo; no se entrenan modelos ni se evalúa el conjunto de prueba.', 'Los resultados describen los datos disponibles para desarrollar el modelo. Los ajustes diagnósticos se restringen al entrenamiento temporal de cada análisis; la comparación del SVR con Persistence se presenta en la sección 3. TEST no se evalúa.')
    a=current.index('## 2.4 ');b=current.index('## 2.5 ',a)
    book.write_text(current[:a]+text+current[b:],encoding='utf-8')
    base_path=ROOT/'book/sections/01_base_datos.md'
    base_text=base_path.read_text(encoding='utf-8')
    explanation=f'Esta relación corresponde al dataset original e incluye identificadores temporales y de entidad. Para la matriz efectiva del SVR, **p=168 rezagos de close por activo**. Los entrenamientos contienen entre {int(s.n.min()):,} y {int(s.n.max()):,} filas elegibles por activo, con n/p entre {s.n_over_p.min():.2f} y {s.n_over_p.max():.2f}; el detalle por fold está en 2.4.1. Estos cocientes no miden tamaño muestral independiente: las ventanas se solapan y los activos pueden presentar dependencia transversal. TEST no interviene en el diagnóstico.'
    base_text,count=re.subn(r'Esta relación corresponde al dataset original[^\n]*',lambda _:explanation,base_text,count=1)
    assert count==1
    base_path.write_text(base_text,encoding='utf-8')
    path=OUT/'entregable1_checklist.csv'
    with path.open(encoding='utf-8',newline='') as h:
        reader=csv.DictReader(h);fields,rows=reader.fieldnames,list(reader)
    for row in rows:
        if row['seccion'].startswith('2.4 '):
            key=row['requisito'].split(' ',1)[0]
            alternative=key in {'2.4.4','2.4.5','2.4.6','2.4.8'}
            row.update(estado='NO APLICA' if alternative else 'RESPONDIDO',
                       evidencia='Sección 2.4: 168 rezagos reales, ajustes por TRAIN/activo, PCA, redundancia, Isolation Forest y periodos conocidos.',
                       archivo_resultado='outputs/tables/multivariate_summary.csv',
                       observaciones='Alternativas no ejecutadas: se usa PCA e Isolation Forest; no se declaran clusters descubiertos.' if alternative else 'Descriptivo; sin TEST, sin selección predictiva ni eliminación de filas. TRAIN de folds solapados.')
    with path.open('w',encoding='utf-8',newline='') as h:
        writer=csv.DictWriter(h,fieldnames=fields);writer.writeheader();writer.writerows(rows)
    if write_notebook:
        intro='# Análisis multivariado de los 168 rezagos\n\nPCA, correlaciones y anomalías sobre TRAIN de cada fold, por activo. El cálculo ejecutable está en `src/12_multivariate_close.py`. Por defecto se presentan las salidas guardadas; `RECALCULAR = True` repite los 25 diagnósticos. TEST permanece reservado.'
        code='''from pathlib import Path
import runpy, json, hashlib
import pandas as pd
from IPython.display import display, Image
ROOT = next(p for p in [Path.cwd(), *Path.cwd().parents]
            if (p/'src/12_multivariate_close.py').is_file())
RECALCULAR = False
if RECALCULAR:
    runpy.run_path(str(ROOT/'src/12_multivariate_close.py'),run_name='__main__')
report=runpy.run_path(str(ROOT/'src/12_render_multivariate_report.py'))
report['render'](write_notebook=False)
meta=json.loads((ROOT/'outputs/tables/multivariate_metadata.json').read_text())
assert meta['hashes']['development']==hashlib.sha256((ROOT/'data/splits/development_80.csv').read_bytes()).hexdigest()
display(pd.read_csv(ROOT/'outputs/tables/multivariate_summary.csv'))
print('Ajustes: 25 PCA y 25 Isolation Forest; solo TRAIN. Filas eliminadas:',meta['rows_removed'])
'''
        description=re.sub(r'```\{figure\}[^\n]*\n.*?```','',text,flags=re.S).replace('../../notebooks/','./').replace('../../outputs/','../outputs/')
        nb=nbformat.v4.new_notebook(cells=[nbformat.v4.new_markdown_cell(intro),nbformat.v4.new_code_cell(code),nbformat.v4.new_markdown_cell(description)])
        for sym in sorted(last.symbol):
            nb.cells.append(nbformat.v4.new_code_cell(f"display(Image(filename=str(ROOT/'book/_static/figures/multivariate_{sym}.png')))"))
        nb.metadata.kernelspec=dict(name='python3',display_name='Python 3',language='python')
        nbformat.write(nb,ROOT/'notebooks/12_multivariate_scope_close.ipynb')
    print('Sección 2.4 y notebook sincronizados con los diagnósticos multivariados.')


if __name__=='__main__':render()
