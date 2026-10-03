"""Apply the verified optimization to Entregable 1, locally, without Git actions."""
import hashlib
import json
import shutil
import zipfile
from pathlib import Path
import nbformat
from nbclient import NotebookClient
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from optimize_minute_svr import ROOT, OUT
from metric_comparison import comparison_tables


def table(frame):
    # Avoid an optional tabulate dependency in the reproducible delivery.
    frame = frame.copy()
    for col in frame.select_dtypes('number'):
        frame[col] = frame[col].map(lambda x: f'{x:.5f}'.rstrip('0').rstrip('.'))
    return '| ' + ' | '.join(frame.columns) + ' |\n| ' + ' | '.join(['---']*len(frame.columns)) + ' |\n' + '\n'.join('| ' + ' | '.join(map(str,row)) + ' |' for row in frame.to_numpy())


def main():
    verification = json.loads((OUT/'verification.json').read_text())
    assert verification['passed'] and verification['configurations'] == 16
    provenance = json.loads((OUT/'provenance.json').read_text())
    for name, digest in provenance['artifact_sha256'].items():
        assert hashlib.sha256((OUT/name).read_bytes()).hexdigest() == digest, name
    section = ROOT/'book/sections/07_estudio_minuto.md'
    archive = ROOT/'book/sections/09_referencia_minuto_original.md'
    if not archive.exists():
        archive.write_text('> **Referencia histórica:** modelo original con precios por minuto como columnas. El [entregable vigente](07_estudio_minuto.md) usa características derivadas y validación creciente.\n\n' + section.read_text(encoding='utf-8'), encoding='utf-8')
    old = archive.read_text(encoding='utf-8')
    eda = old.split('## Exploración\n',1)[1].split('## Resultados\n',1)[0]
    macro = pd.read_csv(OUT/'macro_metrics.csv')
    global_comparison, asset_comparison, interpretation = comparison_tables(macro)
    choices = pd.read_csv(OUT/'selected_inputs.csv')
    validation = pd.read_csv(OUT/'validation_comparison.csv')
    detailed = pd.read_csv(OUT/'selected_metrics.csv')
    calendar = pd.read_csv(OUT/'calendar.csv')
    folds = calendar.groupby(['fold','split']).agg(n=('origin','size'), first=('origin','min'), last=('origin','max')).reset_index()
    audit = pd.read_csv(OUT/'native_cv_audit.csv').groupby('method').agg(cortes=('native_fold','size'), cortes_con_futuro=('future_train_origins',lambda x:int((x>0).sum()))).reset_index()
    fig, axes = plt.subplots(1,2,figsize=(12,4))
    for ax, metric, title in zip(axes,['r2','rmse'],['R²: mayor es mejor','RMSE: menor es mejor']):
        macro[macro.symbol != 'GLOBAL_MACRO'].pivot(index='symbol',columns='model',values=metric).plot.bar(ax=ax,rot=0)
        ax.set_title(title); ax.set_xlabel(''); ax.grid(axis='y',alpha=.2)
    fig.tight_layout()
    fig.savefig(ROOT/'book/figures/optimized_svr_metrics.png',dpi=160)
    plt.close(fig)
    report = r'''# Entregable 1: predicción de volatilidad con SVR lineal

**Jassan Arteta y Mateo Bernal**  
Universidad del Norte · Maestría en Matemáticas · Machine Learning · Profesor Lihki Rubio · 2026-30

## Objetivo y datos

Se predice la volatilidad de BTCUSDT, ETHUSDT, BNBUSDT y XRPUSDT para los siguientes siete días. El periodo es **1 de enero de 2023 a 31 de diciembre de 2025, UTC**. La fuente es Binance, con cierres de **un minuto**. Se conservan 1.578.159 observaciones válidas por activo y 81 minutos faltantes; el día incompleto es el 24 de marzo de 2023. Se excluyen las ventanas afectadas, sin interpolación.

La frecuencia de adquisición es un minuto, la construcción de características resume información intradía y cada muestra supervisada se origina al cierre de un día UTC. Por tanto, millones de observaciones de minuto no equivalen a millones de ejemplos de entrenamiento. Las fechas se etiquetan por el día cuyo cierre ya se conoce.

## Retornos, volatilidad y salidas

Sean \(P_t\) el último cierre del día y \(r_t=\ln(P_t/P_{t-1})\). El objetivo es

$$
\sigma_t^{(w)}=100\sqrt{\frac{1}{w}\sum_{j=0}^{w-1}(r_{t-j}-\bar r_t)^2},\qquad w\in\{7,14,21,28\}.
$$

Se usa `rolling(w).std(ddof=0)`, sin anualizar. Cada salida es \((\sigma_{t+1}^{(w)},\ldots,\sigma_{t+7}^{(w)})\). La volatilidad realizada de los retornos por minuto se emplea como característica; el objetivo sigue siendo la volatilidad de retornos diarios. Son definiciones diferentes.

## Exploración y calidad

'''+eda+r'''

La exploración de 2023–2025 es descriptiva y retrospectiva. Los extremos y la dependencia en retornos al cuadrado motivan estudiar volatilidad y comparar contra persistencia, pero no garantizan capacidad predictiva. Los escaladores y la selección del modelo utilizan únicamente los periodos de desarrollo correspondientes a cada corte.

## Preprocesamiento y características

Se prueban ventanas de entrada de **7, 14, 21 y 28 días**. Para cada día de la ventana se incluyen seis características: retorno diario, su cuadrado, raíz de la suma de cuadrados de todos los retornos por minuto, suma absoluta de retornos por minuto dividida por la raíz de 1.440, semivolatilidad negativa y máximo retorno absoluto por minuto. Se normalizan por la volatilidad actual conocida, o su cuadrado según la unidad.

Se añaden siete características que describen el efecto de la salida de retornos antiguos de la ventana objetivo. Para horizonte \(h\), se conservan los \(w-h\) retornos diarios más recientes y se supone varianza futura igual a la actual para construir una referencia. Esta operación usa únicamente retornos ya observados; no utiliza el valor futuro del objetivo.

En unidades porcentuales, con \(b_t=\max(\sigma_t^{(w)},10^{-8})\), suma \(S\) y suma de cuadrados \(Q\) de esos retornos retenidos, la característica es

$$
\frac{\sqrt{\max((Q+h b_t^2)/w-(S/w)^2,0)}}{b_t}-1.
$$

La dimensión es **6L+7**: 49, 91, 133 o 175 características según la ventana. Este cambio sustituye las 10.080–40.320 columnas de precios del experimento original por características derivadas de los datos de un minuto. La volatilidad actual y las características de salida de la ventana requieren además la historia de la definición objetivo, de hasta 28 retornos diarios. El calendario común exige suficiente historia para la ventana máxima y siete objetivos completos.

## Modelo y justificación

Se mantienen siete regresores **LinearSVR**, uno por horizonte, mediante `MultiOutputRegressor`. Cada uno aprende la corrección relativa \(z_{t,h}=\sigma_{t+h}/b_t-1\); la predicción final es \(\max(b_t(1+\hat z_{t,h}),0)\). El recorte a cero forma parte del procedimiento evaluado. La persistencia repite la volatilidad actual en las siete salidas.

Esta representación reduce la dependencia del nivel nominal del precio y permite al SVR aprender cuándo corregir persistencia. Se ajustan los escaladores de entradas y objetivos exclusivamente dentro de cada entrenamiento. El modelo es lineal respecto a las características transformadas; el procesamiento completo incorpora transformaciones no lineales.

Se usa pérdida `squared_epsilon_insensitive`, `dual=False`, tolerancia 1e-6, máximo 50.000 iteraciones y semilla 42. La búsqueda evalúa C en {0,0001; 0,001; 0,01; 0,1; 1} y epsilon en {0,01; 0,1}, además de las cuatro entradas: **640 candidatos**, cada uno evaluado en seis cortes. Las advertencias de falta de convergencia interrumpen el ajuste.

## Split temporal y validación con tsxv

Se usa `timeseries-cv==0.1.5`, desarrollado con la coautoría de Filipe Roberto Ramos. Se ejecutan las funciones nativas `split_train_val_forwardChaining`, `split_train_val_kFold` y `split_train_val_groupKFold` sobre índices diarios del calendario de 2023–2024. Los índices se vinculan a características de datos por minuto y objetivos diarios; no se interpretan siete minutos como siete días.

La auditoría conserva los cortes nativos:

'''+table(audit)+'''

K-Fold y Group K-Fold contienen entrenamiento posterior a algunas fechas de validación. Se documentan como diagnóstico y se excluyen de la selección del pronóstico. No se convierten mediante agrupación en una copia de Forward Chaining ni se presentan como tres evaluaciones temporales equivalentes.

El método principal conserva seis cortes de entrenamiento nativos de Forward Chaining, con inicio de validación en enero, marzo, mayo, julio, septiembre y noviembre de 2024. Cada validación se extiende a dos meses; este adaptador se declara explícitamente. El entrenamiento crece y sus etiquetas terminan antes del inicio de cada validación. Se mantienen las restricciones conservadoras de separación de la biblioteca. Los últimos siete días de 2024 no se usan como orígenes de validación porque sus objetivos entrarían en 2025.

'''+table(folds)+'''

Se seleccionan C, epsilon y entrada por el RMSE conjunto de las predicciones fuera de muestra de 2024, promediando los siete RMSE por horizonte. Se fijan las 16 configuraciones antes de evaluar 2025. El ajuste final usa todas las muestras elegibles cuyas etiquetas terminan antes de 2025 y permanece fijo durante la prueba. La prueba contiene **358 orígenes diarios** y **40.096 valores pronosticados**.

## Selección y contraste en validación

'''+table(choices)+'''

'''+table(validation)+'''

Las entradas elegidas cambian con el activo y la definición de volatilidad; una ventana más larga no mejora necesariamente el pronóstico. Los C seleccionados favorecen regularización fuerte, coherente con limitar el sobreajuste.

## Resultados de prueba de 2025

### Tabla comparativa global: persistencia y SVR lineal

'''+table(global_comparison)+'''

### Comparación por criptomoneda

'''+table(asset_comparison)+'''

### Interpretación de las métricas

'''+ '\n\n'.join(interpretation)+'''

![R² y RMSE del SVR optimizado frente a persistencia](../figures/optimized_svr_metrics.png)

El SVR supera a persistencia en RMSE en **16 de 16 configuraciones seleccionadas**. El R² macro es **0,7007**, frente a **0,5377**; el RMSE macro baja de **0,7477 a 0,5981**, una reducción aproximada del **20 %**. BNB presenta el mayor R² y ETH el menor entre los cuatro activos. Una mejora media no implica ganar en todos los días ni en cada horizonte individual.

R² se promedia entre siete horizontes; por activo se promedian las cuatro definiciones de volatilidad y el global es la media de las 16 configuraciones. No es el R² de series concatenadas. RMSE es la media de los RMSE por horizonte, no la raíz de un MSE agrupado. RMSE y MAE se expresan en puntos porcentuales de volatilidad, MSE en su cuadrado y MAPE en porcentaje.

### Detalle por ventana objetivo

'''+table(detailed[['symbol','volatility_window','input_window','model','r2','rmse','mae']])+'''

## Conclusiones y límites

La ingeniería de características, la corrección de persistencia, la regularización y la validación creciente permiten una mejora observada frente al baseline. El modelo anterior obtuvo R² macro −16,0817 y RMSE 4,0266 en las mismas fechas y objetivos; se conserva como referencia histórica.

La comparación cambia varias decisiones simultáneamente, incluido el ajuste final con 2023–2024 frente al ajuste anterior solo con 2023. No permite atribuir toda la mejora a una decisión aislada. **2025 ya se había examinado**: es una evaluación retrospectiva, aunque la nueva selección no usa sus métricas. Los horizontes y objetivos móviles se superponen; no son observaciones estadísticamente independientes. No se aporta una prueba de significancia ni se garantiza rendimiento futuro.

Se verificaron hashes de los datos, elección de hiperparámetros, predicciones de modelos guardados, métricas recalculadas y coincidencia de fechas y objetivos con el experimento original. Pasaron 30 pruebas del proyecto. El informe anterior de BDS corresponde al modelo original; no se atribuye al modelo optimizado.

## Reproducción y archivos

Desde la raíz, con las dependencias de `requirements.txt` y `requirements-dashboard.txt` instaladas:

```bash
python src/optimize_minute_svr.py
python src/report_optimized_svr.py
python src/apply_optimized_delivery.py
```

La entrega incluye los datos procesados de minuto, los 16 modelos, las 640 búsquedas, calendarios, auditoría nativa, predicciones, métricas, figuras y notebook ejecutado. Para regenerar los datos desde Binance se incluyen los scripts de descarga y preparación y el registro SHA-256 de fuentes. El entrenamiento tarda varios minutos; no es necesario repetirlo para consultar los resultados.

Notebook vigente: [Entregable_1_Minuto.ipynb](../../notebooks/Entregable_1_Minuto.ipynb). Paquete: [Entregable1_optimizado_2023_2025.zip](../../delivery/Entregable1_optimizado_2023_2025.zip). Resultados: `results/optimized_minute_2023_2025`. El dashboard/API anteriores continúan asociados al modelo original y no sirven los modelos optimizados. El informe vigente se consulta en el HTML incluido o en localhost:8051.

## Referencias

- Binance. [Datos públicos de mercado](https://data.binance.vision/) y [API](https://www.binance.com/en/binance-api).
- [timeseries-cv](https://pypi.org/project/timeseries-cv/), versión 0.1.5.
- Ramos, F. (2021). *Data Science na Modelação e Previsão de Séries Económico-financeiras: das Metodologias Clássicas ao Deep Learning*. Tesis doctoral, Instituto Universitário de Lisboa, ISCTE Business School.
- Scikit-learn. [LinearSVR](https://scikit-learn.org/stable/modules/generated/sklearn.svm.LinearSVR.html).
'''
    report = report.replace('\\(', '$').replace('\\)', '$')
    section.write_text(report, encoding='utf-8')
    intro = '''# Entregable 1: volatilidad con SVR lineal

**Jassan Arteta y Mateo Bernal**

Estudio Binance de BTC, ETH, BNB y XRP durante **2023–2025**, con datos de **un minuto**, características derivadas en ventanas de **7, 14, 21 y 28 días** y siete salidas diarias de volatilidad. La variante vigente usa correcciones relativas a persistencia, seis cortes temporales crecientes para selección en 2024 y ajuste final con etiquetas anteriores a 2025.

En 358 fechas de prueba de 2025: **R² macro 0,7007**, RMSE **0,5981** y mejora de RMSE frente a persistencia en **16/16 configuraciones**. La evaluación es retrospectiva; 2025 ya había sido examinado.

Consultar el [informe completo](sections/07_estudio_minuto.md), el [notebook ejecutado](../notebooks/Entregable_1_Minuto.ipynb) y la [entrega reproducible](../delivery/Entregable1_optimizado_2023_2025.zip).

El [modelo original con precios por minuto](sections/09_referencia_minuto_original.md) y el [estudio diario anterior](sections/08_referencia_diaria.md) quedan como referencias históricas. Los resultados vigentes están en `results/optimized_minute_2023_2025`.
'''
    (ROOT/'book/entregable1_master.md').write_text(intro,encoding='utf-8')
    config = ROOT/'book/myst.yml'
    content = config.read_text(encoding='utf-8')
    if 'sections/09_referencia_minuto_original.md' not in content:
        content = content.replace('    - file: sections/08_referencia_diaria.md','    - file: sections/09_referencia_minuto_original.md\n    - file: sections/08_referencia_diaria.md')
    config.write_text(content,encoding='utf-8')
    readme = intro.replace('(sections/','(book/sections/').replace('(../notebooks/','(notebooks/').replace('(../delivery/','(delivery/')
    readme += '''
## Reproducir

```bash
python -m pip install -r requirements.txt -r requirements-dashboard.txt
python src/optimize_minute_svr.py
python src/report_optimized_svr.py
python src/apply_optimized_delivery.py
```

Para consultar el informe sin entrenar:

```bash
python -m http.server 8051 --bind 127.0.0.1 --directory results/optimized_minute_2023_2025
```

Abrir http://localhost:8051. Los datos procesados están incluidos. Su procedencia y hashes se conservan en `results/minute_2023_2025/sources.csv` y `data_manifest.json`. `src/download_three_years.py` y `src/minute_experiment.py` permiten reconstruirlos desde los archivos mensuales de Binance.

`experiment_active.json` identifica el entregable vigente; `experiment_minute.json` conserva la configuración histórica y del periodo de adquisición. El dashboard de puerto 8050 y `app/api.py` todavía sirven el modelo original. Para las métricas optimizadas usar el informe de puerto 8051. Ningún script de esta reproducción publica en GitHub.
'''
    (ROOT/'README.md').write_text(readme,encoding='utf-8')
    dashboard = ROOT/'DASHBOARD.md'
    content = dashboard.read_text(encoding='utf-8')
    marker = '> **Entregable vigente:**'
    if not content.startswith(marker):
        dashboard.write_text(marker+' las métricas optimizadas están en http://localhost:8051, servidas desde `results/optimized_minute_2023_2025`. El dashboard/API descritos a continuación corresponden al modelo original y permanecen como referencia histórica.\n\n'+content,encoding='utf-8')
    (ROOT/'experiment_active.json').write_text(json.dumps(dict(study='optimized_minute_2023_2025',configuration='results/optimized_minute_2023_2025/configuration.json',report='book/sections/07_estudio_minuto.md',notebook='notebooks/Entregable_1_Minuto.ipynb',publication='local_only'),indent=2),encoding='utf-8')
    notebook = ROOT/'notebooks/Entregable_1_Minuto.ipynb'
    archived_nb = ROOT/'notebooks/Referencia_Minuto_Original.ipynb'
    if not archived_nb.exists(): shutil.copy2(notebook,archived_nb)
    nb = nbformat.read(ROOT/'notebooks/Optimizacion_SVR_Minuto.ipynb',as_version=4)
    nb.cells[0] = nbformat.v4.new_markdown_cell(report.replace('(../figures/','(../book/figures/').replace('(../../notebooks/','(').replace('(../../delivery/','(../delivery/'))
    NotebookClient(nb,timeout=120,kernel_name='python3',resources={'metadata':{'path':str(ROOT)}}).execute()
    nbformat.write(nb,notebook)
    names = ['README.md','DASHBOARD.md','experiment_active.json','experiment_minute.json','requirements.txt','requirements-dashboard.txt',
             'notebooks/Entregable_1_Minuto.ipynb','notebooks/Optimizacion_SVR_Minuto.ipynb','notebooks/Referencia_Minuto_Original.ipynb']
    names += ['src/'+n+'.py' for n in ['optimize_minute_svr','report_optimized_svr','apply_optimized_delivery','metric_comparison','minute_experiment','volatility_experiment','download_three_years','download_minute_history']]
    names += ['results/minute_2023_2025/'+n for n in ['sources.csv','data_manifest.json','descriptive_statistics.csv','predictions.csv.gz','macro_metrics.csv']]
    for pattern in ['results/optimized_minute_2023_2025/**/*','data/processed/minute_2023_2025/*','book/**/*','tests/test_optimized_minute_svr.py']:
        names.extend(str(p.relative_to(ROOT)).replace('\\','/') for p in ROOT.glob(pattern) if p.is_file() and '_build' not in p.parts)
    names = sorted(set(names))
    manifest = {n:hashlib.sha256((ROOT/n).read_bytes()).hexdigest() for n in names}
    dest = ROOT/'delivery'
    (dest/'OPTIMIZED_MANIFEST.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    path = dest/'Entregable1_optimizado_2023_2025.zip'
    with zipfile.ZipFile(path,'w',zipfile.ZIP_DEFLATED) as bundle:
        for name in names: bundle.write(ROOT/name,name)
        bundle.write(dest/'OPTIMIZED_MANIFEST.json','delivery/OPTIMIZED_MANIFEST.json')
    with zipfile.ZipFile(path) as bundle:
        for name,digest in manifest.items(): assert hashlib.sha256(bundle.read(name)).hexdigest() == digest, name
    print(json.dumps(dict(files=len(names),zip_bytes=path.stat().st_size,manifest_verified=True,notebook_executed_cells=sum(c.cell_type=='code' and c.execution_count is not None for c in nb.cells)),indent=2))


if __name__ == '__main__': main()
