"""Synchronize executed daily results to Book, notebook, README and ZIP."""
from pathlib import Path
import base64
import hashlib
import io
import json
import re
import zipfile
import numpy as np
import pandas as pd
import nbformat as nbf
from volatility_experiment import ROOT, configuration


def table(frame):
    def val(v):
        if isinstance(v, (float, np.floating)):
            return f'{v:.5f}' if np.isfinite(v) else 'No estimable'
        return str(v).replace('|', '/')
    return '\n'.join(['| '+' | '.join(map(str, frame.columns))+' |',
        '| '+' | '.join(['---']*len(frame.columns))+' |'] +
        ['| '+' | '.join(map(val, row))+' |' for row in frame.itertuples(index=False, name=None)])


def write(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.strip()+'\n', encoding='utf-8')


def main():
    cfg = configuration(); out = ROOT/'results'; book = ROOT/'book'; sections = book/'sections'
    status = json.loads((out/'run_status.json').read_text())
    if status['status'] != 'complete': raise RuntimeError('Experiment incomplete')
    metrics = pd.read_csv(out/'metrics.csv'); bds = pd.read_csv(out/'bds.csv')
    registry = pd.read_csv(out/'model_registry.csv'); eda = pd.read_csv(out/'eda_summary.csv')
    dates = pd.read_csv(out/'fold_calendar.csv'); sources = pd.read_csv(out/'minute_sources.csv')
    daily = pd.read_csv(ROOT/'data/processed/daily_2020_2025.csv')
    coverage = daily.groupby('symbol').agg(calendar_days=('date','size'), complete_days=('complete','sum'),
        minute_rows=('minutes','sum'), invalid_closes=('invalid_close_times','sum')).reset_index()
    coverage['excluded_days'] = coverage.calendar_days-coverage.complete_days
    coverage.to_csv(out/'coverage.csv', index=False)
    test = metrics.query("split == 'test' and horizon == 0")
    means = test.groupby(['symbol','volatility_window','input_window','model'], as_index=False)[['mape','mae','rmse','mse']].mean()
    selected = means.merge(registry[['symbol','volatility_window','input_window']], on=['symbol','volatility_window','input_window'])
    comparison = selected.pivot(index=['symbol','volatility_window','input_window'], columns='model', values='rmse').reset_index()
    comparison['SVR_minus_Persistence'] = comparison.SVR-comparison.Persistence
    comparison.to_csv(out/'selected_test_comparison.csv', index=False)
    wins = int((comparison.SVR_minus_Persistence < 0).sum())
    sizes = dates.groupby(['fold','split']).size().unstack().reset_index()
    symbols = ', '.join(cfg['symbols'])
    intro = f'''# Pronóstico de volatilidad diaria con SVR lineal

**Entregable 1 — Jassan Arteta y Mateo Bernal**  
Universidad del Norte · Maestría en Matemáticas · Machine Learning · Profesor Lihki Rubio · 2026-30

## Resumen

Se estudian {symbols} entre el 1 de enero de 2020 y el 31 de diciembre de 2025. Se descargaron
{int(sources.minutes.sum()):,} velas de un minuto de los archivos oficiales de Binance, con verificación SHA-256,
y se obtuvieron cierres diarios UTC. El objetivo es pronosticar siete valores futuros de volatilidad móvil.
Se cruzan cuatro ventanas de precios (7, 14, 21 y 28 días) con cuatro ventanas de volatilidad de igual conjunto:
64 configuraciones activo–entrada–objetivo, cinco folds por configuración y siete SVR lineales por modelo multisalida.

La función Group K-Fold de `timeseries-cv` se ejecuta sobre índices temporales comunes y se complementa con filtros
cronológicos. Entrenamiento: 2020–2022; validación: 2023; test: 2024–2025. El escalado y el ajuste usan solo TRAIN;
los hiperparámetros y la ventana de entrada se seleccionan con validación. Se evalúan precios diarios históricos
como entradas, pero **todas las salidas del modelo son volatilidades**, no precios.

Entre las 16 selecciones por activo y definición del objetivo, SVR supera a Persistence en RMSE test en {wins} casos.
Este conteo es descriptivo: los folds se solapan y el muestreo de `tsxv` deja pocas fechas de evaluación.
No se comparan errores absolutos entre objetivos de volatilidad distintos para declarar un ganador global.

## Organización del informe

1. [Datos y problema](sections/01_base_datos.md).
2. [EDA y preprocesamiento](sections/02_eda.md).
3. [Validación y SVR lineal](sections/03_modelo_base.md).
4. [Resultados detallados y residuos](sections/04_resultados_diarios.md).
5. [Conclusiones](sections/05_conclusiones.md).
6. [Dashboard, API y reproducción](sections/06_dashboard.md).

La versión horaria anterior está archivada en `delivery/legacy_hourly_before_daily_restructure.zip`.
Sus métricas y conclusiones no pertenecen a este experimento.
'''
    data_text = f'''# 1. Datos y problema de investigación

## Pregunta y alcance

¿Puede un SVR lineal, usando únicamente cierres diarios pasados, mejorar la persistencia al pronosticar
la volatilidad móvil de BTC, ETH, BNB y XRP durante los siguientes siete días?
Los cuatro activos se modelan por separado; no se incorporan noticias, volumen ni variables externas como predictores.

## Fuente, frecuencia y fechas

Binance Spot, pares contra USDT, del 01/01/2020 al 31/12/2025 inclusive (fin exclusivo 01/01/2026).
Los archivos mensuales oficiales contienen velas de **un minuto** procedentes de `/api/v3/klines`.
Esta descarga conserva la granularidad exigida; el modelado utiliza el último cierre diario UTC para limitar
la dimensión de entrada. La frecuencia de adquisición y la de modelado se documentan por separado.
La [documentación oficial](https://github.com/binance/binance-public-data) describe los archivos,
checksums y el cambio a timestamps en microsegundos desde enero de 2025; el lector normaliza ambos formatos a UTC.

## Calidad y cobertura ejecutada

{table(coverage)}

Un día es elegible si contiene exactamente 1.440 minutos únicos y todas las velas tienen tiempos de cierre válidos.
Los días incompletos quedan como faltantes en el calendario, sin interpolación ni compresión del tiempo.
Se excluyen las muestras que requieren cualquiera de esos días. Es una política conservadora: también descarta
días cuyo cierre final podría existir, pero cuya cobertura intradiaria no es completa. No se atribuyen las anomalías
a una causa sin evidencia. La selección puede cambiar la representatividad del conjunto.

## Diccionario

| Campo | Uso y unidad |
| --- | --- |
| `symbol` | Identificador del par; cuatro modelos separados por objetivo |
| `date` | Día UTC de la vela agregada; disponible al finalizar ese día |
| `close` | Último precio de cierre del día, en USDT; NaN si día inelegible |
| `minutes` | Número observado de velas de un minuto |
| `invalid_close_times` | Velas con final temporal anómalo |
| `complete` | Indicador de día elegible |
| `log_return` | Logaritmo del cociente de dos cierres diarios consecutivos |
| `volatility_w` | 100 por desviación estándar poblacional de w retornos diarios |

El precio es una magnitud positiva; los retornos pueden ser negativos; la volatilidad observada es no negativa.
Los archivos originales incluyen OHLC, volúmenes y operaciones, pero estos campos no entran en el SVR.
`results/minute_sources.csv` registra la URL, hash, cobertura y unidad temporal de cada uno de los 288 archivos.

## Representatividad y uso

La muestra representa cuatro pares de un único exchange, con selección retrospectiva de activos.
No representa todos los mercados ni garantiza generalización a otros periodos o monedas. Se respetan los
[términos del proveedor](https://github.com/binance/binance-public-data/blob/master/TERMS_AND_CONDITIONS.md).
El estudio no contiene datos personales y sus errores de pronóstico no se interpretan como rentabilidad de una estrategia.
'''
    eda_text = '''# 2. EDA y preprocesamiento

## Alcance del análisis exploratorio

Las estadísticas y visualizaciones exploratorias usan exclusivamente **TRAIN, 2020–2022**.
La auditoría de cobertura puede revisar las fechas de todo el periodo; la elección de modelos no utiliza los valores de test.
Las series de precios muestran niveles y cambios temporales; sus retornos facilitan comparar movimientos relativos.
Las colas del histograma y la curtosis describen valores extremos, sin justificar su eliminación automática.
La ACF de retornos explora dependencia lineal; la ACF de retornos al cuadrado explora persistencia de la variabilidad,
pero no constituye por sí sola una prueba concluyente de heterocedasticidad.

Las ACF se calculan sobre el tramo diario continuo más largo del entrenamiento; no se concatenan observaciones
a ambos lados de un hueco. Sus fechas y tamaños quedan en `results/eda_acf.csv`.
'''
    for symbol in cfg['symbols']:
        stats = eda[eda.symbol == symbol]
        ret = stats[stats.variable == 'log_return'].iloc[0]
        eda_text += f'\n## {symbol}\n\n'+table(stats[['variable','count','mean','std','min','50%','max','missing','skew','kurtosis']])
        eda_text += f'\n\nLa media de retornos es {ret["mean"]:.6f} y su desviación estándar {ret["std"]:.6f}. '
        eda_text += f'La asimetría es {ret["skew"]:.3f} y el exceso de curtosis {ret["kurtosis"]:.3f}; '
        eda_text += 'estos valores caracterizan la distribución observada y no implican normalidad ni estabilidad futura.\n\n'
        eda_text += f'![EDA {symbol}](../../notebooks/figs/eda_{symbol}.png)\n'
    eda_text += r'''

## Construcción de entradas y objetivos

Con el pronóstico emitido al terminar el día t:

$$r_t=\log(P_t/P_{t-1}),\qquad
\sigma_t^{(w)}=100\sqrt{\frac1w\sum_{j=0}^{w-1}(r_{t-j}-\bar r_t^{(w)})^2}.$$

La fórmula usa `ddof=0`, retornos diarios y porcentaje, sin anualizar. Requiere w+1 cierres consecutivos.
Se adopta el índice de fin de ventana: el retorno del día t ya se conoce al emitir la predicción.

$$X_t^{(L)}=(P_{t-L+1},\ldots,P_t),\quad
y_t^{(w)}=(\sigma_{t+1}^{(w)},\ldots,\sigma_{t+7}^{(w)}).$$

L y w recorren independientemente {7,14,21,28}. Son 16 combinaciones por activo.
Cada salida es una volatilidad histórica móvil observada en una fecha futura, no una volatilidad calculada
exclusivamente con retornos posteriores al origen. Para w>7, una parte importante de los retornos que forman
las salidas ya es conocida en t; esto facilita persistencia y explica parte de la dependencia entre horizontes.

El máximo historial común exige 29 cierres, por el objetivo/persistencia de 28 retornos. Se usa la misma
máscara de disponibilidad entre activos y las 16 combinaciones, evitando ventajas por evaluar en fechas distintas.
El escalado de entradas y de las siete salidas se ajusta dentro de cada entrenamiento; las métricas se calculan
después de invertir el escalado de la variable objetivo. No se aplica PCA ni eliminación de valores extremos.
'''
    model_text = f'''# 3. Validación temporal y modelo

## Adaptación de la guía

Se conserva Binance y se sustituye el MLP de la guía por **SVR lineal**, según el alcance acordado.
Se instala `timeseries-cv==0.1.5` y se importa `tsxv`. La firma real de la versión instalada es
`split_train_val_test_groupKFold(sequence, numInputs, numOutputs, numJumps)`;
los nombres de argumentos de algunos ejemplos de la guía no coinciden con esta versión.

## Particiones y prevención de fuga

Se invoca la función exigida sobre índices del calendario con `numInputs=28`, `numOutputs=7`, `numJumps=1`.
Las ventanas de índices se traducen después a los cierres (recortados a L) y a la volatilidad del objetivo w.
El calendario de 28 días se fija para todas las comparaciones; no se vuelve a generar un calendario distinto para cada L.
Es una **adaptación de Group K-Fold de tsxv con filtros cronológicos**, no el método nativo sin cambios.

La implementación nativa intercala entrenamiento, validación y test; puede entrenar con fechas posteriores
a las de evaluación y no garantiza por sí sola prevención de fuga. Se conservan únicamente:

- TRAIN: etiquetas completamente anteriores al 01/01/2023.
- Validación: origen desde 01/01/2023 y etiquetas anteriores al 01/01/2024.
- TEST: origen desde 01/01/2024 y etiquetas hasta 31/12/2025.

Se comprueba que la última etiqueta de TRAIN preceda al primer origen de validación y que la última etiqueta
de validación preceda al primer origen de test. El historial conocido puede cruzar hacia un bloque anterior;
eso representa información disponible en el origen. Los modelos de cada fold se ajustan solo con TRAIN.

{table(sizes)}

**Limitación:** numJumps=1 no significa un origen de evaluación diario en esta librería.
El muestreo nativo deja pocas muestras por fold; los cinco folds se solapan y no son cinco réplicas independientes.
Las desviaciones estándar entre folds son descriptivas, no errores estándar ni intervalos de confianza.
La auditoría exacta de fechas se entrega en `results/fold_calendar.csv`.

## SVR lineal multisalida

`StandardScaler` de X seguido de `TransformedTargetRegressor`, con escalado de y y
`MultiOutputRegressor(LinearSVR(...))`: siete regresores independientes para h=1,…,7.
Se usa pérdida `squared_epsilon_insensitive`, `dual=False`, tolerancia 1e-6, máximo 50.000 iteraciones y semilla 42.
Un aviso de no convergencia detiene la ejecución. Las predicciones negativas se conservan y se contabilizan,
pues el SVR no impone la restricción física de volatilidad no negativa.

Se prueba C en {{0.01,0.1,1,10}} y epsilon en {{0.01,0.1}}, en unidades estandarizadas del objetivo.
Por activo, w y L, se selecciona la pareja que minimiza el RMSE de validación promediado sobre horizontes y folds.
Después se elige L por la misma métrica de validación, manteniendo fijo w. TEST no participa en estas selecciones.
Persistence repite la volatilidad conocida en t en los siete horizontes.

## Evaluación y residuos

Se guardan todas las predicciones TRAIN/VAL/TEST con origen y fecha objetivo. MAPE se expresa en porcentaje;
MAE y RMSE en puntos porcentuales de volatilidad; MSE en puntos porcentuales al cuadrado.
La columna horizonte=0 es el promedio de las siete métricas por horizonte; el promedio de RMSE no es la raíz
del MSE global. Los objetivos cero se excluyen solo de MAPE y se cuentan explícitamente.

BDS usa `statsmodels.tsa.stattools.bds`, dimensión 2 y distancia por defecto (1.5 desviaciones estándar).
Se calcula sobre residuos h=1 ordenados temporalmente de cada fold test. Se marca no estimable si hay menos
de seis residuos, si son constantes o si el cálculo no es finito. Seis es un control operativo, no garantía asintótica.
El tamaño reducido y el espaciado entre orígenes limitan fuertemente la inferencia. p>0.05 no demuestra iid;
solo indica no rechazo. El promedio de valores p exigido por la guía es descriptivo, no un contraste combinado.

## Fuentes metodológicas

- [timeseries-cv y atribución a Filipe Roberto Ramos](https://pypi.org/project/timeseries-cv/).
- [LinearSVR](https://scikit-learn.org/stable/modules/generated/sklearn.svm.LinearSVR.html).
- [Regresión multisalida](https://scikit-learn.org/stable/modules/generated/sklearn.multioutput.MultiOutputRegressor.html).
- [Test BDS](https://www.statsmodels.org/stable/generated/statsmodels.tsa.stattools.bds.html).
'''
    result_text = '# 4. Resultados por objetivo, entrada y activo\n\n'
    result_text += 'Las tablas corresponden a TEST. La elección de parámetros e inputs se hizo con validación. '
    result_text += 'Los gráficos mejor/mediano/peor por error test son ilustraciones posteriores a la evaluación, no selección de modelos. '
    result_text += 'BDS dispone solo de 8–9 residuos por fold: sus valores p asintóticos se muestran para completar el diagnóstico, '
    result_text += 'pero no sostienen conclusiones fiables sobre independencia.\n'
    for symbol in cfg['symbols']:
      for w in cfg['volatility_windows']:
        result_text += f'\n## {symbol}: volatilidad de {w} días\n\n'
        selected_row = registry.query('symbol == @symbol and volatility_window == @w').iloc[0]
        result_text += f'Entrada elegida con validación: **{int(selected_row.input_window)} días**, C={selected_row.C:g}, epsilon={selected_row.epsilon:g}.\n\n'
        result_text += table(means.query('symbol == @symbol and volatility_window == @w')[['input_window','model','mape','mae','rmse','mse']])+'\n\n'
        result_text += f'![RMSE {symbol} {w}](../../notebooks/figs/rmse_{symbol}_v{w}.png)\n'
        for lag in cfg['input_windows']:
            block = test.query('symbol == @symbol and volatility_window == @w and input_window == @lag and model == "SVR"').merge(
                bds, on=['symbol','volatility_window','input_window','model','fold'], suffixes=('','_bds'))
            cols = ['fold','n','mape','mae','rmse','mse','bds_pvalue']
            display = block[cols].copy()
            stats = {c: (f'{block[c].mean():.5f} ± {block[c].std():.5f}' if block[c].notna().any()
                         else 'No estimable') for c in cols if c != 'fold'}
            display = pd.concat([display, pd.DataFrame([dict(fold='Mean ± Std', **stats)])], ignore_index=True)
            block.to_csv(out/f'{symbol}_v{w}_l{lag}_folds.csv', index=False)
            detailed = metrics.query('symbol == @symbol and volatility_window == @w and input_window == @lag and split == "test"')
            detailed.to_csv(out/f'{symbol}_v{w}_l{lag}_horizons.csv', index=False)
            result_text += f'\n### Entrada de {lag} días\n\n'+table(display)+'\n\n'
            result_text += f'![Predicciones {symbol} {w} {lag}](../../notebooks/figs/pred_{symbol}_v{w}_l{lag}.png)\n'
        row = comparison.query('symbol == @symbol and volatility_window == @w').iloc[0]
        result_text += f'\nCon la entrada seleccionada, RMSE SVR={row.SVR:.5f} frente a Persistence={row.Persistence:.5f}. '
        result_text += ('SVR mejora' if row.SVR_minus_Persistence < 0 else 'SVR no mejora')+' la referencia en este resumen de test. '
        result_text += 'No se extrapola esta diferencia a significancia estadística ni a objetivos de otra ventana.\n'
    conclusion = f'''# 5. Conclusiones

## Respuesta a la pregunta

{table(comparison)}

La ventana de entrada se seleccionó por validación para cada activo y ventana del objetivo. De las 16
comparaciones resultantes, SVR obtiene menor RMSE test que Persistence en {wins}. Los resultados completos
permiten examinar qué activo y definición de volatilidad favorecen cada modelo, sin reunir objetivos diferentes
en una única clasificación por RMSE. Las métricas corresponden a los modelos de los folds, no al reajuste de despliegue.

## Limitaciones

El Group K-Fold nativo necesita filtros cronológicos; esos filtros reducen las muestras.
Los folds dependientes y los pocos residuos limitan la interpretación de su variación y de BDS.
La volatilidad móvil comparte retornos entre salidas y puede contener retornos ya conocidos al emitir el pronóstico.
El historial de 2020–2022 no representa necesariamente las condiciones de 2024–2025.
Los días incompletos se excluyen con una política conservadora, y los activos se seleccionaron retrospectivamente.
Un SVR lineal puede predecir valores negativos; se reportan en las tablas sin recortarlos después de observar test.

## Continuidad

Una evaluación posterior puede ampliar los orígenes de test con una estrategia temporal más densa,
manteniendo estos resultados como referencia y sin reusar este test para afirmar una nueva evaluación independiente.
Los 16 modelos de API se reajustan con todos los datos elegibles de 2020–2025 una vez fijados los parámetros por validación.
Ese reajuste sirve a la inferencia y no tiene una nueva métrica fuera de muestra.
'''
    reproduction = '''# 6. Dashboard, API y reproducción

## Ejecución

Desde la raíz, con Python 3.10:

```bash
python -m pip install -r requirements.txt
python -m unittest discover -s tests -p 'test_daily*.py' -v
python src/download_minute_history.py
python src/run_daily_experiment.py
python src/build_daily_delivery.py
python src/verify_daily_delivery.py
python src/execute_daily_notebook.py
```

La descarga verifica 288 archivos ZIP oficiales de un minuto y sus checksums. Es reanudable: los archivos
validados se reutilizan. La entrega incluye los datos diarios derivados y la procedencia para reproducir
el modelado sin volver a descargar el histórico. Los ZIP originales permanecen en `data/raw/minute_2020_2025`.

El [notebook completo](../../notebooks/Entregable_1_Completo.ipynb) incluye datos diarios y código en un paquete
integrado; puede ejecutarse fuera del repositorio después de instalar las dependencias. Sus figuras y análisis
guardados corresponden a la ejecución documentada. Una celda de ejecución recalcula las 64 configuraciones.
Las tablas de resultados están en `results/`; las figuras, en `notebooks/figs/`.

## Dashboard

```bash
python -m pip install -r requirements-dashboard.txt
python dashboard.py
```

El dashboard lee exclusivamente las métricas diarias nuevas. Tiene filtros de activo, definición de volatilidad
y tamaño de entrada. Abrir http://localhost:8050. En Render, el comando es `gunicorn dashboard:server`.

## API

```bash
uvicorn app.api:app --host 127.0.0.1 --port 8000
```

`GET /models` indica la longitud de entrada seleccionada para cada activo y ventana de volatilidad.
`POST /predict` recibe `symbol`, `volatility_window` y `lags` (cierres diarios consecutivos, del más antiguo
al más reciente). Devuelve siete volatilidades en porcentaje, horizontes 1–7 y fecha final del entrenamiento.
Los precios deben ser positivos y finitos. La API verifica tamaño y valores, pero el cliente debe garantizar
que corresponden a días consecutivos y cierres UTC. El origen concreto no se infiere de una lista sin fechas.

```bash
docker build -t volatility-svr .
docker run --rm -p 8000:8000 volatility-svr
```

Docker utiliza los modelos ya entrenados; no descarga Binance ni entrena durante el arranque.
El workflow de pruebas comprueba construcción de ventanas, ausencia de información futura, escalado y contrato API.
La presencia de Dockerfile y workflow no implica que se haya completado un despliegue remoto; su estado se informa por separado.

## Artefactos

- [Entrega reproducible](../../delivery/Entregable1_diario.zip).
- [Configuración](../../experiment.json).
- [Datos diarios](../../data/processed/daily_2020_2025.csv).
- [Protocolo y registro de ejecución](../../results/run_status.json).
- [Dependencias](../../requirements.txt).
'''
    docs = {'01_base_datos.md':data_text, '02_eda.md':eda_text, '03_modelo_base.md':model_text,
        '04_resultados_diarios.md':result_text, '05_conclusiones.md':conclusion, '06_dashboard.md':reproduction}
    write(book/'entregable1_master.md', intro)
    for filename, text in docs.items(): write(sections/filename, text)
    write(book/'myst.yml', '''version: 1
project:
  id: cd88d8d9-9d15-4270-aec0-4687c9c73a15
  title: Volatilidad diaria con SVR lineal
  toc:
    - file: entregable1_master.md
    - file: sections/01_base_datos.md
    - file: sections/02_eda.md
    - file: sections/03_modelo_base.md
    - file: sections/04_resultados_diarios.md
    - file: sections/05_conclusiones.md
    - file: sections/06_dashboard.md
site:
  template: book-theme
  options:
    style: _static/custom.css
''')
    write(ROOT/'README.md', intro.replace('](sections/', '](book/sections/')+'\n'+reproduction.replace('../../',''))
    write(ROOT/'DASHBOARD.md', reproduction.replace('../../',''))
    notebook(intro, docs)
    package()


def notebook(intro, docs):
    payload = io.BytesIO()
    sources = ['src/volatility_experiment.py', 'src/run_daily_experiment.py', 'experiment.json',
        'data/processed/daily_2020_2025.csv']
    with zipfile.ZipFile(payload, 'w', zipfile.ZIP_DEFLATED) as z:
        for name in sources: z.write(ROOT/name, name)
    cells = [nbf.v4.new_markdown_cell(intro), nbf.v4.new_markdown_cell(
        '## Reproducción independiente\n\nEl paquete integrado contiene código, configuración y datos diarios. '
        'La celda de ejecución recalcula el experimento completo en una carpeta temporal. Las figuras guardadas '
        'documentan la ejecución entregada; no son predicciones recalculadas al abrir el archivo.')]
    encoded = base64.b64encode(payload.getvalue()).decode()
    cells.append(nbf.v4.new_code_cell("import base64, io, zipfile, tempfile, pathlib, sys\n"
        "workspace = pathlib.Path(tempfile.mkdtemp(prefix='volatility_daily_'))\n"
        f"payload = {encoded!r}\n"
        "with zipfile.ZipFile(io.BytesIO(base64.b64decode(payload))) as archive:\n"
        "    archive.extractall(workspace)\n"
        "sys.path.insert(0, str(workspace / 'src'))\n"
        "print('Directorio de reproducción:', workspace)"))
    cells.append(nbf.v4.new_code_cell("from run_daily_experiment import run\nrun()\n"
        "import pandas as pd\nfrom IPython.display import display, Image\n"
        "display(pd.read_csv(workspace / 'results/model_registry.csv'))"))
    for name, text in docs.items():
        chunks = re.split(r'(!\[[^\]]*\]\([^\)]+\.png\))', text)
        for chunk in chunks:
            match = re.fullmatch(r'!\[([^\]]*)\]\(([^\)]+)\)', chunk)
            if match:
                path = (ROOT/'book/sections'/match.group(2)).resolve()
                rel = path.relative_to(ROOT).as_posix()
                cell = nbf.v4.new_code_cell(f"display(Image(filename=str(workspace / {rel!r})))")
                cell.outputs = [nbf.v4.new_output('display_data', data={
                    'image/png':base64.b64encode(path.read_bytes()).decode(), 'text/plain':match.group(1)})]
                cells.append(cell)
            elif chunk.strip():
                cells.append(nbf.v4.new_markdown_cell(chunk))
    nb = nbf.v4.new_notebook(cells=cells, metadata={'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'}})
    nbf.validate(nb)
    nbf.write(nb, ROOT/'notebooks/Entregable_1_Completo.ipynb')


def package():
    files = []
    for name in ['volatility_experiment.py','run_daily_experiment.py','download_minute_history.py','build_daily_delivery.py',
                 'verify_daily_delivery.py','execute_daily_notebook.py']:
        files.append(ROOT/'src'/name)
    for folder in ['app','results','notebooks/figs','.github/workflows']:
        files.extend(p for p in (ROOT/folder).rglob('*') if p.is_file() and '__pycache__' not in p.parts)
    files.extend((ROOT/'tests').glob('test_daily*.py'))
    files.extend(ROOT/'book/sections'/name for name in ['01_base_datos.md','02_eda.md','03_modelo_base.md',
        '04_resultados_diarios.md','05_conclusiones.md','06_dashboard.md'])
    files.extend(ROOT/name for name in ['experiment.json','README.md','DASHBOARD.md','dashboard.py','app.py',
        'requirements.txt','requirements-api.txt','requirements-dashboard.txt','requirements-lock-windows-py310.txt',
        'Dockerfile','.dockerignore','render.yaml',
        'book/myst.yml','book/entregable1_master.md','data/processed/daily_2020_2025.csv',
        'notebooks/Entregable_1_Completo.ipynb'])
    files.extend(p for p in (ROOT/'book/_static').glob('*.css'))
    manifest = {p.relative_to(ROOT).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(set(files))}
    dest = ROOT/'delivery/Entregable1_diario.zip'
    with zipfile.ZipFile(dest, 'w', zipfile.ZIP_DEFLATED) as z:
        for p in sorted(set(files)): z.write(p, p.relative_to(ROOT))
        z.writestr('MANIFEST.json', json.dumps(manifest, indent=2))
    with zipfile.ZipFile(dest) as z:
        if z.testzip(): raise ValueError('Invalid delivery archive')
    write(ROOT/'delivery/DAILY_MANIFEST.json', json.dumps(manifest, indent=2))
    print(f'Delivery: {dest} ({len(files)} files)', flush=True)


if __name__ == '__main__': main()
