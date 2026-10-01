"""Build the report and executable results notebook from real minute-run outputs."""
import json
import nbformat
import pandas as pd
import numpy as np
import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from minute_experiment import ROOT,OUT,DATA,load_panel,windows,predict_artifact
from volatility_experiment import targets
from compare_cv_methods import scores


def split_diagnostics():
    panel=load_panel(); calendar=pd.read_csv(OUT/'calendar.csv').query("method == 'ForwardChaining'")
    rows=[]
    for symbol in panel:
        minutes=np.load(DATA/f'{symbol}.npy',mmap_mode='r')
        for w in [7,14,21,28]:
            artifact=joblib.load(OUT/'models'/f'{symbol}_v{w}.joblib')
            vol=targets(panel[symbol],w).to_numpy()
            for role,frame in calendar.groupby('split'):
                anchors=panel.index.get_indexer(pd.to_datetime(sorted(frame.origin.unique()),utc=True))
                X=windows(minutes,artifact['input_window'],anchors)
                y=vol[anchors[:,None]+np.arange(1,8)]
                for label,p in [('SVR',predict_artifact(artifact,X)),('Persistence',np.repeat(vol[anchors,None],7,axis=1))]:
                    rows.append(dict(symbol=symbol,volatility_window=w,input_window=artifact['input_window'],
                        split=role,model=label,n_origins=len(anchors),**scores(y,p)))
    frame=pd.DataFrame(rows); frame.to_csv(OUT/'selected_split_metrics.csv',index=False)
    return frame.groupby(['split','model'])[['r2','rmse']].mean()


def eda():
    from statsmodels.graphics.tsaplots import plot_acf
    panel=load_panel(); records=[]; links=[]
    directory=ROOT/'book/figures'; directory.mkdir(parents=True,exist_ok=True)
    for symbol in panel.columns:
        minutes=np.load(DATA/f'{symbol}.npy',mmap_mode='r')
        daily=panel[symbol]; returns=np.log(daily).diff()
        minute_returns=np.diff(np.log(minutes))
        for name,values in [('minute_close',minutes),('minute_log_return',minute_returns),('daily_log_return',returns.to_numpy())]:
            valid=values[np.isfinite(values)]
            records.append(dict(symbol=symbol,series=name,n=len(valid),missing=int((~np.isfinite(values)).sum()),
                mean=valid.mean(),std=valid.std(ddof=0),min=valid.min(),median=np.median(valid),max=valid.max()))
        fig,axes=plt.subplots(3,2,figsize=(12,9))
        daily.plot(ax=axes[0,0],title=f'{symbol}: cierre diario para visualización')
        (returns*100).plot(ax=axes[0,1],title='Retorno diario (%)')
        (returns*100).hist(bins=60,ax=axes[1,0]); axes[1,0].set_title('Histograma de retornos diarios')
        plot_acf(returns.to_numpy(),lags=40,missing='conservative',zero=False,auto_ylims=True,ax=axes[1,1],title='ACF retornos diarios')
        plot_acf(returns.to_numpy()**2,lags=40,missing='conservative',zero=False,auto_ylims=True,ax=axes[2,0],title='ACF retornos diarios al cuadrado')
        for w in [7,14,21,28]: (100*returns.rolling(w).std(ddof=0)).plot(ax=axes[2,1],label=f'{w} días')
        axes[2,1].set_title('Volatilidad (%)'); axes[2,1].legend()
        fig.tight_layout(); fig.savefig(directory/f'minute_eda_{symbol}.png',dpi=120); plt.close(fig)
        links.append(f'![EDA {symbol}](../figures/minute_eda_{symbol}.png)')
    pd.DataFrame(records).to_csv(OUT/'descriptive_statistics.csv',index=False)
    details=[]
    for symbol in panel.columns:
        rows=[r for r in records if r['symbol']==symbol]
        minute,daily=rows[0],rows[2]
        details.append(f"- **{symbol}:** {minute['n']:,} cierres de minuto válidos y {minute['missing']} faltantes. "
            f"Los retornos logarítmicos diarios tienen desviación estándar de {100*daily['std']:.2f} %, "
            f"mínimo {100*daily['min']:.2f} % y máximo {100*daily['max']:.2f} %. "
            "Los extremos justifican comparar errores cuadrados con MAE, menos sensible a valores extremos.")
    return '\n'.join(details)+'\n\n'+'\n\n'.join(links)


def build():
    status=json.loads((OUT/'status.json').read_text())
    assert status['status']=='complete'
    eda_links=eda()
    split_metrics=split_diagnostics()
    cache_note=''
    if (OUT/'selection_cache.csv').exists():
        cache_note=('La ejecución paralela reutilizó búsquedas ya terminadas, conservadas en selection_cache.csv, '
            'y volvió a verificar sus errores de validación. Para esas configuraciones, search.csv contiene la '
            'verificación del candidato seleccionado; para las nuevas, contiene la búsqueda completa. '
            'Una ejecución desde cero, sin caché, evalúa todos los candidatos.')
    macro=pd.read_csv(OUT/'macro_metrics.csv')
    selected=pd.read_csv(OUT/'selected_metrics.csv')
    calendar=pd.read_csv(OUT/'calendar.csv')
    counts=calendar.query("method == 'ForwardChaining'").groupby('split').origin.nunique()
    main=macro.query("scope == 'all_available_test' and method == 'ForwardChaining'")
    choices=pd.read_csv(OUT/'selected_inputs.csv').query("method == 'ForwardChaining'")
    predictions=pd.read_csv(OUT/'predictions.csv.gz').query("method == 'ForwardChaining'")
    predictions=predictions.merge(choices[['method','symbol','volatility_window','input_window']],
        on=['method','symbol','volatility_window','input_window'],validate='many_to_one')
    negatives=int((predictions.svr<0).sum())
    figure=ROOT/'book/figures/minute_2023_2025.png'
    figure.parent.mkdir(exist_ok=True,parents=True)
    fig,axes=plt.subplots(1,2,figsize=(12,4))
    for ax,column in zip(axes,['r2','rmse']):
        main.pivot(index='symbol',columns='model',values=column).plot.bar(ax=ax)
        ax.set_title(column.upper()); ax.tick_params(axis='x',rotation=25)
        ax.axhline(0,color='black',linewidth=.5)
    fig.tight_layout(); fig.savefig(figure,dpi=150); plt.close(fig)
    comparison=selected.query("scope == 'all_available_test' and method == 'ForwardChaining'").pivot(
        index=['symbol','volatility_window'],columns='model',values='rmse')
    wins=int((comparison.SVR<comparison.Persistence).sum())
    table=main[['symbol','model','r2','rmse','mae','mse','mape']].round(5)
    # Avoid optional tabulate dependency.
    markdown='| '+' | '.join(table.columns)+' |\n| '+' | '.join(['---']*len(table.columns))+' |\n'
    markdown+='\n'.join('| '+' | '.join(map(str,row))+' |' for row in table.itertuples(index=False,name=None))
    report=f'''# Estudio de tres años con entradas de un minuto

**Entregable 1 — Jassan Arteta y Mateo Bernal**  
Universidad del Norte · Maestría en Matemáticas · Machine Learning · Profesor Lihki Rubio · 2026-30

Periodo: **1 de enero de 2023 a 31 de diciembre de 2025 (UTC)**. BTCUSDT, ETHUSDT, BNBUSDT y XRPUSDT.
Entrenamiento 2023; validación 2024; evaluación retrospectiva 2025. Esta última ya se ha examinado en experimentos previos.

Las entradas conservan cada cierre de un minuto: 10.080, 20.160, 30.240 o 40.320 valores para ventanas de 7, 14, 21 o 28 días.
Se emite un pronóstico al cierre de cada día elegible. Las siete salidas son volatilidades diarias futuras:
100 veces la desviación estándar poblacional de retornos logarítmicos de cierres diarios, en ventanas de 7, 14, 21 o 28 días.
No se anualiza. Los días con minutos faltantes se excluyen cuando afectan una entrada o un objetivo; no se interpolan.
En este conjunto, el día incompleto es el 24 de marzo de 2023. Todas las ventanas comparten el calendario elegible
definido por la historia máxima, para que su comparación no cambie de fechas según el tamaño de entrada.

El escalado se ajusta exclusivamente en entrenamiento. LinearSVR se resuelve en una base ortonormal del espacio generado
por las filas de entrenamiento, conservando todas las direcciones numéricamente no nulas. Se verifica la reconstrucción
de la matriz de productos internos y se exportan coeficientes para todos los minutos originales.
No se sustituyen las entradas por cierres diarios. C, epsilon y ventana de entrada se eligen por RMSE de validación,
separadamente para cada cripto y definición de volatilidad.

{cache_note}

## Validación

Se ejecutan GroupKFold, KFold y ForwardChaining de timeseries-cv con filtros cronológicos.
Los adaptadores de KFold y ForwardChaining agrupan cortes nativos y, tras filtrar, generan exactamente las mismas muestras.
Sus cinco particiones de evaluación comparten entrenamiento: no son cinco reentrenamientos independientes.
GroupKFold conserva pocas muestras. El archivo all_metrics.csv distingue fechas comunes de todas las fechas disponibles.
Las diferencias de rendimiento entre GroupKFold y los otros adaptadores combinan el método de partición
y la cantidad de entrenamiento retenida; GroupKFold conserva entre 9 y 12 muestras de entrenamiento por fold.
Comparar métodos exige usar common_test; los resultados siguientes usan todas las fechas de ForwardChaining.
KFold y ForwardChaining usan {counts['train']} orígenes únicos de entrenamiento, {counts['val']} de validación
y {counts['test']} de prueba. Hay {status['common_test_origins']} fechas de prueba comunes a los tres métodos.
Los millones de cierres de minuto constituyen las columnas de las ventanas; no equivalen a millones de muestras supervisadas.

## Exploración

descriptive_statistics.csv contiene estadísticas de cierres y retornos de un minuto y de retornos diarios.
Las figuras usan cierres diarios para legibilidad y la frecuencia diaria del objetivo para los diagnósticos.
Los huecos se mantienen; la ACF usa tratamiento conservador de faltantes. La ACF de cuadrados permite inspeccionar
dependencia en la magnitud de los retornos, sin demostrar por sí sola un modelo de heterocedasticidad.

{eda_links}

## Resultados

Se evaluaron {status['configurations']} combinaciones método–cripto–entrada–objetivo.
SVR supera a persistencia en RMSE en **{wins} de 16** selecciones del método principal.
Persistencia repite la última volatilidad conocida en los siete horizontes.
Se conservan {negatives} predicciones negativas del SVR entre {len(predictions)} valores pronosticados seleccionados.
No se aplica recorte antes de evaluar; una predicción negativa no representa una volatilidad válida.
R² se promedia entre horizontes; por cripto se promedia entre cuatro objetivos, y GLOBAL_MACRO entre las 16 selecciones.
El global es un promedio macro, no un R² obtenido concatenando escalas distintas.
RMSE también promedia los siete RMSE por horizonte; no es la raíz de un único MSE global.
RMSE y MAE se expresan en puntos porcentuales de volatilidad, MSE en su cuadrado y MAPE en porcentaje.

{markdown}

![R² y RMSE](../figures/minute_2023_2025.png)

Reducir años y cambiar de cierres diarios a entradas de un minuto modifica simultáneamente muestra y representación.
Por ello no se atribuye una diferencia de métricas exclusivamente a reducir el periodo. Más predictores que observaciones
de entrenamiento puede perjudicar la generalización. Los objetivos de volatilidad móvil comparten retornos conocidos,
lo que favorece un baseline persistente; superar ese baseline sigue siendo la referencia pertinente.

El archivo bds.csv conserva el diagnóstico BDS de residuos del primer horizonte por partición.
GroupKFold tiene menos de seis observaciones por partición y se registra como no estimable.
En los demás métodos, las particiones contienen fechas espaciadas: este diagnóstico es exploratorio
y no equivale a una prueba sobre todos los residuos diarios consecutivos.

Para revisar generalización, selected_split_metrics.csv evalúa los mismos 16 modelos seleccionados en cada periodo,
sin reentrenarlos. Su R² macro de entrenamiento es {split_metrics.loc[('train','SVR'),'r2']:.4f},
frente a {split_metrics.loc[('val','SVR'),'r2']:.4f} en validación y {split_metrics.loc[('test','SVR'),'r2']:.4f} en prueba.
Una brecha grande señala que el ajuste dentro de 2023 no se traslada a años posteriores; acortar el periodo
no resuelve por sí solo la generalización.

## Reproducción

Configuración: experiment_minute.json. Ejecutar, desde la raíz y con dependencias instaladas:

```bash
python src/run_minute_experiment.py
python src/report_minute_experiment.py
python src/verify_minute_experiment.py
```

La entrega incluye los datos procesados de un minuto. Para reconstruirlos desde archivos mensuales de Binance
ya descargados en data/raw/minute_2020_2025, ejecutar python src/minute_experiment.py.
Si faltan los originales, python src/download_three_years.py descarga únicamente los 144 archivos necesarios.
Resultados, calendario, métricas por horizonte y modelos: results/minute_2023_2025.
El estudio anterior con entradas diarias permanece como referencia histórica.
'''
    (ROOT/'book/sections/07_estudio_minuto.md').write_text(report,encoding='utf-8')
    notebook=nbformat.v4.new_notebook(cells=[
        nbformat.v4.new_markdown_cell(report.replace('../figures/','../book/figures/')),
        nbformat.v4.new_code_cell("from pathlib import Path\nimport pandas as pd\nROOT=Path.cwd()\nif not (ROOT/'experiment_minute.json').exists(): ROOT=ROOT.parent\nOUT=ROOT/'results/minute_2023_2025'\nassert OUT.exists()\n"),
        nbformat.v4.new_markdown_cell('La siguiente celda permite repetir el entrenamiento completo usando los datos de un minuto incluidos. Por defecto se consultan los resultados ejecutados. El entrenamiento puede tardar decenas de minutos.'),
        nbformat.v4.new_code_cell("RETRAIN=False\nif RETRAIN:\n    import subprocess, sys\n    subprocess.run([sys.executable,str(ROOT/'src/run_minute_experiment.py')],cwd=ROOT,check=True)"),
        nbformat.v4.new_code_cell("pd.read_csv(OUT/'descriptive_statistics.csv')"),
        nbformat.v4.new_code_cell("pd.read_csv(OUT/'macro_metrics.csv').query(\"scope == 'all_available_test' and method == 'ForwardChaining'\")"),
        nbformat.v4.new_code_cell("pd.read_csv(OUT/'selected_inputs.csv')"),
        nbformat.v4.new_code_cell("pd.read_csv(OUT/'selected_split_metrics.csv').groupby(['split','model'])[['r2','rmse','mae']].mean()"),
        nbformat.v4.new_code_cell("pd.read_csv(OUT/'macro_metrics.csv').query(\"scope == 'common_test'\")"),
        nbformat.v4.new_code_cell("pd.read_csv(OUT/'calendar.csv').groupby(['method','fold','split']).size().unstack()"),
    ],metadata={'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'}})
    from nbclient import NotebookClient
    NotebookClient(notebook,timeout=180,resources={'metadata':{'path':str(ROOT)}}).execute()
    nbformat.write(notebook,ROOT/'notebooks/Entregable_1_Minuto.ipynb')
    print(table.to_string(index=False)); print(f'RMSE wins: {wins}/16')


if __name__=='__main__': build()
