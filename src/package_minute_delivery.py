"""Activate the verified minute report and package reproducible local artifacts."""
from pathlib import Path
import hashlib
import json
import shutil
import zipfile
from minute_experiment import ROOT,OUT


def main():
    assert json.loads((OUT/'verification.json').read_text())['status']=='passed'
    report=ROOT/'book/sections/07_estudio_minuto.md'
    assert report.exists()
    master=ROOT/'book/entregable1_master.md'
    historical=ROOT/'book/sections/08_referencia_diaria.md'
    if not historical.exists():
        historical.write_text(master.read_text(encoding='utf-8').replace('(sections/','('),encoding='utf-8')
    intro='''# Volatilidad con SVR lineal: Binance 2023–2025

**Entregable 1 — Jassan Arteta y Mateo Bernal**

El estudio activo usa BTCUSDT, ETHUSDT, BNBUSDT y XRPUSDT desde enero de 2023 hasta diciembre de 2025.
Conserva cierres de **un minuto** en entradas de 7, 14, 21 y 28 días. Entrena con 2023, selecciona parámetros
con 2024 y evalúa retrospectivamente en 2025. Produce siete valores diarios futuros de volatilidad.

Consulta el [informe vigente](sections/07_estudio_minuto.md), con resultados reales, gráficos, comparación
con persistencia y limitaciones de los adaptadores de validación de timeseries-cv.

La [referencia anterior](sections/08_referencia_diaria.md) y las secciones señaladas como históricas
documentan el experimento de seis años con entradas diarias; no describen la configuración activa.

Descargas: [notebook ejecutado](../notebooks/Entregable_1_Minuto.ipynb) y
[entrega reproducible con datos de un minuto](../delivery/Entregable1_minuto_2023_2025.zip).
'''
    master.write_text(intro,encoding='utf-8')
    readme_intro=intro.replace('(sections/','(book/sections/').replace('(../notebooks/','(notebooks/').replace('(../delivery/','(delivery/')
    readme_intro=readme_intro.replace('(book/sections/08_referencia_diaria.md)',
        '(https://jassan-bit.github.io/MachineLearning_202630/referencia-diaria/)')
    (ROOT/'README.md').write_text(readme_intro+'''

## Reproducción

```bash
python -m pip install -r requirements.txt
python src/run_minute_experiment.py
python src/report_minute_experiment.py
python src/verify_minute_experiment.py
```

Los datos procesados de un minuto están en `data/processed/minute_2023_2025` y sus hashes en
`results/minute_2023_2025/data_manifest.json`. Para reconstruirlos desde ZIP mensuales ya descargados,
ejecutar `python src/minute_experiment.py`. La procedencia original se registra en `sources.csv`.
Si faltan los archivos originales, `python src/download_three_years.py` descarga únicamente los 144 meses–activo necesarios.

Notebook ejecutado: [Entregable_1_Minuto.ipynb](notebooks/Entregable_1_Minuto.ipynb).
Entrega: [Entregable1_minuto_2023_2025.zip](delivery/Entregable1_minuto_2023_2025.zip).
Resultados y modelos: `results/minute_2023_2025`. El notebook consulta resultados; el script anterior
realiza el entrenamiento completo. Los modelos publicados conservan el ajuste de 2023.

## Aplicaciones

Instalar `requirements-dashboard.txt` y ejecutar `python dashboard.py`; abrir http://localhost:8050.
La API se ejecuta con `uvicorn app.api:app --host 127.0.0.1 --port 8000`.
`GET /models` devuelve `n_features` y `input_frequency`; `POST /predict` recibe
`symbol`, `volatility_window` y `lags`: todos los cierres consecutivos de un minuto, del más antiguo
al más reciente. Se espera una ventana terminada en el cierre diario UTC. La API valida valores
y longitud; la lista no contiene marcas temporales para verificar continuidad.

En el repositorio completo, el experimento anterior permanece en `results/`, `experiment.json` y `delivery/Entregable1_diario.zip`.
El ZIP nuevo incluye el estudio vigente y permite repetir su modelado; para regenerar todo el Book con las referencias históricas, usar el repositorio completo.
''',encoding='utf-8')
    (ROOT/'DASHBOARD.md').write_text('''# Dashboard y API del estudio 2023–2025

Ejecutar `python dashboard.py` y abrir http://localhost:8050.
Lee `results/minute_2023_2025` cuando status.json indica complete; conserva compatibilidad con el estudio diario anterior.
Permite filtrar criptomoneda, método, ventana de entrada, ventana de volatilidad y fechas comunes/todas.
Las métricas incluyen R², RMSE y MAE, junto con el número de orígenes y comparación con persistencia.

Para la API: `uvicorn app.api:app --host 127.0.0.1 --port 8000`.
Consultar `/models` antes de enviar `/predict`. `n_features` indica 10.080, 20.160, 30.240 o 40.320 cierres
de un minuto consecutivos. `input_window` se expresa en días. Son siete salidas diarias de volatilidad.
El cliente debe proporcionar una ventana completa que termine en el cierre diario UTC.

Los modelos conservan el entrenamiento de 2023 para reproducir las predicciones reportadas.
El Dockerfile incluye los modelos y metadatos; no descarga datos ni entrena al arrancar.
Un Dockerfile preparado no equivale a un despliegue remoto verificado.
''',encoding='utf-8')
    config=ROOT/'book/myst.yml'
    value=config.read_text(encoding='utf-8')
    value=value.replace('title: Volatilidad diaria con SVR lineal','title: Volatilidad con SVR lineal — Binance 2023–2025')
    if 'sections/07_estudio_minuto.md' not in value:
        value=value.replace('    - file: sections/01_base_datos.md',
            '    - file: sections/07_estudio_minuto.md\n    - file: sections/08_referencia_diaria.md\n    - file: sections/01_base_datos.md')
    config.write_text(value,encoding='utf-8')
    for path in sorted((ROOT/'book/sections').glob('0[1-6]_*.md')):
        content=path.read_text(encoding='utf-8')
        if not content.startswith('> **Referencia histórica'):
            content='> **Referencia histórica:** estudio anterior 2020–2025 con entradas diarias. Consulte el [estudio vigente de tres años](07_estudio_minuto.md).\n\n'+content
            path.write_text(content,encoding='utf-8')
    names=['experiment_minute.json','requirements.txt','requirements-dashboard.txt','requirements-api.txt',
        'dashboard.py','minute_dashboard.py','Dockerfile','README.md','DASHBOARD.md',
        'notebooks/Entregable_1_Minuto.ipynb','.github/workflows/daily-tests.yml']
    names += [f'src/{s}.py' for s in ['minute_experiment','run_minute_experiment','report_minute_experiment',
        'verify_minute_experiment','package_minute_delivery','volatility_experiment','compare_cv_methods',
        'download_three_years','download_minute_history','run_minute_parallel']]
    for pattern in ['data/processed/minute_2023_2025/*','results/minute_2023_2025/*',
                    'results/minute_2023_2025/models/*','app/*.py','tests/test_minute*.py',
                    'tests/test_daily_api.py','tests/test_dashboard.py','tests/test_cv_comparison.py',
                    'book/sections/07_estudio_minuto.md','book/figures/minute*.png']:
        names += [str(p.relative_to(ROOT)).replace('\\','/') for p in ROOT.glob(pattern) if p.is_file()]
    names=sorted(n for n in set(names) if not Path(n).name.startswith('checkpoint.'))
    manifest={n:hashlib.sha256((ROOT/n).read_bytes()).hexdigest() for n in names}
    dest=ROOT/'delivery'; dest.mkdir(exist_ok=True)
    (dest/'MINUTE_MANIFEST.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    with zipfile.ZipFile(dest/'Entregable1_minuto_2023_2025.zip','w',zipfile.ZIP_DEFLATED) as bundle:
        for name in names: bundle.write(ROOT/name,name)
        bundle.write(dest/'MINUTE_MANIFEST.json','delivery/MINUTE_MANIFEST.json')
    print(f'Packaged {len(names)} files; {(dest/"Entregable1_minuto_2023_2025.zip").stat().st_size/1e6:.1f} MB')


if __name__=='__main__': main()
