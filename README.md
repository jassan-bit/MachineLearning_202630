# Volatilidad con SVR lineal: Binance 2023–2025

**Entregable 1 — Jassan Arteta y Mateo Bernal**

El estudio activo usa BTCUSDT, ETHUSDT, BNBUSDT y XRPUSDT desde enero de 2023 hasta diciembre de 2025.
Conserva cierres de **un minuto** en entradas de 7, 14, 21 y 28 días. Entrena con 2023, selecciona parámetros
con 2024 y evalúa retrospectivamente en 2025. Produce siete valores diarios futuros de volatilidad.

Consulta el [informe vigente](book/sections/07_estudio_minuto.md), con resultados reales, gráficos, comparación
con persistencia y limitaciones de los adaptadores de validación de timeseries-cv.

La [referencia anterior](https://jassan-bit.github.io/MachineLearning_202630/referencia-diaria/) y las secciones señaladas como históricas
documentan el experimento de seis años con entradas diarias; no describen la configuración activa.

Descargas: [notebook ejecutado](notebooks/Entregable_1_Minuto.ipynb) y
[entrega reproducible con datos de un minuto](delivery/Entregable1_minuto_2023_2025.zip).


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
