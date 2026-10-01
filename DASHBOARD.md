# 6. Dashboard, API y reproducción

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

El [notebook completo](notebooks/Entregable_1_Completo.ipynb) incluye datos diarios y código en un paquete
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

- [Entrega reproducible](delivery/Entregable1_diario.zip).
- [Configuración](experiment.json).
- [Datos diarios](data/processed/daily_2020_2025.csv).
- [Protocolo y registro de ejecución](results/run_status.json).
- [Dependencias](requirements.txt).
