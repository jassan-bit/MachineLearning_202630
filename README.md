# Pronóstico de volatilidad diaria con SVR lineal

**Entregable 1 — Jassan Arteta y Mateo Bernal**  
Universidad del Norte · Maestría en Matemáticas · Machine Learning · Profesor Lihki Rubio · 2026-30

## Resumen

Se estudian BTCUSDT, ETHUSDT, BNBUSDT, XRPUSDT entre el 1 de enero de 2020 y el 31 de diciembre de 2025. Se descargaron
12,616,618 velas de un minuto de los archivos oficiales de Binance, con verificación SHA-256,
y se obtuvieron cierres diarios UTC. El objetivo es pronosticar siete valores futuros de volatilidad móvil.
Se cruzan cuatro ventanas de precios (7, 14, 21 y 28 días) con cuatro ventanas de volatilidad de igual conjunto:
64 configuraciones activo–entrada–objetivo, cinco folds por configuración y siete SVR lineales por modelo multisalida.

La función Group K-Fold de `timeseries-cv` se ejecuta sobre índices temporales comunes y se complementa con filtros
cronológicos. Entrenamiento: 2020–2022; validación: 2023; test: 2024–2025. El escalado y el ajuste usan solo TRAIN;
los hiperparámetros y la ventana de entrada se seleccionan con validación. Se evalúan precios diarios históricos
como entradas, pero **todas las salidas del modelo son volatilidades**, no precios.

Entre las 16 selecciones por activo y definición del objetivo, SVR supera a Persistence en RMSE test en 0 casos.
Este conteo es descriptivo: los folds se solapan y el muestreo de `tsxv` deja pocas fechas de evaluación.
No se comparan errores absolutos entre objetivos de volatilidad distintos para declarar un ganador global.

## Organización del informe

1. [Datos y problema](book/sections/01_base_datos.md).
2. [EDA y preprocesamiento](book/sections/02_eda.md).
3. [Validación y SVR lineal](book/sections/03_modelo_base.md).
4. [Resultados detallados y residuos](book/sections/04_resultados_diarios.md).
5. [Conclusiones](book/sections/05_conclusiones.md).
6. [Dashboard, API y reproducción](book/sections/06_dashboard.md).

La versión horaria anterior está archivada en `delivery/legacy_hourly_before_daily_restructure.zip`.
Sus métricas y conclusiones no pertenecen a este experimento.

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
