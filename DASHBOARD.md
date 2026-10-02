> **Entregable vigente:** las métricas optimizadas están en http://localhost:8051, servidas desde `results/optimized_minute_2023_2025`. El dashboard/API descritos a continuación corresponden al modelo original y permanecen como referencia histórica.

# Dashboard y API del estudio 2023–2025

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


## Dashboard comparativo de siete modelos

### Publicar el comparativo en Render

`render.yaml` apunta a `volatility_dashboard.app:server`. Instala el runtime
con `requirements-render.txt` y sirve la aplicación con Gunicorn en `$PORT`.

1. Publicar los archivos actualizados en el repositorio conectado a Render.
2. En Render, crear un **Web Service** conectado a ese repositorio y seleccionar
   **Python 3**. Dejar **Root Directory** vacío si estos archivos están en la raíz.
3. Usar **Build Command**: `pip install -r requirements-render.txt`.
4. Usar **Start Command**: `gunicorn volatility_dashboard.app:server --bind 0.0.0.0:$PORT --workers 1 --threads 2 --timeout 180 --access-logfile - --error-logfile -`.
5. Configurar `PYTHON_VERSION=3.11.11` y `OMP_NUM_THREADS=1`,
   `OPENBLAS_NUM_THREADS=1`, `MKL_NUM_THREADS=1`. Health check: `/`.
6. Crear el servicio y revisar los logs. Cuando termine el despliegue, abrir
   la URL pública y comprobar Contexto, EDA y Comparación de modelos.

También se puede crear un Blueprint a partir de `render.yaml`.
No cambiar el servicio a sitio estático: los filtros requieren un servidor Python.

El repositorio remoto debe incluir `volatility_dashboard/`, `src/`,
`data/processed/minute_2023_2025/`, `results/minute_2023_2025/data_manifest.json`,
las siete carpetas de resultados enumeradas en `volatility_dashboard/data_loader.py`
(incluidos modelos, predicciones y verificaciones), y
`book/sections/10_dashboard_comparativo.md`. Mantener sus rutas relativas.
Si los artefactos usan Git LFS, verificar que se descarguen los archivos reales.
El dashboard audita y carga modelos guardados; no entrena al arrancar.
Comprobar memoria y tiempos con los datos completos antes de elegir el plan.
Una validación local no confirma que el servicio remoto esté desplegado.

Documentación: https://render.com/docs/web-services

Implementacion modular en volatility_dashboard/, con exactamente tres pestanas: contexto, EDA y comparacion. Reutiliza el dataset y los modelos existentes; no entrena ni modifica resultados.

Desde la raiz:

    .\.venv-repro\Scripts\python.exe -m volatility_dashboard.app

Abrir http://127.0.0.1:8050/. Si otro dashboard ocupa 8050, detener esa instancia antes de iniciar este comando. Las dependencias estan en requirements.txt, requirements-dashboard.txt y requirements-xgboost.txt.

El loader audita hashes, fechas, objetivos, calendario, medias del scaler y predicciones serializadas. Las inconsistencias se excluyen con su motivo. BTC, ETH, BNB y XRP estan disponibles; SOL no tiene datos comparables.

Informe detallado: book/sections/10_dashboard_comparativo.md. No hay volumen procesado, pruebas residuales alineadas ni tiempos separados completos. MAPE no se incluye automaticamente. Las importancias de k-NN/MLP son diagnosticos descriptivos por grupos de lags, sin tuning.

Validar:

    .\.venv-repro\Scripts\python.exe -m unittest discover -s tests -p test_comparative_dashboard.py
