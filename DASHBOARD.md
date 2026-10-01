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
