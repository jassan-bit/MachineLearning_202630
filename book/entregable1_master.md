# Volatilidad con SVR lineal: Binance 2023–2025

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
