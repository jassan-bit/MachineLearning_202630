# Entregable 1: volatilidad con SVR lineal

**Jassan Arteta y Mateo Bernal**

Estudio Binance de BTC, ETH, BNB y XRP durante **2023–2025**, con datos de **un minuto**, características derivadas en ventanas de **7, 14, 21 y 28 días** y siete salidas diarias de volatilidad. La variante vigente usa correcciones relativas a persistencia, seis cortes temporales crecientes para selección en 2024 y ajuste final con etiquetas anteriores a 2025.

En 358 fechas de prueba de 2025: **R² macro 0,7007**, RMSE **0,5981** y mejora de RMSE frente a persistencia en **16/16 configuraciones**. La evaluación es retrospectiva; 2025 ya había sido examinado.

Consultar el [informe completo](sections/07_estudio_minuto.md), el [notebook ejecutado](../notebooks/Entregable_1_Minuto.ipynb) y la [entrega reproducible](../delivery/Entregable1_optimizado_2023_2025.zip).

El [modelo original con precios por minuto](sections/09_referencia_minuto_original.md) y el [estudio diario anterior](sections/08_referencia_diaria.md) quedan como referencias históricas. Los resultados vigentes están en `results/optimized_minute_2023_2025`.
