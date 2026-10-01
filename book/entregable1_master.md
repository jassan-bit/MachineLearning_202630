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

1. [Datos y problema](sections/01_base_datos.md).
2. [EDA y preprocesamiento](sections/02_eda.md).
3. [Validación y SVR lineal](sections/03_modelo_base.md).
4. [Resultados detallados y residuos](sections/04_resultados_diarios.md).
5. [Conclusiones](sections/05_conclusiones.md).
6. [Dashboard, API y reproducción](sections/06_dashboard.md).

La versión horaria anterior está archivada en `delivery/legacy_hourly_before_daily_restructure.zip`.
Sus métricas y conclusiones no pertenecen a este experimento.
