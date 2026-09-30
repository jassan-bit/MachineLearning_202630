# Proyecto de Investigación

## Pronóstico de la Volatilidad Realizada a 24 Horas de BTC, ETH, BNB, XRP y SOL mediante SVR Lineal con Datos Horarios de Binance Spot (2020–2026)

**Materia:** Machine Learning

**Estudiantes:** Jassan Arteta y Mateo Bernal

**Profesor:** Lihki Rubio

**Universidad:** Universidad del Norte

**Programa:** Maestría en Matemáticas

**Periodo académico:** 2026-30

## Resumen

Este proyecto estudia el pronóstico de la volatilidad futura de BTCUSDT, ETHUSDT, BNBUSDT, XRPUSDT y SOLUSDT a partir de datos horarios de Binance Spot. La variable objetivo es la desviación estándar de los próximos 24 retornos logarítmicos horarios, centrados en su media, con divisor 24 (`ddof=0`) y expresada en porcentaje. No se anualiza ni se interpreta como la volatilidad acumulada del retorno de un día.

Se comparan Persistence y SVR lineal mediante cinco folds cronológicos de ventana creciente dentro de DEVELOPMENT. El SVR recibe 168 cierres horarios consecutivos por activo. Cada ventana de entradas, referencia y etiqueta permanece contenida en su bloque, y el escalado se ajusta únicamente con entrenamiento. TEST permanece reservado para una evaluación independiente posterior.

El RMSE medio entre los 25 bloques activo-fold es **0,389006 puntos porcentuales** para Persistence y **0,623259** para SVR. El SVR no supera la referencia temporal en las condiciones evaluadas. La diferencia de RMSE es **0,234253 puntos porcentuales**, con intervalo bootstrap por bloques del 95 % **[0,204497; 0,269379]**. Esta incertidumbre está condicionada a la configuración seleccionada y no elimina el posible optimismo de la selección en los mismos folds.

## Organización del informe

1. [Base de datos](sections/01_base_datos.md): problema, selección de la fuente, diccionario, estructura, calidad y representatividad.
2. [Análisis exploratorio](sections/02_eda.md): distribuciones, relaciones entre variables, estructura multivariada, dependencia temporal, disponibilidad de información y preprocesamiento.
3. [Modelo base](sections/03_modelo_base.md): formulación del objetivo, partición cronológica, validación con separación temporal, Persistence, SVR, métricas, residuos, incertidumbre y curva de aprendizaje.
4. [Conclusiones](sections/05_conclusiones.md): resultados, limitaciones, reproducibilidad y continuidad del proyecto.

Los resultados corresponden a validación interna en DEVELOPMENT. Las conclusiones se limitan a los cinco activos, el proveedor y los periodos estudiados.
