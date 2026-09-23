# Pronóstico de Volatilidad en Mercados de Criptomonedas mediante MLP Multisalida

**Autor:** Jassan Alberto Arteta Chavarro

Este libro presenta un proyecto de pronóstico de volatilidad con datos reales de **Binance Spot**, a frecuencia de **una hora (1h)**, para **BTCUSDT, ETHUSDT, BNBUSDT, XRPUSDT y SOLUSDT**. Cada activo se procesa por separado, respetando el orden temporal y los segmentos continuos de negociación.

A partir del precio de cierre se calcula el retorno logarítmico (`log_return`). La variable objetivo, `volatility`, es la desviación estándar móvil de **30 retornos horarios**, sin anualizar. Los cálculos se reinician ante discontinuidades para evitar retornos y ventanas que crucen huecos temporales.

El modelo usa exclusivamente historia de `volatility`, con ventanas de entrada de **7, 14, 21 y 28 observaciones**, para pronosticar simultáneamente las siguientes **7 horas**. Se emplea un MLP multisalida con capas ocultas `(64, 32)`, dos escaladores `StandardScaler` ajustados únicamente con train y **cinco folds cronológicos expansivos** por configuración. Las cinco criptomonedas y cuatro ventanas dan lugar a 100 modelos ya entrenados.

La evaluación utiliza **MAE, MSE, RMSE y MAPE** en la escala original, por horizonte y conjunto, y el **test BDS** sobre los residuos de TEST del horizonte h=1. El BDS es un diagnóstico de residuos, no una medida de precisión ni una prueba de calidad global del modelo; sus p-values se interpretan por fold.

El libro organiza la base de datos, el EDA, el modelado y las conclusiones a partir de las celdas y salidas guardadas de `model_evaluation.ipynb`, el notebook maestro final. Los resultados, figuras y particiones existentes se conservan. La elección de ventanas por RMSE TEST es retrospectiva y sus limitaciones se explican en la evaluación y las conclusiones.

## Contenido

1. [Base de Datos](database.md)
2. [Análisis Exploratorio de Datos (EDA)](eda.ipynb)
3. [Pronóstico de Volatilidad con MLP Multisalida](model_base.ipynb)
4. [Conclusiones, Limitaciones y Reproducibilidad](conclusions.ipynb)

## Validación temporal y publicación

El diagnóstico de `timeseries-cv` (`tsxv.splitTrainValTest`) mostró particiones intercaladas y ventanas que podían atravesar discontinuidades. La adaptación documentada construye cada ventana dentro de un mismo `segment_id` y distribuye las muestras en siete bloques consecutivos para formar cinco folds expansivos. La purga de fronteras garantiza `train_end < validation_start` y `validation_end < test_start`, considerando toda la ventana de entrada y salida.

Los capítulos conservan las salidas guardadas del maestro: celdas 1–14 en EDA, 15–25 en modelado y 26 en conclusiones. Las celdas de código llevan `skip-execution`; la publicación del libro utiliza esas salidas existentes. Los capítulos son una presentación del flujo completo documentado en el maestro y comparten sus dependencias de datos y variables.
