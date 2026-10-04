# 3. Modelo base y SVR lineal

Los apartados 3.1–3.8 siguen los requisitos de la guía para el modelo base.
El problema es de regresión temporal; el modelo entrenado es un SVR lineal
y su referencia mínima es persistencia. Los requisitos sin evidencia del
ajuste vigente se indican como pendientes.

## 3.1 Definición de la variable objetivo

Se pronostica la volatilidad no anualizada de los retornos logarítmicos
diarios de BTC, ETH, BNB y XRP, en puntos porcentuales. Para cada ventana
$w\in\{7,14,21,28\}$ días, la salida en el origen $t$ es
$(\sigma_{t+1}^{(w)},\ldots,\sigma_{t+7}^{(w)})$: siete horizontes diarios.
Cada ventana define un objetivo distinto.

La fórmula y el uso de `rolling(w).std(ddof=0)` se documentan en
[2.1.1 Retornos, volatilidad y salidas](12_eda.md).
Los datos de un minuto aportan características; el objetivo se calcula
con retornos diarios.

## 3.2 Línea base trivial: persistencia

Para cada origen, activo y ventana, la referencia repite la volatilidad
actual en los siete horizontes: $\hat y_{t,h}=\sigma_t^{(w)}$.
No requiere entrenamiento y utiliza únicamente información conocida.
El único modelo entrenado de esta entrega es el SVR lineal.

## 3.3 División temporal y validación con tsxv

Se usa `timeseries-cv==0.1.5`, desarrollado con la coautoría de Filipe Roberto Ramos. Se ejecutan las funciones nativas `split_train_val_forwardChaining`, `split_train_val_kFold` y `split_train_val_groupKFold` sobre índices diarios del calendario de 2023–2024. Los índices se vinculan a características de datos por minuto y objetivos diarios; no se interpretan siete minutos como siete días.

La auditoría conserva los cortes nativos:

| method | cortes | cortes_con_futuro |
| --- | --- | --- |
| forwardChaining | 668 | 0 |
| groupKFold | 5 | 5 |
| kFold | 668 | 640 |

K-Fold y Group K-Fold contienen entrenamiento posterior a algunas fechas de validación. Se documentan como diagnóstico y se excluyen de la selección del pronóstico. No se convierten mediante agrupación en una copia de Forward Chaining ni se presentan como tres evaluaciones temporales equivalentes.

El método principal conserva seis cortes de entrenamiento nativos de Forward Chaining, con inicio de validación en enero, marzo, mayo, julio, septiembre y noviembre de 2024. Cada validación se extiende a dos meses; este adaptador se declara explícitamente. El entrenamiento crece y sus etiquetas terminan antes del inicio de cada validación. Se mantienen las restricciones conservadoras de separación de la biblioteca. Los últimos siete días de 2024 no se usan como orígenes de validación porque sus objetivos entrarían en 2025.

| fold | split | n | first | last |
| --- | --- | --- | --- | --- |
| 1 | train | 274 | 2023-01-29 00:00:00+00:00 | 2023-12-04 00:00:00+00:00 |
| 1 | validation | 60 | 2024-01-01 00:00:00+00:00 | 2024-02-29 00:00:00+00:00 |
| 2 | train | 334 | 2023-01-29 00:00:00+00:00 | 2024-02-02 00:00:00+00:00 |
| 2 | validation | 61 | 2024-03-01 00:00:00+00:00 | 2024-04-30 00:00:00+00:00 |
| 3 | train | 395 | 2023-01-29 00:00:00+00:00 | 2024-04-03 00:00:00+00:00 |
| 3 | validation | 61 | 2024-05-01 00:00:00+00:00 | 2024-06-30 00:00:00+00:00 |
| 4 | train | 456 | 2023-01-29 00:00:00+00:00 | 2024-06-03 00:00:00+00:00 |
| 4 | validation | 62 | 2024-07-01 00:00:00+00:00 | 2024-08-31 00:00:00+00:00 |
| 5 | train | 518 | 2023-01-29 00:00:00+00:00 | 2024-08-04 00:00:00+00:00 |
| 5 | validation | 61 | 2024-09-01 00:00:00+00:00 | 2024-10-31 00:00:00+00:00 |
| 6 | train | 579 | 2023-01-29 00:00:00+00:00 | 2024-10-04 00:00:00+00:00 |
| 6 | validation | 54 | 2024-11-01 00:00:00+00:00 | 2024-12-24 00:00:00+00:00 |

Se seleccionan C, epsilon y entrada por el RMSE conjunto de las predicciones fuera de muestra de 2024, promediando los siete RMSE por horizonte. Se fijan las 16 configuraciones antes de evaluar 2025. El ajuste final usa todas las muestras elegibles cuyas etiquetas terminan antes de 2025 y permanece fijo durante la prueba. La prueba contiene **358 orígenes diarios** y **40.096 valores pronosticados**.

La partición determina qué muestras pueden intervenir en cada ajuste del
preprocesamiento. **2025 ya se había explorado** en experimentos anteriores;
por ello, esta prueba es retrospectiva y no acredita una reserva inicial
intacta del conjunto de prueba.

## 3.4 Entrenamiento mediante Pipeline

`src/optimize_minute_svr.py` construye cada ajuste con
`make_pipeline(StandardScaler(), estimator(C, epsilon))`.
El estimador incluye `TransformedTargetRegressor`, un escalador de objetivos
y `MultiOutputRegressor` con siete `LinearSVR`. Los escaladores se ajustan
solo con el entrenamiento de cada corte; la selección utiliza la validación
temporal del apartado 3.3.

### 3.4.1 Preprocesamiento y características

Se prueban ventanas de entrada de **7, 14, 21 y 28 días**. Para cada día de la ventana se incluyen seis características: retorno diario, su cuadrado, raíz de la suma de cuadrados de todos los retornos por minuto, suma absoluta de retornos por minuto dividida por la raíz de 1.440, semivolatilidad negativa y máximo retorno absoluto por minuto. Se normalizan por la volatilidad actual conocida, o su cuadrado según la unidad.

Se añaden siete características que describen el efecto de la salida de retornos antiguos de la ventana objetivo. Para horizonte $h$, se conservan los $w-h$ retornos diarios más recientes y se supone varianza futura igual a la actual para construir una referencia. Esta operación usa únicamente retornos ya observados; no utiliza el valor futuro del objetivo.

En unidades porcentuales, con $b_t=\max(\sigma_t^{(w)},10^{-8})$, suma $S$ y suma de cuadrados $Q$ de esos retornos retenidos, la característica es

$$
\frac{\sqrt{\max((Q+h b_t^2)/w-(S/w)^2,0)}}{b_t}-1.
$$

La dimensión es **6L+7**: 49, 91, 133 o 175 características según la ventana. Este cambio sustituye las 10.080–40.320 columnas de precios del experimento original por características derivadas de los datos de un minuto. La volatilidad actual y las características de salida de la ventana requieren además la historia de la definición objetivo, de hasta 28 retornos diarios. El calendario común exige suficiente historia para la ventana máxima y siete objetivos completos.


### 3.4.2 Modelo y justificación

Se mantienen siete regresores **LinearSVR**, uno por horizonte, mediante `MultiOutputRegressor`. Cada uno aprende la corrección relativa $z_{t,h}=\sigma_{t+h}/b_t-1$; la predicción final es $\max(b_t(1+\hat z_{t,h}),0)$. El recorte a cero forma parte del procedimiento evaluado. La persistencia repite la volatilidad actual en las siete salidas.

Esta representación reduce la dependencia del nivel nominal del precio y permite al SVR aprender cuándo corregir persistencia. Se ajustan los escaladores de entradas y objetivos exclusivamente dentro de cada entrenamiento. El modelo es lineal respecto a las características transformadas; el procesamiento completo incorpora transformaciones no lineales.

Se usa pérdida `squared_epsilon_insensitive`, `dual=False`, tolerancia 1e-6, máximo 50.000 iteraciones y semilla 42. La búsqueda evalúa C en {0,0001; 0,001; 0,01; 0,1; 1} y epsilon en {0,01; 0,1}, además de las cuatro entradas: **640 candidatos**, cada uno evaluado en seis cortes. Las advertencias de falta de convergencia interrumpen el ajuste.


### 3.4.3 Selección y contraste en validación

| symbol | volatility_window | input_window | C | epsilon | validation_rmse |
| --- | --- | --- | --- | --- | --- |
| BTCUSDT | 7 | 14 | 0.001 | 0.01 | 0.72018 |
| BTCUSDT | 14 | 14 | 0.001 | 0.01 | 0.38693 |
| BTCUSDT | 21 | 21 | 0.001 | 0.01 | 0.27354 |
| BTCUSDT | 28 | 7 | 0.1 | 0.01 | 0.20782 |
| ETHUSDT | 7 | 21 | 0.001 | 0.01 | 0.92546 |
| ETHUSDT | 14 | 28 | 0.001 | 0.01 | 0.57944 |
| ETHUSDT | 21 | 7 | 0.01 | 0.01 | 0.39737 |
| ETHUSDT | 28 | 7 | 0.01 | 0.01 | 0.31532 |
| BNBUSDT | 7 | 21 | 0.001 | 0.01 | 0.82457 |
| BNBUSDT | 14 | 7 | 0.01 | 0.01 | 0.48086 |
| BNBUSDT | 21 | 21 | 0.001 | 0.01 | 0.32762 |
| BNBUSDT | 28 | 7 | 0.001 | 0.01 | 0.24959 |
| XRPUSDT | 7 | 14 | 0.001 | 0.01 | 1.35439 |
| XRPUSDT | 14 | 7 | 0.001 | 0.01 | 0.84652 |
| XRPUSDT | 21 | 7 | 0.0001 | 0.01 | 0.63939 |
| XRPUSDT | 28 | 7 | 0.001 | 0.01 | 0.51241 |

| symbol | volatility_window | svr_rmse | persistence_rmse |
| --- | --- | --- | --- |
| BTCUSDT | 7 | 0.72018 | 0.92859 |
| BTCUSDT | 14 | 0.38693 | 0.47452 |
| BTCUSDT | 21 | 0.27354 | 0.35376 |
| BTCUSDT | 28 | 0.20782 | 0.28442 |
| ETHUSDT | 7 | 0.92546 | 1.2107 |
| ETHUSDT | 14 | 0.57944 | 0.74367 |
| ETHUSDT | 21 | 0.39737 | 0.50431 |
| ETHUSDT | 28 | 0.31532 | 0.41457 |
| BNBUSDT | 7 | 0.82457 | 1.00883 |
| BNBUSDT | 14 | 0.48086 | 0.64359 |
| BNBUSDT | 21 | 0.32762 | 0.44948 |
| BNBUSDT | 28 | 0.24959 | 0.32945 |
| XRPUSDT | 7 | 1.35439 | 1.5476 |
| XRPUSDT | 14 | 0.84652 | 0.98402 |
| XRPUSDT | 21 | 0.63939 | 0.7044 |
| XRPUSDT | 28 | 0.51241 | 0.56346 |

Las entradas elegidas cambian con el activo y la definición de volatilidad; una ventana más larga no mejora necesariamente el pronóstico. Los C seleccionados favorecen regularización fuerte, coherente con limitar el sobreajuste.

## 3.5 Evaluación y comparación con la línea base

Las tablas y su interpretación se presentan en
[4. Evaluación, interpretación y limitaciones](14_evaluacion.md).
Se reportan R², RMSE, MAE, MSE y MAPE sobre las mismas fechas y objetivos
para persistencia y SVR lineal. El SVR obtiene R² macro **0,70073** frente
a **0,53773** de persistencia y RMSE **0,59808** frente a **0,74770**.
La comparación es retrospectiva y las métricas macro promedian
configuraciones y horizontes.

**Pendiente:** intervalos de confianza para las métricas principales mediante
un procedimiento que respete la dependencia temporal, como bootstrap por
bloques. Las métricas de clasificación de la guía no aplican al objetivo
continuo de este estudio.

## 3.6 Diagnóstico de residuos según la estructura de los datos

**Pendiente para el SVR vigente:** análisis de normalidad, heterocedasticidad
y autocorrelación temporal de los residuos mediante ACF. La dependencia
residual puede señalar información temporal que el modelo no captura.
Los diagnósticos del modelo histórico no sustituyen los de este ajuste.

El diagnóstico espacial mediante I de Moran **no aplica**: no hay coordenadas
ni unidades geográficas en el dataset, como se explica en el apartado 2.7.

## 3.7 Curva de aprendizaje

**Pendiente:** comparar los errores de entrenamiento y validación con tamaños
crecientes de muestra, manteniendo el orden temporal y el procedimiento
fijado. La selección de ventanas e hiperparámetros del apartado 3.4.3 no
sustituye una curva de aprendizaje ni demuestra ausencia de sobreajuste.

## 3.8 Interpretación de coeficientes y métricas

La [sección 4](14_evaluacion.md) interpreta las métricas, sus unidades,
la agregación entre activos, ventanas y horizontes y las mejoras frente
a persistencia.

**Pendiente:** análisis de coeficientes por característica y horizonte,
considerando el escalado de entradas y objetivos. El SVR es lineal respecto
a las características transformadas; sus coeficientes no se interpretan
directamente como efectos sobre los precios originales ni como efectos
causales.

## 3.9 Nota crítica ante un desempeño alto

En BNB con ventana de 28 días el SVR alcanza R² **0,94272**, mientras
persistencia ya obtiene **0,88655**. Este resultado debe revisarse junto
con los controles y limitaciones de la guía:

- Verificar fuga de datos y disponibilidad de los predictores, conforme
  a los apartados 2.5 y 3.3–3.4.
- Comparar siempre con persistencia sobre las mismas fechas y objetivos.
- Respetar la estructura temporal en la validación. Las ventanas móviles
  comparten retornos y pueden favorecer un R² alto en ambos métodos.
- Revisar la dificultad del objetivo: aumentar la ventana cambia y suaviza
  la variable pronosticada; un R² mayor no demuestra por sí solo una mejor
  capacidad predictiva para otro objetivo.
- Documentar los diagnósticos pendientes y confirmar el procedimiento en
  un periodo nunca explorado. El conocimiento previo de 2025 impide
  presentarlo como una prueba independiente de generalización.

