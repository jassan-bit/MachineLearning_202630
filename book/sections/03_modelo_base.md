# 3. Modelo base

## 3.1 Objetivo y alcance de la evaluación

El modelo base es **SVR lineal** y la referencia temporal es **persistencia**. El objetivo continuo es la volatilidad realizada futura de 24 retornos horarios, calculada como su desviación estándar centrada con divisor 24 (`ddof=0`). Se expresa en porcentaje y no se anualiza. No se utiliza regresión logística porque la tarea es de regresión, ni se confunden estos resultados con la regresión OLS diagnóstica de 2.5.

Para el ancla s, que identifica la apertura de la última vela observada, la predicción se realiza después de su cierre. Con retorno horario $r_{i,s}=\ln(C_{i,s}/C_{i,s-1})$:

$$
y_{i,s}=100\sqrt{\frac{1}{24}\sum_{j=1}^{24}(r_{i,s+j}-\bar r^+_{i,s})^2},\qquad
\bar r^+_{i,s}=\frac{1}{24}\sum_{j=1}^{24}r_{i,s+j}.
$$

Los resultados de esta sección proceden exclusivamente de validación cronológica en DEVELOPMENT. **TEST permanece reservado** hasta fijar los procedimientos de los modelos que se compararán al final. No se presenta esta validación como una evaluación final independiente. La sección 2.1 utiliza la misma definición del objetivo; cada experimento conserva sus propias reglas de cobertura e historial disponible.

## 3.2 Entrada común y referencia de persistencia

Se fija antes del entrenamiento una ventana de **168 cierres horarios** (7 días de observaciones), con entradas desde `close(s)` hasta `close(s−167)`. El horizonte sigue siendo 24 horas y la ventana de volatilidad contiene 24 retornos; son tres parámetros distintos. Esta es la configuración de este experimento, no una conclusión de que 168 sea la mejor ventana. Los futuros modelos deberán usar esta misma información para compararse en este escenario.

`close` es la única variable explicativa original. Sus rezagos forman columnas diferentes y pueden presentar multicolinealidad. Se ajusta un modelo por activo; no se incorporan volumen, calendario o identificadores como entradas. Se exige una ventana completa de cierres válidos, un objetivo válido y disponibilidad de la referencia. Los cinco activos y ambos métodos se evalúan sobre timestamps comunes; no se filtran por errores obtenidos.

Persistencia utiliza la volatilidad reciente ya observable:

$$
\widehat y^{\mathrm{persistencia}}_{i,s}=100\,\operatorname{std}(r_{i,s-23},\ldots,r_{i,s};\mathrm{ddof}=0).
$$

Esos 24 retornos requieren 25 cierres que están contenidos en la ventana disponible al SVR. La referencia utiliza menos historia por su regla propia; no recibe información futura ni información de origen adicional. No necesita ajuste de parámetros. No es la media del entrenamiento utilizada como referencia en 2.5.

## 3.3 Partición y prevención de fuga

**Separación entre entrenamiento, validación y TEST.** DEVELOPMENT contiene el 80 % inicial de los timestamps observados y TEST el 20 % final. Los cinco folds se construyen exclusivamente dentro de DEVELOPMENT. **TEST no se lee ni se utiliza para entrenar, escalar, seleccionar hiperparámetros o calcular estas métricas.**

| Conjunto | Función |
|---|---|
| Entrenamiento de cada fold | Ajustar escalador y SVR con historias y etiquetas contenidas en su bloque. |
| Validación de cada fold | Comparar configuraciones sobre ventanas completas contenidas en su bloque, sin reajustar el escalador. |
| TEST final | Evaluar procedimientos fijados; no se reportan resultados de este conjunto. |

**RMSE, MAPE y R² son de validación en DEVELOPMENT.** Los mismos folds se utilizan para elegir hiperparámetros, por lo que estas métricas pueden ser optimistas y no constituyen una evaluación final independiente. En el esquema creciente, una validación anterior puede incorporarse al entrenamiento de un fold posterior cuando su información ya está disponible. Los folds no son muestras independientes.

La partición DEVELOPMENT/TEST existente permanece intacta. Dentro de DEVELOPMENT se utiliza [TimeSeriesSplit](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.TimeSeriesSplit.html) con cinco folds de entrenamiento creciente. Se divide la rejilla horaria completa, no las filas apiladas de los activos: cada frontera es común a los cinco. Los huecos permanecen en el calendario y luego se aplica la elegibilidad de ventanas.

Se adopta **confinamiento estricto por bloque**, fijado en el protocolo antes de repetir la búsqueda. Para un bloque con primeras y últimas aperturas A y B, una ancla s solo es elegible si:

$$
s-167\text{ horas}\ge A,\qquad s+24\text{ horas}\le B.
$$

La primera condición contiene los 168 cierres de entrada y los 25 cierres de Persistence; la segunda contiene los 24 retornos futuros. Además, todos los cierres deben ser consecutivos y convencionales. Se aplican las mismas reglas en entrenamiento, validación y en cada prefijo de la curva de aprendizaje. El ajuste final utiliza ventanas contenidas en DEVELOPMENT. La evaluación futura de TEST deberá formar su historial dentro de TEST y excluir sus últimas 24 anclas, sin utilizar contexto de DEVELOPMENT.

La regla elimina las primeras 167 anclas potenciales de cada validación y las últimas 24. Los conteos adicionales sobre filas ya elegibles pueden ser menores por huecos o por exclusiones que ya existían en el extremo de DEVELOPMENT. La auditoría distingue ambos motivos. Los timestamps excluidos se fijan por disponibilidad y fronteras, nunca por errores de los modelos.

El entrenamiento termina antes del inicio del bloque validado: sus últimas 24 anclas se purgan para que ninguna etiqueta atraviese la frontera. La separación se comprueba en horas reales; no se interpreta como independencia estadística. Las fechas de los bloques son aperturas de velas: la predicción se emite alrededor de s+1 hora y la etiqueta se conoce alrededor de s+25 horas, tras los cierres respectivos.

El uso de contexto histórico anterior a validación puede ser causal en otros protocolos. Aquí se excluye para cumplir la condición estricta del experimento. Esta restricción reduce cobertura y no demuestra por sí sola ausencia universal de fuga.

| Fold | Inicio bloque validación UTC | Fin bloque validación UTC | Primera ancla evaluada | Última ancla evaluada | n train | n val |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 2021-06-04 21:00 | 2022-03-29 10:00 | 2021-06-11 20:00 | 2022-03-28 10:00 | 5781 | 6370 |
| 2 | 2022-03-29 11:00 | 2023-01-21 00:00 | 2022-04-05 10:00 | 2023-01-20 00:00 | 12342 | 6951 |
| 3 | 2023-01-21 01:00 | 2023-11-14 14:00 | 2023-01-28 00:00 | 2023-11-13 14:00 | 19484 | 6758 |
| 4 | 2023-11-14 15:00 | 2024-09-07 04:00 | 2023-11-21 14:00 | 2024-09-06 04:00 | 26433 | 6951 |
| 5 | 2024-09-07 05:00 | 2025-07-01 18:00 | 2024-09-14 04:00 | 2025-06-30 18:00 | 33575 | 6951 |

La tabla distingue el bloque de calendario de las anclas efectivamente evaluadas. Los tamaños son comunes a los cinco activos por la intersección de elegibilidad. El archivo `base_folds.csv` registra también el inicio del historial y la última vela objetivo de ambos conjuntos.

| Fold | Última ancla train | Última vela objetivo train | Inicio historial validación | Última vela objetivo validación | Excluidas por historia | Excluidas por horizonte |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 2021-06-03 20:00 | 2021-06-04 20:00 | 2021-06-04 21:00 | 2022-03-29 10:00 | 167 | 24 |
| 2 | 2022-03-28 10:00 | 2022-03-29 10:00 | 2022-03-29 11:00 | 2023-01-21 00:00 | 167 | 24 |
| 3 | 2023-01-20 00:00 | 2023-01-21 00:00 | 2023-01-21 01:00 | 2023-11-14 14:00 | 167 | 24 |
| 4 | 2023-11-13 14:00 | 2023-11-14 14:00 | 2023-11-14 15:00 | 2024-09-07 04:00 | 167 | 24 |
| 5 | 2024-09-06 04:00 | 2024-09-07 04:00 | 2024-09-07 05:00 | 2025-07-01 18:00 | 167 | 0 |

Sobre las filas previamente elegibles, se excluyen 835 anclas por historial y 96 por horizonte, por activo. Se evalúan 33,981 anclas por activo y 169,905 predicciones por método. La pérdida inicial de historial no es una imputación ni se rellena con TRAIN.

Las exclusiones no garantizan que los periodos conservados representen a los descartados. Estos resultados sustituyen las métricas previas calculadas con otras fronteras; no se comparan como si procedieran de las mismas observaciones.

## 3.4 Pipeline, formulación y búsqueda acotada

Se utiliza `Pipeline(StandardScaler, LinearSVR)`. Cada escalador se ajusta con su entrenamiento y se aplica sin reajuste a validación. [LinearSVR](https://scikit-learn.org/stable/modules/generated/sklearn.svm.LinearSVR.html) implementa regresión de vectores de soporte lineal; se elige por su adecuación computacional a muchas observaciones. La configuración usa pérdida **epsilon-insensible al cuadrado**, `dual=False`, tolerancia $10^{-5}$, máximo 20.000 iteraciones y semilla 42. El intercepto de esta implementación está regularizado. No es OLS ni se afirma identidad numérica con `SVR(kernel="linear")` y su pérdida epsilon-insensible no cuadrática habitual.

La función predicha es lineal en los cierres estandarizados. Se penaliza la magnitud de los coeficientes y los errores que exceden el tubo de ancho epsilon. No se transforma ni escala la etiqueta. C controla la penalización del error frente a la regularización; epsilon está expresado en puntos porcentuales de volatilidad.

Antes de entrenar se fija la rejilla C ∈ {0,01; 0,1; 1} y epsilon ∈ {0,01; 0,1}: **seis configuraciones**, cinco folds y cinco activos, 150 ajustes de búsqueda. El criterio de selección es el RMSE medio con igual peso por activo y fold. Se selecciona una configuración común, sin elegir una diferente para cada activo. El presupuesto usa ejecución secuencial en CPU y un límite de iteraciones por ajuste; se guardan duración e iteraciones. No se amplía adaptativamente la rejilla según los resultados. Persistence queda exenta de búsqueda.

| C | epsilon (pp) | RMSE medio |
| --- | --- | --- |
| 0.01 | 0.01 | 0.623259 |
| 0.1 | 0.01 | 0.627220 |
| 1.0 | 0.01 | 0.627994 |
| 0.01 | 0.1 | 0.640616 |
| 0.1 | 0.1 | 0.644733 |
| 1.0 | 0.1 | 0.645424 |

Se seleccionan **C=0.01 y epsilon=0.01**. La ejecución de los 170 ajustes y diagnósticos tomó 204.8 segundos en el entorno local; no es una comparación de coste entre algoritmos.

La curva de aprendizaje añade 15 ajustes y el ajuste final sobre DEVELOPMENT añade cinco: 170 en total. Se guarda un pipeline final por activo, todavía sin evaluación en TEST. Las ventanas futuras, otros algoritmos y presupuestos deberán identificarse como experimentos propios bajo el protocolo común.

## 3.5 Comparación de desempeño

RMSE y MAE están en puntos porcentuales de volatilidad; MAPE se expresa en porcentaje y R² es adimensional. MAPE no se informa como finito si hay etiquetas iguales a cero; no se usan denominadores artificiales. Un R² negativo indica peor error cuadrático que la referencia constante de la media observada en ese bloque, no que el error sea negativo.

| Modelo | RMSE media ± DE | MAPE media ± DE (%) | R² media ± DE | MAE media ± DE |
| --- | --- | --- | --- | --- |
| persistence | 0.3890 ± 0.1084 | 40.5560 ± 6.8341 | -0.0447 ± 0.2412 | 0.2611 ± 0.0628 |
| svr | 0.6233 ± 0.3720 | 109.4179 ± 41.1307 | -1.9054 ± 2.7367 | 0.5372 ± 0.3339 |

La media y desviación se calculan sobre 25 combinaciones activo-fold con pesos iguales. La desviación entre bloques no es un intervalo de confianza. Los resultados por activo se resumen a continuación como media de sus cinco folds; el detalle completo conserva cada fold.

| Activo | Modelo | RMSE | MAPE (%) | R² |
| --- | --- | --- | --- | --- |
| BNBUSDT | persistence | 0.3334 | 39.22 | -0.0066 |
| BNBUSDT | svr | 0.5508 | 123.48 | -1.9659 |
| BTCUSDT | persistence | 0.2764 | 46.37 | -0.1394 |
| BTCUSDT | svr | 0.3596 | 98.43 | -0.9698 |
| ETHUSDT | persistence | 0.3423 | 41.02 | -0.1267 |
| ETHUSDT | svr | 0.5284 | 103.35 | -2.3778 |
| SOLUSDT | persistence | 0.4816 | 34.33 | 0.1194 |
| SOLUSDT | svr | 0.9426 | 105.16 | -3.0102 |
| XRPUSDT | persistence | 0.5113 | 41.84 | -0.0704 |
| XRPUSDT | svr | 0.7348 | 116.67 | -1.2035 |

El RMSE medio es **0.3890** para Persistence y **0.6233** para SVR; la diferencia SVR menos Persistence es **0.2343 puntos porcentuales**. El SVR no supera la referencia en este experimento. Se registran **8 predicciones negativas** del SVR. Los resultados corresponden a las nuevas ventanas contenidas y no deben mezclarse con las métricas de la versión anterior.

No se recortan predicciones negativas después de observar su desempeño. Se contabilizan y se reconocen como valores incompatibles con la no negatividad de la volatilidad. Cualquier restricción posterior deberá definirse como parte de un procedimiento distinto antes de evaluarlo.

## 3.6 Incertidumbre

Se utiliza bootstrap circular de bloques de **168 horas**, con 499 réplicas y semilla 42, separado dentro de cada fold. Se remuestrean conjuntamente los mismos bloques para los activos y los dos métodos, manteniendo los huecos del calendario. Se calcula en cada réplica el RMSE por activo y fold y luego su media con pesos iguales. La diferencia se define como RMSE del SVR menos RMSE de persistencia; un valor negativo favorece SVR.

| Métrica | Modelo o diferencia | Estimación | Límite 2,5 % | Límite 97,5 % |
| --- | --- | --- | --- | --- |
| macro_rmse | svr | 0.6233 | 0.5946 | 0.6497 |
| macro_rmse | persistence | 0.3890 | 0.3533 | 0.4185 |
| delta_macro_rmse | svr_minus_persistence | 0.2343 | 0.2045 | 0.2694 |

**Sensibilidad a la longitud del bloque.** Se mantienen las predicciones, la semilla y las 499 réplicas; no se vuelve a seleccionar el modelo. Se contrastan 24 horas (horizonte del objetivo), 168 horas (una semana, análisis principal) y 336 horas (dos semanas).

| Bloque (h) | Modelo o diferencia | RMSE o diferencia (pp) | IC 95 %: inferior | IC 95 %: superior |
| --- | --- | --- | --- | --- |
| 24 | svr | 0.6233 | 0.6068 | 0.6380 |
| 24 | persistence | 0.3890 | 0.3683 | 0.4112 |
| 24 | svr_minus_persistence | 0.2343 | 0.2143 | 0.2526 |
| 168 | svr | 0.6233 | 0.5946 | 0.6497 |
| 168 | persistence | 0.3890 | 0.3533 | 0.4185 |
| 168 | svr_minus_persistence | 0.2343 | 0.2045 | 0.2694 |
| 336 | svr | 0.6233 | 0.5918 | 0.6535 |
| 336 | persistence | 0.3890 | 0.3501 | 0.4206 |
| 336 | svr_minus_persistence | 0.2343 | 0.1965 | 0.2742 |

En las tres longitudes, el intervalo de la diferencia queda por encima de cero: la desventaja del SVR es consistente en esta comprobación de sensibilidad. Estos tres escenarios no identifican una longitud óptima ni corrigen el sesgo de selección. La réplica de 168 horas reproduce los intervalos originales. [Resultados completos de sensibilidad](../../outputs/tables/base_bootstrap_sensitivity.csv).

Los intervalos percentiles del 95 % son exploratorios y condicionados a las predicciones y a la configuración seleccionada. No incluyen la incertidumbre del proceso de selección ni un reajuste del modelo en cada réplica; los folds comparten entrenamiento y pueden reflejar regímenes distintos. La longitud de bloque fijada tampoco prueba independencia entre bloques. Por ello no se interpreta este intervalo como evidencia final independiente de superioridad. Esa conclusión requiere TEST reservado y los procedimientos fijados de antemano.

## 3.7 Residuos y dependencia temporal

Se define el residuo como observado menos predicho. Se analizan **ambos modelos en los cinco folds y los cinco activos: 50 diagnósticos**. Los paneles muestran residuos en el tiempo, gráfico Q-Q normal, dispersión frente a la predicción y ACF residual hasta 168 horas. Se resta la media del residuo del bloque; para cada rezago k se suman los productos centrados de los pares disponibles separados por k horas y se divide por la suma de cuadrados centrados de todas las observaciones del bloque. El denominador es común a los rezagos. Se excluyen los pares con faltantes, sin comprimir el calendario, y se guarda el número de pares. Con huecos, la pérdida de pares afecta la estimación; no se dibujan bandas iid. Las líneas temporales se interrumpen en los huecos. No se concatenan folds para calcular dependencia.

| Activo | Modelo | Media residuo: rango entre folds | ACF 1h: rango | ACF 24h: rango | ACF 168h: rango |
| --- | --- | --- | --- | --- | --- |
| BNBUSDT | persistence | -0.0047 a 0.0012 | 0.9570 a 0.9712 | -0.5048 a -0.2455 | -0.0265 a 0.1793 |
| BNBUSDT | svr | -0.8337 a -0.1815 | 0.9809 a 0.9900 | 0.3451 a 0.6334 | 0.1515 a 0.3003 |
| BTCUSDT | persistence | -0.0030 a 0.0020 | 0.9554 a 0.9683 | -0.4238 a -0.2528 | 0.0407 a 0.3717 |
| BTCUSDT | svr | -0.2668 a -0.1344 | 0.9825 a 0.9889 | 0.3871 a 0.5809 | 0.2675 a 0.3046 |
| ETHUSDT | persistence | -0.0021 a 0.0025 | 0.9547 a 0.9650 | -0.4178 a -0.2559 | 0.0366 a 0.2468 |
| ETHUSDT | svr | -0.8271 a -0.0327 | 0.9802 a 0.9901 | 0.3219 a 0.6249 | 0.1617 a 0.4784 |
| SOLUSDT | persistence | -0.0012 a 0.0046 | 0.9583 a 0.9652 | -0.3313 a -0.1666 | 0.0073 a 0.2061 |
| SOLUSDT | svr | -1.6559 a -0.1491 | 0.9849 a 0.9957 | 0.4839 a 0.6808 | 0.1656 a 0.4444 |
| XRPUSDT | persistence | -0.0002 a 0.0085 | 0.9590 a 0.9680 | -0.3762 a -0.2445 | 0.0136 a 0.1530 |
| XRPUSDT | svr | -0.8027 a -0.2645 | 0.9797 a 0.9933 | 0.2140 a 0.6437 | 0.0364 a 0.3614 |

Los rangos resumen cinco folds, no son intervalos de confianza. Una media residual negativa indica sobreestimación media; una positiva, subestimación. El confinamiento de ventanas no elimina la dependencia inducida por objetivos solapados dentro de un mismo bloque.

La media residual es negativa en 36 de los 50 casos. La ACF a una hora está entre 0.9547 y 0.9957; esta persistencia impide tratar los errores horarios como observaciones independientes. No se atribuye toda la dependencia a una variable omitida.

| Activo | Modelo | Spearman magnitud–predicción: rango | Varianza segunda/primera mitad: rango | Asimetría: rango | Exceso de curtosis: rango |
| --- | --- | --- | --- | --- | --- |
| BNBUSDT | persistence | 0.3019 a 0.4157 | 0.3385 a 1.5258 | -0.0506 a 0.6696 | 4.6593 a 11.8201 |
| BNBUSDT | svr | 0.1441 a 0.5854 | 0.4160 a 1.0075 | 1.6496 a 3.0637 | 3.4407 a 15.1787 |
| BTCUSDT | persistence | 0.2172 a 0.3403 | 0.8007 a 1.1603 | -0.0283 a 0.2514 | 1.6892 a 4.5435 |
| BTCUSDT | svr | 0.0540 a 0.5117 | 0.6645 a 0.9618 | 1.0089 a 1.8194 | 1.5343 a 4.9472 |
| ETHUSDT | persistence | 0.2671 a 0.3282 | 0.7107 a 1.6012 | 0.1497 a 0.7756 | 2.9643 a 6.7092 |
| ETHUSDT | svr | 0.1042 a 0.7271 | 0.6662 a 1.3008 | 0.3888 a 2.1300 | 0.0308 a 8.3171 |
| SOLUSDT | persistence | 0.2784 a 0.3809 | 0.3319 a 1.5235 | 0.0562 a 0.7510 | 2.6209 a 8.4363 |
| SOLUSDT | svr | -0.0046 a 0.8761 | 0.5963 a 1.4108 | 0.4583 a 3.3604 | 1.4074 a 17.0453 |
| XRPUSDT | persistence | 0.2937 a 0.4744 | 0.6153 a 3.5264 | 0.1056 a 2.2496 | 4.8443 a 36.4702 |
| XRPUSDT | svr | -0.0893 a 0.6607 | 0.5264 a 3.1501 | 1.2572 a 5.9250 | 2.4842 a 53.9137 |

[Los 50 diagnósticos individuales](../../outputs/tables/base_residual_diagnostics_all.csv) y [correlaciones con número de pares por rezago](../../outputs/tables/base_residual_correlations_all.csv) permiten revisar la heterogeneidad sin agrupar residuos de folds diferentes. La división en mitades para el cociente de varianzas usa el punto medio del calendario de cada fold.

```{figure} ../_static/figures/residuals_all_BNBUSDT.png
:alt: Residuos de ambos modelos en los cinco folds para BNBUSDT.

BNBUSDT: cada fila corresponde a un fold; azul SVR y naranja Persistence.
```

```{figure} ../_static/figures/residuals_all_BTCUSDT.png
:alt: Residuos de ambos modelos en los cinco folds para BTCUSDT.

BTCUSDT: cada fila corresponde a un fold; azul SVR y naranja Persistence.
```

```{figure} ../_static/figures/residuals_all_ETHUSDT.png
:alt: Residuos de ambos modelos en los cinco folds para ETHUSDT.

ETHUSDT: cada fila corresponde a un fold; azul SVR y naranja Persistence.
```

```{figure} ../_static/figures/residuals_all_SOLUSDT.png
:alt: Residuos de ambos modelos en los cinco folds para SOLUSDT.

SOLUSDT: cada fila corresponde a un fold; azul SVR y naranja Persistence.
```

```{figure} ../_static/figures/residuals_all_XRPUSDT.png
:alt: Residuos de ambos modelos en los cinco folds para XRPUSDT.

XRPUSDT: cada fila corresponde a un fold; azul SVR y naranja Persistence.
```

La correlación de la magnitud del residuo con la predicción y el cociente entre varianzas de la segunda y primera mitad del bloque son diagnósticos descriptivos de dispersión variable; no son pruebas concluyentes de heterocedasticidad. El archivo de diagnóstico incluye Jarque–Bera, cuyos p-valores de referencia iid no se interpretan como inferencia calibrada bajo dependencia temporal. El gráfico Q-Q permite examinar desviaciones de normalidad sin confundir normalidad marginal con independencia.

La autocorrelación residual representa información temporal no capturada o dependencia inducida por las etiquetas solapadas. No demuestra por sí sola que esa información sea explotable por otro modelo. No corresponde aplicar Moran u otros diagnósticos espaciales porque no existen coordenadas.

## 3.8 Curvas de aprendizaje

Se ajusta la configuración seleccionada con el 25 %, 50 % y 100 % inicial del **calendario de entrenamiento** del último fold. Dentro de cada prefijo se exige que historial y objetivo estén completos y contenidos en él; el porcentaje no se aplica a filas ya filtradas. La validación permanece fija y el escalador se ajusta de nuevo en cada caso. Son prefijos cronológicos, no submuestras aleatorias. La auditoría de la curva registra inicio del historial, fin del objetivo y fronteras del prefijo.

| Activo | Fracción calendario train | n train | RMSE train | RMSE validación |
| --- | --- | --- | --- | --- |
| BNBUSDT | 0.25 | 7441 | 0.7349 | 1.2022 |
| BNBUSDT | 0.5 | 15912 | 0.6245 | 0.4996 |
| BNBUSDT | 1.0 | 33575 | 0.5377 | 0.3468 |
| BTCUSDT | 0.25 | 7441 | 0.4083 | 0.9371 |
| BTCUSDT | 0.5 | 15912 | 0.3671 | 0.4759 |
| BTCUSDT | 1.0 | 33575 | 0.3405 | 0.3379 |
| ETHUSDT | 0.25 | 7441 | 0.4935 | 0.7302 |
| ETHUSDT | 0.5 | 15912 | 0.4718 | 0.4108 |
| ETHUSDT | 1.0 | 33575 | 0.4423 | 0.3388 |
| SOLUSDT | 0.25 | 7441 | 0.9437 | 1.0403 |
| SOLUSDT | 0.5 | 15912 | 0.8086 | 0.5231 |
| SOLUSDT | 1.0 | 33575 | 0.7466 | 0.4623 |
| XRPUSDT | 0.25 | 7441 | 0.9164 | 1.8766 |
| XRPUSDT | 0.5 | 15912 | 0.7727 | 0.9567 |
| XRPUSDT | 1.0 | 33575 | 0.6495 | 1.0520 |

Evolución del error de validación al ampliar el prefijo: BNBUSDT: disminuye en ambos incrementos; BTCUSDT: disminuye en ambos incrementos; ETHUSDT: disminuye en ambos incrementos; SOLUSDT: disminuye en ambos incrementos; XRPUSDT: no disminuye de forma monótona. No se concluye que más historia resuelva uniformemente el problema.

La distancia entre ambos errores y su evolución orientan el diagnóstico de sobreajuste, insuficiencia de información o cambio de distribución. Al ampliar el prefijo también cambia el periodo representado y la cercanía al bloque validado; por tanto, la curva no aísla únicamente el efecto del tamaño muestral ni demuestra que la muestra sea suficiente. La configuración se eligió previamente con los cinco folds, así que esta curva es diagnóstica y no una evaluación adicional independiente.

```{figure} ../_static/figures/base_BNBUSDT.png
:alt: Predicciones, residuos y curva cronológica con ventanas contenidas para BNBUSDT.

BNBUSDT: validación del último fold y curva de aprendizaje con fronteras estrictas.
```

```{figure} ../_static/figures/base_BTCUSDT.png
:alt: Predicciones, residuos y curva cronológica con ventanas contenidas para BTCUSDT.

BTCUSDT: validación del último fold y curva de aprendizaje con fronteras estrictas.
```

```{figure} ../_static/figures/base_ETHUSDT.png
:alt: Predicciones, residuos y curva cronológica con ventanas contenidas para ETHUSDT.

ETHUSDT: validación del último fold y curva de aprendizaje con fronteras estrictas.
```

```{figure} ../_static/figures/base_SOLUSDT.png
:alt: Predicciones, residuos y curva cronológica con ventanas contenidas para SOLUSDT.

SOLUSDT: validación del último fold y curva de aprendizaje con fronteras estrictas.
```

```{figure} ../_static/figures/base_XRPUSDT.png
:alt: Predicciones, residuos y curva cronológica con ventanas contenidas para XRPUSDT.

XRPUSDT: validación del último fold y curva de aprendizaje con fronteras estrictas.
```

## 3.9 Interpretación de coeficientes

Los coeficientes se guardan para los pipelines finales ajustados sobre DEVELOPMENT. Un coeficiente estandarizado expresa el cambio predicho en puntos porcentuales de volatilidad ante una desviación estándar de aumento en ese rezago, manteniendo los demás fijos. Se guarda también el coeficiente por unidad de USDT al dividirlo por la escala de su columna.

| Activo | Rezago (h) | Coeficiente estandarizado | Coeficiente por USDT |
| --- | --- | --- | --- |
| BNBUSDT | 0 | -0.320641 | -0.00172800 |
| BTCUSDT | 0 | -0.243138 | -0.00000947 |
| ETHUSDT | 0 | -0.386527 | -0.00040421 |
| SOLUSDT | 0 | -0.204474 | -0.00289488 |
| XRPUSDT | 0 | 0.059053 | 0.09141999 |

La tabla muestra el rezago con mayor coeficiente absoluto estandarizado por activo, no una selección de variables. Dada la fuerte correlación entre cierres consecutivos, los signos y magnitudes individuales pueden ser inestables y mantener otros rezagos fijos puede describir combinaciones poco habituales. No se interpretan como efectos causales ni como importancias robustas. La regularización ayuda a controlar coeficientes, pero no elimina la redundancia de la entrada.

## 3.10 Auditoría crítica y limitaciones

El máximo R² del SVR entre las 25 evaluaciones es 0.0300. No alcanza la alerta de 0,8–0,9 de la guía; un desempeño bajo tampoco demuestra ausencia de fuga.

Las pruebas `tests/test_temporal_boundaries.py` verifican que recalcular un bloque aislado produzca las mismas entradas, etiquetas y persistencia que filtrarlo por fechas; que cambiar cierres futuros no altere entradas o persistencia anteriores; que los huecos invaliden las ventanas correspondientes; y que entrenamiento, validación y prefijos respeten ambos extremos. El script verifica además las fronteras reales de todos los folds. Los ajustes no presentaron advertencias de falta de convergencia; las predicciones son finitas y ambos modelos se comparan sobre las mismas filas. Estas comprobaciones no constituyen una prueba universal de ausencia de fuga.

El EDA previo utilizó DEVELOPMENT y pudo orientar decisiones metodológicas. La selección y las métricas de esta sección usan los mismos cinco folds, por lo que pueden ser optimistas. La evidencia se limita a cinco activos, un proveedor y los periodos observados. Los movimientos extremos, los huecos y los cambios de distribución pueden afectar el desempeño futuro. No se afirma que otro modelo será mejor ni se cambia de dataset a partir de un resultado aislado.

## 3.11 Reproducibilidad y entregables

El protocolo se registra en `outputs/tables/base_model_protocol.json` antes del entrenamiento y se verifica su SHA-256. El procedimiento está en `src/18_base_model.py`; `src/18_render_base_report.py` genera este informe desde las tablas. Las predicciones incluyen inicio del historial, fin del objetivo y tiempos de predicción/disponibilidad nominales. Las tablas, coeficientes y metadatos están en `outputs/tables/base_*.csv` y `base_metadata.json`; los cinco pipelines finales se guardan en `outputs/models/linear_svr_*.joblib`. Se conserva el SHA-256 de DEVELOPMENT y no se lee TEST.

Para reproducir: ejecutar `python -m unittest discover -s tests -v`, después `python src/18_base_model.py`, `python src/19_extended_diagnostics.py` y finalmente `python src/18_render_base_report.py`. El segundo script amplía residuos y bootstrap desde las predicciones guardadas, sin volver a ajustar modelos. Sus metadatos conservan las huellas de las entradas; el generador del informe rechaza resultados desactualizados. El notebook carga por defecto los resultados guardados, verifica que correspondan al protocolo vigente y permite repetir los ajustes y diagnósticos con `REENTRENAR = True`.

[Dependencias fijadas](../../requirements.txt). El notebook registra la ejecución y los resultados del procedimiento. La entrega requiere el enlace publicado del Jupyter Book y el archivo `.ipynb`; localhost es una vista local, no un enlace accesible al profesor desde otro equipo. La evaluación final en TEST queda separada y no se presenta como realizada.

[Notebook del modelo base con resultados guardados](../../notebooks/18_base_model.ipynb).
