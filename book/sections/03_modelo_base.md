# 3. Modelo base

## 3.1 Objetivo y alcance de la evaluación

El modelo base es **SVR lineal** y la referencia temporal es **persistencia**. El objetivo continuo es la volatilidad realizada futura de 24 retornos horarios, calculada como su desviación estándar centrada con divisor 24 (`ddof=0`). Se expresa en porcentaje y no se anualiza. No se utiliza regresión logística porque la tarea es de regresión, ni se confunden estos resultados con la regresión OLS diagnóstica de 2.5.

Para el ancla s, que identifica la apertura de la última vela observada, la predicción se realiza después de su cierre. Con retorno horario $r_{i,s}=\ln(C_{i,s}/C_{i,s-1})$:

$$
y_{i,s}=100\sqrt{\frac{1}{24}\sum_{j=1}^{24}(r_{i,s+j}-\bar r^+_{i,s})^2},\qquad
\bar r^+_{i,s}=\frac{1}{24}\sum_{j=1}^{24}r_{i,s+j}.
$$

Los resultados de esta sección proceden exclusivamente de validación cronológica en DEVELOPMENT. **TEST permanece reservado** hasta fijar los procedimientos de los modelos que se compararán al final. No se presenta esta validación como una evaluación final independiente. Las cifras antiguas de 2.1 no se reutilizan para construir el objetivo.

## 3.2 Entrada común y referencia de persistencia

Se fija antes del entrenamiento una ventana de **168 cierres horarios** (7 días de observaciones), con entradas desde `close(s)` hasta `close(s−167)`. El horizonte sigue siendo 24 horas y la ventana de volatilidad contiene 24 retornos; son tres parámetros distintos. Esta es la configuración de este experimento, no una conclusión de que 168 sea la mejor ventana. Los futuros modelos deberán usar esta misma información para compararse en este escenario.

`close` es la única variable explicativa original. Sus rezagos forman columnas diferentes y pueden presentar multicolinealidad. Se ajusta un modelo por activo; no se incorporan volumen, calendario o identificadores como entradas. Se exige una ventana completa de cierres válidos, un objetivo válido y disponibilidad de la referencia. Los cinco activos y ambos métodos se evalúan sobre timestamps comunes; no se filtran por errores obtenidos.

Persistencia utiliza la volatilidad reciente ya observable:

$$
\widehat y^{\mathrm{persistencia}}_{i,s}=100\,\operatorname{std}(r_{i,s-23},\ldots,r_{i,s};\mathrm{ddof}=0).
$$

Esos 24 retornos requieren 25 cierres que están contenidos en la ventana disponible al SVR. La referencia utiliza menos historia por su regla propia; no recibe información futura ni información de origen adicional. No necesita ajuste de parámetros. No es la media del entrenamiento utilizada como referencia en 2.5.

## 3.3 Partición y prevención de fuga

**Separación entre entrenamiento, validación y TEST.** DEVELOPMENT contiene el 80 % inicial de los timestamps observados y TEST el 20 % final. Los cinco folds de entrenamiento y validación se construyen exclusivamente dentro de DEVELOPMENT. **El archivo TEST no se leyó ni se utilizó para entrenar, ajustar el escalador, seleccionar hiperparámetros o calcular las métricas presentadas.**

| Conjunto | Función en este experimento |
|---|---|
| Entrenamiento de cada fold | Ajustar el escalador y el SVR con observaciones anteriores al bloque validado y etiquetas disponibles antes de su frontera. |
| Validación de cada fold | Evaluar las configuraciones y seleccionar C y epsilon, sin reajustar el escalador con sus datos. |
| TEST final | Reservado para evaluar los procedimientos fijados; todavía no se reportan métricas sobre este conjunto. |

La separación es cronológica, no una afirmación de independencia estadística entre observaciones temporales. En el esquema creciente, un periodo validado en un fold puede pasar a formar parte del entrenamiento de un fold posterior; siempre se respeta el orden temporal y la disponibilidad de las etiquetas.

**Los R², RMSE y MAPE mostrados son de validación en DEVELOPMENT.** Como los mismos folds se utilizaron para elegir los hiperparámetros, sus métricas no constituyen una evaluación final independiente. El ajuste posterior de los pipelines con todo DEVELOPMENT no cambia el origen de las métricas: siguen siendo las predicciones de validación de cada fold.

La partición DEVELOPMENT/TEST existente permanece intacta. Dentro de DEVELOPMENT se utiliza [TimeSeriesSplit](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.TimeSeriesSplit.html) con cinco folds de entrenamiento creciente. Se divide la rejilla horaria completa, no las filas apiladas de los activos: cada frontera es común a los cinco. Los huecos permanecen en el calendario y luego se aplica la elegibilidad de ventanas.

Una etiqueta de entrenamiento solo se admite si su última vela objetivo es anterior a la primera hora del bloque de validación. La comprobación se hace por fechas reales. Los horizontes que exceden DEVELOPMENT también quedan excluidos. Los históricos de validación pueden contener cierres anteriores al corte ya disponibles; no contienen observaciones posteriores al instante de predicción.

| Fold | Inicio entrenamiento | Última ancla entrenamiento | Inicio validación | Fin validación | n train | n val |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 2020-08-18 05:00 | 2021-06-03 20:00 | 2021-06-04 21:00 | 2022-03-29 10:00 | 5781 | 6561 |
| 2 | 2020-08-18 05:00 | 2022-03-28 10:00 | 2022-03-29 11:00 | 2023-01-21 00:00 | 12342 | 7142 |
| 3 | 2020-08-18 05:00 | 2023-01-20 00:00 | 2023-01-21 01:00 | 2023-11-14 14:00 | 19484 | 6949 |
| 4 | 2020-08-18 05:00 | 2023-11-13 14:00 | 2023-11-14 15:00 | 2024-09-07 04:00 | 26433 | 7142 |
| 5 | 2020-08-18 05:00 | 2024-09-06 04:00 | 2024-09-07 05:00 | 2025-06-30 18:00 | 33575 | 7118 |


Las fechas de la tabla son horas de apertura UTC de las velas ancla; el archivo de auditoría registra además la fecha final de las etiquetas de entrenamiento. Los tamaños son comunes a los activos por la intersección de elegibilidad. Esta regla puede reducir la cobertura y no garantiza que los periodos excluidos sean representativos de los incluidos.

## 3.4 Pipeline, formulación y búsqueda acotada

Se utiliza `Pipeline(StandardScaler, LinearSVR)`. Cada escalador se ajusta con su entrenamiento y se aplica sin reajuste a validación. [LinearSVR](https://scikit-learn.org/stable/modules/generated/sklearn.svm.LinearSVR.html) implementa regresión de vectores de soporte lineal; se elige por su adecuación computacional a muchas observaciones. La configuración usa pérdida **epsilon-insensible al cuadrado**, `dual=False`, tolerancia $10^{-5}$, máximo 20.000 iteraciones y semilla 42. El intercepto de esta implementación está regularizado. No es OLS ni se afirma identidad numérica con `SVR(kernel="linear")` y su pérdida epsilon-insensible no cuadrática habitual.

La función predicha es lineal en los cierres estandarizados. Se penaliza la magnitud de los coeficientes y los errores que exceden el tubo de ancho epsilon. No se transforma ni escala la etiqueta. C controla la penalización del error frente a la regularización; epsilon está expresado en puntos porcentuales de volatilidad.

Antes de entrenar se fija la rejilla C ∈ {0,01; 0,1; 1} y epsilon ∈ {0,01; 0,1}: **seis configuraciones**, cinco folds y cinco activos, 150 ajustes de búsqueda. El criterio de selección es el RMSE medio con igual peso por activo y fold. Se selecciona una configuración común, sin elegir una diferente para cada activo. El presupuesto usa ejecución secuencial en CPU y un límite de iteraciones por ajuste; se guardan duración e iteraciones. No se amplía adaptativamente la rejilla según los resultados. Persistence queda exenta de búsqueda.

| C | epsilon (pp) | RMSE medio |
| --- | --- | --- |
| 0.01 | 0.01 | 0.618792 |
| 0.1 | 0.01 | 0.622726 |
| 1.0 | 0.01 | 0.623492 |
| 0.01 | 0.1 | 0.636124 |
| 0.1 | 0.1 | 0.640217 |
| 1.0 | 0.1 | 0.640901 |

Se seleccionan **C=0,01 y epsilon=0,01**. La ejecución completa tomó aproximadamente 207.8 segundos en el entorno local; no constituye una comparación de coste entre algoritmos.


La curva de aprendizaje añade 15 ajustes y el ajuste final sobre DEVELOPMENT añade cinco: 170 en total. Se guarda un pipeline final por activo, todavía sin evaluación en TEST. Las ventanas futuras, otros algoritmos y presupuestos deberán identificarse como experimentos propios bajo el protocolo común.

## 3.5 Comparación de desempeño

RMSE y MAE están en puntos porcentuales de volatilidad; MAPE se expresa en porcentaje y R² es adimensional. MAPE no se informa como finito si hay etiquetas iguales a cero; no se usan denominadores artificiales. Un R² negativo indica peor error cuadrático que la referencia constante de la media observada en ese bloque, no que el error sea negativo.

| Modelo | RMSE media ± DE | MAPE media ± DE (%) | R² media ± DE | MAE media ± DE |
| --- | --- | --- | --- | --- |
| persistence | 0.3881 ± 0.1082 | 40.4611 ± 6.7403 | -0.0424 ± 0.2371 | 0.2610 ± 0.0627 |
| svr | 0.6188 ± 0.3668 | 108.0650 ± 39.9438 | -1.8495 ± 2.6235 | 0.5322 ± 0.3271 |


La media y desviación se calculan sobre 25 combinaciones activo-fold con pesos iguales. La desviación entre bloques no es un intervalo de confianza. Los resultados por activo se resumen a continuación como media de sus cinco folds; el detalle completo conserva cada fold.

| Activo | Modelo | RMSE | MAPE (%) | R² |
| --- | --- | --- | --- | --- |
| BNBUSDT | persistence | 0.3332 | 39.05 | 0.0076 |
| BNBUSDT | svr | 0.5485 | 121.81 | -1.8886 |
| BTCUSDT | persistence | 0.2759 | 46.21 | -0.1396 |
| BTCUSDT | svr | 0.3569 | 97.10 | -0.9450 |
| ETHUSDT | persistence | 0.3404 | 40.83 | -0.1217 |
| ETHUSDT | svr | 0.5242 | 102.01 | -2.3083 |
| SOLUSDT | persistence | 0.4809 | 34.28 | 0.1213 |
| SOLUSDT | svr | 0.9336 | 103.57 | -2.8956 |
| XRPUSDT | persistence | 0.5099 | 41.93 | -0.0792 |
| XRPUSDT | svr | 0.7307 | 115.83 | -1.2101 |


Persistencia obtiene menor RMSE medio: **0,3881**, frente a **0,6188** del SVR. La diferencia es **0,2307 puntos porcentuales** a favor de persistencia. El SVR no supera la referencia en este experimento; los R² medios negativos y el MAPE elevado muestran sus limitaciones con cierres rezagados como entrada lineal. Se registraron **8 predicciones negativas** del SVR en las 174,560 predicciones de validación.

No se recortan predicciones negativas después de observar su desempeño. Se contabilizan y se reconocen como valores incompatibles con la no negatividad de la volatilidad. Cualquier restricción posterior deberá definirse como parte de un procedimiento distinto antes de evaluarlo.

## 3.6 Incertidumbre

Se utiliza bootstrap circular de bloques de **168 horas**, con 499 réplicas y semilla 42, separado dentro de cada fold. Se remuestrean conjuntamente los mismos bloques para los activos y los dos métodos, manteniendo los huecos del calendario. Se calcula en cada réplica el RMSE por activo y fold y luego su media con pesos iguales. La diferencia se define como RMSE del SVR menos RMSE de persistencia; un valor negativo favorece SVR.

| Métrica | Modelo o diferencia | Estimación | Límite 2,5 % | Límite 97,5 % |
| --- | --- | --- | --- | --- |
| macro_rmse | svr | 0.6188 | 0.5920 | 0.6427 |
| macro_rmse | persistence | 0.3881 | 0.3541 | 0.4173 |
| delta_macro_rmse | svr_minus_persistence | 0.2307 | 0.1994 | 0.2628 |


Los intervalos percentiles del 95 % son exploratorios y condicionados a las predicciones y a la configuración seleccionada. No incluyen la incertidumbre del proceso de selección ni un reajuste del modelo en cada réplica; los folds comparten entrenamiento y pueden reflejar regímenes distintos. La longitud de bloque fijada tampoco prueba independencia entre bloques. Por ello no se interpreta este intervalo como evidencia final independiente de superioridad. Esa conclusión requiere TEST reservado y los procedimientos fijados de antemano.

## 3.7 Residuos y dependencia temporal

Se define el residuo como observado menos predicho. Los paneles muestran el último fold: serie observada y predicha, residuos en el tiempo, gráfico Q-Q normal, dispersión frente a la predicción y autocorrelación residual hasta 168 horas. Los rezagos se calculan sobre el calendario, sin unir artificialmente extremos de huecos.

| Activo | Media residuo | ACF 1h | ACF 24h | ACF 168h | Correlación entre magnitud del residuo y predicción |
| --- | --- | --- | --- | --- | --- |
| BNBUSDT | -0.1812 | 0.9900 | 0.5410 | 0.2793 | 0.1818 |
| BTCUSDT | -0.2232 | 0.9889 | 0.5099 | 0.2989 | 0.2984 |
| ETHUSDT | -0.0375 | 0.9823 | 0.3918 | 0.2204 | 0.1029 |
| SOLUSDT | -0.1565 | 0.9886 | 0.5598 | 0.2570 | 0.2397 |
| XRPUSDT | -0.7272 | 0.9932 | 0.6425 | 0.3687 | 0.6667 |

Las medias residuales negativas indican sobreestimación media del SVR en el último fold. La autocorrelación a una hora supera 0,98 en los cinco activos y sigue siendo positiva a 24 y 168 horas. Las desviaciones del gráfico Q-Q y la variación de dispersión aconsejan evitar supuestos iid o gaussianos para cuantificar la incertidumbre.


La correlación de la magnitud del residuo con la predicción y el cociente entre varianzas de la segunda y primera mitad del bloque son diagnósticos descriptivos de dispersión variable; no son pruebas concluyentes de heterocedasticidad. El archivo de diagnóstico incluye Jarque–Bera, cuyos p-valores de referencia iid no se interpretan como inferencia calibrada bajo dependencia temporal. El gráfico Q-Q permite examinar desviaciones de normalidad sin confundir normalidad marginal con independencia.

La autocorrelación residual representa información temporal no capturada o dependencia inducida por las etiquetas solapadas. No demuestra por sí sola que esa información sea explotable por otro modelo. No corresponde aplicar Moran u otros diagnósticos espaciales porque no existen coordenadas.

## 3.8 Curvas de aprendizaje

Se ajusta la configuración seleccionada con el 25 %, 50 % y 100 % inicial del entrenamiento del último fold, manteniendo fija su validación y ajustando de nuevo el escalador en cada caso. Son prefijos cronológicos, no submuestras aleatorias. Se comparan los RMSE de entrenamiento y validación en los paneles.

| Activo | Fracción train | n train | RMSE train | RMSE validación |
| --- | --- | --- | --- | --- |
| BNBUSDT | 0.25 | 8393 | 0.7297 | 0.9929 |
| BNBUSDT | 0.5 | 16787 | 0.6161 | 0.4848 |
| BNBUSDT | 1.0 | 33575 | 0.5377 | 0.3443 |
| BTCUSDT | 0.25 | 8393 | 0.4055 | 0.8022 |
| BTCUSDT | 0.5 | 16787 | 0.3623 | 0.4821 |
| BTCUSDT | 1.0 | 33575 | 0.3405 | 0.3352 |
| ETHUSDT | 0.25 | 8393 | 0.5090 | 0.5845 |
| ETHUSDT | 0.5 | 16787 | 0.4653 | 0.4089 |
| ETHUSDT | 1.0 | 33575 | 0.4423 | 0.3373 |
| SOLUSDT | 0.25 | 8393 | 0.9330 | 0.9551 |
| SOLUSDT | 0.5 | 16787 | 0.8083 | 0.5199 |
| SOLUSDT | 1.0 | 33575 | 0.7466 | 0.4631 |
| XRPUSDT | 0.25 | 8393 | 0.9044 | 1.4231 |
| XRPUSDT | 0.5 | 16787 | 0.7630 | 0.9603 |
| XRPUSDT | 1.0 | 33575 | 0.6495 | 1.0422 |

El RMSE de validación disminuye al ampliar los prefijos en BNB, BTC, ETH y SOL. XRP mejora del 25 % al 50 %, pero empeora del 50 % al 100 %. No hay evidencia de que simplemente añadir más observaciones resuelva de forma uniforme las limitaciones del modelo.


La distancia entre ambos errores y su evolución orientan el diagnóstico de sobreajuste, insuficiencia de información o cambio de distribución. Al ampliar el prefijo también cambia el periodo representado y la cercanía al bloque validado; por tanto, la curva no aísla únicamente el efecto del tamaño muestral ni demuestra que la muestra sea suficiente. La configuración se eligió previamente con los cinco folds, así que esta curva es diagnóstica y no una evaluación adicional independiente.

```{figure} ../_static/figures/base_BNBUSDT.png
:alt: Predicciones, residuos y curva de aprendizaje del SVR lineal para BNBUSDT.

BNBUSDT: último fold de validación y curva de aprendizaje cronológica.
```

```{figure} ../_static/figures/base_BTCUSDT.png
:alt: Predicciones, residuos y curva de aprendizaje del SVR lineal para BTCUSDT.

BTCUSDT: último fold de validación y curva de aprendizaje cronológica.
```

```{figure} ../_static/figures/base_ETHUSDT.png
:alt: Predicciones, residuos y curva de aprendizaje del SVR lineal para ETHUSDT.

ETHUSDT: último fold de validación y curva de aprendizaje cronológica.
```

```{figure} ../_static/figures/base_SOLUSDT.png
:alt: Predicciones, residuos y curva de aprendizaje del SVR lineal para SOLUSDT.

SOLUSDT: último fold de validación y curva de aprendizaje cronológica.
```

```{figure} ../_static/figures/base_XRPUSDT.png
:alt: Predicciones, residuos y curva de aprendizaje del SVR lineal para XRPUSDT.

XRPUSDT: último fold de validación y curva de aprendizaje cronológica.
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

El máximo R² del SVR entre las 25 evaluaciones es 0.0258; no aparece un desempeño cercano a 0,8–0,9 que active esa alerta particular. Un resultado bajo tampoco demuestra ausencia de fuga. La inferioridad frente a persistencia muestra que el problema no queda resuelto por esta configuración lineal.

Se verificó que cambiar cierres futuros no altere la ventana de entrada de un ancla anterior, que los huecos invaliden las ventanas correspondientes y que los periodos objetivo de entrenamiento no invadan validación. Los ajustes no presentaron advertencias de falta de convergencia; las predicciones son finitas y se comparan sobre las mismas filas. Estas comprobaciones no constituyen una prueba universal de ausencia de fuga.

El EDA previo utilizó DEVELOPMENT y pudo orientar decisiones metodológicas. La selección y las métricas de esta sección usan los mismos cinco folds, por lo que pueden ser optimistas. La evidencia se limita a cinco activos, un proveedor y los periodos observados. Los movimientos extremos, los huecos y los cambios de distribución pueden afectar el desempeño futuro. No se afirma que otro modelo será mejor ni se cambia de dataset a partir de un resultado aislado.

## 3.11 Reproducibilidad y entregables

El protocolo se registra en `outputs/tables/base_model_protocol.json` antes del entrenamiento. El procedimiento está en `src/18_base_model.py`; las tablas, predicciones de validación, coeficientes y metadatos están en `outputs/tables/base_*.csv` y `base_metadata.json`. Los cinco pipelines finales se guardan en `outputs/models/linear_svr_*.joblib`. Se conserva el SHA-256 de DEVELOPMENT y no se lee TEST.

[Dependencias fijadas](../../requirements.txt). El notebook registra la ejecución y los resultados del procedimiento. La entrega requiere el enlace publicado del Jupyter Book y el archivo `.ipynb`; localhost es una vista local, no un enlace accesible al profesor desde otro equipo. La evaluación final en TEST queda separada y no se presenta como realizada.

[Notebook del modelo base con resultados guardados](../../notebooks/18_base_model.ipynb).
