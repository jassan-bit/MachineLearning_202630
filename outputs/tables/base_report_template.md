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

La partición DEVELOPMENT/TEST existente permanece intacta. Dentro de DEVELOPMENT se utiliza [TimeSeriesSplit](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.TimeSeriesSplit.html) con cinco folds de entrenamiento creciente. Se divide la rejilla horaria completa, no las filas apiladas de los activos: cada frontera es común a los cinco. Los huecos permanecen en el calendario y luego se aplica la elegibilidad de ventanas.

Una etiqueta de entrenamiento solo se admite si su última vela objetivo es anterior a la primera hora del bloque de validación. La comprobación se hace por fechas reales. Los horizontes que exceden DEVELOPMENT también quedan excluidos. Los históricos de validación pueden contener cierres anteriores al corte ya disponibles; no contienen observaciones posteriores al instante de predicción.

{{FOLDS}}

Las fechas de la tabla son horas de apertura UTC de las velas ancla; el archivo de auditoría registra además la fecha final de las etiquetas de entrenamiento. Los tamaños son comunes a los activos por la intersección de elegibilidad. Esta regla puede reducir la cobertura y no garantiza que los periodos excluidos sean representativos de los incluidos.

## 3.4 Pipeline, formulación y búsqueda acotada

Se utiliza `Pipeline(StandardScaler, LinearSVR)`. Cada escalador se ajusta con su entrenamiento y se aplica sin reajuste a validación. [LinearSVR](https://scikit-learn.org/stable/modules/generated/sklearn.svm.LinearSVR.html) implementa regresión de vectores de soporte lineal; se elige por su adecuación computacional a muchas observaciones. La configuración usa pérdida **epsilon-insensible al cuadrado**, `dual=False`, tolerancia $10^{-5}$, máximo 20.000 iteraciones y semilla 42. El intercepto de esta implementación está regularizado. No es OLS ni se afirma identidad numérica con `SVR(kernel="linear")` y su pérdida epsilon-insensible no cuadrática habitual.

La función predicha es lineal en los cierres estandarizados. Se penaliza la magnitud de los coeficientes y los errores que exceden el tubo de ancho epsilon. No se transforma ni escala la etiqueta. C controla la penalización del error frente a la regularización; epsilon está expresado en puntos porcentuales de volatilidad.

Antes de entrenar se fija la rejilla C ∈ {0,01; 0,1; 1} y epsilon ∈ {0,01; 0,1}: **seis configuraciones**, cinco folds y cinco activos, 150 ajustes de búsqueda. El criterio de selección es el RMSE medio con igual peso por activo y fold. Se selecciona una configuración común, sin elegir una diferente para cada activo. El presupuesto usa ejecución secuencial en CPU y un límite de iteraciones por ajuste; se guardan duración e iteraciones. No se amplía adaptativamente la rejilla según los resultados. Persistence queda exenta de búsqueda.

{{SELECTION}}

La curva de aprendizaje añade 15 ajustes y el ajuste final sobre DEVELOPMENT añade cinco: 170 en total. Se guarda un pipeline final por activo, todavía sin evaluación en TEST. Las ventanas futuras, otros algoritmos y presupuestos deberán identificarse como experimentos propios bajo el protocolo común.

## 3.5 Comparación de desempeño

RMSE y MAE están en puntos porcentuales de volatilidad; MAPE se expresa en porcentaje y R² es adimensional. MAPE no se informa como finito si hay etiquetas iguales a cero; no se usan denominadores artificiales. Un R² negativo indica peor error cuadrático que la referencia constante de la media observada en ese bloque, no que el error sea negativo.

{{SUMMARY}}

La media y desviación se calculan sobre 25 combinaciones activo-fold con pesos iguales. La desviación entre bloques no es un intervalo de confianza. Los resultados por activo se resumen a continuación como media de sus cinco folds; el detalle completo conserva cada fold.

{{ASSETS}}

{{COMPARISON}}

No se recortan predicciones negativas después de observar su desempeño. Se contabilizan y se reconocen como valores incompatibles con la no negatividad de la volatilidad. Cualquier restricción posterior deberá definirse como parte de un procedimiento distinto antes de evaluarlo.

## 3.6 Incertidumbre

Se utiliza bootstrap circular de bloques de **168 horas**, con 499 réplicas y semilla 42, separado dentro de cada fold. Se remuestrean conjuntamente los mismos bloques para los activos y los dos métodos, manteniendo los huecos del calendario. Se calcula en cada réplica el RMSE por activo y fold y luego su media con pesos iguales. La diferencia se define como RMSE del SVR menos RMSE de persistencia; un valor negativo favorece SVR.

{{CI}}

Los intervalos percentiles del 95 % son exploratorios y condicionados a las predicciones y a la configuración seleccionada. No incluyen la incertidumbre del proceso de selección ni un reajuste del modelo en cada réplica; los folds comparten entrenamiento y pueden reflejar regímenes distintos. La longitud de bloque fijada tampoco prueba independencia entre bloques. Por ello no se interpreta este intervalo como evidencia final independiente de superioridad. Esa conclusión requiere TEST reservado y los procedimientos fijados de antemano.

## 3.7 Residuos y dependencia temporal

Se define el residuo como observado menos predicho. Los paneles muestran el último fold: serie observada y predicha, residuos en el tiempo, gráfico Q-Q normal, dispersión frente a la predicción y autocorrelación residual hasta 168 horas. Los rezagos se calculan sobre el calendario, sin unir artificialmente extremos de huecos.

{{RESIDUALS}}

La correlación de la magnitud del residuo con la predicción y el cociente entre varianzas de la segunda y primera mitad del bloque son diagnósticos descriptivos de dispersión variable; no son pruebas concluyentes de heterocedasticidad. El archivo de diagnóstico incluye Jarque–Bera, cuyos p-valores de referencia iid no se interpretan como inferencia calibrada bajo dependencia temporal. El gráfico Q-Q permite examinar desviaciones de normalidad sin confundir normalidad marginal con independencia.

La autocorrelación residual representa información temporal no capturada o dependencia inducida por las etiquetas solapadas. No demuestra por sí sola que esa información sea explotable por otro modelo. No corresponde aplicar Moran u otros diagnósticos espaciales porque no existen coordenadas.

## 3.8 Curvas de aprendizaje

Se ajusta la configuración seleccionada con el 25 %, 50 % y 100 % inicial del entrenamiento del último fold, manteniendo fija su validación y ajustando de nuevo el escalador en cada caso. Son prefijos cronológicos, no submuestras aleatorias. Se comparan los RMSE de entrenamiento y validación en los paneles.

{{LEARNING}}

La distancia entre ambos errores y su evolución orientan el diagnóstico de sobreajuste, insuficiencia de información o cambio de distribución. Al ampliar el prefijo también cambia el periodo representado y la cercanía al bloque validado; por tanto, la curva no aísla únicamente el efecto del tamaño muestral ni demuestra que la muestra sea suficiente. La configuración se eligió previamente con los cinco folds, así que esta curva es diagnóstica y no una evaluación adicional independiente.

{{FIGURES}}

## 3.9 Interpretación de coeficientes

Los coeficientes se guardan para los pipelines finales ajustados sobre DEVELOPMENT. Un coeficiente estandarizado expresa el cambio predicho en puntos porcentuales de volatilidad ante una desviación estándar de aumento en ese rezago, manteniendo los demás fijos. Se guarda también el coeficiente por unidad de USDT al dividirlo por la escala de su columna.

{{COEFFICIENTS}}

La tabla muestra el rezago con mayor coeficiente absoluto estandarizado por activo, no una selección de variables. Dada la fuerte correlación entre cierres consecutivos, los signos y magnitudes individuales pueden ser inestables y mantener otros rezagos fijos puede describir combinaciones poco habituales. No se interpretan como efectos causales ni como importancias robustas. La regularización ayuda a controlar coeficientes, pero no elimina la redundancia de la entrada.

## 3.10 Auditoría crítica y limitaciones

{{HIGH_SCORE}}

Se verificó que cambiar cierres futuros no altere la ventana de entrada de un ancla anterior, que los huecos invaliden las ventanas correspondientes y que los periodos objetivo de entrenamiento no invadan validación. Los ajustes no presentaron advertencias de falta de convergencia; las predicciones son finitas y se comparan sobre las mismas filas. Estas comprobaciones no constituyen una prueba universal de ausencia de fuga.

El EDA previo utilizó DEVELOPMENT y pudo orientar decisiones metodológicas. La selección y las métricas de esta sección usan los mismos cinco folds, por lo que pueden ser optimistas. La evidencia se limita a cinco activos, un proveedor y los periodos observados. Los movimientos extremos, los huecos y los cambios de distribución pueden afectar el desempeño futuro. No se afirma que otro modelo será mejor ni se cambia de dataset a partir de un resultado aislado.

## 3.11 Reproducibilidad y entregables

El protocolo se registra en `outputs/tables/base_model_protocol.json` antes del entrenamiento. El procedimiento está en `src/18_base_model.py`; las tablas, predicciones de validación, coeficientes y metadatos están en `outputs/tables/base_*.csv` y `base_metadata.json`. Los cinco pipelines finales se guardan en `outputs/models/linear_svr_*.joblib`. Se conserva el SHA-256 de DEVELOPMENT y no se lee TEST.

[Dependencias fijadas](../../requirements.txt). El notebook registra la ejecución y los resultados del procedimiento. La entrega requiere el enlace publicado del Jupyter Book y el archivo `.ipynb`; localhost es una vista local, no un enlace accesible al profesor desde otro equipo. La evaluación final en TEST queda separada y no se presenta como realizada.
