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

{{FOLDS}}

La tabla distingue el bloque de calendario de las anclas efectivamente evaluadas. Los tamaños son comunes a los cinco activos por la intersección de elegibilidad. El archivo `base_folds.csv` registra también el inicio del historial y la última vela objetivo de ambos conjuntos.

{{BOUNDARY_AUDIT}}

Las exclusiones no garantizan que los periodos conservados representen a los descartados. Estos resultados sustituyen las métricas previas calculadas con otras fronteras; no se comparan como si procedieran de las mismas observaciones.

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

Se define el residuo como observado menos predicho. Se analizan **ambos modelos en los cinco folds y los cinco activos: 50 diagnósticos**. Los paneles muestran residuos en el tiempo, gráfico Q-Q normal, dispersión frente a la predicción y ACF residual hasta 168 horas. Se resta la media del residuo del bloque; para cada rezago k se suman los productos centrados de los pares disponibles separados por k horas y se divide por la suma de cuadrados centrados de todas las observaciones del bloque. El denominador es común a los rezagos. Se excluyen los pares con faltantes, sin comprimir el calendario, y se guarda el número de pares. Con huecos, la pérdida de pares afecta la estimación; no se dibujan bandas iid. Las líneas temporales se interrumpen en los huecos. No se concatenan folds para calcular dependencia.

{{RESIDUALS}}

La correlación de la magnitud del residuo con la predicción y el cociente entre varianzas de la segunda y primera mitad del bloque son diagnósticos descriptivos de dispersión variable; no son pruebas concluyentes de heterocedasticidad. El archivo de diagnóstico incluye Jarque–Bera, cuyos p-valores de referencia iid no se interpretan como inferencia calibrada bajo dependencia temporal. El gráfico Q-Q permite examinar desviaciones de normalidad sin confundir normalidad marginal con independencia.

La autocorrelación residual representa información temporal no capturada o dependencia inducida por las etiquetas solapadas. No demuestra por sí sola que esa información sea explotable por otro modelo. No corresponde aplicar Moran u otros diagnósticos espaciales porque no existen coordenadas.

## 3.8 Curvas de aprendizaje

Se ajusta la configuración seleccionada con el 25 %, 50 % y 100 % inicial del **calendario de entrenamiento** del último fold. Dentro de cada prefijo se exige que historial y objetivo estén completos y contenidos en él; el porcentaje no se aplica a filas ya filtradas. La validación permanece fija y el escalador se ajusta de nuevo en cada caso. Son prefijos cronológicos, no submuestras aleatorias. La auditoría de la curva registra inicio del historial, fin del objetivo y fronteras del prefijo.

{{LEARNING}}

La distancia entre ambos errores y su evolución orientan el diagnóstico de sobreajuste, insuficiencia de información o cambio de distribución. Al ampliar el prefijo también cambia el periodo representado y la cercanía al bloque validado; por tanto, la curva no aísla únicamente el efecto del tamaño muestral ni demuestra que la muestra sea suficiente. La configuración se eligió previamente con los cinco folds, así que esta curva es diagnóstica y no una evaluación adicional independiente.

{{FIGURES}}

## 3.9 Interpretación de coeficientes

Los coeficientes se guardan para los pipelines finales ajustados sobre DEVELOPMENT. Un coeficiente estandarizado expresa el cambio predicho en puntos porcentuales de volatilidad ante una desviación estándar de aumento en ese rezago, manteniendo los demás fijos. Se guarda también el coeficiente por unidad de USDT al dividirlo por la escala de su columna.

{{COEFFICIENTS}}

La tabla muestra el rezago con mayor coeficiente absoluto estandarizado por activo, no una selección de variables. Dada la fuerte correlación entre cierres consecutivos, los signos y magnitudes individuales pueden ser inestables y mantener otros rezagos fijos puede describir combinaciones poco habituales. No se interpretan como efectos causales ni como importancias robustas. La regularización ayuda a controlar coeficientes, pero no elimina la redundancia de la entrada.

## 3.10 Auditoría crítica y limitaciones

{{HIGH_SCORE}}

Las pruebas `tests/test_temporal_boundaries.py` verifican que recalcular un bloque aislado produzca las mismas entradas, etiquetas y persistencia que filtrarlo por fechas; que cambiar cierres futuros no altere entradas o persistencia anteriores; que los huecos invaliden las ventanas correspondientes; y que entrenamiento, validación y prefijos respeten ambos extremos. El script verifica además las fronteras reales de todos los folds. Los ajustes no presentaron advertencias de falta de convergencia; las predicciones son finitas y ambos modelos se comparan sobre las mismas filas. Estas comprobaciones no constituyen una prueba universal de ausencia de fuga.

El EDA previo utilizó DEVELOPMENT y pudo orientar decisiones metodológicas. La selección y las métricas de esta sección usan los mismos cinco folds, por lo que pueden ser optimistas. La evidencia se limita a cinco activos, un proveedor y los periodos observados. Los movimientos extremos, los huecos y los cambios de distribución pueden afectar el desempeño futuro. No se afirma que otro modelo será mejor ni se cambia de dataset a partir de un resultado aislado.

## 3.11 Reproducibilidad y entregables

El protocolo se registra en `outputs/tables/base_model_protocol.json` antes del entrenamiento y se verifica su SHA-256. El procedimiento está en `src/18_base_model.py`; `src/18_render_base_report.py` genera este informe desde las tablas. Las predicciones incluyen inicio del historial, fin del objetivo y tiempos de predicción/disponibilidad nominales. Las tablas, coeficientes y metadatos están en `outputs/tables/base_*.csv` y `base_metadata.json`; los cinco pipelines finales se guardan en `outputs/models/linear_svr_*.joblib`. Se conserva el SHA-256 de DEVELOPMENT y no se lee TEST.

Para reproducir: ejecutar `python -m unittest discover -s tests -v`, después `python src/18_base_model.py`, `python src/19_extended_diagnostics.py` y finalmente `python src/18_render_base_report.py`. El segundo script amplía residuos y bootstrap desde las predicciones guardadas, sin volver a ajustar modelos. Sus metadatos conservan las huellas de las entradas; el generador del informe rechaza resultados desactualizados. El notebook carga por defecto los resultados guardados, verifica que correspondan al protocolo vigente y permite repetir los ajustes y diagnósticos con `REENTRENAR = True`.

[Dependencias fijadas](../../requirements.txt). El notebook registra la ejecución y los resultados del procedimiento. La entrega requiere el enlace publicado del Jupyter Book y el archivo `.ipynb`; localhost es una vista local, no un enlace accesible al profesor desde otro equipo. La evaluación final en TEST queda separada y no se presenta como realizada.

[Notebook del modelo base con resultados guardados](../../notebooks/18_base_model.ipynb).
