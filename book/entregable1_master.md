# Proyecto de Investigación

## Pronóstico de la Volatilidad Realizada a 24 Horas de BTC, ETH, BNB, XRP y SOL mediante SVR Lineal con Datos Horarios de Binance Spot (2020–2026)

**Materia:** Machine Learning

**Estudiantes:** Jassan Arteta y Mateo Bernal

**Profesor:** Lihki Rubio

**Universidad:** Universidad del Norte

**Programa:** Maestría en Matemáticas

**Periodo académico:** 2026-30

## Criterio común de comparabilidad entre modelos

Todos los modelos predictivos de regresión utilizarán una definición común de volatilidad basada en los retornos logarítmicos horarios y en la desviación de dichos retornos dentro de una ventana temporal fija. Esta definición se mantendrá constante entre modelos para garantizar que las diferencias de desempeño provengan de la metodología predictiva y no de cambios en la variable objetivo.

Cada algoritmo conservará su formulación matemática, arquitectura, procedimiento de entrenamiento y función de pérdida. En los modelos residuales, la red podrá aprender una corrección respecto de una referencia temporal; sin embargo, la predicción final será reconstruida en términos de la misma volatilidad utilizada por los demás modelos.

La comparación final se realizará únicamente entre modelos evaluados sobre la misma variable objetivo, el mismo horizonte, los mismos timestamps del conjunto de prueba y las mismas métricas.

### Retorno logarítmico horario

Para cada activo $i$ y hora $t$:

$$
r_{i,t}=\ln\left(\frac{P_{i,t}}{P_{i,t-1}}\right),
$$

donde $P_{i,t}$ es el precio de cierre del activo en la hora $t$ y $P_{i,t-1}$ es el cierre de la hora anterior. El retorno solo es válido cuando ambas observaciones corresponden a horas consecutivas. No se construyen retornos a través de huecos temporales.

### Volatilidad común

La volatilidad histórica/realizada se define como la desviación de los retornos respecto de su media dentro de una ventana de $n$ horas:

$$
\bar r_{i,t}=\frac{1}{n}\sum_{k=1}^{n}r_{i,t-k},
\qquad
\sigma_{i,t}=\sqrt{\frac{1}{n}\sum_{k=1}^{n}\left(r_{i,t-k}-\bar r_{i,t}\right)^2}.
$$

Inicialmente se fija $n=24$:

$$
\bar r_{i,t}=\frac{1}{24}\sum_{k=1}^{24}r_{i,t-k},
\qquad
\sigma_{i,t}^{(24)}=\sqrt{\frac{1}{24}\sum_{k=1}^{24}\left(r_{i,t-k}-\bar r_{i,t}\right)^2}.
$$

El divisor es **24**, no 23: corresponde a una desviación estándar con `ddof=0`. La magnitud base se expresa en unidades decimales de retorno horario, sin anualización ni multiplicación por $\sqrt{24}$. La ventana contiene 24 retornos, pero su desviación estándar no es la volatilidad acumulada de un retorno de 24 horas. Si se presenta como porcentaje, se multiplica por 100 tanto el valor observado como la predicción y se utiliza esa misma escala para todos los modelos y métricas.

Para evitar ambigüedad temporal, $t$ identifica el instante de referencia posterior a la disponibilidad del cierre de la hora $t-1$. Así, $\sigma_{i,t}$ utiliza únicamente los retornos $r_{i,t-24},\ldots,r_{i,t-1}$, ya disponibles. El objetivo a horizonte $h$ es $\sigma_{i,t+h}^{(24)}$. Para $h=24$, utiliza $r_{i,t},\ldots,r_{i,t+23}$, cuya realización es futura respecto del instante de referencia. El tamaño de ventana $n$ y el horizonte $h$ son conceptos distintos, aunque ambos sean inicialmente 24 horas.

La ventana debe contar con 24 retornos horarios válidos, construidos a partir de 25 cierres consecutivos. Toda regla de elegibilidad relativa a huecos o cierres irregulares debe ser común a los modelos, documentarse antes de la evaluación y no ajustarse a partir de TEST.

### Formulación por tipo de modelo

La definición financiera permanece fija. Entre modelos pueden cambiar la función predictiva, la arquitectura, la pérdida, la regularización, la construcción de la predicción, los hiperparámetros y el tratamiento residual propio del método. La salida final evaluada siempre será $\widehat{\sigma}_{i,t+h}$ frente a $\sigma_{i,t+h}$.

#### Predicción directa

SVR lineal, Ridge, Lasso, k-NN Regression, Random Forest, XGBoost, SVR con kernel, MLP directo, LSTM y Transformer podrán plantearse como:

$$
X_{i,t}\longrightarrow\widehat{\sigma}_{i,t+h}.
$$

En particular, SVR directo estima la volatilidad futura común a partir de información disponible en $t$, conservando su formulación y procedimiento de entrenamiento. Esta regla no modifica las ecuaciones internas de ninguno de los algoritmos ni obliga a utilizar la misma función de pérdida durante su entrenamiento.

#### Persistence

Persistence mantiene su papel de línea base temporal: utiliza la volatilidad reciente disponible como predicción de la futura, con la misma definición y ventana:

$$
\widehat{\sigma}^{\mathrm{Persistence}}_{i,t+h}=\sigma_{i,t}.
$$

Si la referencia reciente no puede calcularse con una ventana válida, no se sustituye por un valor futuro ni se completa utilizando TEST para tomar decisiones. La elegibilidad debe resolverse mediante la regla común de evaluación.

#### MLP residual

El MLP residual conserva su metodología. Su objetivo interno puede ser la corrección:

$$
\Delta_{i,t+h}=\sigma_{i,t+h}-\sigma_{i,t},
\qquad
\widehat{\Delta}_{i,t+h}=f_\theta(X_{i,t}).
$$

La salida se reconstruye como:

$$
\widehat{\sigma}_{i,t+h}=\sigma_{i,t}+\widehat{\Delta}_{i,t+h}.
$$

La comparación final evalúa esta volatilidad reconstruida, no la corrección aislada. El modelo no se convierte en un MLP directo.

#### MLP residual multivariado

Se conserva la metodología multisalida/multihorizonte. Para los horizontes $h_1,\ldots,h_H$, el objetivo interno puede escribirse como:

$$
\mathbf{y}_{i,t}=\begin{bmatrix}
\Delta_{i,t+h_1}\\
\Delta_{i,t+h_2}\\
\vdots\\
\Delta_{i,t+h_H}
\end{bmatrix},
\qquad
\Delta_{i,t+h}=\sigma_{i,t+h}-\sigma_{i,t}.
$$

La red estima $\widehat{\boldsymbol{\Delta}}_{i,t}=f_\theta(X_{i,t})$ y reconstruye cada componente mediante:

$$
\widehat{\sigma}_{i,t+h_j}=\sigma_{i,t}+\widehat{\Delta}_{i,t+h_j},\qquad j=1,\ldots,H.
$$

Esta regla no modifica su arquitectura residual. Cada horizonte se compara con el mismo horizonte de los demás modelos; no se mezclan horizontes distintos como si resolvieran el mismo problema.

### Evaluación final común

La comparación entre regresores exige simultáneamente:

- La misma definición y escala de volatilidad.
- El mismo horizonte $h$ y la misma ventana $n$.
- Los mismos activos, timestamps y observaciones válidas.
- El mismo conjunto TEST, reservado para la evaluación final.
- Las mismas métricas y el mismo procedimiento de agregación entre activos.

Las métricas comunes serán **RMSE, MAPE y $R^2$**, calculadas sobre la volatilidad final reconstruida cuando corresponda. El análisis de residuos, su ACF y los intervalos de confianza podrán complementar la comparación. Los residuos se definirán de manera uniforme como $e_{i,t+h}=\sigma_{i,t+h}-\widehat{\sigma}_{i,t+h}$ y su análisis deberá considerar la dependencia temporal.

MAPE no está definido cuando el valor observado es cero y puede ser inestable cerca de cero. Esa limitación se informará expresamente: no se introducirán denominadores artificiales ni se excluirán filas de forma diferente para cada modelo. Si existen ceros en el conjunto común, no se reportará un MAPE finito como si fuera el MAPE convencional del conjunto completo; cualquier evaluación complementaria deberá identificarse y usar una regla común previamente fijada. Del mismo modo, $R^2$ requiere variabilidad en los valores observados del conjunto evaluado.

La elegibilidad y la agregación se fijarán antes de la evaluación final, sin seleccionar timestamps por los errores obtenidos por cada algoritmo. No se permitirá que cada modelo mejore artificialmente su comparación al evaluar solo sus propias filas más favorables. Las transformaciones y parámetros aprendidos se estimarán dentro de cada entrenamiento cronológico; TEST no se utilizará para seleccionarlos.

### Protocolo de comparación en condiciones comunes

De la guía del proyecto MLP se adoptan los principios de ventanas temporales, validación cronológica, escalado dentro del entrenamiento, evaluación por horizonte y análisis de residuos. Los ejemplos de frecuencia, arquitectura, descarga y despliegue no sustituyen las decisiones ya establecidas para este proyecto. La comparación busca aproximar las condiciones de información y evaluación entre modelos, conservando sus diferencias metodológicas.

#### Tres parámetros temporales distintos

| Parámetro | Significado | Regla común |
|-----------|-------------|-------------|
| Frecuencia | Separación entre observaciones | Una hora. |
| Ventana de volatilidad $n$ | Retornos usados para calcular cada valor de volatilidad | 24 retornos horarios; definición del profesor con divisor 24. |
| Horizonte $h$ | Distancia desde el instante de predicción hasta el objetivo | Comparación principal a 24 horas. |
| Ventana de entrada $L$ | Historia disponible para construir los predictores | Misma longitud y mismas variables de origen para los modelos entrenables de cada experimento. |

Los tamaños de entrada de 7, 14, 21 y 28 **días** del ejemplo equivalen a 168, 336, 504 y 672 **horas**. No equivalen a 7, 14, 21 y 28 observaciones de este dataset. Se registran como una posible rejilla común, no como una selección definitiva ni como experimentos ya ejecutados. La rejilla y su coste deberán fijarse antes de comparar modelos mediante validación en DEVELOPMENT.

Cada longitud $L$ describe la historia de los predictores y no modifica $n=24$. Si una variable de entrada es a su vez una volatilidad móvil, deberá contabilizarse también el historial adicional necesario para calcularla: $L$ valores de una variable derivada pueden requerir más de $L$ precios originales.

Si posteriormente se incorporan siete horizontes diarios, deberán identificarse explícitamente como $h\in\{24,48,72,96,120,144,168\}$ horas. Siete pasos horarios serían otra tarea. Esa ampliación no se adopta en esta decisión: el horizonte principal continúa siendo $h=24$. En una evaluación multihorizonte se comparará cada modelo en los mismos horizontes; un promedio de siete horizontes no se enfrentará al resultado de un modelo evaluado únicamente a 24 horas.

#### Información de entrada y arquitectura

Los modelos entrenables compartirán, dentro de cada experimento, las variables de origen, los rezagos, la ventana histórica y el instante de disponibilidad. Un modelo tabular podrá recibir los mismos datos aplanados en un vector y uno secuencial como una matriz temporal. El cambio de representación no deberá incorporar observaciones adicionales ni información futura. Si se ensayan distintos conjuntos de variables, se identificarán como experimentos separados.

No se exige el mismo número de capas a algoritmos diferentes. SVR no tiene capas neuronales y la profundidad de un bosque no equivale a la de un MLP. Incluso entre MLP, LSTM y Transformer, igualar capas no iguala capacidad ni coste. Se registrarán las capas y unidades cuando correspondan, el número de parámetros entrenables en redes, los hiperparámetros propios de cada método y el tiempo de entrenamiento e inferencia.

La comparación distinguirá dos perspectivas: desempeño con una configuración de entrada común y desempeño de cada método tras una búsqueda acotada bajo el mismo protocolo de validación. No se mezclarán resultados por ventana fija con resultados de ventanas seleccionadas individualmente sin identificarlo. La configuración final de cada método se elegirá utilizando únicamente validación en DEVELOPMENT.

Persistence conserva su regla simple y no se fuerza a utilizar toda la ventana $L$: toma la volatilidad reciente válida. El MLP residual y el residual multivariado conservan su referencia y sus correcciones. Esta referencia también podrá estar disponible como variable explicativa común para los modelos directos, de modo que el modelo residual no reciba información histórica privilegiada. Las salidas residuales siempre se reconstruyen antes de evaluar.

#### Particiones, ajuste y presupuesto

Todos los modelos utilizarán las mismas fronteras temporales de entrenamiento y validación, con los cinco activos alineados por timestamp. Se mantendrá la partición DEVELOPMENT/TEST existente. Los bloques internos utilizados para elegir configuraciones pertenecen a DEVELOPMENT, aunque un ejemplo de código externo los denomine test; no se confundirá ese nombre con el TEST final reservado.

Se eliminará de cada entrenamiento cualquier etiqueta cuyo periodo objetivo invada el bloque de validación, verificando sus fechas reales. La separación deberá cubrir el mayor horizonte evaluado y respetar la disponibilidad de las etiquetas. Los escaladores, PCA, imputaciones y demás transformaciones aprendidas se ajustarán solo sobre el entrenamiento de cada fold. Un modelo que no necesite escalado podrá conservar su procedimiento propio; recibirá la misma información de origen.

Antes de ejecutar búsquedas se documentará un presupuesto comparable: mismos folds y criterio de selección, límite común de configuraciones evaluadas y un límite de recursos explícito. Se registrará el presupuesto realmente consumido; el mismo número de pruebas no garantiza idéntico coste computacional. No se permitirá una búsqueda extensa para un método y una configuración arbitraria sin ajustar para otro. Persistence queda exento de una búsqueda de hiperparámetros que su formulación no requiere.

Para métodos estocásticos se fijará una lista común de semillas y se reportará la variabilidad de sus resultados; no se elegirá la mejor semilla según TEST. El número de épocas y las reglas de parada podrán ser propios de cada arquitectura, dentro del presupuesto declarado, y cualquier parada temprana utilizará validación interna de DEVELOPMENT.

#### Observaciones y selección del modelo

La tabla comparativa principal utilizará un conjunto común de observaciones elegibles, determinado por disponibilidad de datos e historial requerido, nunca por la magnitud del error. Al comparar diferentes ventanas $L$, se utilizarán timestamps comunes para evitar que una ventana parezca mejor por evaluarse en un periodo más fácil. Se informarán las observaciones descartadas y su motivo. Una ausencia de predicciones o un fallo numérico no autoriza a eliminar silenciosamente las filas afectadas.

Para elegir configuraciones en DEVELOPMENT se establece como criterio principal el **RMSE medio entre activos con igual ponderación**, calculado sobre la volatilidad final y a $h=24$. Se reportarán también MAPE y $R^2$, con las limitaciones ya documentadas, y podrán añadirse MAE y MSE como métricas complementarias de la guía MLP. MSE y RMSE no se tratarán como dos evidencias independientes, pues una es el cuadrado de la otra en el mismo conjunto.

Los resultados se presentarán por activo y por horizonte antes de resumirlos. En validación se informarán las métricas por fold y su media y dispersión, sin interpretar automáticamente esa dispersión como un intervalo de confianza entre observaciones independientes. Los diagnósticos de residuos y su ACF complementarán las métricas; la prueba BDS de la guía podrá incorporarse como diagnóstico adicional y no como criterio automático para declarar un ganador.

La configuración final y el criterio de clasificación quedarán fijados antes de evaluar TEST. Este se utilizará para comparar los procedimientos ya fijados, sin volver a ajustar el modelo que obtenga peor resultado. Se distinguirá el menor RMSE observado de una superioridad estadísticamente sustentada. Si se estiman intervalos para diferencias de error, deberán respetar la dependencia temporal y entre activos. Los costes y la estabilidad se reportarán junto con la precisión.

La tabla final identificará, como mínimo, modelo, ventana $L$, horizonte $h$, activos, número de predicciones sobre timestamps comunes, RMSE, MAPE, $R^2$ y coste computacional. Las capas solo aparecerán donde sean aplicables. No se incorporan en esta etapa obligaciones de API, Docker o despliegue del ejemplo MLP, ya que no determinan la comparabilidad predictiva.

### Métodos que no son regresores finales equivalentes

PCA puede emplearse para reducción de dimensionalidad, exploración o preprocesamiento, pero no es por sí mismo un modelo predictivo final de regresión. Si forma parte de un pipeline, se evaluará el regresor completo, no PCA como predictor independiente.

Clustering es aprendizaje no supervisado y no se compara directamente mediante RMSE, MAPE o $R^2$. Un clasificador bayesiano que convierta la volatilidad continua en clases resuelve otro problema y tampoco es directamente comparable. Estos métodos no se incluirán en una tabla final de RMSE como si fueran regresores equivalentes.

### Alcance de esta decisión y resultados existentes

Esta decisión es exclusivamente documental. La definición común aquí establecida sustituye, para el trabajo posterior, la raíz de la suma de retornos futuros al cuadrado utilizada en la sección 2.1 existente. Ambas magnitudes difieren por el centrado en la media y por el divisor de la ventana; no deben tratarse como objetivos equivalentes ni sustituirse mediante un simple cambio de nombre.

Los resultados, tablas, figuras y notebooks previos se conservan sin cambios como resultados de la definición anterior. No constituyen todavía evidencia calculada para la nueva variable objetivo y deberán actualizarse en una tarea posterior expresamente autorizada. Asimismo, la propuesta anterior de MAE y RMSE no reemplaza el conjunto común de métricas fijado aquí: RMSE, MAPE y $R^2$.

No se recalcula volatilidad, no se ejecutan modelos ni validación cruzada y no se modifica ningún dataset ni TEST al registrar esta decisión.
