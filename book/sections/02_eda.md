# 2. Análisis Exploratorio de Datos (EDA)

El análisis exploratorio utiliza exclusivamente DEVELOPMENT, del 11 de agosto de 2020 a las 06:00 UTC al 1 de julio de 2025 a las 18:00 UTC. TEST permanece reservado desde la partición cronológica inicial. Los resultados describen los datos disponibles para desarrollar el modelo; no se entrenan modelos ni se evalúa el conjunto de prueba.

Las decisiones posteriores de imputación, transformación, escalado, selección de variables o tratamiento de extremos deberán ajustarse únicamente con la porción de entrenamiento de cada partición temporal. Las estadísticas globales de DEVELOPMENT que se presentan aquí son descriptivas y no constituyen parámetros de preprocesamiento para validación.

## 2.1 Análisis de la variable objetivo

### 2.1.1 Definición y disponibilidad temporal

La variable objetivo es la **volatilidad realizada futura a 24 horas**, una magnitud cuantitativa continua. Por tratarse de regresión, las frecuencias por clase, el desbalance de clases y el tamaño de la clase minoritaria no aplican.

Se utiliza la **definición de volatilidad**, adaptada de días a horas. Sea $P_{i,t}$ el precio de cierre (`close`) del activo $i$ en la hora $t$. El retorno logarítmico horario es:

$$
r_{i,t}=\ln\left(\frac{P_{i,t}}{P_{i,t-1}}\right).
$$

Para una ventana de $n=24$ retornos horarios, la media y la volatilidad son:

$$
\bar r_{i,t}=\frac{1}{24}\sum_{k=1}^{24}r_{i,t-k},
\qquad
\sigma_{i,t}^{(24)}=\sqrt{\frac{1}{24}\sum_{k=1}^{24}\left(r_{i,t-k}-\bar r_{i,t}\right)^2}.
$$

La volatilidad es la **desviación estándar de los retornos dentro de la ventana**, centrada en su media y con divisor **24** (`ddof=0`), no 23. No se anualiza ni se multiplica por $\sqrt{24}$. La fórmula está expresada en unidades decimales; para presentarla como porcentaje se multiplica por 100, utilizando la misma escala para las observaciones y las predicciones de todos los modelos.

El instante de referencia $t$ se sitúa después de la disponibilidad del cierre de la hora $t-1$. La volatilidad reciente $\sigma_{i,t}^{(24)}$ utiliza los retornos $r_{i,t-24},\ldots,r_{i,t-1}$, ya disponibles. Para un horizonte de 24 horas, el objetivo es la misma magnitud calculada sobre los retornos futuros:

$$
\bar r_{i,t+24}=\frac{1}{24}\sum_{j=0}^{23}r_{i,t+j},
\qquad
y_{i,t}=\sigma_{i,t+24}^{(24)}
=\sqrt{\frac{1}{24}\sum_{j=0}^{23}\left(r_{i,t+j}-\bar r_{i,t+24}\right)^2}.
$$

Su valor solo se conoce al terminar el horizonte futuro. Los retornos deben corresponder a horas consecutivas, sin atravesar huecos temporales. Las variables explicativas deben estar disponibles en el instante de predicción o antes. Esta convención coincide con el criterio metodológico común; en los resultados anteriores, la referencia temporal se etiquetaba mediante la apertura de la última vela observada.

### 2.1.2 Ventanas válidas y cobertura

Se exige una secuencia de 25 cierres horarios consecutivos para calcular los 24 retornos. Cada cierre debe corresponder al último milisegundo de su hora, según la regla de consistencia definida en 1.7.6. Las ventanas que incluyen horas ausentes o alguno de los 28 cierres no convencionales quedan sin objetivo en esta definición conservadora. Esta exclusión afecta a la tabla derivada, no elimina filas del dataset original ni declara erróneas las velas señaladas.

También se excluyen las últimas 24 horas de referencia de DEVELOPMENT, porque su horizonte futuro sobrepasa la partición. No se consultan datos de TEST para completarlo. Los motivos se contabilizan de forma excluyente: primero límite temporal, después horas ausentes y finalmente cierres irregulares.

| Activo | Velas observadas | Objetivos válidos | Borde final | Huecos | Cierres irregulares |
| --- | ---: | ---: | ---: | ---: | ---: |
| BTCUSDT | 42,833 | 42,539 | 24 | 240 | 30 |
| ETHUSDT | 42,833 | 42,539 | 24 | 240 | 30 |
| BNBUSDT | 42,833 | 42,539 | 24 | 240 | 30 |
| XRPUSDT | 42,833 | 42,564 | 24 | 240 | 5 |
| SOLUSDT | 42,833 | 42,564 | 24 | 240 | 5 |

En total se obtienen **212,745 objetivos válidos**; 1,420 velas de referencia quedan sin objetivo: 120 por el borde final, 1,200 por huecos y 100 adicionales por cierres irregulares. Una misma vela irregular afecta a varias ventanas solapadas.

Como sensibilidad, aceptar cierres irregulares manteniendo las demás condiciones produciría 42,569 objetivos por activo. Frente a la definición conservadora, añadiría 30 ventanas en BTC, ETH y BNB, y 5 en XRP y SOL. El cambio absoluto en la media es inferior a 0.0006 puntos porcentuales por activo y los máximos no cambian. Esta comparación no prueba la validez de los cierres irregulares; se mantiene la definición conservadora. Los resultados completos están en la [tabla de sensibilidad](../../outputs/tables/eda_target_sensitivity.csv).

### 2.1.3 Distribución, asimetría y valores extremos

| Activo | Mínimo (%) | Mediana (%) | Media (%) | P95 (%) | P99 (%) | Máximo (%) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| BTCUSDT | 0.2192 | 2.4239 | 2.7522 | 5.9095 | 8.8148 | 16.5971 |
| ETHUSDT | 0.2659 | 3.0945 | 3.5639 | 7.5489 | 12.0423 | 25.1058 |
| BNBUSDT | 0.3879 | 2.7903 | 3.4342 | 7.9181 | 13.6605 | 29.7777 |
| XRPUSDT | 0.5611 | 3.3650 | 4.3678 | 10.9599 | 18.6078 | 39.0647 |
| SOLUSDT | 0.9167 | 4.7852 | 5.6996 | 12.3089 | 19.6207 | 34.6655 |

No se observan objetivos iguales a cero. SOL presenta la mayor mediana (4.7852%) y BTC la menor (2.4239%). XRP alcanza el máximo más alto (39.0647%), aunque su mediana es inferior a la de SOL. Por ello, el nivel habitual de volatilidad y la intensidad de los episodios extremos deben distinguirse. Los cuartiles y la desviación estándar están en el [resumen completo](../../outputs/tables/eda_target_summary.csv).

```{figure} ../_static/figures/eda_target_distribution.png
:alt: Histogramas, boxplots y distribución del logaritmo de la volatilidad futura a 24 horas, separados por activo y calculados solo en DEVELOPMENT.

Distribución del objetivo por activo. La primera columna muestra frecuencias en escala logarítmica; la segunda utiliza boxplots con cercas de 1.5 IQR; la tercera muestra el logaritmo natural del objetivo como diagnóstico visual. No se transforma la variable utilizada en las tablas originales.
```

| Activo | Asimetría | Exceso de curtosis | Señalados por IQR | Porcentaje |
| --- | ---: | ---: | ---: | ---: |
| BTCUSDT | 2.000 | 7.101 | 2,058 | 4.84% |
| ETHUSDT | 2.208 | 8.878 | 1,965 | 4.62% |
| BNBUSDT | 3.225 | 17.727 | 2,429 | 5.71% |
| XRPUSDT | 3.171 | 15.266 | 3,261 | 7.66% |
| SOLUSDT | 2.607 | 11.012 | 2,246 | 5.28% |

La asimetría es positiva en todos los activos y la media supera la mediana. Los excesos de curtosis positivos, entre 7.101 y 17.727, y los valores extremos visibles indican colas empíricas pronunciadas frente a una referencia normal. Estas estadísticas no demuestran una ley de colas específica ni prueban normalidad o independencia. BNB y XRP presentan los mayores excesos de curtosis; XRP también tiene la mayor proporción señalada por IQR.

Los valores fuera de las cercas $Q_1-1.5\,IQR$ y $Q_3+1.5\,IQR$ son candidatos a extremos, no errores demostrados. Se conservan todos los objetivos válidos, incluidos los elevados. El IQR no se utiliza aquí para recortar ni winsorizar la variable objetivo.

### 2.1.4 Necesidad de transformación

El logaritmo permite examinar si la escala positiva y la asimetría del objetivo se representan de forma más equilibrada. La tabla compara la asimetría y el exceso de curtosis antes y después del logaritmo, únicamente como diagnóstico.

| Activo | Asimetría original | Asimetría de ln(y) | Exceso de curtosis original | Exceso de curtosis de ln(y) |
| --- | ---: | ---: | ---: | ---: |
| BTCUSDT | 2.000 | -0.280 | 7.101 | 0.564 |
| ETHUSDT | 2.208 | -0.138 | 8.878 | 0.425 |
| BNBUSDT | 3.225 | 0.176 | 17.727 | 0.259 |
| XRPUSDT | 3.171 | 0.454 | 15.266 | 0.358 |
| SOLUSDT | 2.607 | 0.304 | 11.012 | 0.280 |

Una menor asimetría no demuestra normalidad ni garantiza mejores pronósticos. No se adopta una transformación definitiva. Box–Cox es una alternativa para valores estrictamente positivos, pero su parámetro no se estima globalmente en esta etapa: si se evalúa posteriormente, deberá ajustarse dentro de cada partición de entrenamiento. Lo mismo aplica a cualquier desplazamiento utilizado si aparecen ceros en datos futuros.

Si se modelara el logaritmo del objetivo, minimizar el error en esa escala cambiaría la importancia relativa de los errores grandes y pequeños. Además, exponenciar un pronóstico en escala logarítmica no produce automáticamente la media condicional en la escala original. La elección deberá evaluarse cronológicamente y las métricas finales deberán incluir la escala original.

### 2.1.5 Comportamiento temporal

```{figure} ../_static/figures/eda_target_time.png
:alt: Evolución horaria de la volatilidad futura a 24 horas por activo y su mediana mensual en DEVELOPMENT.

Evolución del objetivo. El eje horizontal identifica la apertura de la vela de referencia, no el momento en que se conoce el objetivo. La línea naranja resume la mediana mensual de los objetivos válidos; es descriptiva y no se utiliza como predictor.
```

| Activo | Apertura de referencia (UTC) | Máximo (%) |
| --- | ---: | ---: |
| BTCUSDT | 2021-05-19 00:00 | 16.5971 |
| ETHUSDT | 2021-05-19 09:00 | 25.1058 |
| BNBUSDT | 2021-05-19 02:00 | 29.7777 |
| XRPUSDT | 2021-02-01 02:00 | 39.0647 |
| SOLUSDT | 2022-11-09 14:00 | 34.6655 |

Los máximos de BTC, ETH y BNB corresponden a ventanas iniciadas el 19 de mayo de 2021; los de XRP y SOL ocurren en fechas diferentes. Las fechas identifican ventanas futuras de 24 horas y no un único retorno ni una causa económica comprobada. La evolución muestra episodios de distinta intensidad, lo que exige evaluar el desempeño en varios bloques cronológicos.

| Activo | Correlación del objetivo a 1h | Correlación del objetivo a 24h |
| --- | ---: | ---: |
| BTCUSDT | 0.990 | 0.595 |
| ETHUSDT | 0.991 | 0.650 |
| BNBUSDT | 0.994 | 0.711 |
| XRPUSDT | 0.991 | 0.625 |
| SOLUSDT | 0.992 | 0.689 |

Las correlaciones se calculan por pares disponibles sobre la cuadrícula horaria, sin comprimir los huecos. La elevada asociación a una hora está influida mecánicamente por el solapamiento de 23 retornos entre etiquetas. A 24 horas las ventanas de retornos ya no se solapan, aunque persiste asociación descriptiva. Ninguno de estos coeficientes sustituye una evaluación fuera de muestra.

No se realiza análisis espacial: el dataset carece de coordenadas geográficas. Los activos son entidades del panel y no ubicaciones espaciales.

### 2.1.6 Relación con las variables explicativas disponibles

Como exploración inicial vinculada al análisis bivariado, se compara el objetivo con las nueve variables numéricas de la vela cerrada, el retorno de esa hora y la volatilidad realizada de las 24 horas anteriores, definida como $100\sqrt{\sum_{j=0}^{23}r_{i,t-j}^{2}}$. Esta última usa exclusivamente información pasada y se calcula con la misma regla de continuidad y cierres convencionales. No se ejecuta el baseline Persistence ni se evalúa un pronóstico.

Las fechas actúan como índices temporales y `symbol` identifica el activo. No se asignan códigos numéricos arbitrarios a los símbolos para calcular correlaciones. Las asociaciones se calculan por activo y con los pares disponibles para cada variable; por eso su tamaño de muestra puede variar. Las cifras completas se incluyen en la [tabla de correlaciones y tamaños de muestra](../../outputs/tables/eda_target_correlations.csv).

```{figure} ../_static/figures/eda_target_correlations.png
:alt: Correlaciones Pearson y Spearman entre cada variable disponible y la volatilidad futura, por activo.

Pearson resume asociación lineal y Spearman asociación monótona. Los coeficientes son descriptivos; no se calculan valores p suponiendo independencia de las observaciones.
```

```{figure} ../_static/figures/eda_target_relations.png
:alt: Densidad conjunta del objetivo frente a volatilidad pasada y volumen actual para cada activo.

Relaciones con la volatilidad pasada y el volumen en USDT. Cada panel incorpora todos los pares disponibles; los hexágonos más oscuros indican mayor frecuencia, en escala logarítmica. Las escalas numéricas de los activos se muestran por separado.
```

La volatilidad pasada de 24 horas muestra correlaciones de Pearson entre 0.595 y 0.711 con el objetivo, y de Spearman entre 0.579 y 0.697. Sus retornos no se solapan con los del objetivo futuro, aunque ambas ventanas comparten el cierre que actúa como frontera. Esta asociación motiva estudiar la persistencia más adelante, sin afirmar todavía el desempeño del baseline.

El volumen en USDT presenta asociaciones distintas según el activo: Spearman es 0.384 para BTC, 0.503 para ETH, 0.529 para BNB, 0.562 para XRP y 0.029 para SOL. Los volúmenes taker presentan patrones similares a sus volúmenes totales. El número de operaciones también varía: Spearman va de 0.006 en SOL a 0.539 en XRP.

Los cuatro precios OHLC muestran asociaciones parecidas dentro de cada activo y más débiles en valor absoluto que la volatilidad pasada. El retorno horario con signo tiene una correlación de Pearson cercana a cero en los cinco activos (entre -0.040 y 0.003); esto no descarta relaciones no lineales ni una posible relación con su magnitud absoluta. Los diagramas de densidad permiten observar dispersión y extremos que un coeficiente aislado no describe.

Estos resultados no constituyen una selección definitiva de predictores ni un análisis multivariado condicional. Las asociaciones con precios y volúmenes pueden depender del periodo y de cambios de régimen; una correlación no implica causalidad ni capacidad predictiva fuera de muestra. La selección y las transformaciones deberán contrastarse después mediante validación temporal, sin TEST.

### 2.1.7 Implicaciones para métricas y validación

Se propone reportar MAE y RMSE en puntos porcentuales de volatilidad, por activo y como promedio de las métricas de los cinco activos con igual ponderación. MAE describe el tamaño absoluto del error y RMSE da mayor peso a errores grandes, por lo que su lectura conjunta permite examinar el desempeño durante episodios de alta volatilidad. No se reportan valores de estas métricas porque todavía no se han generado pronósticos. Los errores porcentuales relativos requieren cautela cuando el objetivo se aproxima a cero.

La validación deberá ser cronológica, con ventanas de entrenamiento anteriores a las de validación, y mantener todos los activos de una misma fecha en el mismo bloque temporal. No se utilizará una partición aleatoria de filas. La [documentación de validación temporal de scikit-learn](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.TimeSeriesSplit.html) describe la separación ordenada y el uso de un intervalo entre entrenamiento y evaluación; en este panel, ese intervalo debe aplicarse a timestamps globales, no a un número de filas mezcladas entre activos.

Los objetivos de horas consecutivas comparten 23 de sus 24 retornos. Por ello, la dependencia entre etiquetas no puede interpretarse como evidencia automática de capacidad predictiva y el número de filas no equivale a observaciones independientes. En cada frontera se excluirán del entrenamiento las etiquetas cuyo horizonte alcance el inicio de la validación: una separación conservadora de 24 horas entre las horas de referencia evita el solapamiento de retornos futuros. Además, toda etiqueta de entrenamiento debe estar disponible antes de realizar la primera predicción de validación. Las ventanas se comprobarán por tiempo real, especialmente alrededor de huecos.

Todos los parámetros aprendidos de los datos deberán ajustarse de nuevo dentro de cada entrenamiento. TEST se mantendrá reservado para la evaluación final del procedimiento ya fijado.

### 2.1.8 Reproducibilidad

El cálculo está en `src/08_eda_target_development.py`. Los resultados, cobertura, sensibilidad y trazabilidad se guardan en `outputs/tables/eda_target_*.csv` y `outputs/tables/eda_target_metadata.json`. La tabla derivada se guarda separadamente en `outputs/tables/eda_target_development_derived.csv`: no se modifica el CSV maestro ni las particiones. Se verifica cada objetivo válido contra una suma directa de 24 retornos y se comprueba la conservación de DEVELOPMENT mediante SHA-256.


## 2.2 Análisis unidimensional de close

El análisis se centra en el precio de cierre (`close`) de los cinco activos, usando exclusivamente DEVELOPMENT: 214,165 observaciones, con 42,833 por activo. De esta variable se obtienen los retornos logarítmicos necesarios para construir la volatilidad futura, que continúa siendo la variable objetivo del pronóstico. Las demás columnas no se analizan en esta sección; las fechas y los símbolos se utilizan únicamente para identificar las observaciones y separar los activos.

### 2.2.1 Tipo de variable

`close` es una variable numérica continua, expresada en USDT por unidad del activo base. No es una variable categórica: no corresponden frecuencias de clases, categorías raras ni tratamiento de desbalance. Los precios se describen por activo para evitar mezclar escalas diferentes. No se detectaron valores nulos en esta columna dentro de DEVELOPMENT.

### 2.2.2 Estadísticos descriptivos

Cada fila de la tabla resume 42,833 observaciones. Se presentan media, mediana, desviación estándar muestral y percentiles 5 y 95. La [tabla completa de close](../../outputs/tables/univariate_close_summary.csv) incluye también mínimo, máximo, percentiles 1 y 99 y cuartiles.

Los percentiles utilizan interpolación lineal. La desviación estándar descriptiva de los precios utiliza divisor N−1 (`ddof=1`); no representa la volatilidad de los retornos, cuya definición mantiene divisor 24 (`ddof=0`).

| Activo | Media | Mediana | Desv. estándar | P5 | P95 |
| --- | ---: | ---: | ---: | ---: | ---: |
| BTCUSDT | 45,804.96 | 40,554.17 | 25,338.61 | 15,372.90 | 98,647.39 |
| ETHUSDT | 2,247.83 | 2,110.24 | 967.56 | 450.26 | 3,888.85 |
| BNBUSDT | 376.40 | 328.00 | 187.84 | 29.04 | 661.28 |
| XRPUSDT | 0.8047 | 0.5577 | 0.6362 | 0.2712 | 2.3561 |
| SOLUSDT | 82.97 | 46.35 | 70.70 | 2.4129 | 207.08 |

### 2.2.3 Distribución, asimetría y curtosis

Los histogramas incluyen todos los cierres y utilizan 60 intervalos de igual anchura. El eje de frecuencias es logarítmico para hacer visibles los intervalos menos frecuentes; los precios permanecen en sus unidades originales. La asimetría y el exceso de curtosis se calculan con estimadores corregidos por sesgo. La referencia normal del exceso de curtosis es cero.

```{figure} ../_static/figures/univariate_close.png
:alt: Histogramas y boxplots de close para los cinco activos de DEVELOPMENT.

Distribución de close por activo. Izquierda: histograma y mediana discontinua. Derecha: boxplot en unidades originales. Las escalas son propias de cada panel.
```

### 2.2.4 Valores extremos: boxplots e IQR

Se señalan cierres inferiores a Q1−1.5 IQR o superiores a Q3+1.5 IQR, con IQR=Q3−Q1, por separado para cada activo. En los boxplots, la caja representa Q1–Q3, la línea central es la mediana y los bigotes alcanzan las últimas observaciones dentro de las cercas. Los puntos exteriores son candidatos a extremos, no errores demostrados.

| Activo | Asimetría | Exceso de curtosis | Extremos IQR | Porcentaje |
| --- | ---: | ---: | ---: | ---: |
| BTCUSDT | 0.803 | -0.188 | 0 | 0.00% |
| ETHUSDT | 0.128 | -0.478 | 0 | 0.00% |
| BNBUSDT | -0.065 | -0.822 | 0 | 0.00% |
| XRPUSDT | 1.870 | 2.526 | 5,809 | 13.56% |
| SOLUSDT | 0.556 | -1.021 | 0 | 0.00% |

Los cierres señalados se conservan. No se eliminan, recortan ni winsorizan precios. Un nivel excepcional puede corresponder a otro periodo de cotización; la regla global de IQR no certifica errores de medición ni identifica por sí sola retornos extremos.

### 2.2.5 Normalidad

No se aplica una prueba de normalidad como requisito para utilizar `close`: la normalidad marginal del predictor no es una condición necesaria de los regresores considerados. Además, los precios forman una serie temporal y no se justifica interpretar las pruebas convencionales como si las observaciones fueran independientes. Con 42,833 valores por activo, una prueba puede detectar desviaciones pequeñas sin determinar su relevancia predictiva.

La [documentación de Shapiro–Wilk de SciPy](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.shapiro.html) advierte que el valor p puede no ser preciso para muestras mayores de 5,000. Se utilizan los histogramas y las medidas de forma como descripción, sin afirmar normalidad ni extraer una submuestra arbitraria para obtener un valor p.

### 2.2.6 Interpretación y relevancia

El precio de cierre presenta comportamientos diferentes entre activos. BTC y ETH tienen medianas de 40,554.17 y 2,110.24 USDT, respectivamente; esos niveles no permiten concluir cuál tiene mayor riesgo, porque corresponden a activos y escalas de precio diferentes. La desviación estándar del precio durante varios años tampoco equivale a la volatilidad de sus retornos en una ventana de 24 horas.

XRP tiene la mayor asimetría del cierre (1.870) y un exceso de curtosis positivo (2.526). La regla IQR señala 5,809 precios de cierre, equivalentes al 13.56%, todos por encima del límite superior de 1.411585 USDT. Esto describe precios altos frente a la distribución global de DEVELOPMENT; no demuestra que esas cotizaciones sean errores ni que los retornos de esas horas sean extremos.

En BTC, ETH, BNB y SOL no se señalan cierres bajo la regla IQR global. Ese resultado no implica normalidad ni ausencia de episodios volátiles: las cercas describen el nivel de precio acumulado durante todo el periodo. BNB presenta asimetría cercana a cero (-0.065), mientras que SOL combina una mediana de 46.35 USDT con una media de 82.97 USDT. La forma completa del histograma aporta información que no queda resumida por una única medida.

Para construir retornos se emplean cierres de horas consecutivas, sin atravesar huecos. `close` solo está disponible una vez cerrada la vela. Su presencia como predictor, sus rezagos y cualquier transformación deberán respetar ese instante de disponibilidad. No se eliminan los cierres señalados ni se decide una transformación a partir de estos gráficos.

### 2.2.7 Reproducibilidad

Los resultados corresponden al cálculo previo sobre DEVELOPMENT realizado mediante `src/09_univariate_development.py`. Se conservan sus filas de `close` en `outputs/tables/univariate_close_summary.csv`, sin recalcular estadísticas. La figura es `book/_static/figures/univariate_close.png`. El [notebook centrado en close](../../notebooks/10_close_univariate.ipynb) carga y presenta esos resultados guardados con su interpretación. Los archivos del análisis más amplio se conservan como antecedentes; no forman parte de la presentación de esta sección.


## 2.3 Análisis bidimensional

El análisis se centra en `close` y en variables construidas exclusivamente a partir de sus precios históricos. No se incorporan volúmenes ni otros campos de mercado como predictores. `symbol` identifica las entidades y las fechas permiten alinear las observaciones. Todos los cálculos utilizan DEVELOPMENT; TEST permanece reservado.

### 2.3.1 Variables, objetivo y alineación temporal

El objetivo de esta sección se calcula con la desviación estándar de los 24 retornos futuros, centrados en su media y con divisor 24 (`ddof=0`). Se presenta en porcentaje, sin anualizar. Estos resultados son nuevos y no reutilizan las cifras de la definición anterior de 2.1.

Para precisar los índices, sea $s$ la hora de apertura de la última vela observada y $C_{i,s}$ su cierre. La predicción se sitúa después del cierre de esa vela, alrededor de $s+1$ hora. Con $r_{i,s}=\ln(C_{i,s}/C_{i,s-1})$:

$$
\bar r^+_{i,s}=\frac{1}{24}\sum_{j=1}^{24}r_{i,s+j},\qquad
y_{i,s}=100\sqrt{\frac{1}{24}\sum_{j=1}^{24}(r_{i,s+j}-\bar r^+_{i,s})^2}.
$$

Esta notación etiqueta las filas mediante la última vela observada; corresponde a la definición común con instante de referencia $t=s+1$. El factor 100 convierte la desviación en porcentaje, sin cambiar su significado financiero.

| Variable | Construcción y disponibilidad |
|----------|-------------------------------|
| `close` | Cierre de la última vela observada. |
| `close_lag_1h` | Cierre una hora antes, con desplazamiento por calendario. |
| `close_lag_24h` | Cierre 24 horas antes, con desplazamiento por calendario. |
| `return_1h_pct` | $100r_{i,s}$, retorno de la última hora observada. |
| `abs_return_1h_pct` | $100\lvert r_{i,s}\rvert$, magnitud del movimiento reciente. |
| `sigma_past_24h_pct` | Desviación estándar de los retornos $r_{i,s-23},\ldots,r_{i,s}$, con divisor 24 y multiplicada por 100. |
| `sigma_future_24h_pct` | Objetivo $y_{i,s}$, conocido únicamente al finalizar el horizonte futuro. |

Los rezagos de una y 24 horas son candidatos exploratorios, no una selección definitiva de la ventana de entrada de los modelos. Las ventanas de volatilidad requieren 25 cierres consecutivos y convencionales; los huecos, cierres irregulares y horizontes que sobrepasan DEVELOPMENT no se imputan. Los retornos pasados y futuros no se solapan, aunque comparten el cierre de frontera.

Para comparar asociaciones y VIF dentro de cada activo se usa la misma muestra de filas completas en las seis variables candidatas y el objetivo. Esto evita que los coeficientes difieran solo por utilizar conjuntos distintos de observaciones.

| Activo | Velas observadas | Objetivos válidos | Filas completas |
| --- | ---: | ---: | ---: |
| BTCUSDT | 42,833 | 42,539 | 42,251 |
| ETHUSDT | 42,833 | 42,539 | 42,251 |
| BNBUSDT | 42,833 | 42,539 | 42,251 |
| XRPUSDT | 42,833 | 42,564 | 42,300 |
| SOLUSDT | 42,833 | 42,564 | 42,300 |

### 2.3.2 Numéricas frente a numéricas: gráficos y correlaciones

Se calculan Pearson y Spearman por activo. Pearson describe asociación lineal; Spearman describe asociación monótona mediante rangos. Los coeficientes se interpretan como tamaños de asociación descriptivos y no se acompañan de valores p basados en independencia de filas horarias.

Los gráficos hexbin incluyen todos los pares de la muestra completa. El color representa su frecuencia en escala logarítmica y permite visualizar concentraciones y observaciones dispersas sin ocultarlas mediante un muestreo de puntos.


#### BTCUSDT

| Candidato | Pearson | Spearman | MI (nats) |
| --- | ---: | ---: | ---: |
| close | -0.034 | 0.060 | 0.635 |
| close_lag_1h | -0.034 | 0.060 | 0.595 |
| close_lag_24h | -0.027 | 0.063 | 0.550 |
| return_1h_pct | -0.029 | -0.001 | 0.085 |
| abs_return_1h_pct | 0.359 | 0.344 | 0.077 |
| sigma_past_24h_pct | 0.596 | 0.582 | 0.435 |

```{figure} ../_static/figures/bivariate_BTCUSDT.png
:alt: Diagramas hexbin de los seis candidatos derivados de close frente a la volatilidad futura de BTCUSDT.

Relaciones por pares en BTCUSDT. La barra de cada panel muestra el número de observaciones por hexágono.
```

#### ETHUSDT

| Candidato | Pearson | Spearman | MI (nats) |
| --- | ---: | ---: | ---: |
| close | -0.007 | 0.036 | 0.497 |
| close_lag_1h | -0.006 | 0.036 | 0.460 |
| close_lag_24h | 0.007 | 0.043 | 0.436 |
| return_1h_pct | -0.042 | -0.004 | 0.087 |
| abs_return_1h_pct | 0.386 | 0.357 | 0.088 |
| sigma_past_24h_pct | 0.649 | 0.623 | 0.476 |

```{figure} ../_static/figures/bivariate_ETHUSDT.png
:alt: Diagramas hexbin de los seis candidatos derivados de close frente a la volatilidad futura de ETHUSDT.

Relaciones por pares en ETHUSDT. La barra de cada panel muestra el número de observaciones por hexágono.
```

#### BNBUSDT

| Candidato | Pearson | Spearman | MI (nats) |
| --- | ---: | ---: | ---: |
| close | -0.124 | -0.071 | 0.506 |
| close_lag_1h | -0.123 | -0.070 | 0.468 |
| close_lag_24h | -0.117 | -0.067 | 0.436 |
| return_1h_pct | -0.013 | 0.007 | 0.125 |
| abs_return_1h_pct | 0.463 | 0.406 | 0.137 |
| sigma_past_24h_pct | 0.709 | 0.697 | 0.544 |

```{figure} ../_static/figures/bivariate_BNBUSDT.png
:alt: Diagramas hexbin de los seis candidatos derivados de close frente a la volatilidad futura de BNBUSDT.

Relaciones por pares en BNBUSDT. La barra de cada panel muestra el número de observaciones por hexágono.
```

#### XRPUSDT

| Candidato | Pearson | Spearman | MI (nats) |
| --- | ---: | ---: | ---: |
| close | 0.125 | 0.213 | 0.495 |
| close_lag_1h | 0.125 | 0.213 | 0.464 |
| close_lag_24h | 0.123 | 0.208 | 0.445 |
| return_1h_pct | 0.002 | 0.010 | 0.103 |
| abs_return_1h_pct | 0.413 | 0.370 | 0.106 |
| sigma_past_24h_pct | 0.625 | 0.638 | 0.472 |

```{figure} ../_static/figures/bivariate_XRPUSDT.png
:alt: Diagramas hexbin de los seis candidatos derivados de close frente a la volatilidad futura de XRPUSDT.

Relaciones por pares en XRPUSDT. La barra de cada panel muestra el número de observaciones por hexágono.
```

#### SOLUSDT

| Candidato | Pearson | Spearman | MI (nats) |
| --- | ---: | ---: | ---: |
| close | -0.206 | -0.184 | 0.484 |
| close_lag_1h | -0.206 | -0.184 | 0.457 |
| close_lag_24h | -0.204 | -0.184 | 0.434 |
| return_1h_pct | -0.007 | 0.000 | 0.093 |
| abs_return_1h_pct | 0.416 | 0.349 | 0.099 |
| sigma_past_24h_pct | 0.687 | 0.668 | 0.466 |

```{figure} ../_static/figures/bivariate_SOLUSDT.png
:alt: Diagramas hexbin de los seis candidatos derivados de close frente a la volatilidad futura de SOLUSDT.

Relaciones por pares en SOLUSDT. La barra de cada panel muestra el número de observaciones por hexágono.
```


La volatilidad pasada presenta correlaciones de Pearson entre 0.596 y 0.709 y de Spearman entre 0.582 y 0.697 con la volatilidad futura. El retorno con signo muestra asociación lineal pequeña, mientras que su magnitud absoluta alcanza Pearson entre 0.359 y 0.463. Esto sugiere que la intensidad del movimiento reciente describe mejor la variabilidad futura que su dirección, sin demostrar desempeño predictivo fuera de muestra.

El precio de cierre tiene asociaciones de Pearson débiles en valor absoluto, con signos diferentes entre activos. Las figuras muestran dispersión, cambios de escala y relaciones que no quedan resumidas por una recta. Los patrones de `close` y sus rezagos son muy parecidos, lo que debe contrastarse con la revisión de multicolinealidad.

### 2.3.3 Predictoras frente al objetivo: información mutua

Las tablas anteriores incluyen información mutua (MI), estimada mediante vecinos próximos con `n_neighbors=5` y semilla 42, expresada en nats. Se utiliza [mutual_info_regression de scikit-learn](https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.mutual_info_regression.html). Para estabilidad numérica, los predictores se estandarizan dentro de la muestra descriptiva de cada activo; ese escalado no se guarda ni se utiliza para entrenar modelos.

La MI puede detectar dependencia que Pearson no resume, pero no indica el signo de una relación, no es una correlación entre cero y uno y no es un valor p. En estas estimaciones, `close` presenta MI entre 0.484 y 0.635 nats, pese a sus asociaciones lineales débiles; la volatilidad pasada presenta entre 0.435 y 0.544 nats.

No se interpreta la MI alta del nivel de precio como prueba de una relación no lineal estable: la dependencia temporal, las ventanas de objetivo solapadas y la mezcla de periodos de mercado pueden producir asociaciones elevadas en toda la muestra. Tampoco se ordenan definitivamente las variables por esta cifra. Su utilidad deberá comprobarse en bloques cronológicos no usados para ajustar cada modelo.

### 2.3.4 Categórica frente a numérica: activo y volatilidad

Se compara `symbol` con la volatilidad futura calculada a partir de `close`. La comparación se realiza sobre 42,539 timestamps con objetivo válido simultáneamente en los cinco activos. Se evita comparar niveles nominales de precio como si un activo más caro tuviera necesariamente mayor riesgo.

| Activo | Media (%) | Mediana (%) | Q1 (%) | Q3 (%) |
| --- | ---: | ---: | ---: | ---: |
| BTCUSDT | 0.5507 | 0.4854 | 0.3335 | 0.6738 |
| ETHUSDT | 0.7129 | 0.6196 | 0.4355 | 0.8769 |
| BNBUSDT | 0.6875 | 0.5589 | 0.3731 | 0.8298 |
| XRPUSDT | 0.8756 | 0.6764 | 0.4628 | 1.0177 |
| SOLUSDT | 1.1414 | 0.9607 | 0.6814 | 1.3717 |

```{figure} ../_static/figures/bivariate_groups.png
:alt: Boxplots de la volatilidad futura centrada con divisor 24 por activo, sobre timestamps comunes de DEVELOPMENT.

Distribución de la volatilidad futura por activo, en la misma escala porcentual y sobre las mismas horas de referencia.
```

Las observaciones de distintos activos están emparejadas por fecha y pueden compartir movimientos. Además, cada serie presenta dependencia temporal y las etiquetas consecutivas comparten retornos. Por ello no se aplican pruebas t, ANOVA, Mann–Whitney o Kruskal–Wallis como si se tratara de grupos de observaciones independientes. Se emplea una alternativa de remuestreo temporal pareado para comparar medias.

### 2.3.5 Comparación entre grupos y rigor estadístico

Se utiliza bootstrap circular por bloques: 1,999 réplicas, semilla 42 y longitud principal de 168 horas, fijada antes de examinar los resultados para conservar segmentos de una semana y superar el solapamiento de 24 horas del objetivo. Se remuestrea la misma secuencia de bloques para los cinco activos, conservando los huecos sobre la cuadrícula horaria y calculando las medias únicamente sobre observaciones comunes válidas. El último bloque se recorta para mantener la longitud original del calendario. La [documentación de bootstrap temporal de arch](https://arch.readthedocs.io/en/stable/bootstrap/timeseries-bootstraps.html) describe el remuestreo circular mediante bloques de longitud fija.

Para cada par se informa la diferencia media $\bar d=\operatorname{media}(y_A-y_B)$ en puntos porcentuales, el efecto estandarizado pareado $d_z=\bar d/s_d$, un intervalo percentil del 95% y un valor p bilateral aproximado obtenido de las diferencias bootstrap centradas. El valor p se calcula como $(1+\#\{|\bar d^*-\bar d|\geq|\bar d|\})/(B+1)$, con $B=1999$. La hipótesis contrastada es diferencia media nula; no igualdad de todas las características de las distribuciones.

Se aplica Holm a la familia de diez comparaciones por pares. Los intervalos mostrados son individuales, no simultáneos; la corrección múltiple se aplica a los valores p. La resolución mínima del valor p sin ajustar es 0.0005: no se reportan probabilidades exactamente cero.

| Par A−B | Diferencia (pp) | Efecto pareado | IC 95% individual | p sin ajustar | p Holm |
| --- | ---: | ---: | ---: | ---: | ---: |
| BTCUSDT − ETHUSDT | -0.1622 | -0.7557 | [-0.1809; -0.1448] | 0.0005 | 0.0050 |
| BTCUSDT − BNBUSDT | -0.1368 | -0.3689 | [-0.1747; -0.1031] | 0.0005 | 0.0050 |
| BTCUSDT − XRPUSDT | -0.3249 | -0.5688 | [-0.3752; -0.2789] | 0.0005 | 0.0050 |
| BTCUSDT − SOLUSDT | -0.5907 | -1.0671 | [-0.6423; -0.5410] | 0.0005 | 0.0050 |
| ETHUSDT − BNBUSDT | 0.0254 | 0.0715 | [-0.0100; 0.0589] | 0.1385 | 0.1385 |
| ETHUSDT − XRPUSDT | -0.1627 | -0.2915 | [-0.2120; -0.1167] | 0.0005 | 0.0050 |
| ETHUSDT − SOLUSDT | -0.4285 | -0.8641 | [-0.4736; -0.3837] | 0.0005 | 0.0050 |
| BNBUSDT − XRPUSDT | -0.1881 | -0.3304 | [-0.2382; -0.1409] | 0.0005 | 0.0050 |
| BNBUSDT − SOLUSDT | -0.4539 | -0.9031 | [-0.4996; -0.4112] | 0.0005 | 0.0050 |
| XRPUSDT − SOLUSDT | -0.2658 | -0.3994 | [-0.3272; -0.2076] | 0.0005 | 0.0050 |

Con bloques de 168 horas, nueve comparaciones tienen $p_{Holm}=0.005$ y ETH–BNB no muestra evidencia suficiente de diferencia media ($p_{Holm}=0.1385$). La diferencia ETH–BNB es pequeña, 0.0254 puntos porcentuales, con efecto estandarizado 0.0715; su intervalo incluye cero. No rechazar la hipótesis no demuestra igualdad entre ambos activos.

Se examinó la sensibilidad a bloques de 24 y 336 horas. En ETH–BNB, el valor p ajustado cambia de 0.005 con 24 horas a 0.1385 con 168 y 0.2345 con 336. Esta sensibilidad impide presentar la separación entre ambos como un hallazgo estable. Las otras nueve comparaciones mantienen $p_{Holm}=0.005$ en las tres longitudes. La corrección se aplica por separado en cada escenario; los escenarios alternativos son sensibilidad y no se elige el de menor valor p. Los [resultados completos](../../outputs/tables/bivariate_group_comparisons.csv) incluyen las tres longitudes.

El bootstrap no elimina todas las limitaciones: su interpretación depende de que los bloques representen adecuadamente la dependencia y de una estabilidad temporal suficiente. Los cambios de régimen y la unión circular del final con el inicio pueden afectar la inferencia. Los valores p e intervalos son aproximaciones exploratorias condicionadas al periodo, no evidencia causal ni garantía de generalización. La importancia sustantiva se evalúa con las diferencias y tamaños de efecto, no solo con la significancia.

### 2.3.6 Categóricas frente a categóricas

No aplica: solo se dispone de `symbol` como variable categórica del alcance analizado. No existe una segunda variable categórica para construir una tabla de contingencia, realizar chi-cuadrado o calcular V de Cramér. No se discretiza el objetivo ni se crean clases artificiales para cumplir este apartado, pues eso cambiaría la pregunta de regresión.

### 2.3.7 Matrices de correlación y multicolinealidad

```{figure} ../_static/figures/bivariate_heatmap_pearson.png
:alt: Matrices Pearson por activo entre close, sus derivados y la volatilidad futura.

Correlaciones lineales en la misma muestra completa de cada activo. El objetivo aparece como referencia y no entra en los predictores del cálculo VIF.
```

```{figure} ../_static/figures/bivariate_heatmap_spearman.png
:alt: Matrices Spearman por activo entre close, sus derivados y la volatilidad futura.

Correlaciones por rangos. Una asociación alta entre precios rezagados puede reflejar persistencia del nivel y no utilidad adicional para pronosticar volatilidad.
```

Se calcula $VIF_j=1/(1-R_j^2)$ para cada uno de los seis predictores, regresándolo sobre los otros cinco con intercepto. Son regresiones auxiliares de diagnóstico, no modelos de pronóstico entrenados o evaluados. Se estandarizan las columnas para estabilidad numérica. El objetivo se excluye del cálculo. El VIF describe [multicolinealidad de la matriz de diseño](https://www.statsmodels.org/stable/generated/statsmodels.stats.outliers_influence.variance_inflation_factor.html), no asociación con el objetivo.

| Candidato | BTCUSDT | ETHUSDT | BNBUSDT | XRPUSDT | SOLUSDT |
| --- | ---: | ---: | ---: | ---: | ---: |
| close | 33,075.03 | 13,899.32 | 14,756.26 | 7,249.48 | 6,174.40 |
| close_lag_1h | 33,244.55 | 13,987.80 | 14,854.04 | 7,351.22 | 6,304.62 |
| close_lag_24h | 298.40 | 102.97 | 151.75 | 123.86 | 163.86 |
| return_1h_pct | 5.16 | 6.13 | 4.55 | 2.80 | 1.86 |
| abs_return_1h_pct | 1.27 | 1.31 | 1.41 | 1.35 | 1.33 |
| sigma_past_24h_pct | 1.26 | 1.31 | 1.42 | 1.37 | 1.37 |

Los VIF de `close` y su rezago de una hora superan 6,000 en todos los activos; el rezago de 24 horas también presenta valores superiores a 100. Incluir simultáneamente los tres niveles aporta una fuerte redundancia lineal. Los VIF de la magnitud del retorno y la volatilidad pasada se sitúan aproximadamente entre 1.26 y 1.42 en esta matriz. Los VIF elevados pueden dificultar la interpretación de coeficientes; no son una regla universal de eliminación para todos los algoritmos.

### 2.3.8 Interpretación y selección preliminar

Los resultados motivan considerar la volatilidad pasada y la magnitud del retorno como candidatos derivados de `close`, y revisar la conveniencia de introducir simultáneamente varios niveles de precio muy correlacionados. El retorno con signo conserva una asociación lineal pequeña, pero ello no descarta relaciones condicionales o no lineales. No se eliminan variables ni se fija la matriz final de entrada en esta etapa.

La selección preliminar es una hipótesis de trabajo: cualquier comparación de conjuntos de variables, regularización o transformación se realizará dentro de la validación cronológica de DEVELOPMENT. Los modelos deberán compartir la información disponible y las observaciones de evaluación según el protocolo común. No se utilizan los resultados de esta exploración para afirmar qué algoritmo será mejor.

### 2.3.9 Reproducibilidad

El [notebook ejecutado del análisis bidimensional](../../notebooks/11_bivariate_close.ipynb) contiene el cálculo, los resultados y las figuras. El procedimiento también está en `src/11_bivariate_close.py`. Las tablas y trazabilidad se guardan en `outputs/tables/bivariate_*.csv` y `outputs/tables/bivariate_metadata.json`; las figuras se guardan en `book/_static/figures/bivariate_*.png`. Cada objetivo válido se verifica contra el cálculo directo de la desviación estándar de sus 24 retornos futuros. Se comprueban la alineación temporal y la conservación de DEVELOPMENT mediante SHA-256. Los datos originales y TEST permanecen intactos.

## 2.4 Análisis multivariado

### 2.4.1 Alcance y dimensionalidad

La única variable explicativa original es el precio de cierre, `close`. El alcance de esta sección considera una columna de cierre por activo; `symbol` identifica la criptomoneda y la fecha ordena las observaciones. La volatilidad futura es la variable objetivo y no se incorpora como entrada para reducir dimensionalidad, detectar anomalías o formar grupos de predictores.

Con esta representación, el análisis de los predictores es unidimensional. Los derivados examinados en 2.3 permiten explorar asociaciones, pero no constituyen por sí mismos una decisión de incluirlos simultáneamente en los modelos. Por ello, no se construye aquí una matriz adicional de variables solo para aplicar técnicas multivariadas.

### 2.4.2 Reducción de dimensionalidad

PCA no aporta reducción de dimensionalidad sobre una única columna de `close`: si su varianza es positiva, solo puede obtenerse un componente, que concentra el 100 % de esa varianza. Esta es una propiedad matemática de una entrada unidimensional, no un resultado de un PCA ejecutado sobre los datos. No se aplican UMAP ni t-SNE, porque no existe una representación de alta dimensión que requiera resumirse visualmente en este alcance.

Si posteriormente se utilizan ventanas de varios cierres históricos, cada rezago será una columna distinta. En ese caso sí podrá evaluarse PCA sobre esa matriz, ajustando el escalado y los componentes exclusivamente con el entrenamiento de cada partición temporal y aplicándolos después a validación. La dimensión efectiva de esas ventanas no se determina en esta sección.

### 2.4.3 Valores atípicos

Con una sola entrada no se evalúan anomalías multivariadas. En una dimensión y con varianza positiva, la distancia de Mahalanobis se reduce a la distancia absoluta a la media dividida por la desviación estándar. No añade un diagnóstico de relaciones entre variables al análisis unidimensional de `close` presentado en 2.2.

Isolation Forest puede utilizarse con una sola variable, pero ello sería una detección unidimensional adicional. No se ejecuta en esta sección ni se presentan resultados de ese método. Los precios extremos no se consideran automáticamente errores y no se eliminan observaciones a partir de este apartado.

### 2.4.4 Grupos y subpoblaciones

Los activos identificados por `symbol` constituyen grupos conocidos, no grupos descubiertos mediante clustering. Sus precios tienen escalas distintas; agrupar los cierres brutos mezclando criptomonedas podría reflejar principalmente esas diferencias de nivel, sin demostrar la existencia de regímenes de mercado.

No se aplica clustering exploratorio en este alcance. La comparación por activo se conserva en los apartados anteriores. Identificar regímenes temporales exigiría definir una representación histórica adecuada y comprobar la estabilidad de los grupos; no se afirma que dichos regímenes hayan sido detectados.

### 2.4.5 Interpretación

Para una sola columna de `close`, la dimensión de entrada es uno y la multicolinealidad entre predictores no aplica. Esto no implica que los cierres sucesivos sean independientes ni que una ventana de rezagos esté libre de redundancia. Cuando varios cierres históricos se introducen simultáneamente como entradas, pueden estar fuertemente correlacionados aunque procedan de la misma variable original.

En consecuencia, se justifica la no aplicación de reducción de dimensionalidad, detección multivariada y clustering en la representación actual. Este apartado no selecciona ventanas, transforma datos ni modifica el protocolo común de comparación de modelos. Cualquier decisión posterior sobre representaciones temporales se evaluará dentro de DEVELOPMENT, manteniendo TEST reservado.

El [notebook de alcance del análisis multivariado](../../notebooks/12_multivariate_scope_close.ipynb) recoge esta justificación metodológica. Es un documento sin celdas de cálculo; no requiere ejecutar análisis.

## 2.5 Auditoría de fuga de datos (*Data Leakage*)

### 2.5.1 Disponibilidad en el instante de predicción

La entrada original es `close`. Para una vela cuya apertura es s, su cierre solo está disponible una vez finalizada la hora, alrededor de s+1 hora. La predicción se realiza después de recibir ese cierre, no al abrir la vela. Los timestamps del archivo permiten verificar el orden temporal, pero no acreditan la latencia real de publicación o recepción de la API; una implementación operativa deberá registrar esa disponibilidad.

| Campo o construcción | Disponibilidad y función | Decisión |
|---|---|---|
| `close` de la última vela cerrada | Observado antes de predecir. No se usa el cierre de una vela en curso. | Entrada admitida. |
| Rezagos de `close` | Disponibles si pertenecen a velas anteriores ya cerradas. | Solo si se define una ventana común; no implican nuevas variables de origen. |
| Retornos y volatilidad pasada | Disponibles si se calculan exclusivamente hacia atrás con cierres consecutivos. | Derivados exploratorios; no se incorporan automáticamente al modelo. |
| Volatilidad futura de 24 horas | Requiere 24 retornos posteriores y se conoce al finalizar el horizonte. | Etiqueta; excluida de las entradas. |
| `symbol` | Identifica el activo. | Clave de agrupación, no predictor en este diagnóstico. |
| `open_time`, `close_time` | Ordenan las velas y verifican su cierre. | Metadatos de alineación, no predictores. |
| Resto de campos de mercado | No forman parte de la entrada definida para el proyecto. | Excluidos por alcance, sin afirmar que sean fugas por sí mismos. |

### 2.5.2 Objetivo, derivados y posibles proxies

El objetivo se calcula como 100 veces la desviación estándar de los 24 retornos logarítmicos futuros, centrados en su media y con divisor 24. A partir del ancla s utiliza los retornos s+1 a s+24. El desplazamiento hacia el futuro se usa únicamente para construir la etiqueta; no se permite en las entradas.

Compartir el cierre de frontera entre el último dato observado y el primer retorno futuro no constituye fuga: ese precio ya se conoce al predecir. En cambio, usar cierres futuros, ventanas centradas, rellenos hacia atrás desde el futuro, la propia etiqueta o transformaciones de ella como predictores introduciría información no disponible. No se usan esas construcciones en el diagnóstico de esta sección. La correlación o información mutua por sí solas no demuestran disponibilidad temporal ni ausencia de fuga.

### 2.5.3 Diagnóstico predictivo de close

AUC no aplica al objetivo continuo de regresión. Se ajusta una regresión lineal diagnóstica con intercepto y una sola entrada, `close`, sin búsqueda de hiperparámetros. Se reserva el último 20 % de los timestamps observados de DEVELOPMENT para validación interna, con frontera común en **2024-07-09 20:00 UTC** (hora de apertura de la vela ancla). No es el TEST final ni una comparación definitiva de modelos.

El escalado de `close` utiliza únicamente el entrenamiento. Se excluyen 24 etiquetas por activo en la frontera: la última vela requerida por cada etiqueta de entrenamiento debe preceder al primer ancla de validación. Los huecos y cierres irregulares invalidan las ventanas correspondientes; no se imputan. Tampoco se construyen etiquetas que excedan DEVELOPMENT. RMSE y MAE se expresan en puntos porcentuales de volatilidad; R² es adimensional. La referencia constante utiliza solo la media de las etiquetas de entrenamiento.

| Activo | Entrenamiento | Validación | RMSE close | MAE close | R² close | RMSE media de entrenamiento |
|---|---:|---:|---:|---:|---:|---:|
| BNBUSDT | 33972 | 8543 | 0.3551 | 0.2962 | -0.4413 | 0.3683 |
| BTCUSDT | 33972 | 8543 | 0.3362 | 0.2891 | -0.9333 | 0.2594 |
| ETHUSDT | 33972 | 8543 | 0.3512 | 0.2561 | -0.0173 | 0.3510 |
| SOLUSDT | 33997 | 8543 | 0.4685 | 0.3724 | -0.2479 | 0.5533 |
| XRPUSDT | 33997 | 8543 | 0.9824 | 0.8353 | -1.4772 | 0.6242 |

Los R² son negativos en los cinco activos: esta relación lineal con un único cierre no explica bien la volatilidad futura en el bloque evaluado. No aparece un desempeño casi perfecto que active esa señal de alarma. Sin embargo, un desempeño bajo no prueba ausencia de fuga ni descarta que ventanas históricas o relaciones no lineales tengan utilidad. Un R² negativo compara con la media del bloque evaluado; la referencia de la última columna es distinta porque utiliza la media de entrenamiento.

### 2.5.4 Duplicados, entidades y fronteras temporales

La comprobación actual en DEVELOPMENT encuentra **0 filas duplicadas exactas** y **0 claves (`symbol`, `open_time`) duplicadas**. Los registros existentes de la partición sitúan su última apertura en **2025-07-01 18:00 UTC** y la primera de TEST en **2025-07-01 19:00 UTC**, con el mismo corte para los cinco activos y sin mezcla aleatoria.

Según esos registros, los intervalos no se solapan; por tanto, no comparten claves temporales ni filas completas idénticas que incluyan esas fechas. Esta conclusión utiliza la auditoría guardada de la partición, no una nueva lectura de TEST. No se certifica aquí la ausencia de vectores de precios iguales o casi duplicados entre archivos ignorando la fecha. Un cierre repetido, por sí solo, no identifica un registro duplicado.

BTC, ETH, BNB, XRP y SOL aparecen en ambas particiones. Esta repetición de entidades es coherente con pronosticar el futuro de los mismos activos; no evalúa generalización a criptomonedas desconocidas. Los activos de una misma hora deberán permanecer en el mismo bloque en la validación temporal.

La separación de filas por fecha no basta para separar etiquetas futuras. En cada fold se deben purgar las etiquetas cuyo periodo objetivo invada la validación. El diagnóstico ejecutado comprueba esa condición por fechas reales. Los folds definitivos y sus preprocesadores deberán superar la misma verificación cuando se construyan. Utilizar historia anterior al corte como contexto de una predicción posterior es admisible si ya estaba disponible; entrenar con sus resultados futuros no lo es.

### 2.5.5 Transformaciones y alcance de la evidencia

Escaladores, imputaciones, PCA, selección de variables, umbrales e hiperparámetros deberán ajustarse exclusivamente dentro del entrenamiento de cada fold. El EDA previo sobre DEVELOPMENT completo no equivale a una validación fuera de muestra: sus estadísticas y transformaciones auxiliares no deben reutilizarse como preprocesadores ya ajustados. Si sus conclusiones guían decisiones, las métricas internas tienen ese contexto exploratorio y TEST debe conservarse para la evaluación final.

En este diagnóstico se verificaron el orden temporal, la purga, la disponibilidad de las etiquetas dentro de DEVELOPMENT y el ajuste del escalado solo en entrenamiento. No se audita todavía un pipeline final entrenado ni la latencia de una implementación en producción. Las cifras antiguas de 2.1 no se reutilizan: el objetivo se recalcula con la definición común de desviación estándar.

### 2.5.6 Interpretación y decisiones

Se conserva `close` de velas cerradas como entrada original. Se excluyen de la entrada el objetivo, sus transformaciones y cualquier observación futura. Identificadores y fechas se mantienen como claves; los demás campos quedan fuera del alcance acordado. Las ventanas de rezagos y los derivados históricos quedan sujetos a comprobar su disponibilidad y a una definición común antes de comparar modelos.

Las comprobaciones realizadas no detectan infracciones de las reglas temporales implementadas en el diagnóstico. Esta evidencia no constituye una garantía general de ausencia de fuga: siguen siendo necesarias la auditoría de los folds y pipelines definitivos y la verificación operativa de disponibilidad. No se utiliza TEST para elegir variables, ventanas o configuraciones.

### 2.5.7 Reproducibilidad

El procedimiento está en `src/13_leakage_audit.py`. Las métricas se guardan en `outputs/tables/leakage_close_diagnostic.csv` y la trazabilidad en `outputs/tables/leakage_audit_metadata.json`. Se verifica que el SHA-256 de DEVELOPMENT no cambie durante la ejecución. Los datos no se modifican y el archivo TEST no se lee; sus fronteras se consultan exclusivamente en los registros existentes de la partición.

[Notebook ejecutado de auditoría de fuga de datos](../../notebooks/13_leakage_audit.ipynb).

## 2.6 Componente temporal

### 2.6.1 Validación de fechas, frecuencia y cobertura

Se analiza `close` por activo exclusivamente en DEVELOPMENT, desde **2020-08-11 06:00 UTC hasta 2025-07-01 18:00 UTC**, según la apertura de las velas. Las fechas se interpretan como ISO 8601 con zona UTC; se verifica que las aperturas estén alineadas a horas enteras. La frecuencia nominal es horaria, pero el calendario observado presenta huecos. Se conserva una rejilla horaria con valores ausentes, sin comprimir el tiempo ni interpolar precios.

| Activo | Horas observadas | Horas esperadas | Ausencias | Cierres irregulares | Fechas duplicadas |
| --- | --- | --- | --- | --- | --- |
| BNBUSDT | 42833 | 42853 | 20 | 6 | 0 |
| BTCUSDT | 42833 | 42853 | 20 | 6 | 0 |
| ETHUSDT | 42833 | 42853 | 20 | 6 | 0 |
| SOLUSDT | 42833 | 42853 | 20 | 5 | 0 |
| XRPUSDT | 42833 | 42853 | 20 | 5 | 0 |

La cobertura de timestamps es **99,9533 %** por activo. Las 20 horas ausentes no incluyen los cierres irregulares: estos son registros presentes cuyo cierre temporal no coincide con el final convencional de la hora. Para los cálculos de esta sección se enmascaran esos cierres, manteniendo los archivos originales. Por ello, la cobertura analítica válida es ligeramente menor que la cobertura de timestamps. Una fecha repetida entre activos es esperada; la unicidad se verifica dentro de cada activo.

### 2.6.2 Series completas y agregaciones

Los paneles por activo muestran toda la serie horaria de DEVELOPMENT, el último cierre de cada día completo y medias semanales y mensuales. Un día completo requiere 24 cierres horarios válidos. Las medias semanales y mensuales utilizan las observaciones válidas disponibles; los periodos con huecos o en los extremos pueden ser parciales. Estas agregaciones son descriptivas y no se introducen como entradas de predicción.

Las escalas de precio se mantienen separadas por activo, en USDT por unidad. Esto permite observar tendencias y fluctuaciones sin interpretar las diferencias de precio nominal entre criptomonedas como diferencias de riesgo.

### 2.6.3 Ciclos de calendario y descomposición

Se presentan boxplots de `close` por hora UTC, día de la semana y mes. Los puntos extremos se ocultan solo para facilitar la lectura; no se eliminan de los cálculos. Estos gráficos mezclan años y niveles de tendencia: una diferencia entre meses no demuestra una estacionalidad repetible. Los patrones por hora o día tampoco deben confundirse con una señal predictiva validada.

Se aplica [STL robusto](https://www.statsmodels.org/stable/generated/statsmodels.tsa.seasonal.STL.html) al logaritmo del cierre diario, con periodo de **7 días**. Se utiliza el tramo diario continuo más largo, seleccionado por cobertura y no por resultados: **2023-03-25 a 2025-06-30, 829 días**, común a los cinco activos. No se rellenan huecos. La descomposición separa tendencia, componente semanal y residuo; no evalúa ciclos horarios o anuales.

Como resumen descriptivo se calcula la fuerza semanal: máximo entre cero y uno menos la varianza del residuo dividida por la varianza de la suma del residuo y el componente estacional. No es una prueba de significación ni una proporción de varianza explicada del precio completo.

| Activo | Fuerza semanal STL |
| --- | --- |
| BNBUSDT | 0.0016 |
| BTCUSDT | 0.0343 |
| ETHUSDT | 0.0000 |
| SOLUSDT | 0.0796 |
| XRPUSDT | 0.0537 |

Los valores entre 0 y 0,080 indican una componente semanal débil según este indicador y esta configuración. No justifican introducir automáticamente variables de calendario. STL utiliza información a ambos lados de cada punto y aquí es solo exploratorio: sus componentes calculados sobre todo el tramo no pueden trasladarse directamente a un predictor disponible en tiempo real.

### 2.6.4 Estacionariedad

Se muestran medias y varianzas móviles de `close` sobre 30 días completos, con ventanas hacia atrás; los huecos interrumpen su cálculo. Además, se aplican pruebas al logaritmo del cierre diario y a su primera diferencia dentro del mismo tramo continuo. El retorno diario es un diagnóstico derivado de `close`, no una nueva variable original ni un cambio del objetivo horario.

[ADF](https://www.statsmodels.org/stable/generated/statsmodels.tsa.stattools.adfuller.html) contrasta la hipótesis nula de raíz unitaria; se usa constante y selección AIC entre 0 y 14 rezagos diarios. [KPSS](https://www.statsmodels.org/stable/generated/statsmodels.tsa.stattools.kpss.html) contrasta estacionariedad alrededor de una constante, con selección automática de rezagos. Son pruebas con hipótesis distintas. Los p-valores de KPSS en los límites de la tabla se presentan como cotas; los valores numéricos de ADF inferiores a 0,0001 se muestran con esa cota, no como probabilidades exactamente nulas.

| Activo | Serie | n | p ADF | p KPSS |
| --- | --- | --- | --- | --- |
| BNBUSDT | log close | 829 | 0.8375 | ≤0.01 |
| BNBUSDT | Retorno diario | 828 | <0.0001 | ≥0.10 |
| BTCUSDT | log close | 829 | 0.8571 | ≤0.01 |
| BTCUSDT | Retorno diario | 828 | <0.0001 | ≥0.10 |
| ETHUSDT | log close | 829 | 0.3622 | ≤0.01 |
| ETHUSDT | Retorno diario | 828 | <0.0001 | ≥0.10 |
| SOLUSDT | log close | 829 | 0.5509 | ≤0.01 |
| SOLUSDT | Retorno diario | 828 | <0.0001 | ≥0.10 |
| XRPUSDT | log close | 829 | 0.8778 | ≤0.01 |
| XRPUSDT | Retorno diario | 828 | <0.0001 | ≥0.10 |

En los cinco activos, ADF no rechaza raíz unitaria en el nivel logarítmico y KPSS rechaza estacionariedad alrededor de una constante al 5 %. En los retornos diarios, ADF rechaza raíz unitaria y KPSS no rechaza su hipótesis nula. El patrón es compatible con mayor estabilidad del retorno que del nivel en este tramo, pero no demuestra independencia, varianza constante ni estacionariedad durante todo DEVELOPMENT. Las pruebas son exploratorias, sin selección automática de transformaciones; cambios estructurales y la especificación determinista pueden afectar sus conclusiones.

### 2.6.5 Dependencia y correlaciones con rezagos

ACF y PACF se calculan hasta 30 rezagos **diarios** sobre las dos series del tramo continuo. La ACF del nivel logarítmico muestra persistencia y decaimiento lento; en retornos la dependencia lineal es mucho menor. Las curvas no incluyen bandas inferenciales ni constituyen una prueba de independencia. La ACF del nivel potencialmente no estacionario se interpreta descriptivamente.

La relación con el objetivo se examina mediante Pearson y Spearman entre `close(s-k)` y la volatilidad futura etiquetada en s, para k = 0, 1, 6, 12, 24, 48 y 168 **horas**. El objetivo utiliza 24 retornos horarios futuros, divisor 24 y escala porcentual. Solo se emparejan observaciones válidas mediante desplazamientos sobre el calendario completo. Los tamaños de muestra por rezago se guardan en la tabla; pueden diferir. No se usan rezagos futuros como entradas ni se elige aquí la ventana que maximiza una correlación.

Las asociaciones del precio en nivel pueden depender de tendencias y regímenes. Su interpretación no es causal ni demuestra capacidad predictiva fuera de muestra. Las etiquetas consecutivas comparten retornos futuros, lo que genera dependencia adicional entre observaciones.

### 2.6.6 Cambios de nivel, dispersión y deriva temporal

Las tendencias y varianzas móviles muestran cambios de nivel y dispersión. Se comparan por año las distribuciones de `close` y de la volatilidad futura mediante tamaño de muestra, media, desviación, mediana y percentiles 10 y 90. La tabla siguiente resume las medianas; 2020 y 2025 son periodos parciales, por lo que no equivalen a años completos.

| Activo | Variable | 2020 | 2021 | 2022 | 2023 | 2024 | 2025 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| BNBUSDT | close | 28.2091 | 379.3000 | 299.1000 | 248.8500 | 576.3000 | 642.3900 |
| BNBUSDT | target_pct | 0.6957 | 0.9254 | 0.5802 | 0.3640 | 0.5003 | 0.3807 |
| BTCUSDT | close | 12410.0500 | 47895.9900 | 23148.2050 | 27714.8700 | 64203.0950 | 96786.6800 |
| BTCUSDT | target_pct | 0.4600 | 0.7065 | 0.5176 | 0.3373 | 0.4554 | 0.3879 |
| ETHUSDT | close | 407.0300 | 2612.2400 | 1686.8500 | 1807.7950 | 3087.3100 | 2494.0200 |
| ETHUSDT | target_pct | 0.6747 | 0.8790 | 0.6959 | 0.3782 | 0.5411 | 0.6474 |
| SOLUSDT | close | 2.2224 | 38.6865 | 40.0250 | 22.1000 | 149.0850 | 152.0900 |
| SOLUSDT | target_pct | 1.5346 | 1.3455 | 0.9509 | 0.7685 | 0.7891 | 0.7646 |
| XRPUSDT | close | 0.2578 | 0.8873 | 0.4246 | 0.5016 | 0.5573 | 2.3048 |
| XRPUSDT | target_pct | 0.6816 | 1.0484 | 0.6810 | 0.5021 | 0.5506 | 0.7201 |

`close` se expresa en USDT y `target_pct` en porcentaje de volatilidad. Los cambios de medianas y dispersión evidencian diferencias marginales entre periodos. Un cambio en la distribución de `close` es una señal de cambio en las entradas; no basta para establecer covariate shift en sentido estricto, que además supone estabilidad de la relación condicional con el objetivo. Tampoco un cambio en la distribución del objetivo demuestra concept drift: este exige evidencia de cambio en la relación entre entradas y objetivo.

Se identifican variaciones descriptivas, no puntos de cambio formalmente estimados. No se atribuyen fluctuaciones a eventos externos sin un calendario documentado, ni se crean indicadores de feriados cuya pertinencia no haya sido justificada. La validación por periodos y el seguimiento posterior de errores permitirán evaluar si estos cambios afectan la capacidad predictiva.

### 2.6.7 Panel por activo

Los cinco activos comparten el calendario y el periodo analizado, pero difieren en niveles de precio, amplitud de fluctuaciones y distribución de volatilidad. Cada figura utiliza el mismo procedimiento y conserva su escala propia. Las magnitudes de variación absoluta de precios no son comparables directamente entre activos; la volatilidad en retornos permite una comparación en unidades comunes.

```{figure} ../_static/figures/temporal_BNBUSDT.png
:alt: Serie, agregaciones, ciclos, STL, estacionariedad visual y dependencia temporal de BNBUSDT.

BNBUSDT: diagnóstico temporal de close y volatilidad futura, limitado a DEVELOPMENT.
```

```{figure} ../_static/figures/temporal_BTCUSDT.png
:alt: Serie, agregaciones, ciclos, STL, estacionariedad visual y dependencia temporal de BTCUSDT.

BTCUSDT: diagnóstico temporal de close y volatilidad futura, limitado a DEVELOPMENT.
```

```{figure} ../_static/figures/temporal_ETHUSDT.png
:alt: Serie, agregaciones, ciclos, STL, estacionariedad visual y dependencia temporal de ETHUSDT.

ETHUSDT: diagnóstico temporal de close y volatilidad futura, limitado a DEVELOPMENT.
```

```{figure} ../_static/figures/temporal_SOLUSDT.png
:alt: Serie, agregaciones, ciclos, STL, estacionariedad visual y dependencia temporal de SOLUSDT.

SOLUSDT: diagnóstico temporal de close y volatilidad futura, limitado a DEVELOPMENT.
```

```{figure} ../_static/figures/temporal_XRPUSDT.png
:alt: Serie, agregaciones, ciclos, STL, estacionariedad visual y dependencia temporal de XRPUSDT.

XRPUSDT: diagnóstico temporal de close y volatilidad futura, limitado a DEVELOPMENT.
```

### 2.6.8 Consecuencias para el modelado

La partición y la validación deben respetar el orden cronológico, con ventanas crecientes o deslizantes dentro de DEVELOPMENT y el mismo corte para todos los activos. No se mezclan filas aleatoriamente. El gap debe comprobar la disponibilidad real de las etiquetas y excluir del entrenamiento cualquier horizonte objetivo que invada la validación; no basta con restar un número de filas a un panel de cinco activos.

Los rezagos y estadísticas móviles de entrada se construyen solo con información pasada. Las transformaciones aprendidas se ajustan dentro del entrenamiento de cada fold. La persistencia del precio no garantiza buena predicción de volatilidad, y la menor autocorrelación del retorno no implica ausencia de dependencia en su magnitud. Las diferencias entre periodos motivan informar desempeño por activo y por bloque temporal, además del promedio.

Esta exploración no cambia el horizonte de 24 horas, no selecciona una ventana definitiva y no entrena SVR, persistencia u otros modelos predictivos. TEST permanece reservado. La información temporal apoya el diseño de la validación, sin convertir los patrones retrospectivos en predictores disponibles en el pasado.

### 2.6.9 Reproducibilidad

El procedimiento está en `src/14_temporal_close.py`; las tablas se guardan en `outputs/tables/temporal_*.csv` y la trazabilidad en `outputs/tables/temporal_metadata.json`. Se verifica la conservación de DEVELOPMENT mediante SHA-256. Las pruebas, gráficos y descomposición no modifican los datos originales.

[Notebook ejecutado del componente temporal](../../notebooks/14_temporal_close.ipynb).

## 2.7 Componente espacial

### 2.7.1 Aplicabilidad al dataset

El componente espacial no aplica al alcance actual del proyecto. El encabezado del CSV maestro contiene doce campos: `open_time`, `open`, `high`, `low`, `close`, `volume`, `close_time`, `quote_asset_volume`, `number_of_trades`, `taker_buy_base_asset_volume`, `taker_buy_quote_asset_volume` y `symbol`. No contiene latitud, longitud, geometrías ni unidades geográficas asociadas a las observaciones.

La unidad de observación es una vela horaria de un activo. `symbol` identifica la criptomoneda y las marcas de tiempo sitúan la vela en el calendario; ninguno de estos campos representa una ubicación espacial. La existencia de cinco activos constituye un panel de entidades, no una distribución de puntos geográficos.

### 2.7.2 Técnicas espaciales y justificación

| Elemento de la guía | Aplicabilidad y justificación |
|---|---|
| Coordenadas, CRS, rangos y precisión | No aplica: no hay coordenadas ni un sistema de referencia espacial que validar. |
| Mapas de puntos, densidad y coropletas | No aplica: las observaciones no tienen localización ni una unidad territorial de agregación. |
| Cobertura, distancias y agrupamiento espacial | No aplica: no se pueden definir distancias geográficas, vecinos o zonas sin cobertura a partir de estos campos. |
| Pesos espaciales, Moran, LISA, hotspots y semivariograma | No aplica: no existe una estructura de vecindad geográfica documentada. |
| Heterogeneidad regional y MAUP | No aplica: los activos no son regiones y no se utilizan unidades de agregación territorial. |
| Características espaciales: H3, geohash, distancias y rezagos espaciales | No aplica: no hay ubicaciones con las que construir estas variables. |
| Validación por bloques espaciales y buffers | No aplica: la estructura de dependencia pertinente para este proyecto es temporal y por activo. |

No se asignan ubicaciones a las criptomonedas ni a las velas a partir de la sede del proveedor o de otra referencia externa. Esa asignación no representaría la localización de las observaciones del dataset. Tampoco se interpreta la correlación entre activos como autocorrelación espacial.

### 2.7.3 Consecuencias para el modelado e interpretación

Se mantiene `close` como única variable explicativa original y se conserva la validación cronológica descrita en 2.6, con los activos alineados por timestamp y control del horizonte de las etiquetas. No se incorporan variables, distancias ni particiones espaciales.

No se presentan conclusiones sobre patrones geográficos o generalización a zonas nuevas, porque los datos no permiten evaluarlos. La ausencia de información espacial delimita el alcance del análisis; no implica que se haya demostrado ausencia de efectos geográficos en el mercado.

### 2.7.4 Verificación y reproducibilidad

La comprobación se limita a leer el encabezado del CSV maestro. No se ejecutan análisis espaciales, no se modifican datos y no se consulta TEST. El notebook de esta sección documenta la justificación de no aplicabilidad y no requiere celdas de cálculo.

[Notebook del alcance espacial](../../notebooks/15_spatial_scope.ipynb).

## 2.8 Componente espacio-temporal

### 2.8.1 Aplicabilidad al dataset

El componente espacio-temporal no aplica al alcance actual. El dataset contiene marcas de tiempo y un identificador de activo, pero no coordenadas, geometrías ni zonas geográficas. Por tanto, permite analizar la evolución temporal de las criptomonedas, pero no su distribución o desplazamiento en el espacio geográfico.

`symbol` identifica entidades financieras, no celdas o regiones. Comparar series de BTC, ETH, BNB, XRP y SOL constituye un análisis temporal de panel; no convierte el dataset en un conjunto de observaciones geográficas. La ausencia de coordenadas documentada en 2.7 se mantiene en este apartado.

### 2.8.2 Técnicas y alcance

| Elemento de la guía | Aplicabilidad y justificación |
|---|---|
| Mapas por periodo y animaciones geográficas | No aplica: las observaciones no tienen una ubicación que representar en cada periodo. |
| Series por celda o zona y comparación territorial | No aplica: no existen unidades geográficas. Las series por activo se presentan en 2.6. |
| Interacción espacio-tiempo, hotspots emergentes y prueba de Knox | No aplica: no es posible definir proximidad geográfica ni desplazamientos de concentraciones espaciales. |
| Autocorrelación espacio-temporal | No aplica: hay orden temporal, pero no una estructura de vecindad espacial documentada. La dependencia temporal se examina en 2.6. |
| Particiones por bloques espacio-temporales | No aplica: no existen bloques geográficos que combinar con los periodos de evaluación. |

No se asignan coordenadas artificiales a los activos ni se interpreta su correlación como proximidad geográfica. No se generan mapas, animaciones ni resultados de pruebas espacio-temporales sin datos que los sustenten.

### 2.8.3 Objetivo de generalización y partición

El objetivo es pronosticar la volatilidad futura a 24 horas de los mismos cinco activos, utilizando información histórica disponible al predecir. No se evalúa generalización a zonas nuevas ni a criptomonedas no observadas durante el desarrollo.

Se conserva la partición cronológica DEVELOPMENT/TEST y la validación temporal interna, con los cinco activos alineados por timestamp. La comparación de modelos utilizará los mismos bloques, horizontes y observaciones elegibles. No se añade una partición espacial ni se presentan resultados de esquemas territoriales inexistentes.

### 2.8.4 Interpretación y riesgo de fuga

Los riesgos pertinentes son temporales: usar cierres futuros, permitir que las etiquetas de entrenamiento invadan la validación, ajustar transformaciones con datos posteriores o distribuir aleatoriamente ventanas que comparten información. Se mantienen los controles de disponibilidad, purga de etiquetas y ajuste dentro de cada entrenamiento descritos en 2.5 y 2.6.

La dependencia entre activos puede coexistir con la dependencia temporal, pero no constituye evidencia de vecindad espacial. La falta de ubicaciones impide evaluar fuga por proximidad geográfica; no demuestra que el fenómeno financiero carezca de factores geográficos externos.

En consecuencia, esta sección delimita la no aplicabilidad del componente espacio-temporal y mantiene el análisis temporal por activo como enfoque del proyecto. No se ejecutan análisis adicionales, no se modifican datos y no se consulta TEST.

[Notebook del alcance espacio-temporal](../../notebooks/16_spatiotemporal_scope.ipynb). Este documento metodológico no requiere celdas de cálculo.

## 2.9 Preprocesamiento

### 2.9.1 Pipeline y alcance

La entrada original continúa siendo `close`. Se implementa un pipeline de preprocesamiento con `StandardScaler`, independiente para cada activo en la comprobación de esta sección. Cada instancia se ajusta exclusivamente con las observaciones elegibles del entrenamiento cronológico; la validación utiliza únicamente `transform`. No se reutilizan escaladores del EDA calculados sobre todo DEVELOPMENT.

El [Pipeline de scikit-learn](https://scikit-learn.org/stable/modules/generated/sklearn.pipeline.Pipeline.html) agrupa las transformaciones que se ajustan. El parseo de fechas, la alineación horaria, la construcción determinista de la etiqueta y la selección de ventanas válidas se realizan antes, conservando la correspondencia entre X, y y fechas. Estas operaciones no estiman parámetros con validación. El escalado aprendido queda dentro del pipeline. Cuando se entrene el regresor, se integrará como etapa final y el pipeline completo se ajustará de nuevo en cada fold.

La ejecución actual verifica el preprocesamiento con una columna de cierre, sin entrenar OLS, SVR ni persistencia. Esta comprobación no fija la longitud definitiva de entrada de los modelos y no genera métricas de desempeño predictivo.

### 2.9.2 Faltantes y elegibilidad temporal

Los huecos horarios se mantienen en una rejilla de calendario. No se rellenan con ceros, medias, interpolaciones ni observaciones posteriores. No se ha establecido un mecanismo de ausencia que justifique una imputación específica; además, imputar precios alteraría los retornos y la volatilidad construida a partir de ellos.

Los cierres con final horario irregular se enmascaran para la construcción analítica, sin modificar los registros originales. Una entrada histórica requiere todos sus cierres válidos; una etiqueta requiere 25 cierres consecutivos para obtener los 24 retornos futuros. Se excluyen del conjunto supervisado las ventanas incompletas y se mantienen sus fechas para auditar la cobertura. Esto no elimina filas de los archivos fuente ni demuestra que la muestra elegible esté libre de sesgo de selección.

La exclusión de etiquetas inválidas afecta al entrenamiento y a la evaluación, no obliga a conocer el futuro para emitir una predicción: en operación, la elegibilidad de la entrada se determina solo con el histórico disponible. La etiqueta se comprueba cuando el horizonte se realiza. Los modelos se compararán sobre las mismas observaciones evaluables, con motivos de exclusión documentados.

### 2.9.3 Transformaciones y escalado

Se adopta como configuración inicial el cierre sin transformación logarítmica como entrada y su estandarización: a cada columna se le resta la media de entrenamiento y se divide por su escala de entrenamiento. [StandardScaler](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html) utiliza la desviación con divisor n y es sensible a valores extremos. La estandarización no vuelve normal ni estacionaria la serie y no elimina la multicolinealidad entre rezagos.

La transformación logarítmica se utiliza para construir los retornos del objetivo, no se aplica automáticamente a `close` como predictor. El objetivo conserva la desviación estándar centrada de 24 retornos, `ddof=0`, expresada en porcentaje y sin anualización. No se escala la etiqueta en esta comprobación. Una transformación posterior de entradas o del objetivo deberá evaluarse dentro de la validación temporal; cualquier transformación del objetivo exigirá volver a su escala común al calcular métricas.

La variación de niveles entre entrenamiento y validación observada en 2.6 puede producir valores estandarizados alejados de cero. No se centra de nuevo la validación para ocultar ese cambio ni se recortan sus valores a un rango arbitrario.

### 2.9.4 Codificación y tratamiento de valores extremos

No se requiere codificación categórica: `symbol` se utiliza para separar activos y las fechas para ordenar y dividir observaciones. No se incluyen como columnas predictoras. Los otros campos de mercado permanecen fuera del alcance acordado.

Los extremos válidos de `close` se conservan. Los hallazgos del EDA no justifican asumir que todo precio extremo sea un error. No se aplica eliminación por IQR ni winsorización y no se reutilizan umbrales estimados sobre todo DEVELOPMENT. Se exige que los precios utilizados sean finitos y positivos para construir retornos logarítmicos válidos. Si posteriormente se compara un escalado robusto o un tratamiento adicional, sus parámetros se ajustarán dentro de cada entrenamiento y su elección dependerá de validación interna.

### 2.9.5 Ingeniería temporal

La función `prepare_close` construye ventanas de L cierres: el último cierre observado y sus L−1 rezagos horarios. L debe especificarse explícitamente; no se selecciona en esta sección. Los desplazamientos se realizan sobre el calendario horario de cada activo, de modo que un hueco no se convierta artificialmente en una observación consecutiva. El ancla identifica la apertura de la última vela utilizada y la predicción se sitúa después de su cierre.

La comprobación ejecutada utiliza L=1 únicamente para verificar el pipeline básico. Si se amplía L, cada rezago será una columna y se ajustará su escala con el entrenamiento. La fuente seguirá siendo `close`, aunque puedan aparecer dependencias fuertes entre columnas. No se añaden automáticamente retornos, volatilidad pasada, medias móviles, senos, cosenos ni indicadores de calendario como entradas. La estacionalidad semanal débil de 2.6 no aporta por sí sola evidencia suficiente para incorporarlos. No existen variables espaciales aplicables.

### 2.9.6 Decisiones vinculadas al EDA

| Hallazgo | Decisión de preprocesamiento |
|---|---|
| Huecos y cierres irregulares | Mantener el calendario y excluir ventanas incompletas; no imputar. |
| Diferencias de nivel y escala entre activos y periodos | Verificar escalado por activo, ajustado solo en entrenamiento. |
| Precios extremos que pueden ser movimientos reales | Conservar extremos válidos, sin recorte automático. |
| Persistencia y redundancia entre rezagos | Mantener L explícito; comparar ventanas dentro de validación, sin selección definitiva aquí. |
| Componente semanal débil en el tramo estudiado | No incorporar características de calendario automáticamente. |
| Ausencia de predictores categóricos y coordenadas | No usar codificación categórica ni ingeniería espacial. |
| Etiquetas que abarcan 24 horas futuras | Purgar por la fecha final de la etiqueta, antes de ajustar el pipeline. |

### 2.9.7 Verificación ejecutada

Se utiliza la frontera interna **2024-07-09 20:00 UTC**, dentro de DEVELOPMENT, como en la auditoría 2.5. Se excluyen del entrenamiento las etiquetas cuya última vela objetivo alcanza o supera esa frontera. Se ajustan cinco pipelines de escalado, uno por activo, con una sola columna de cierre. Las cifras siguientes son parámetros y tamaños de muestra, no resultados de un modelo predictivo.

| Activo | Filas entrenamiento | Filas validación | Etiquetas purgadas | Media entrenamiento (USDT) | Escala entrenamiento (USDT) |
|---|---:|---:|---:|---:|---:|
| BNBUSDT | 33972 | 8543 | 24 | 316.821586 | 159.145884 |
| BTCUSDT | 33972 | 8543 | 24 | 36050.904935 | 16026.223094 |
| ETHUSDT | 33972 | 8543 | 24 | 2140.664848 | 1013.130489 |
| SOLUSDT | 33997 | 8543 | 24 | 61.749436 | 60.355696 |
| XRPUSDT | 33997 | 8543 | 24 | 0.587047 | 0.271218 |

Las comprobaciones confirman que la media aprendida corresponde al entrenamiento, que transformar validación no cambia los parámetros, que las salidas son finitas y que la transformación inversa recupera los cierres de validación. La media estandarizada de entrenamiento es aproximadamente cero; no se exige que la de validación lo sea. Esta verificación de una frontera no sustituye las comprobaciones de los folds definitivos.

### 2.9.8 Reproducibilidad

El procedimiento está en `src/17_preprocessing_close.py`. La auditoría se guarda en `outputs/tables/preprocessing_close_audit.csv` y la trazabilidad en `outputs/tables/preprocessing_metadata.json`. El script verifica el SHA-256 de DEVELOPMENT antes y después de ejecutarse. No se modifica ningún dataset ni se lee TEST. Los pipelines de esta comprobación no se guardan como modelos finales ni se reutilizan para otros folds.

[Notebook ejecutado de preprocesamiento](../../notebooks/17_preprocessing_close.ipynb).
