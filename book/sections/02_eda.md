# 2. Análisis Exploratorio de Datos (EDA)

El análisis exploratorio utiliza exclusivamente DEVELOPMENT, del 11 de agosto de 2020 a las 06:00 UTC al 1 de julio de 2025 a las 18:00 UTC. TEST permanece reservado desde la partición cronológica inicial. Los resultados describen los datos disponibles para desarrollar el modelo. Los ajustes diagnósticos se restringen al entrenamiento temporal de cada análisis; la comparación del SVR con Persistence se presenta en la sección 3. TEST no se evalúa.

Las decisiones posteriores de imputación, transformación, escalado, selección de variables o tratamiento de extremos deberán ajustarse únicamente con la porción de entrenamiento de cada partición temporal. Las estadísticas globales de DEVELOPMENT que se presentan aquí son descriptivas y no constituyen parámetros de preprocesamiento para validación.

## 2.1 Análisis de la variable objetivo

### 2.1.1 Definición y disponibilidad temporal

La variable objetivo es la **desviación estándar de los retornos horarios de las próximas 24 horas**, una magnitud continua. Se adopta la fórmula de volatilidad del profesor, adaptando la frecuencia de días a horas y fijando una ventana de 24 retornos. No es la volatilidad acumulada del retorno de un día.

Sea $C_{i,s}$ el cierre de la vela del activo $i$ cuya apertura UTC es $s$. Se define:

$$
r_{i,s}=\ln\left(\frac{C_{i,s}}{C_{i,s-1}}\right),\qquad
\bar r^+_{i,s}=\frac1{24}\sum_{j=1}^{24}r_{i,s+j},
$$

$$
\boxed{y_{i,s}=100\sqrt{\frac1{24}\sum_{j=1}^{24}(r_{i,s+j}-\bar r^+_{i,s})^2}}.
$$

El divisor es **24**, no 23: se utiliza `ddof=0`. El factor 100 expresa el resultado en porcentaje; no se anualiza ni se multiplica por $\sqrt{24}$. Esta es la misma definición y escala del objetivo de las secciones 2.3 y 3.

La predicción se emite **después del cierre de la vela ancla $s$ y una vez disponible ese cierre**, alrededor de $s+1$ hora. El objetivo utiliza los retornos $r_{i,s+1},\ldots,r_{i,s+24}$, construidos a partir de los 25 cierres $C_{i,s},\ldots,C_{i,s+24}$. Solo se conoce después del cierre de la última vela del horizonte, alrededor de $s+25$ horas. Por ejemplo, para el ancla de apertura 10:00 se predice alrededor de las 11:00 y se evalúan los 24 retornos posteriores. No se predice a la apertura de la vela ancla usando su cierre futuro.

En la notación del profesor, $\sigma_t$ usa los retornos $r_{t-24},\ldots,r_{t-1}$. Con $t=s+1$, la referencia histórica disponible es $\sigma_{s+1}$ y el objetivo aquí almacenado es $y_{i,s}=100\sigma_{i,s+25}$. Así se reconcilia la notación de la fórmula con el índice `open_time` del código sin desplazar las observaciones una hora por error.

Las frecuencias y el desbalance de clases no aplican: el problema es de regresión temporal. Las variables explicativas utilizan únicamente la vela ya cerrada o información anterior.

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

Como sensibilidad, aceptar cierres irregulares manteniendo las demás condiciones produciría 42,569 objetivos por activo. Añadiría 30 ventanas en BTC, ETH y BNB, y 5 en XRP y SOL. La diferencia absoluta máxima entre las medias estricta y relajada es 0.000121 puntos porcentuales; los máximos no cambian. Se mantiene la regla conservadora por consistencia temporal, no porque esta sensibilidad demuestre la validez de los cierres irregulares. Véase la [tabla de sensibilidad](../../outputs/tables/eda_target_sensitivity.csv).

### 2.1.3 Distribución, asimetría y valores extremos

| Activo | Mínimo (%) | Mediana (%) | Media (%) | P95 (%) | P99 (%) | Máximo (%) |
| --- | --- | --- | --- | --- | --- | --- |
| BTCUSDT | 0.0445 | 0.4854 | 0.5507 | 1.1818 | 1.7748 | 3.3765 |
| ETHUSDT | 0.0542 | 0.6196 | 0.7129 | 1.5124 | 2.4151 | 5.1053 |
| BNBUSDT | 0.0787 | 0.5589 | 0.6875 | 1.5804 | 2.6840 | 5.9566 |
| XRPUSDT | 0.1144 | 0.6767 | 0.8756 | 2.1890 | 3.6531 | 7.9232 |
| SOLUSDT | 0.1853 | 0.9605 | 1.1413 | 2.4680 | 3.9244 | 7.0645 |

No se observan objetivos iguales a cero. SOL presenta la mayor mediana (0.9605%) y BTC la menor (0.4854%). XRP alcanza el máximo más alto (7.9232%), aunque su mediana es inferior a la de SOL. El nivel habitual y la intensidad de los episodios extremos deben distinguirse. Los cuartiles y la desviación estándar están en el [resumen completo](../../outputs/tables/eda_target_summary.csv). Las pequeñas diferencias de XRP y SOL respecto de 2.3.4 obedecen a que aquí se usan todas las etiquetas válidas de cada activo y allí la intersección de timestamps de los cinco.

```{figure} ../_static/figures/eda_target_distribution.png
:alt: Histogramas, boxplots y distribución del logaritmo de la volatilidad futura a 24 horas, separados por activo y calculados solo en DEVELOPMENT.

Distribución del objetivo por activo. La primera columna muestra frecuencias en escala logarítmica; la segunda utiliza boxplots con cercas de 1.5 IQR; la tercera muestra el logaritmo natural del objetivo como diagnóstico visual. No se transforma la variable utilizada en las tablas originales.
```

| Activo | Asimetría | Exceso de curtosis | Señalados por IQR | Porcentaje |
| --- | --- | --- | --- | --- |
| BTCUSDT | 2.036 | 7.440 | 2,109 | 4.96% |
| ETHUSDT | 2.228 | 9.064 | 2,006 | 4.72% |
| BNBUSDT | 3.221 | 17.822 | 2,467 | 5.80% |
| XRPUSDT | 3.162 | 15.250 | 3,242 | 7.62% |
| SOLUSDT | 2.608 | 10.972 | 2,294 | 5.39% |

La asimetría es positiva en todos los activos y la media supera la mediana. El exceso de curtosis va de 7.440 a 17.822, compatible con colas empíricas pronunciadas respecto de una referencia normal. No demuestra una ley de colas ni independencia. BNB y XRP presentan los mayores excesos de curtosis; XRP tiene la mayor proporción señalada por IQR.

Los valores fuera de las cercas $Q_1-1.5\,IQR$ y $Q_3+1.5\,IQR$ son candidatos a extremos, no errores demostrados. Se conservan todos los objetivos válidos, incluidos los elevados. El IQR no se utiliza aquí para recortar ni winsorizar la variable objetivo.

### 2.1.4 Necesidad de transformación

El logaritmo permite examinar si la escala positiva y la asimetría del objetivo se representan de forma más equilibrada. La tabla compara la asimetría y el exceso de curtosis antes y después del logaritmo, únicamente como diagnóstico.

| Activo | Asimetría original | Asimetría de ln(y) | Exceso de curtosis original | Exceso de curtosis de ln(y) |
| --- | --- | --- | --- | --- |
| BTCUSDT | 2.036 | -0.271 | 7.440 | 0.580 |
| ETHUSDT | 2.228 | -0.126 | 9.064 | 0.437 |
| BNBUSDT | 3.221 | 0.175 | 17.822 | 0.257 |
| XRPUSDT | 3.162 | 0.454 | 15.250 | 0.356 |
| SOLUSDT | 2.608 | 0.313 | 10.972 | 0.283 |

Una menor asimetría no demuestra normalidad ni garantiza mejores pronósticos. No se adopta una transformación definitiva. Box–Cox es una alternativa para valores estrictamente positivos, pero su parámetro no se estima globalmente en esta etapa: si se evalúa posteriormente, deberá ajustarse dentro de cada partición de entrenamiento. Lo mismo aplica a cualquier desplazamiento utilizado si aparecen ceros en datos futuros.

Si se modelara el logaritmo del objetivo, minimizar el error en esa escala cambiaría la importancia relativa de los errores grandes y pequeños. Además, exponenciar un pronóstico en escala logarítmica no produce automáticamente la media condicional en la escala original. La elección deberá evaluarse cronológicamente y las métricas finales deberán incluir la escala original.

### 2.1.5 Comportamiento temporal

```{figure} ../_static/figures/eda_target_time.png
:alt: Evolución horaria de la volatilidad futura a 24 horas por activo y su mediana mensual en DEVELOPMENT.

Evolución del objetivo. El eje horizontal identifica la apertura de la vela de referencia, no el momento en que se conoce el objetivo. La línea naranja resume la mediana mensual de los objetivos válidos; es descriptiva y no se utiliza como predictor.
```

| Activo | Apertura de referencia (UTC) | Máximo (%) |
| --- | --- | --- |
| BTCUSDT | 2021-05-19 06:00 | 3.3765 |
| ETHUSDT | 2021-05-19 09:00 | 5.1053 |
| BNBUSDT | 2021-05-19 03:00 | 5.9566 |
| XRPUSDT | 2021-02-01 02:00 | 7.9232 |
| SOLUSDT | 2022-11-09 14:00 | 7.0645 |

Los máximos de BTC, ETH y BNB corresponden a ventanas iniciadas el 19 de mayo de 2021; los de XRP y SOL ocurren en fechas diferentes. Las fechas identifican ventanas futuras de 24 horas y no un único retorno ni una causa económica comprobada. La evolución muestra episodios de distinta intensidad, lo que exige evaluar el desempeño en varios bloques cronológicos.

| Activo | Correlación del objetivo a 1h | Correlación del objetivo a 24h |
| --- | --- | --- |
| BTCUSDT | 0.989 | 0.596 |
| ETHUSDT | 0.991 | 0.649 |
| BNBUSDT | 0.993 | 0.709 |
| XRPUSDT | 0.991 | 0.625 |
| SOLUSDT | 0.992 | 0.687 |

Las correlaciones se calculan por pares disponibles sobre la cuadrícula horaria, sin comprimir los huecos. La elevada asociación a una hora está influida mecánicamente por el solapamiento de 23 retornos entre etiquetas. A 24 horas las ventanas de retornos ya no se solapan, aunque persiste asociación descriptiva. Ninguno de estos coeficientes sustituye una evaluación fuera de muestra.

No se realiza análisis espacial: el dataset carece de coordenadas geográficas. Los activos son entidades del panel y no ubicaciones espaciales.

### 2.1.6 Relación con las variables explicativas disponibles

Como exploración inicial se compara el objetivo con las nueve variables numéricas de la vela cerrada, el retorno de esa hora y la volatilidad pasada $100\,\operatorname{std}(r_{i,s-23},\ldots,r_{i,s};\mathrm{ddof}=0)$. Esta última centra los retornos en su media, utiliza divisor 24 y requiere 25 cierres consecutivos convencionales. Es la referencia disponible de Persistence en la sección 3; aquí solo se estudia su asociación, sin evaluar pronósticos.

Las fechas actúan como índices temporales y `symbol` identifica el activo. No se asignan códigos numéricos arbitrarios a los símbolos para calcular correlaciones. Las asociaciones se calculan por activo y con los pares disponibles para cada variable; por eso su tamaño de muestra puede variar. Las cifras completas se incluyen en la [tabla de correlaciones y tamaños de muestra](../../outputs/tables/eda_target_correlations.csv).

```{figure} ../_static/figures/eda_target_correlations.png
:alt: Correlaciones Pearson y Spearman entre cada variable disponible y la volatilidad futura, por activo.

Pearson resume asociación lineal y Spearman asociación monótona. Los coeficientes son descriptivos; no se calculan valores p suponiendo independencia de las observaciones.
```

```{figure} ../_static/figures/eda_target_relations.png
:alt: Densidad conjunta del objetivo frente a volatilidad pasada y volumen actual para cada activo.

Relaciones con la volatilidad pasada y el volumen en USDT. Cada panel incorpora todos los pares disponibles; los hexágonos más oscuros indican mayor frecuencia, en escala logarítmica. Las escalas numéricas de los activos se muestran por separado.
```

La volatilidad pasada de 24 horas muestra correlaciones de Pearson entre 0.596 y 0.709, y de Spearman entre 0.582 y 0.697. Sus retornos no se solapan con los futuros, aunque comparten el cierre de frontera. La asociación motiva la referencia Persistence, cuyo desempeño se evalúa separadamente en la sección 3.

El volumen en USDT presenta asociaciones Spearman distintas por activo (BTCUSDT: 0.387, ETHUSDT: 0.505, BNBUSDT: 0.531, XRPUSDT: 0.565, SOLUSDT: 0.030). Para el número de operaciones, Spearman va de 0.007 a 0.541. Estas diferencias no establecen utilidad predictiva fuera de muestra; las asociaciones de todos los campos de volumen se conservan en la tabla completa.

Los precios OHLC tienen asociaciones descriptivas que pueden reflejar tendencias y regímenes; no se interpretan como causalidad. El retorno horario con signo presenta Pearson entre -0.041 y 0.002. Su asociación lineal pequeña no descarta relaciones no lineales ni relaciones con su magnitud absoluta. Los diagramas muestran dispersión y extremos que un coeficiente aislado no resume.

Estos resultados no constituyen una selección definitiva de predictores ni un análisis multivariado condicional. Las asociaciones con precios y volúmenes pueden depender del periodo y de cambios de régimen; una correlación no implica causalidad ni capacidad predictiva fuera de muestra. La selección y las transformaciones deberán contrastarse después mediante validación temporal, sin TEST.

### 2.1.7 Implicaciones para métricas y validación

Las métricas comunes son RMSE, MAPE y $R^2$; MAE es complementaria. RMSE y MAE se expresan en puntos porcentuales de volatilidad, MAPE en porcentaje y $R^2$ es adimensional. Se reportan por activo y fold antes del promedio con pesos iguales. MAPE es indefinido con objetivos cero y sensible a valores cercanos a cero; no se añaden denominadores artificiales. $R^2$ requiere variabilidad del objetivo. Esta sección describe el objetivo; las métricas predictivas se presentan en la sección 3.

La validación deberá ser cronológica, con ventanas de entrenamiento anteriores a las de validación, y mantener todos los activos de una misma fecha en el mismo bloque temporal. No se utilizará una partición aleatoria de filas. La [documentación de validación temporal de scikit-learn](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.TimeSeriesSplit.html) describe la separación ordenada y el uso de un intervalo entre entrenamiento y evaluación; en este panel, ese intervalo debe aplicarse a timestamps globales, no a un número de filas mezcladas entre activos.

Los objetivos de horas consecutivas comparten 23 de sus 24 retornos. Por ello, la dependencia entre etiquetas no puede interpretarse como evidencia automática de capacidad predictiva y el número de filas no equivale a observaciones independientes. En cada frontera se excluirán del entrenamiento las etiquetas cuyo horizonte alcance el inicio de la validación: una separación conservadora de 24 horas entre las horas de referencia evita el solapamiento de retornos futuros. Además, toda etiqueta de entrenamiento debe estar disponible antes de realizar la primera predicción de validación. Las ventanas se comprobarán por tiempo real, especialmente alrededor de huecos.

Todos los parámetros aprendidos de los datos deberán ajustarse de nuevo dentro de cada entrenamiento. TEST se mantendrá reservado para la evaluación final del procedimiento ya fijado.

### 2.1.8 Reproducibilidad

El cálculo está en `src/08_eda_target_development.py`; `src/08_render_target_report.py` sincroniza estas tablas e interpretaciones con los resultados guardados. Ejecutar ambos scripts en ese orden desde el entorno de `requirements.txt`. Los resultados y metadatos se guardan en `outputs/tables/eda_target_*`; la tabla derivada permanece separada de los datos originales. Cada objetivo válido se contrasta con el cálculo directo centrado de sus 24 retornos, con divisor 24; se verifica el borde de DEVELOPMENT y su SHA-256. El notebook ejecuta el cálculo y muestra las cuatro figuras. No se consulta TEST ni se reentrenan modelos para esta corrección.


## 2.2 Análisis unidimensional de las variables originales

El análisis se centra en el precio de cierre (`close`) de los cinco activos, usando exclusivamente DEVELOPMENT: 214,165 observaciones, con 42,833 por activo. De esta variable se obtienen los retornos logarítmicos necesarios para construir la volatilidad futura, que continúa siendo la variable objetivo del pronóstico. Los apartados 2.2.1–2.2.7 detallan close; los apartados siguientes completan el análisis de las demás columnas, incluidas las fechas y los símbolos.

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

Los resultados corresponden al cálculo previo sobre DEVELOPMENT realizado mediante `src/09_univariate_development.py`. Se conservan sus filas de `close` en `outputs/tables/univariate_close_summary.csv`, sin recalcular estadísticas. La figura es `book/_static/figures/univariate_close.png`. El [notebook centrado en close](../../notebooks/10_close_univariate.ipynb) carga y presenta esos resultados guardados con su interpretación. Los resultados de las restantes columnas se incorporan a continuación como parte del EDA completo.

### 2.2.8 Cobertura de las doce columnas originales

El EDA univariado cubre las nueve columnas numéricas, las dos marcas temporales y `symbol`. La inclusión de una variable en el EDA no implica incorporarla al SVR. Todos los resúmenes corresponden a DEVELOPMENT antes de filtrar ventanas; no se leen valores de TEST.

| Variable | Tipo | Cardinalidad global | Nulos |
| --- | --- | --- | --- |
| open_time | Temporal | 42833 | 0 |
| open | Numérica continua | 136971 | 0 |
| high | Numérica continua | 129249 | 0 |
| low | Numérica continua | 130210 | 0 |
| close | Numérica continua | 136956 | 0 |
| volume | Numérica continua | 214107 | 0 |
| close_time | Temporal | 42852 | 0 |
| quote_asset_volume | Numérica continua | 214156 | 0 |
| number_of_trades | Numérica discreta | 92879 | 0 |
| taker_buy_base_asset_volume | Numérica continua | 214041 | 0 |
| taker_buy_quote_asset_volume | Numérica continua | 214156 | 0 |
| symbol | Categórica nominal | 5 | 0 |

Los resúmenes numéricos se separan por activo: tienen 42.833 observaciones cada uno. Se usa desviación estándar descriptiva con `ddof=1`, percentiles interpolados linealmente y exceso de curtosis de Fisher (referencia normal cero). El objetivo conserva `ddof=0`. Los histogramas tienen eje de frecuencias logarítmico; no se transforman las observaciones. Los boxplots usan 1,5 IQR. Los umbrales globales de DEVELOPMENT son descriptivos y no se reutilizan para limpiar los folds.

[Tabla completa con mínimos, máximos, cuartiles, percentiles 1/5/95/99, ceros y límites IQR](../../outputs/tables/univariate_summary.csv). No se aplican pruebas marginales de normalidad: no son un supuesto del SVR y la dependencia temporal impide interpretar sus p-valores iid de manera convencional.

#### Precio de apertura: `open`

Unidad: USDT por unidad del activo.

| Activo | Media | Mediana | DE | P5 | P95 |
| --- | --- | --- | --- | --- | --- |
| BNBUSDT | 376.39 | 328 | 187.84 | 29.043 | 661.28 |
| BTCUSDT | 45803 | 40552 | 25337 | 15372 | 98639 |
| ETHUSDT | 2247.8 | 2110.2 | 967.6 | 450.02 | 3888.9 |
| SOLUSDT | 82.965 | 46.313 | 70.698 | 2.4122 | 207.09 |
| XRPUSDT | 0.80467 | 0.5577 | 0.6362 | 0.27124 | 2.356 |

| Activo | Asimetría | Exceso curtosis | Señaladas IQR | % IQR |
| --- | --- | --- | --- | --- |
| BNBUSDT | -0.065 | -0.822 | 0 | 0.00 |
| BTCUSDT | 0.803 | -0.188 | 0 | 0.00 |
| ETHUSDT | 0.128 | -0.478 | 0 | 0.00 |
| SOLUSDT | 0.556 | -1.021 | 0 | 0.00 |
| XRPUSDT | 1.870 | 2.528 | 5807 | 13.56 |

La mayor proporción señalada por IQR es 13.56 % en XRPUSDT. Estas marcas describen colas respecto al rango intercuartílico; no demuestran errores ni justifican eliminar registros. Los niveles de precio mezclan periodos y regímenes; su dispersión no equivale a la volatilidad de retornos. Comparar precios absolutos entre activos no mide cuál tiene mayor riesgo.

```{figure} ../_static/figures/univariate_open.png
:alt: Histogramas y boxplots de open por activo.

Precio de apertura: cinco activos, sin mezclar sus escalas.
```

#### Precio máximo: `high`

Unidad: USDT por unidad del activo.

| Activo | Media | Mediana | DE | P5 | P95 |
| --- | --- | --- | --- | --- | --- |
| BNBUSDT | 378.26 | 329.7 | 188.62 | 29.215 | 664.17 |
| BTCUSDT | 45997 | 40785 | 25423 | 15441 | 98988 |
| ETHUSDT | 2260.1 | 2126.5 | 972.79 | 453.55 | 3910.4 |
| SOLUSDT | 83.637 | 46.9 | 71.227 | 2.4495 | 208.99 |
| XRPUSDT | 0.81068 | 0.5615 | 0.64155 | 0.27347 | 2.3731 |

| Activo | Asimetría | Exceso curtosis | Señaladas IQR | % IQR |
| --- | --- | --- | --- | --- |
| BNBUSDT | -0.068 | -0.823 | 0 | 0.00 |
| BTCUSDT | 0.799 | -0.194 | 0 | 0.00 |
| ETHUSDT | 0.128 | -0.479 | 0 | 0.00 |
| SOLUSDT | 0.556 | -1.017 | 0 | 0.00 |
| XRPUSDT | 1.867 | 2.517 | 5846 | 13.65 |

La mayor proporción señalada por IQR es 13.65 % en XRPUSDT. Estas marcas describen colas respecto al rango intercuartílico; no demuestran errores ni justifican eliminar registros. Los niveles de precio mezclan periodos y regímenes; su dispersión no equivale a la volatilidad de retornos. Comparar precios absolutos entre activos no mide cuál tiene mayor riesgo.

```{figure} ../_static/figures/univariate_high.png
:alt: Histogramas y boxplots de high por activo.

Precio máximo: cinco activos, sin mezclar sus escalas.
```

#### Precio mínimo: `low`

Unidad: USDT por unidad del activo.

| Activo | Media | Mediana | DE | P5 | P95 |
| --- | --- | --- | --- | --- | --- |
| BNBUSDT | 374.34 | 326.2 | 186.98 | 28.864 | 658.01 |
| BTCUSDT | 45601 | 40342 | 25252 | 15306 | 98300 |
| ETHUSDT | 2234.7 | 2096.1 | 961.91 | 447.38 | 3864.1 |
| SOLUSDT | 82.272 | 45.561 | 70.149 | 2.3751 | 205.25 |
| XRPUSDT | 0.79832 | 0.55318 | 0.63072 | 0.26854 | 2.3389 |

| Activo | Asimetría | Exceso curtosis | Señaladas IQR | % IQR |
| --- | --- | --- | --- | --- |
| BNBUSDT | -0.061 | -0.821 | 0 | 0.00 |
| BTCUSDT | 0.807 | -0.180 | 0 | 0.00 |
| ETHUSDT | 0.127 | -0.476 | 0 | 0.00 |
| SOLUSDT | 0.556 | -1.024 | 0 | 0.00 |
| XRPUSDT | 1.874 | 2.540 | 5751 | 13.43 |

La mayor proporción señalada por IQR es 13.43 % en XRPUSDT. Estas marcas describen colas respecto al rango intercuartílico; no demuestran errores ni justifican eliminar registros. Los niveles de precio mezclan periodos y regímenes; su dispersión no equivale a la volatilidad de retornos. Comparar precios absolutos entre activos no mide cuál tiene mayor riesgo.

```{figure} ../_static/figures/univariate_low.png
:alt: Histogramas y boxplots de low por activo.

Precio mínimo: cinco activos, sin mezclar sus escalas.
```

#### Volumen base: `volume`

Unidad: Unidades del activo base.

| Activo | Media | Mediana | DE | P5 | P95 |
| --- | --- | --- | --- | --- | --- |
| BNBUSDT | 50242 | 21400 | 87244 | 4166.1 | 1.8598e+05 |
| BTCUSDT | 3330 | 1732.2 | 4743 | 386.57 | 11795 |
| ETHUSDT | 25975 | 17412 | 28345 | 4588 | 75060 |
| SOLUSDT | 1.8677e+05 | 1.2236e+05 | 2.1638e+05 | 33978 | 5.4414e+05 |
| XRPUSDT | 1.9753e+07 | 1.2131e+07 | 2.9695e+07 | 3.1277e+06 | 5.9985e+07 |

| Activo | Asimetría | Exceso curtosis | Señaladas IQR | % IQR |
| --- | --- | --- | --- | --- |
| BNBUSDT | 6.200 | 67.183 | 4356 | 10.17 |
| BTCUSDT | 4.815 | 45.974 | 4359 | 10.18 |
| ETHUSDT | 4.173 | 31.259 | 3015 | 7.04 |
| SOLUSDT | 5.031 | 46.883 | 3299 | 7.70 |
| XRPUSDT | 10.628 | 274.125 | 3771 | 8.80 |

La mayor proporción señalada por IQR es 10.18 % en BTCUSDT. Estas marcas describen colas respecto al rango intercuartílico; no demuestran errores ni justifican eliminar registros. Las cantidades están en unidades de cada criptomoneda y no son directamente comparables entre activos. Los extremos pueden reflejar actividad concentrada; no se atribuyen a eventos sin evidencia adicional.

```{figure} ../_static/figures/univariate_volume.png
:alt: Histogramas y boxplots de volume por activo.

Volumen base: cinco activos, sin mezclar sus escalas.
```

#### Volumen cotizado: `quote_asset_volume`

Unidad: USDT.

| Activo | Media | Mediana | DE | P5 | P95 |
| --- | --- | --- | --- | --- | --- |
| BNBUSDT | 1.4218e+07 | 6.3251e+06 | 2.5222e+07 | 1.4148e+06 | 5.2555e+07 |
| BTCUSDT | 1.0953e+08 | 7.39e+07 | 1.1777e+08 | 1.6823e+07 | 3.1795e+08 |
| ETHUSDT | 5.2987e+07 | 3.6067e+07 | 5.8842e+07 | 7.7969e+06 | 1.5367e+08 |
| SOLUSDT | 1.5268e+07 | 8.0076e+06 | 2.4689e+07 | 1.7394e+05 | 5.2677e+07 |
| XRPUSDT | 1.5638e+07 | 7.6647e+06 | 2.8588e+07 | 1.4986e+06 | 5.5313e+07 |

| Activo | Asimetría | Exceso curtosis | Señaladas IQR | % IQR |
| --- | --- | --- | --- | --- |
| BNBUSDT | 6.940 | 98.251 | 4356 | 10.17 |
| BTCUSDT | 3.933 | 31.362 | 2818 | 6.58 |
| ETHUSDT | 4.484 | 40.357 | 2844 | 6.64 |
| SOLUSDT | 6.914 | 101.087 | 3116 | 7.27 |
| XRPUSDT | 7.695 | 100.455 | 4469 | 10.43 |

La mayor proporción señalada por IQR es 10.43 % en XRPUSDT. Estas marcas describen colas respecto al rango intercuartílico; no demuestran errores ni justifican eliminar registros. La unidad USDT facilita comparar cantidades cotizadas; los cambios de precio y de actividad siguen afectando la distribución. El volumen comprador taker, cuando corresponde, es un subconjunto del volumen y no representa todo el flujo comprador.

```{figure} ../_static/figures/univariate_quote_asset_volume.png
:alt: Histogramas y boxplots de quote_asset_volume por activo.

Volumen cotizado: cinco activos, sin mezclar sus escalas.
```

#### Número de operaciones: `number_of_trades`

Unidad: Operaciones.

| Activo | Media | Mediana | DE | P5 | P95 |
| --- | --- | --- | --- | --- | --- |
| BNBUSDT | 23747 | 14177 | 29353 | 3949.6 | 75197 |
| BTCUSDT | 1.0919e+05 | 66326 | 1.179e+05 | 20763 | 3.4355e+05 |
| ETHUSDT | 56240 | 34227 | 69307 | 9677.4 | 1.788e+05 |
| SOLUSDT | 31999 | 13664 | 56215 | 830.6 | 1.1879e+05 |
| XRPUSDT | 26463 | 10827 | 51217 | 2536.6 | 99334 |

| Activo | Asimetría | Exceso curtosis | Señaladas IQR | % IQR |
| --- | --- | --- | --- | --- |
| BNBUSDT | 5.152 | 64.957 | 3472 | 8.11 |
| BTCUSDT | 2.946 | 13.025 | 3579 | 8.36 |
| ETHUSDT | 4.381 | 32.556 | 3798 | 8.87 |
| SOLUSDT | 7.499 | 114.785 | 3700 | 8.64 |
| XRPUSDT | 7.939 | 115.569 | 4772 | 11.14 |

La mayor proporción señalada por IQR es 11.14 % en XRPUSDT. Estas marcas describen colas respecto al rango intercuartílico; no demuestran errores ni justifican eliminar registros. Es un conteo horario discreto de operaciones, no de personas ni de participantes independientes. Su dispersión describe actividad, sin demostrar capacidad predictiva de volatilidad futura.

```{figure} ../_static/figures/univariate_number_of_trades.png
:alt: Histogramas y boxplots de number_of_trades por activo.

Número de operaciones: cinco activos, sin mezclar sus escalas.
```

#### Volumen comprador taker base: `taker_buy_base_asset_volume`

Unidad: Unidades del activo base.

| Activo | Media | Mediana | DE | P5 | P95 |
| --- | --- | --- | --- | --- | --- |
| BNBUSDT | 25212 | 10726 | 43942 | 2061.3 | 94309 |
| BTCUSDT | 1647.6 | 850.17 | 2368.2 | 177.81 | 5856 |
| ETHUSDT | 12922 | 8647.4 | 14083 | 2195.6 | 37538 |
| SOLUSDT | 92854 | 60219 | 1.1002e+05 | 15892 | 2.7338e+05 |
| XRPUSDT | 9.7901e+06 | 6.0241e+06 | 1.4789e+07 | 1.4841e+06 | 2.9685e+07 |

| Activo | Asimetría | Exceso curtosis | Señaladas IQR | % IQR |
| --- | --- | --- | --- | --- |
| BNBUSDT | 6.171 | 66.764 | 4431 | 10.34 |
| BTCUSDT | 4.831 | 46.419 | 4381 | 10.23 |
| ETHUSDT | 4.054 | 29.290 | 3043 | 7.10 |
| SOLUSDT | 5.242 | 52.684 | 3284 | 7.67 |
| XRPUSDT | 10.463 | 261.765 | 3761 | 8.78 |

La mayor proporción señalada por IQR es 10.34 % en BNBUSDT. Estas marcas describen colas respecto al rango intercuartílico; no demuestran errores ni justifican eliminar registros. Las cantidades están en unidades de cada criptomoneda y no son directamente comparables entre activos. Los extremos pueden reflejar actividad concentrada; no se atribuyen a eventos sin evidencia adicional.

```{figure} ../_static/figures/univariate_taker_buy_base_asset_volume.png
:alt: Histogramas y boxplots de taker_buy_base_asset_volume por activo.

Volumen comprador taker base: cinco activos, sin mezclar sus escalas.
```

#### Volumen comprador taker cotizado: `taker_buy_quote_asset_volume`

Unidad: USDT.

| Activo | Media | Mediana | DE | P5 | P95 |
| --- | --- | --- | --- | --- | --- |
| BNBUSDT | 7.1451e+06 | 3.1711e+06 | 1.2658e+07 | 6.8719e+05 | 2.6467e+07 |
| BTCUSDT | 5.4102e+07 | 3.6389e+07 | 5.8911e+07 | 7.8466e+06 | 1.5841e+08 |
| ETHUSDT | 2.6365e+07 | 1.7948e+07 | 2.9158e+07 | 3.7794e+06 | 7.6567e+07 |
| SOLUSDT | 7.6199e+06 | 3.9545e+06 | 1.2473e+07 | 78362 | 2.6519e+07 |
| XRPUSDT | 7.7525e+06 | 3.7934e+06 | 1.4291e+07 | 7.0912e+05 | 2.7529e+07 |

| Activo | Asimetría | Exceso curtosis | Señaladas IQR | % IQR |
| --- | --- | --- | --- | --- |
| BNBUSDT | 6.773 | 91.850 | 4392 | 10.25 |
| BTCUSDT | 3.943 | 31.703 | 2799 | 6.53 |
| ETHUSDT | 4.282 | 36.098 | 2853 | 6.66 |
| SOLUSDT | 7.086 | 107.088 | 3200 | 7.47 |
| XRPUSDT | 7.752 | 101.823 | 4513 | 10.54 |

La mayor proporción señalada por IQR es 10.54 % en XRPUSDT. Estas marcas describen colas respecto al rango intercuartílico; no demuestran errores ni justifican eliminar registros. La unidad USDT facilita comparar cantidades cotizadas; los cambios de precio y de actividad siguen afectando la distribución. El volumen comprador taker, cuando corresponde, es un subconjunto del volumen y no representa todo el flujo comprador.

```{figure} ../_static/figures/univariate_taker_buy_quote_asset_volume.png
:alt: Histogramas y boxplots de taker_buy_quote_asset_volume por activo.

Volumen comprador taker cotizado: cinco activos, sin mezclar sus escalas.
```

### 2.2.9 Identificador y marcas temporales

| Activo | Frecuencia | % |
| --- | --- | --- |
| BTCUSDT | 42833 | 20.0 |
| ETHUSDT | 42833 | 20.0 |
| BNBUSDT | 42833 | 20.0 |
| XRPUSDT | 42833 | 20.0 |
| SOLUSDT | 42833 | 20.0 |

Las cinco categorías tienen la misma frecuencia y no hay categorías raras. `symbol` identifica series y no es una clase objetivo. Se mantiene como clave de agrupación; no necesita codificación en los modelos separados por activo.

```{figure} ../_static/figures/univariate_symbol.png
:alt: Frecuencia de cada activo en DEVELOPMENT.

Distribución de symbol.
```

| Activo | Campo | Cardinalidad | Inicio UTC | Fin UTC |
| --- | --- | --- | --- | --- |
| BTCUSDT | open_time | 42833 | 2020-08-11T06:00:00+00:00 | 2025-07-01T18:00:00+00:00 |
| BTCUSDT | close_time | 42833 | 2020-08-11T06:59:59.999000+00:00 | 2025-07-01T18:59:59.999000+00:00 |
| ETHUSDT | open_time | 42833 | 2020-08-11T06:00:00+00:00 | 2025-07-01T18:00:00+00:00 |
| ETHUSDT | close_time | 42833 | 2020-08-11T06:59:59.999000+00:00 | 2025-07-01T18:59:59.999000+00:00 |
| BNBUSDT | open_time | 42833 | 2020-08-11T06:00:00+00:00 | 2025-07-01T18:00:00+00:00 |
| BNBUSDT | close_time | 42833 | 2020-08-11T06:59:59.999000+00:00 | 2025-07-01T18:59:59.999000+00:00 |
| XRPUSDT | open_time | 42833 | 2020-08-11T06:00:00+00:00 | 2025-07-01T18:00:00+00:00 |
| XRPUSDT | close_time | 42833 | 2020-08-11T06:59:59.999000+00:00 | 2025-07-01T18:59:59.999000+00:00 |
| SOLUSDT | open_time | 42833 | 2020-08-11T06:00:00+00:00 | 2025-07-01T18:00:00+00:00 |
| SOLUSDT | close_time | 42833 | 2020-08-11T06:59:59.999000+00:00 | 2025-07-01T18:59:59.999000+00:00 |

Las fechas se interpretan como coordenadas temporales, no como magnitudes con media o normalidad marginal. La distribución mensual registra exposición: los meses extremos son parciales y los meses tienen distinta duración; sus conteos no demuestran estacionalidad del mercado. El análisis temporal audita los huecos y los cierres no convencionales. Estos resúmenes incluyen registros originales, mientras que el modelado excluye las ventanas afectadas.

```{figure} ../_static/figures/univariate_time_coverage.png
:alt: Número de velas observadas por mes y activo.

Cobertura mensual; los conteos coinciden entre los cinco activos.
```

### 2.2.10 Reproducción del análisis completo

Ejecutar `python src/09_univariate_development.py` y `python src/09_render_univariate_report.py`. Los metadatos registran la huella del archivo, versiones y convenciones estadísticas. El informe comprueba la huella antes de incorporar los resultados. [Notebook de las doce columnas](../../notebooks/09_univariate_development.ipynb). No se imputan ni eliminan valores por este análisis.

## 2.3 Análisis bidimensional

El análisis se centra en `close` y en variables construidas exclusivamente a partir de sus precios históricos. No se incorporan volúmenes ni otros campos de mercado como predictores. `symbol` identifica las entidades y las fechas permiten alinear las observaciones. Todos los cálculos utilizan DEVELOPMENT; TEST permanece reservado.

### 2.3.1 Variables, objetivo y alineación temporal

El objetivo de esta sección se calcula con la desviación estándar de los 24 retornos futuros, centrados en su media y con divisor 24 (`ddof=0`). Se presenta en porcentaje, sin anualizar. La definición coincide con la sección 2.1; las muestras descriptivas se especifican en cada comparación.

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

## 2.4 Análisis multivariado de los 168 rezagos

### 2.4.1 Matriz real y alcance temporal

El modelo utiliza una variable de origen (`close`), pero **168 predictores distintos**: $X_{i,s}=(C_{i,s},C_{i,s-1},\ldots,C_{i,s-167})$. Por ello, la matriz sí requiere análisis multivariado. Se estudia cada activo por separado; no se mezclan precios nominales de criptomonedas distintas ni se incluyen `symbol`, timestamps, el objetivo futuro o Persistence como columnas del análisis.

Se reproducen exactamente las anclas de TRAIN de los cinco folds de la sección 3, incluida la intersección de elegibilidad entre activos y el confinamiento de historias y etiquetas al bloque. La elegibilidad verifica que el objetivo exista, pero **su valor no se utiliza para ajustar PCA, el escalador ni el detector de anomalías**. Cada ajuste usa solamente su TRAIN; no se ajusta ni se elige nada con VALIDATION o TEST. Los TRAIN son crecientes y se solapan entre folds; sus resultados no representan 25 muestras independientes.

| Fold | Inicio anclas TRAIN UTC | Fin anclas TRAIN UTC | n por activo | p | n/p |
| --- | --- | --- | --- | --- | --- |
| 1 | 2020-08-18 05:00 | 2021-06-03 20:00 | 5781 | 168 | 34.41 |
| 2 | 2020-08-18 05:00 | 2022-03-28 10:00 | 12342 | 168 | 73.46 |
| 3 | 2020-08-18 05:00 | 2023-01-20 00:00 | 19484 | 168 | 115.98 |
| 4 | 2020-08-18 05:00 | 2023-11-13 14:00 | 26433 | 168 | 157.34 |
| 5 | 2020-08-18 05:00 | 2024-09-06 04:00 | 33575 | 168 | 199.85 |

La relación n/p describe filas por predictor, no observaciones estadísticamente independientes. Dos ventanas contiguas comparten 167 cierres. Se conservan los huecos del calendario, sin imputación ni eliminación de valores extremos. Los paneles detallados muestran TRAIN del fold 5 y las tablas de estabilidad comparan los cinco TRAIN.

### 2.4.2 Redundancia, rango y correlación

Se estandarizan las 168 columnas con `StandardScaler` ajustado de nuevo en cada TRAIN y activo. Se calcula su matriz de Pearson. Se informa la mediana del valor absoluto de las 14.028 correlaciones fuera de la diagonal y la proporción con $|r|>0.99$; ese umbral es descriptivo, no una regla de selección.

La SVD de la matriz estandarizada y centrada permite distinguir rango numérico y redundancia: el rango cuenta los valores singulares mayores que $\max(n,p)\,\epsilon_{\mathrm{mach}}\,s_{max}$, donde $\epsilon_{\mathrm{mach}}$ es la precisión de máquina. El número de condición es $\kappa=s_{max}/s_{min}$ cuando el rango es completo. Un número elevado señala direcciones con escalas de variación muy distintas; no prueba fuga ni determina automáticamente qué rezagos eliminar. El VIF de los candidatos de 2.3 se complementa aquí con el espectro de la matriz completa, en lugar de añadir 168 regresiones auxiliares. Para anomalías se utiliza Isolation Forest, que no requiere invertir la covarianza como la distancia de Mahalanobis convencional.

| Activo · TRAIN fold 5 | Mediana \|r\| | Pares con \|r\| > 0.99 (%) | Rango numérico | Número de condición |
| --- | --- | --- | --- | --- |
| BNBUSDT | 0.99130 | 55.69 | 168 | 1469.4 |
| BTCUSDT | 0.99383 | 73.00 | 168 | 1699.3 |
| ETHUSDT | 0.99054 | 52.45 | 168 | 1434.7 |
| SOLUSDT | 0.99352 | 68.76 | 168 | 1724.9 |
| XRPUSDT | 0.97949 | 24.52 | 168 | 965.2 |

Las altas asociaciones entre niveles de precio rezagados pueden reflejar persistencia, tendencia y mezcla de periodos. La estandarización no elimina esa estructura ni vuelve estacionarias las series. La redundancia ayuda a explicar por qué los coeficientes individuales del SVR deben interpretarse con cautela.

### 2.4.3 PCA y dimensión efectiva

Se ejecuta [PCA de scikit-learn](https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html) con SVD completa (`svd_solver="full"`) sobre cada matriz estandarizada de TRAIN. PCA centra las columnas; el escalado se realiza explícitamente antes. Se conservan los 168 componentes para diagnosticar el espectro, sin introducir PCA en el SVR ni seleccionar variables a partir del resultado.

Se reporta el mínimo número de componentes para alcanzar 90 %, 95 % y 99 % de la varianza, con umbrales fijados en el protocolo antes del cálculo. Además, con $q_j=\lambda_j/\sum_k\lambda_k$, la dimensión efectiva por entropía es:

$$
d_{efectiva}=\exp\left(-\sum_{j:q_j>0}q_j\ln q_j\right).
$$

Esta dimensión continua resume la concentración del espectro y no equivale al rango numérico ni a un número óptimo de predictores para pronosticar.

| Activo · TRAIN fold 5 | PC1 (%) | PC2 (%) | k90 | k95 | k99 | Dimensión efectiva |
| --- | --- | --- | --- | --- | --- | --- |
| BNBUSDT | 98.963 | 0.657 | 1 | 1 | 2 | 1.077 |
| BTCUSDT | 99.304 | 0.417 | 1 | 1 | 1 | 1.055 |
| ETHUSDT | 98.906 | 0.679 | 1 | 1 | 2 | 1.081 |
| SOLUSDT | 99.237 | 0.472 | 1 | 1 | 1 | 1.059 |
| XRPUSDT | 97.656 | 1.415 | 1 | 1 | 2 | 1.162 |

En TRAIN del fold 5, PC1 concentra entre 97.656% y 99.304% de la varianza estandarizada. Para alcanzar 95 % se requieren entre 1 y 1 componentes; para 99 %, entre 1 y 2. La dimensión efectiva varía entre 1.055 y 1.162, mientras que el rango numérico se informa separadamente.

La alineación absoluta de PC1 con el vector uniforme $\mathbf{1}/\sqrt{168}$ permite examinar si el primer componente representa principalmente el nivel conjunto de los cierres. Los pesos de PC1 y PC2 se grafican por rezago. El signo de un componente es arbitrario; para dibujarlo se orienta hacia suma de pesos no negativa, sin cambiar la varianza explicada.

| Activo | \|coseno(PC1, nivel uniforme)\| |
| --- | --- |
| BNBUSDT | 0.999997 |
| BTCUSDT | 0.999999 |
| ETHUSDT | 0.999997 |
| SOLUSDT | 0.999998 |
| XRPUSDT | 0.999985 |

La alineación mínima observada es 0.999985. La cercanía a uno respalda interpretar PC1 principalmente como un movimiento conjunto de nivel en estos entrenamientos, no como una medida de volatilidad futura.

**Explicar varianza de precios no equivale a explicar volatilidad futura.** Las direcciones de baja varianza pueden contener información predictiva. No se interpreta un PC1 dominante como evidencia de que un único componente baste para el pronóstico ni como razón para cambiar ahora el modelo base.

### 2.4.4 Estabilidad descriptiva entre entrenamientos

| Activo | PC1 mín.–máx. (%) | k95 mín.–máx. | k99 mín.–máx. | Dimensión efectiva mín.–máx. |
| --- | --- | --- | --- | --- |
| BNBUSDT | 98.505–98.963 | 1–1 | 2–2 | 1.077–1.103 |
| BTCUSDT | 98.911–99.304 | 1–1 | 1–2 | 1.055–1.082 |
| ETHUSDT | 98.453–98.944 | 1–1 | 2–2 | 1.079–1.109 |
| SOLUSDT | 98.499–99.237 | 1–1 | 1–2 | 1.059–1.114 |
| XRPUSDT | 96.755–97.803 | 1–1 | 2–3 | 1.152–1.213 |

Los rangos resumen los cinco TRAIN crecientes por activo. No son intervalos de confianza y no permiten una prueba de estabilidad independiente: los folds comparten historia y al ampliarlos cambian los periodos y regímenes representados. Cada PCA tiene su propia base; las coordenadas de componentes de distintos ajustes no se comparan directamente como si compartieran orientación.

### 2.4.5 Anomalías multivariadas

Se ajusta [Isolation Forest](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.IsolationForest.html) directamente sobre las **168 columnas estandarizadas**, sin reducirlas antes mediante PCA. Se fijan 200 árboles, submuestras de 256 ventanas, todas las columnas disponibles, `contamination="auto"`, semilla 42 y ejecución secuencial. El puntaje usado es $a(X)=-\operatorname{score\_samples}(X)$: valores mayores representan mayor atipicidad según el detector.

La señalización utiliza un umbral propio y explícito: percentil 99 de los puntajes del mismo TRAIN, con desigualdad estricta `score > umbral`. No se utiliza el umbral automático de `predict`. El percentil se fija como criterio descriptivo antes de observar los resultados; no es una probabilidad de error ni una tasa de falsos positivos calibrada. Señalar aproximadamente el 1 % de TRAIN es una consecuencia de esa regla, no un descubrimiento de que el 1 % de los datos sea erróneo.

| Activo · TRAIN fold 5 | Umbral TRAIN | Anclas señaladas | Porcentaje | Secuencias consecutivas |
| --- | --- | --- | --- | --- |
| BNBUSDT | 0.64312 | 336 | 1.001 | 10 |
| BTCUSDT | 0.60678 | 336 | 1.001 | 14 |
| ETHUSDT | 0.65635 | 336 | 1.001 | 14 |
| SOLUSDT | 0.67656 | 336 | 1.001 | 1 |
| XRPUSDT | 0.71600 | 336 | 1.001 | 6 |

Se agrupan anclas señaladas consecutivas separadas exactamente por una hora en secuencias descriptivas. Los huecos interrumpen las secuencias. Estas secuencias no identifican eventos económicos independientes: las ventanas se solapan y dos secuencias pueden compartir cierres.

| Activo | Inicio secuencia UTC | Fin secuencia UTC | Anclas | Puntaje máximo |
| --- | --- | --- | --- | --- |
| BNBUSDT | 2024-06-07 11:00 | 2024-06-13 20:00 | 154 | 0.70798 |
| BTCUSDT | 2024-03-14 14:00 | 2024-03-18 23:00 | 106 | 0.63051 |
| ETHUSDT | 2021-11-07 22:00 | 2021-11-19 04:00 | 271 | 0.71224 |
| SOLUSDT | 2021-11-06 11:00 | 2021-11-20 10:00 | 336 | 0.74273 |
| XRPUSDT | 2021-04-16 06:00 | 2021-04-19 01:00 | 68 | 0.75560 |

La tabla localiza, por activo, la secuencia que contiene el puntaje máximo de TRAIN del fold 5. No atribuye causas a las fechas. Los precios de nivel extremo, las transiciones de nivel o las trayectorias poco habituales pueden generar puntuaciones altas sin ser errores de cotización. Las ventanas señaladas se conservan; no se recortan ni se reentrena el SVR para mejorar métricas después de observarlas.

| Activo | 2020 (% señalado) | 2021 (% señalado) | 2022 (% señalado) | 2023 (% señalado) | 2024 (% señalado) |
| --- | --- | --- | --- | --- | --- |
| BNBUSDT | 0.00 | 2.41 | 0.00 | 0.00 | 2.61 |
| BTCUSDT | 5.66 | 0.00 | 0.00 | 0.00 | 2.98 |
| ETHUSDT | 0.00 | 4.49 | 0.00 | 0.00 | 0.00 |
| SOLUSDT | 0.00 | 4.49 | 0.00 | 0.00 | 0.00 |
| XRPUSDT | 0.00 | 4.49 | 0.00 | 0.00 | 0.00 |

Los porcentajes anuales usan como denominador las anclas elegibles de cada año, con extremos parciales. Se calculan con un único detector ajustado sobre todo TRAIN del fold 5. Son una descripción retrospectiva, no detección en línea disponible en cada fecha histórica ni evidencia de deterioro futuro. Los puntajes tampoco están calibrados para comparar riesgo entre activos.

### 2.4.6 Estructura, subpoblaciones y gráficos

Los activos son grupos conocidos, tratados por separado. Las proyecciones PC1–PC2 se colorean por año UTC para examinar cómo se distribuyen los periodos en la representación. Solo para esa visualización se muestran hasta 5.000 anclas uniformemente espaciadas en orden temporal; PCA, correlaciones y anomalías utilizan todas las filas elegibles de TRAIN. Los gráficos por activo tienen bases propias y no permiten comparar directamente coordenadas entre criptomonedas.

Las matrices de correlación y las curvas de varianza acumulada muestran escalas ampliadas, identificadas en sus ejes, para hacer visible la redundancia sin ocultar las diferencias entre rezagos. No parten necesariamente de cero; los valores completos se conservan en las tablas y matrices numéricas.

La distribución temporal en la proyección puede reflejar niveles de precio y cambios de escala, sin demostrar clusters o regímenes discretos. No se ejecuta clustering, UMAP ni t-SNE: el objetivo de esta sección es caracterizar la matriz real, no elegir un número de regímenes sin un criterio temporal de estabilidad. El clustering de regímenes exigiría una representación pertinente, validación de estabilidad y una pregunta adicional; no se declaran grupos descubiertos a partir de estos gráficos.

```{figure} ../_static/figures/multivariate_BNBUSDT.png
:alt: Correlación de 168 rezagos, PCA y anomalías de BNBUSDT, solo TRAIN fold 5.

BNBUSDT: matriz real de entrada, varianza acumulada, proyección por año, pesos y anomalías retrospectivas.
```

```{figure} ../_static/figures/multivariate_BTCUSDT.png
:alt: Correlación de 168 rezagos, PCA y anomalías de BTCUSDT, solo TRAIN fold 5.

BTCUSDT: matriz real de entrada, varianza acumulada, proyección por año, pesos y anomalías retrospectivas.
```

```{figure} ../_static/figures/multivariate_ETHUSDT.png
:alt: Correlación de 168 rezagos, PCA y anomalías de ETHUSDT, solo TRAIN fold 5.

ETHUSDT: matriz real de entrada, varianza acumulada, proyección por año, pesos y anomalías retrospectivas.
```

```{figure} ../_static/figures/multivariate_SOLUSDT.png
:alt: Correlación de 168 rezagos, PCA y anomalías de SOLUSDT, solo TRAIN fold 5.

SOLUSDT: matriz real de entrada, varianza acumulada, proyección por año, pesos y anomalías retrospectivas.
```

```{figure} ../_static/figures/multivariate_XRPUSDT.png
:alt: Correlación de 168 rezagos, PCA y anomalías de XRPUSDT, solo TRAIN fold 5.

XRPUSDT: matriz real de entrada, varianza acumulada, proyección por año, pesos y anomalías retrospectivas.
```

### 2.4.7 Consecuencias y reproducibilidad

El análisis establece la dimensión nominal, la concentración de varianza, la redundancia y la atipicidad conjunta de la entrada utilizada. Motiva estudiar, en un experimento posterior, representaciones de retornos o reducción de dimensionalidad; no prueba que mejoren el pronóstico. Cualquier uso predictivo de PCA, selección de componentes o tratamiento de anomalías deberá ajustarse dentro del entrenamiento y elegirse con validación temporal, conservando TEST reservado. Los resultados del SVR y Persistence permanecen intactos.

El protocolo previo está en `outputs/tables/multivariate_protocol.json`. El cálculo está en `src/12_multivariate_close.py`; `src/12_render_multivariate_report.py` sincroniza informe y notebook. Se guardan los 25 resúmenes, el espectro completo, parámetros de escalado, matrices de correlación, pesos de PC1/PC2, puntajes del fold 5 y secuencias señaladas en `outputs/tables/multivariate_*`. Los metadatos registran semillas, versiones, huellas y alcance de ajuste. Las pruebas verifican dimensiones conocidas, rechazo de la etiqueta como entrada y las fronteras temporales compartidas con el modelo.

Para reproducir: `python -m unittest discover -s tests -v`, `python src/12_multivariate_close.py` y `python src/12_render_multivariate_report.py`. El notebook carga los resultados verificados por defecto; `RECALCULAR = True` repite los 25 diagnósticos.

[Notebook multivariado con resultados ejecutados](../../notebooks/12_multivariate_scope_close.ipynb).

## 2.5 Prevención de fuga de datos (*Data Leakage*)

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

### 2.5.3 Diagnóstico individual de todas las entradas

Se reemplaza el diagnóstico limitado a close por **179 regresiones univariadas por activo y fold**: 168 cierres rezagados (`lag_0` es close), las otras ocho columnas numéricas originales, retorno horario, retorno absoluto y volatilidad histórica de 24 horas. Son **4.475 evaluaciones** en los mismos cinco folds cronológicos de la sección 3. Cada OLS con intercepto utiliza una única columna y escalado ajustado únicamente con TRAIN; no hay búsqueda de hiperparámetros. El modelo base sigue siendo el SVR con 168 cierres: estas OLS son diagnósticas.

Se exige la misma elegibilidad y el mismo confinamiento temporal del modelo: historia de 168 cierres, objetivo y referencia completos dentro de cada bloque. Las muestras de entrenamiento y validación coinciden con la verificación `base_folds.csv`. La referencia constante usa la media de y de TRAIN. RMSE se expresa en puntos porcentuales y R² compara con la media observada del bloque validado. AUC no aplica a regresión.

| Activo | Evaluaciones de rezagos | RMSE mínimo–máximo | R² mínimo–máximo | Alertas R² ≥ 0,8 |
| --- | --- | --- | --- | --- |
| BNBUSDT | 840 | 0.3386 a 0.9362 | -4.7710 a -0.3646 | 0 |
| BTCUSDT | 840 | 0.3272 a 0.3965 | -1.4282 a -0.0679 | 0 |
| ETHUSDT | 840 | 0.3451 a 0.9696 | -8.0806 a -0.0062 | 0 |
| SOLUSDT | 840 | 0.4571 a 2.0484 | -12.1281 a -0.1905 | 0 |
| XRPUSDT | 840 | 0.4925 a 1.0796 | -3.4348 a -0.3066 | 0 |

Los rangos reúnen los 168 rezagos y cinco folds de cada activo, no son intervalos de confianza ni resultados de una selección. Los candidatos originales y derivados se resumen sobre los 25 bloques activo-fold:

| Variable | RMSE medio | R² mínimo–máximo | Alertas R² ≥ 0,8 |
| --- | --- | --- | --- |
| abs_return_1h_pct | 0.4544 | -2.5252 a 0.1615 | 0 |
| high | 0.6193 | -10.7323 a -0.0069 | 0 |
| low | 0.6130 | -9.6104 a -0.0053 | 0 |
| number_of_trades | 0.6234 | -13.1447 a 0.1248 | 0 |
| open | 0.6167 | -10.2647 a -0.0063 | 0 |
| past_volatility_24h_pct | 0.3622 | -0.4974 a 0.4176 | 0 |
| quote_asset_volume | 0.5197 | -10.3582 a 0.1231 | 0 |
| return_1h_pct | 0.5036 | -3.9231 a -0.0038 | 0 |
| taker_buy_base_asset_volume | 0.4567 | -4.0814 a 0.2243 | 0 |
| taker_buy_quote_asset_volume | 0.5196 | -10.1956 a 0.1197 | 0 |
| volume | 0.4559 | -4.0797 a 0.2279 | 0 |

Se registran **0 alertas** con el umbral descriptivo R² ≥ 0,8 de la guía. Un resultado alto es una señal para revisar disponibilidad y alineación, no prueba de fuga; un resultado bajo tampoco acredita su ausencia. Se conservan todas las evaluaciones, sin escoger columnas según su mejor fold ni interpretar el máximo entre miles de evaluaciones como evidencia confirmatoria.

[Resultados de las 4.475 evaluaciones](../../outputs/tables/leakage_all_features_scores.csv). El archivo guarda parámetros del escalado, coeficiente, intercepto, conteos y error de la media de TRAIN. No se realizan pruebas de significación ni se usan estos resultados para modificar el modelo.

**Disponibilidad.** Para cada rezago k, el cierre procede de la vela s−k y está disponible nominalmente en s−k+1 hora. Los ocho campos numéricos de la vela actual se consideran disponibles solo cuando termina esa vela, en s+1 hora. El retorno necesita los cierres s−1 y s; la volatilidad pasada necesita los cierres s−24 a s. Todos preceden o coinciden con la emisión nominal s+1 hora. Las claves symbol y fechas no son predictores. La etiqueta futura nunca aparece en las entradas.

[Registro de disponibilidad de las 179 variables](../../outputs/tables/leakage_feature_availability.csv). La disponibilidad es nominal tras el cierre; los archivos históricos no registran latencia real de recepción. Un sistema operativo deberá esperar confirmación de la vela cerrada. El objetivo conserva sus propias 24 horas futuras y solo se usa como etiqueta en el ajuste o la evaluación.

La volatilidad pasada comparte fórmula con el objetivo, pero utiliza otro intervalo temporal: no es fuga por ese solo hecho. Se mantienen separadas las variables disponibles, los candidatos excluidos por diseño y los datos futuros prohibidos. Este diagnóstico no certifica ausencia universal de fuga ni decide transformaciones a partir de TEST.

### 2.5.4 Duplicados, entidades y fronteras temporales

La comprobación actual en DEVELOPMENT encuentra **0 filas duplicadas exactas** y **0 claves (`symbol`, `open_time`) duplicadas**. Los registros existentes de la partición sitúan su última apertura en **2025-07-01 18:00 UTC** y la primera de TEST en **2025-07-01 19:00 UTC**, con el mismo corte para los cinco activos y sin mezcla aleatoria.

Según esos registros, los intervalos no se solapan; por tanto, no comparten claves temporales ni filas completas idénticas que incluyan esas fechas. Esta conclusión utiliza la verificación guardada de la partición, no una nueva lectura de TEST. No se certifica aquí la ausencia de vectores de precios iguales o casi duplicados entre archivos ignorando la fecha. Un cierre repetido, por sí solo, no identifica un registro duplicado.

BTC, ETH, BNB, XRP y SOL aparecen en ambas particiones. Esta repetición de entidades es coherente con pronosticar el futuro de los mismos activos; no evalúa generalización a criptomonedas desconocidas. Los activos de una misma hora deberán permanecer en el mismo bloque en la validación temporal.

La separación de filas por fecha no basta para separar etiquetas futuras. En cada fold se purgan las etiquetas cuyo periodo objetivo invada la validación. Los folds del modelo base y sus preprocesadores se verifican mediante fechas reales y la verificación de la sección 2.9. En este experimento se exige además que la historia completa de cada ejemplo esté contenida en su bloque; se aplica el mismo criterio a ambos modelos y a los prefijos de la curva de aprendizaje.

### 2.5.5 Transformaciones y alcance de la evidencia

Escaladores, imputaciones, PCA, selección de variables, umbrales e hiperparámetros deberán ajustarse exclusivamente dentro del entrenamiento de cada fold. El EDA previo sobre DEVELOPMENT completo no equivale a una validación fuera de muestra: sus estadísticas y transformaciones auxiliares no deben reutilizarse como preprocesadores ya ajustados. Si sus conclusiones guían decisiones, las métricas internas tienen ese contexto exploratorio y TEST debe conservarse para la evaluación final.

Se verificaron el orden temporal, la purga, la disponibilidad de las etiquetas dentro de DEVELOPMENT y el ajuste del escalado solo en entrenamiento. La verificación de la sección 2.9 contrasta las 168 entradas en los 25 bloques activo-fold del pipeline del modelo base. Las pruebas de fronteras y perturbación de datos futuros complementan esta comprobación. El objetivo de la sección 2.1 y el modelo comparten la desviación estándar centrada con `ddof=0`. Los archivos históricos no permiten verificar la latencia de recepción en producción.

### 2.5.6 Interpretación y decisiones

Se conserva `close` de velas cerradas como entrada original y se construyen 168 rezagos consecutivos. Se excluyen de la entrada el objetivo, sus transformaciones y cualquier observación futura. Identificadores y fechas se mantienen como claves; los demás campos se analizan como diagnóstico y no se incorporan al SVR. Las ventanas y la referencia histórica se calculan por activo, con controles de continuidad, disponibilidad y confinamiento en el bloque.

Las comprobaciones realizadas no detectan infracciones de las reglas temporales implementadas en los folds y pipelines evaluados. Esta evidencia no constituye una garantía general de ausencia de fuga. La disponibilidad operativa deberá verificarse antes de una implementación en producción. No se utiliza TEST para elegir variables, ventanas o configuraciones.

### 2.5.7 Reproducibilidad

Ejecutar `python src/13_feature_diagnostics.py` y `python src/13_render_feature_diagnostics.py`. Las tablas `leakage_all_features_scores.csv` y `leakage_feature_availability.csv` y los metadatos guardan el diagnóstico vigente y sus huellas. El script anterior `src/13_leakage_audit.py` se conserva como antecedente; su diagnóstico de un solo cierre no sustituye esta evaluación de todas las entradas. [Notebook de verificación individual](../../notebooks/13_feature_diagnostics.ipynb).

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

La implementación de la sección 3 exige además que la historia de cada entrada y su objetivo queden contenidos en el bloque al que pertenece el ancla. Con L=168 y horizonte 24, se comprueba `ancla − 167h ≥ inicio_bloque` y `ancla + 24h ≤ fin_bloque`. Se registran por separado las pérdidas iniciales de historial y finales de horizonte; se usa la misma regla para SVR y Persistence y para los prefijos de entrenamiento de la curva de aprendizaje.

Los rezagos y estadísticas móviles de entrada se construyen solo con información pasada. Las transformaciones aprendidas se ajustan dentro del entrenamiento de cada fold. La persistencia del precio no garantiza buena predicción de volatilidad, y la menor autocorrelación del retorno no implica ausencia de dependencia en su magnitud. Las diferencias entre periodos motivan informar desempeño por activo y por bloque temporal, además del promedio.

Esta exploración no cambia el horizonte de 24 horas, no selecciona una ventana definitiva y no entrena SVR, persistencia u otros modelos predictivos. TEST permanece reservado. La información temporal apoya el diseño de la validación, sin convertir los patrones retrospectivos en predictores disponibles en el pasado.

### 2.6.9 Reproducibilidad

El procedimiento está en `src/14_temporal_close.py`; las tablas se guardan en `outputs/tables/temporal_*.csv` y la trazabilidad en `outputs/tables/temporal_metadata.json`. Se verifica la conservación de DEVELOPMENT mediante SHA-256. Las pruebas, gráficos y descomposición no modifican los datos originales.

[Notebook ejecutado del componente temporal](../../notebooks/14_temporal_close.ipynb).

### 2.6.10 Análisis horario del objetivo definitivo

El análisis anterior del precio se complementa con el objetivo realmente pronosticado: desviación estándar centrada de los 24 retornos logarítmicos horarios futuros, multiplicada por 100, con divisor 24 y sin anualización. Cada etiqueta exige 25 cierres consecutivos y convencionales. Se conserva la rejilla horaria, se invalidan las ventanas afectadas por huecos y se excluyen las últimas 24 anclas de DEVELOPMENT. TEST no se abre.

**Este análisis es retrospectivo.** El objetivo futuro, sus momentos móviles y su descomposición STL no son entradas disponibles al emitir la predicción; no se incorporan al pipeline. Se incluyen para diagnosticar la serie. Los patrones de calendario se agrupan por el instante nominal de emisión (apertura del ancla + 1 hora, UTC).

La serie completa, los momentos móviles y los ciclos utilizan todas las etiquetas válidas de DEVELOPMENT. ACF/PACF, ADF, KPSS y STL requieren una secuencia regular: se usa exclusivamente el tramo válido consecutivo más largo, elegido por cobertura, sin interpolar. Los resultados de ese tramo no representan necesariamente los periodos excluidos.

| Activo | Etiquetas válidas | n del tramo | Inicio ancla UTC | Fin ancla UTC |
| --- | --- | --- | --- | --- |
| BNBUSDT | 42539 | 19901 | 2023-03-24 14:00:00+00:00 | 2025-06-30 18:00:00+00:00 |
| BTCUSDT | 42539 | 19901 | 2023-03-24 14:00:00+00:00 | 2025-06-30 18:00:00+00:00 |
| ETHUSDT | 42539 | 19901 | 2023-03-24 14:00:00+00:00 | 2025-06-30 18:00:00+00:00 |
| SOLUSDT | 42564 | 19901 | 2023-03-24 14:00:00+00:00 | 2025-06-30 18:00:00+00:00 |
| XRPUSDT | 42564 | 19901 | 2023-03-24 14:00:00+00:00 | 2025-06-30 18:00:00+00:00 |

#### Dependencia horaria y efecto del solapamiento

Se calculan ACF y PACF hasta 168 horas; PACF utiliza Levinson–Durbin sin ajuste de sesgo (`ldbiased`). Los primeros rezagos se resumen a continuación. No se presentan bandas iid como evidencia inferencial. Los objetivos consecutivos comparten 23 retornos: una ACF elevada no demuestra capacidad predictiva ni independencia de las filas. Como contraste descriptivo se calcula la ACF de los retornos absolutos y al cuadrado, que no son objetivos de 24 horas superpuestos.

| Activo | Rezago h | ACF y | PACF y | ACF \|r\| | ACF r² |
| --- | --- | --- | --- | --- | --- |
| BNBUSDT | 1 | 0.9892 | 0.9892 | 0.2947 | 0.1668 |
| BNBUSDT | 24 | 0.5252 | 0.0362 | 0.1621 | 0.0792 |
| BNBUSDT | 168 | 0.3109 | 0.0029 | 0.1094 | 0.0410 |
| BTCUSDT | 1 | 0.9865 | 0.9865 | 0.2664 | 0.1883 |
| BTCUSDT | 24 | 0.4241 | 0.0091 | 0.1358 | 0.0567 |
| BTCUSDT | 168 | 0.3270 | -0.0153 | 0.1490 | 0.0691 |
| ETHUSDT | 1 | 0.9860 | 0.9860 | 0.2494 | 0.1192 |
| ETHUSDT | 24 | 0.4700 | 0.0069 | 0.1397 | 0.0351 |
| ETHUSDT | 168 | 0.3127 | -0.0015 | 0.1226 | 0.0458 |
| SOLUSDT | 1 | 0.9879 | 0.9879 | 0.2311 | 0.1719 |
| SOLUSDT | 24 | 0.5385 | 0.0065 | 0.1471 | 0.0682 |
| SOLUSDT | 168 | 0.2496 | 0.0077 | 0.0920 | 0.0379 |
| XRPUSDT | 1 | 0.9883 | 0.9883 | 0.3545 | 0.1650 |
| XRPUSDT | 24 | 0.5130 | 0.0114 | 0.1772 | 0.0398 |
| XRPUSDT | 168 | 0.2851 | -0.0006 | 0.1193 | 0.0135 |

La ACF del objetivo a una hora va de 0.9860 a 0.9892 entre activos; a 168 horas va de 0.2496 a 0.3270. La dependencia no se limita al primer rezago. Su interpretación requiere considerar tanto el solapamiento como la evolución del proceso.

La ACF de retornos transformados se calcula en su propio tramo consecutivo más largo, cuyos límites se guardan en el CSV; no se supone que coincida exactamente con el del objetivo. La relación con los rezagos de close continúa documentada en 2.6.5 y no se interpreta como causalidad.

#### Estacionariedad: hipótesis y límites

ADF se ejecuta con constante, máximo 48 rezagos horarios y selección AIC dentro de ese máximo; su hipótesis nula es raíz unitaria. KPSS se ejecuta con constante y rezagos automáticos; su hipótesis nula es estacionariedad en nivel. Se reportan ambas salidas; no rechazar una hipótesis no equivale a demostrarla. El horizonte solapado, la selección de tramo y los cambios de distribución limitan la extrapolación. Son diez contrastes exploratorios, no un criterio de selección de variables ni de hiperparámetros; no se declara significación conjunta.

| Activo | ADF estadístico | ADF p | Rezagos ADF | KPSS estadístico | KPSS p tabulado | Rezagos KPSS |
| --- | --- | --- | --- | --- | --- | --- |
| BNBUSDT | -9.3010 | 1.1126e-15 | 48 | 1.1256 | ≤0.01 | 85 |
| BTCUSDT | -10.4822 | 1.2058e-18 | 48 | 1.8912 | ≤0.01 | 85 |
| ETHUSDT | -9.2731 | 1.311e-15 | 48 | 6.5880 | ≤0.01 | 85 |
| SOLUSDT | -9.0074 | 6.2561e-15 | 48 | 0.4626 | 0.05018 | 85 |
| XRPUSDT | -9.5304 | 2.901e-16 | 48 | 2.0598 | ≤0.01 | 85 |

BNBUSDT: ADF rechaza raíz unitaria y KPSS rechaza estacionariedad en nivel, usando 0,05 solo como referencia descriptiva. BTCUSDT: ADF rechaza raíz unitaria y KPSS rechaza estacionariedad en nivel, usando 0,05 solo como referencia descriptiva. ETHUSDT: ADF rechaza raíz unitaria y KPSS rechaza estacionariedad en nivel, usando 0,05 solo como referencia descriptiva. SOLUSDT: ADF rechaza raíz unitaria y KPSS no rechaza estacionariedad en nivel, usando 0,05 solo como referencia descriptiva. XRPUSDT: ADF rechaza raíz unitaria y KPSS rechaza estacionariedad en nivel, usando 0,05 solo como referencia descriptiva.

Los p-valores KPSS pueden ser límites de la tabla (0,01 o 0,10), no valores exactos. Las advertencias originales se conservan en `temporal_target_stationarity.csv`. Rechazar ambas hipótesis puede indicar que ninguna simplificación describe bien el tramo; no se resuelve declarando la serie estacionaria por una sola prueba.

#### Calendario, momentos móviles y evolución trimestral

Los paneles incluyen la media y la varianza móviles sobre 168 horas consecutivas con `min_periods=168`; no se calculan a través de faltantes. Las agregaciones diaria, semanal y mensual son promedios de etiquetas horarias válidas, no volatilidades recalculadas a otra frecuencia; la cobertura puede variar. Los boxplots muestran hora, día de semana y mes de emisión, ocultando únicamente los puntos extremos para facilitar la lectura, sin eliminarlos de las estadísticas.

La STL usa periodo diario de 24 horas, ajuste robusto e interpolación de los suavizadores cada tres puntos (`seasonal_jump=trend_jump=low_pass_jump=3`). Es una descomposición retrospectiva del tramo, no un filtro causal ni una prueba de estacionalidad estable. Se muestran tendencia, componente estacional y residuo. La descomposición semanal del precio analizada previamente responde a otra serie y frecuencia.

| Activo | Mediana mínima–máxima por hora (%) | Día semanal de menor mediana | Día semanal de mayor mediana |
| --- | --- | --- | --- |
| BNBUSDT | 0.5530 a 0.5652 | sábado | lunes |
| BTCUSDT | 0.4822 a 0.4876 | sábado | miércoles |
| ETHUSDT | 0.6138 a 0.6241 | sábado | miércoles |
| SOLUSDT | 0.9551 a 0.9683 | sábado | miércoles |
| XRPUSDT | 0.6726 a 0.6808 | sábado | lunes |

El rango entre medianas por hora permite valorar la magnitud del ciclo horario; cada objetivo cubre un día completo, lo que suaviza diferencias intradiarias. Los días con menor y mayor mediana son resúmenes del periodo observado, no efectos causales ni una regla garantizada para el futuro.

| Activo | Mínima mediana trimestral (%) | Máxima mediana trimestral (%) | Rango de DE trimestral (pp) |
| --- | --- | --- | --- |
| BNBUSDT | 0.2548 | 1.3237 | 0.1862 a 0.9410 |
| BTCUSDT | 0.2334 | 0.8989 | 0.1741 a 0.4642 |
| ETHUSDT | 0.2426 | 1.1125 | 0.1799 a 0.6826 |
| SOLUSDT | 0.6031 | 2.1078 | 0.2929 a 1.0460 |
| XRPUSDT | 0.4335 | 1.6374 | 0.2572 a 1.2107 |

Los resúmenes trimestrales comparan distribuciones del objetivo y complementan la deriva de close. Los trimestres extremos son parciales. Variación de la distribución marginal no prueba un cambio de la relación condicional entre predictores y objetivo (concept drift). La sección 2.6.11 amplía este diagnóstico con fechas candidatas de cambio e incertidumbre condicional y una cronología documentada de eventos. No se atribuyen causalmente los movimientos a esos eventos.

Las diferencias por calendario mezclan años y regímenes: no demuestran efectos horarios permanentes. Los paneles por activo permiten revisar heterogeneidad sin mezclar niveles. Estas comprobaciones sustentan la validación cronológica, la evaluación por fold y la incertidumbre por bloques; no modifican retrospectivamente la configuración del modelo ni consultan TEST.

```{figure} ../_static/figures/temporal_target_BNBUSDT.png
:alt: EDA temporal horario del objetivo y retornos de BNBUSDT.

BNBUSDT: objetivo completo, agregaciones, calendario, dependencia y STL del tramo continuo.
```

```{figure} ../_static/figures/temporal_target_BTCUSDT.png
:alt: EDA temporal horario del objetivo y retornos de BTCUSDT.

BTCUSDT: objetivo completo, agregaciones, calendario, dependencia y STL del tramo continuo.
```

```{figure} ../_static/figures/temporal_target_ETHUSDT.png
:alt: EDA temporal horario del objetivo y retornos de ETHUSDT.

ETHUSDT: objetivo completo, agregaciones, calendario, dependencia y STL del tramo continuo.
```

```{figure} ../_static/figures/temporal_target_SOLUSDT.png
:alt: EDA temporal horario del objetivo y retornos de SOLUSDT.

SOLUSDT: objetivo completo, agregaciones, calendario, dependencia y STL del tramo continuo.
```

```{figure} ../_static/figures/temporal_target_XRPUSDT.png
:alt: EDA temporal horario del objetivo y retornos de XRPUSDT.

XRPUSDT: objetivo completo, agregaciones, calendario, dependencia y STL del tramo continuo.
```

#### Reproducción

Ejecutar `python src/14_temporal_target.py` y `python src/14_render_target_temporal.py`. Las tablas `outputs/tables/temporal_target_*.csv` conservan cobertura, ciclos, trimestres, componentes y diagnósticos. Los metadatos fijan convenciones y huella de DEVELOPMENT. [Notebook temporal del objetivo](../../notebooks/14_temporal_target.ipynb).

### 2.6.11 Puntos de cambio y cronología de eventos

**Alcance retrospectivo.** Se estudia un cambio dominante de media, no se asume que la serie tenga exactamente dos regímenes ni se utiliza este análisis para modificar el SVR, sus variables o TEST. El protocolo de este diagnóstico está fijado en `change_events_protocol.json` antes de ejecutar sus cálculos. Es una ampliación posterior al EDA inicial, no un estudio confirmatorio preregistrado.

#### Serie diaria sin solapamiento de retornos

Se toma la etiqueta horaria de cada ancla de las 23:00 UTC y se asigna al día siguiente, su fecha nominal de emisión. Esa etiqueta contiene los retornos de las 00:00 a las 23:00 de ese día: conserva la misma desviación estándar centrada, divisor 24 y escala porcentual. No es una media de etiquetas ni volatilidad acumulada de un retorno diario. Días consecutivos no comparten retornos, aunque comparten el cierre de frontera y pueden seguir siendo dependientes.

Solo se admiten días con los 25 cierres consecutivos y convencionales. Los puntos de cambio se calculan en el tramo diario completo más largo: 829 días, del 25 de marzo de 2023 al 30 de junio de 2025, común a los cinco activos. No se concatenan días separados por huecos. La cronología de eventos usa los días válidos de todo DEVELOPMENT; por eso también incluye 2022.

#### Método y elección de fecha candidata

Se ajustan dos medias por mínimos cuadrados y se examinan todas las divisiones con al menos 90 días a cada lado. La fecha candidata es el primer día del segundo segmento en la división que minimiza la suma de errores cuadrados. El estadístico es la reducción de esa suma respecto de una sola media, dividida por la suma total de cuadrados. Se contrasta su sensibilidad a mínimos de 60 y 180 días. Es un diagnóstico de media, sensible a extremos y cambios graduales; no detecta necesariamente cambios de varianza, todos los regímenes ni deriva condicional.

El procedimiento siempre propone una división si la serie no es constante. Por ello una fecha candidata, por sí sola, **no demuestra una ruptura**. Para evaluar la mejora bajo una aproximación de media constante se remuestrea la serie centrada mediante bloques circulares de 7, 14 y 28 días, con 499 réplicas y semilla 42. En cada réplica se vuelve a buscar la mejor división. El valor p usa (1 + réplicas con mejora al menos tan grande)/(499 + 1), evitando ceros.

Se aplica Holm a los cinco activos, por separado en cada longitud de bloque. El bootstrap supone dependencia local y estabilidad suficiente bajo la hipótesis nula; la muestra y los cambios graduales pueden incumplir esa aproximación. Los valores p son exploratorios. No se selecciona la longitud que produzca el menor p ni se interpreta la sensibilidad como tres confirmaciones independientes.

| Activo | Fecha candidata | Media antes (%) | Media después (%) | Diferencia (pp) | Reducción SSE (%) | p Holm, bloque 14d |
| --- | --- | --- | --- | --- | --- | --- |
| BNBUSDT | 2023-10-23 | 0.3638 | 0.5375 | 0.1736 | 6.19 | 0.048 |
| BTCUSDT | 2023-12-11 | 0.3468 | 0.4770 | 0.1302 | 6.34 | 0.016 |
| ETHUSDT | 2024-02-12 | 0.4047 | 0.6558 | 0.2511 | 14.41 | 0.010 |
| SOLUSDT | 2023-10-20 | 0.7071 | 0.9226 | 0.2155 | 4.53 | 0.072 |
| XRPUSDT | 2024-11-10 | 0.6086 | 1.0066 | 0.3980 | 10.33 | 0.016 |

| Activo | p Holm 7d | p Holm 14d | p Holm 28d | Fechas candidatas al variar mínimo 60/90/180d |
| --- | --- | --- | --- | --- |
| BNBUSDT | 0.010 | 0.048 | 0.152 | 2023-10-23 / 2023-10-23 / 2023-10-23 |
| BTCUSDT | 0.010 | 0.016 | 0.032 | 2023-12-11 / 2023-12-11 / 2023-12-11 |
| ETHUSDT | 0.010 | 0.010 | 0.020 | 2024-02-12 / 2024-02-12 / 2024-02-12 |
| SOLUSDT | 0.018 | 0.072 | 0.152 | 2023-10-20 / 2023-10-20 / 2023-10-20 |
| XRPUSDT | 0.010 | 0.016 | 0.066 | 2024-11-10 / 2024-11-10 / 2024-11-10 |

#### Incertidumbre de la localización

Condicionando al modelo de dos medias estimado, se remuestrean los residuos en bloques circulares **por separado dentro de cada segmento**, se añaden sus respectivas medias y se vuelve a estimar la división. Los percentiles 2,5 y 97,5 de 499 localizaciones dan el intervalo exploratorio mostrado. No son intervalos simultáneos ni contemplan la incertidumbre entre cero, uno o varios cambios; tampoco garantizan cobertura nominal si la forma de dos medias es inadecuada. Se calculan para las tres longitudes y se presenta 14 días como referencia fijada.

| Activo | Localización: percentil 2,5 | Localización: percentil 97,5 |
| --- | --- | --- |
| BNBUSDT | 2023-07-24 | 2024-10-28 |
| BTCUSDT | 2023-10-30 | 2024-05-29 |
| ETHUSDT | 2024-01-23 | 2024-04-21 |
| SOLUSDT | 2023-07-26 | 2025-01-09 |
| XRPUSDT | 2024-09-27 | 2025-03-16 |

Con umbral exploratorio 0,05, mantienen rechazo en las tres longitudes: BTCUSDT, ETHUSDT. La decisión cambia con la longitud en: BNBUSDT, SOLUSDT, XRPUSDT. Los intervalos amplios y esta sensibilidad impiden presentar todas las fechas como rupturas precisas o estables. Las dos medias describen el contraste del tramo; no autorizan a atribuirlo a un evento concreto.

[Resultados por longitud de bloque](../../outputs/tables/change_points.csv) y [sensibilidad al tamaño mínimo del segmento](../../outputs/tables/change_points_sensitivity.csv).

#### Cronología contrastada con fuentes primarias

Se eligen tres eventos ilustrativos de tecnología, proveedor y regulación a partir de fuentes primarias. La lista no es exhaustiva y se incorpora retrospectivamente; no se seleccionan nuevos eventos buscando coincidencias con los máximos de estos gráficos. Las fuentes acreditan fecha y hecho, no su efecto en la volatilidad.

| Fecha documentada | Evento | Fuente primaria |
| --- | --- | --- |
| 2022-09-15 | The Merge de Ethereum | [Fuente](https://ethereum.org/roadmap/merge/) |
| 2023-06-05 | Anuncio de cargos de la SEC contra Binance | [Fuente](https://www.sec.gov/newsroom/press-releases/2023-101-sec-files-13-charges-against-binance-entities-founder-changpeng-zhao) |
| 2024-01-10 | Aprobación de cotización de ETP spot de bitcoin | [Fuente](https://www.sec.gov/newsroom/speeches-statements/gensler-statement-spot-bitcoin-011023) |

El comunicado de la SEC de 2023 se describe como anuncio de cargos, no como una sentencia ni como descripción del estado actual del litigio. La fecha de enero de 2024 corresponde a la aprobación de cotización de ETP spot de bitcoin, no a una aprobación general de las criptomonedas.

Se comparan los 14 días calendario anteriores y los 14 posteriores a cada fecha, excluyendo el día del evento de las medias. Se utiliza una convención de días UTC, sin inventar una hora exacta de anuncio. Cada etiqueta está contenida en su día y no atraviesa de la ventana anterior a la posterior. Los gráficos sí muestran el día cero como contexto. Se conservan los conteos de días válidos; no se imputan días faltantes.

| Evento | Activo | n antes/después | Media antes (%) | Media después (%) | Diferencia (pp) |
| --- | --- | --- | --- | --- | --- |
| The Merge de Ethereum | BNBUSDT | 14/14 | 0.5216 | 0.5313 | 0.0097 |
| Anuncio de cargos de la SEC contra Binance | BNBUSDT | 14/14 | 0.2238 | 0.7308 | 0.5069 |
| Aprobación de cotización de ETP spot de bitcoin | BNBUSDT | 14/14 | 0.7208 | 0.5021 | -0.2187 |
| The Merge de Ethereum | BTCUSDT | 14/14 | 0.5569 | 0.6524 | 0.0955 |
| Anuncio de cargos de la SEC contra Binance | BTCUSDT | 14/14 | 0.2748 | 0.3884 | 0.1136 |
| Aprobación de cotización de ETP spot de bitcoin | BTCUSDT | 14/14 | 0.5048 | 0.4924 | -0.0124 |
| The Merge de Ethereum | ETHUSDT | 14/14 | 0.7583 | 0.8364 | 0.0781 |
| Anuncio de cargos de la SEC contra Binance | ETHUSDT | 14/14 | 0.2795 | 0.4189 | 0.1395 |
| Aprobación de cotización de ETP spot de bitcoin | ETHUSDT | 14/14 | 0.5489 | 0.5166 | -0.0323 |
| The Merge de Ethereum | SOLUSDT | 14/14 | 0.8358 | 0.8186 | -0.0172 |
| Anuncio de cargos de la SEC contra Binance | SOLUSDT | 14/14 | 0.4559 | 0.9991 | 0.5431 |
| Aprobación de cotización de ETP spot de bitcoin | SOLUSDT | 14/14 | 1.2748 | 1.0019 | -0.2729 |
| The Merge de Ethereum | XRPUSDT | 14/14 | 0.5576 | 1.5597 | 1.0022 |
| Anuncio de cargos de la SEC contra Binance | XRPUSDT | 14/14 | 0.4590 | 0.7986 | 0.3396 |
| Aprobación de cotización de ETP spot de bitcoin | XRPUSDT | 14/14 | 0.6937 | 0.5057 | -0.1880 |

Las diferencias tienen signos y magnitudes distintos según evento y activo. Son asociaciones descriptivas alrededor de una fecha: pueden intervenir anticipación, otros anuncios, tendencia, condiciones de mercado y dependencia entre activos. No hay grupo de control ni identificación causal; no se estiman efectos causales ni se añaden p-valores iid a estas ventanas cortas. Los eventos y las fechas candidatas de cambio se investigan por separado y no se emparejan automáticamente.

[Ventanas y fuentes](../../outputs/tables/event_windows.csv); [serie diaria y faltantes](../../outputs/tables/daily_nonoverlapping_volatility.csv).

```{figure} ../_static/figures/change_events_BNBUSDT.png
:alt: Cambio de media candidato y ventanas de eventos para BNBUSDT.

BNBUSDT: tramo continuo y ventanas de calendario; relaciones descriptivas, no causales.
```

```{figure} ../_static/figures/change_events_BTCUSDT.png
:alt: Cambio de media candidato y ventanas de eventos para BTCUSDT.

BTCUSDT: tramo continuo y ventanas de calendario; relaciones descriptivas, no causales.
```

```{figure} ../_static/figures/change_events_ETHUSDT.png
:alt: Cambio de media candidato y ventanas de eventos para ETHUSDT.

ETHUSDT: tramo continuo y ventanas de calendario; relaciones descriptivas, no causales.
```

```{figure} ../_static/figures/change_events_SOLUSDT.png
:alt: Cambio de media candidato y ventanas de eventos para SOLUSDT.

SOLUSDT: tramo continuo y ventanas de calendario; relaciones descriptivas, no causales.
```

```{figure} ../_static/figures/change_events_XRPUSDT.png
:alt: Cambio de media candidato y ventanas de eventos para XRPUSDT.

XRPUSDT: tramo continuo y ventanas de calendario; relaciones descriptivas, no causales.
```

#### Reproducción

Ejecutar `python src/22_change_events.py` y `python src/23_render_change_events.py`. Las pruebas verifican una ruptura conocida, una serie constante, invariancia de fecha ante cambios de escala y rechazo de faltantes. Los metadatos guardan las huellas del protocolo y DEVELOPMENT. [Notebook de cambios y eventos](../../notebooks/22_change_events.ipynb). TEST no se lee y los modelos predictivos no se modifican.

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

El [Pipeline de scikit-learn](https://scikit-learn.org/stable/modules/generated/sklearn.pipeline.Pipeline.html) agrupa las transformaciones que se ajustan. El parseo de fechas, la alineación horaria, la construcción determinista de la etiqueta y la selección de ventanas válidas se realizan antes, conservando la correspondencia entre X, y y fechas. Estas operaciones no estiman parámetros con validación. El escalado aprendido queda dentro del pipeline. En el modelo vigente, LinearSVR es la etapa final y el pipeline completo se ajusta de nuevo en cada fold, como se verifica en la sección 3.

La verificación de esta sección cubre las 168 columnas y los 25 entrenamientos del protocolo vigente. Comprueba el escalado sin volver a ajustar el SVR; sus métricas predictivas están en la sección 3.

### 2.9.2 Faltantes y elegibilidad temporal

Los huecos horarios se mantienen en una rejilla de calendario. No se rellenan con ceros, medias, interpolaciones ni observaciones posteriores. No se ha establecido un mecanismo de ausencia que justifique una imputación específica; además, imputar precios alteraría los retornos y la volatilidad construida a partir de ellos.

Los cierres con final horario irregular se enmascaran para la construcción analítica, sin modificar los registros originales. Una entrada histórica requiere todos sus cierres válidos; una etiqueta requiere 25 cierres consecutivos para obtener los 24 retornos futuros. Se excluyen del conjunto supervisado las ventanas incompletas y se mantienen sus fechas para auditar la cobertura. Esto no elimina filas de los archivos fuente ni demuestra que la muestra elegible esté libre de sesgo de selección.

En el modelo de la sección 3, `contained_block` añade el control de los dos extremos del bloque después de comprobar la continuidad. No basta con que el ancla esté en validación: su historial no puede comenzar antes del bloque ni su etiqueta terminar después. Este filtro determinista precede al ajuste del pipeline y utiliza las mismas filas para ambos métodos.

La exclusión de etiquetas inválidas afecta al entrenamiento y a la evaluación, no obliga a conocer el futuro para emitir una predicción: en operación, la elegibilidad de la entrada se determina solo con el histórico disponible. La etiqueta se comprueba cuando el horizonte se realiza. Los modelos se compararán sobre las mismas observaciones evaluables, con motivos de exclusión documentados.

### 2.9.3 Transformaciones y escalado

Se adopta como configuración inicial el cierre sin transformación logarítmica como entrada y su estandarización: a cada columna se le resta la media de entrenamiento y se divide por su escala de entrenamiento. [StandardScaler](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html) utiliza la desviación con divisor n y es sensible a valores extremos. La estandarización no vuelve normal ni estacionaria la serie y no elimina la multicolinealidad entre rezagos.

La transformación logarítmica se utiliza para construir los retornos del objetivo, no se aplica automáticamente a `close` como predictor. El objetivo conserva la desviación estándar centrada de 24 retornos, `ddof=0`, expresada en porcentaje y sin anualización. No se escala la etiqueta en esta comprobación. Una transformación posterior de entradas o del objetivo deberá evaluarse dentro de la validación temporal; cualquier transformación del objetivo exigirá volver a su escala común al calcular métricas.

La variación de niveles entre entrenamiento y validación observada en 2.6 puede producir valores estandarizados alejados de cero. No se centra de nuevo la validación para ocultar ese cambio ni se recortan sus valores a un rango arbitrario.

### 2.9.4 Codificación y tratamiento de valores extremos

No se requiere codificación categórica: `symbol` se utiliza para separar activos y las fechas para ordenar y dividir observaciones. No se incluyen como columnas predictoras. Los otros campos de mercado permanecen fuera del alcance acordado.

Los extremos válidos de `close` se conservan. Los hallazgos del EDA no justifican asumir que todo precio extremo sea un error. No se aplica eliminación por IQR ni winsorización y no se reutilizan umbrales estimados sobre todo DEVELOPMENT. Se exige que los precios utilizados sean finitos y positivos para construir retornos logarítmicos válidos. Si posteriormente se compara un escalado robusto o un tratamiento adicional, sus parámetros se ajustarán dentro de cada entrenamiento y su elección dependerá de validación interna.

### 2.9.5 Ingeniería temporal

La función `prepare` del modelo construye ventanas de 168 cierres: el último cierre observado y sus 167 rezagos horarios. Esta longitud está fijada en el protocolo, no optimizada en esta sección. Los desplazamientos se realizan sobre el calendario horario de cada activo, de modo que un hueco no se convierta artificialmente en una observación consecutiva. El ancla identifica la apertura de la última vela utilizada y la predicción se sitúa después de su cierre.

Cada uno de los 168 rezagos es una columna y su escala se ajusta con el entrenamiento del fold. La fuente seguirá siendo `close`, aunque puedan aparecer dependencias fuertes entre columnas. No se añaden automáticamente retornos, volatilidad pasada, medias móviles, senos, cosenos ni indicadores de calendario como entradas. La estacionalidad semanal débil de 2.6 no aporta por sí sola evidencia suficiente para incorporarlos. No existen variables espaciales aplicables.

### 2.9.6 Decisiones vinculadas al EDA

| Hallazgo | Decisión de preprocesamiento |
|---|---|
| Huecos y cierres irregulares | Mantener el calendario y excluir ventanas incompletas; no imputar. |
| Diferencias de nivel y escala entre activos y periodos | Verificar escalado por activo, ajustado solo en entrenamiento. |
| Precios extremos que pueden ser movimientos reales | Conservar extremos válidos, sin recorte automático. |
| Persistencia y redundancia entre rezagos | Fijar L=168 para este experimento; cualquier comparación posterior de ventanas deberá usar validación de DEVELOPMENT. |
| Componente semanal débil en el tramo estudiado | No incorporar características de calendario automáticamente. |
| Ausencia de predictores categóricos y coordenadas | No usar codificación categórica ni ingeniería espacial. |
| Etiquetas que abarcan 24 horas futuras | Purgar por la fecha final de la etiqueta, antes de ajustar el pipeline. |

### 2.9.7 Verificación ejecutada

Se comprueban los 25 bloques de entrenamiento de la sección 3, con 168 entradas y confinamiento de historia y objetivo. Para cada uno se verifica que las medias del escalador coincidan con TRAIN, que transformar validación no modifique medias, varianzas ni escalas, que las salidas sean finitas y que la transformación inversa recupere los cierres. Una perturbación artificial de validación confirma que se aplican los mismos parámetros ya aprendidos. No se exige media cero en validación.

| Activo | Fold | n TRAIN | n VALIDATION | Máxima media TRAIN estandarizada absoluta |
| --- | --- | --- | --- | --- |
| BNBUSDT | 1 | 5781 | 6370 | 5.43e-15 |
| BTCUSDT | 1 | 5781 | 6370 | 4.55e-15 |
| ETHUSDT | 1 | 5781 | 6370 | 4.40e-15 |
| SOLUSDT | 1 | 5781 | 6370 | 3.39e-15 |
| XRPUSDT | 1 | 5781 | 6370 | 4.49e-15 |
| BNBUSDT | 2 | 12342 | 6951 | 5.11e-15 |
| BTCUSDT | 2 | 12342 | 6951 | 8.94e-15 |
| ETHUSDT | 2 | 12342 | 6951 | 9.32e-15 |
| SOLUSDT | 2 | 12342 | 6951 | 4.79e-15 |
| XRPUSDT | 2 | 12342 | 6951 | 8.38e-15 |
| BNBUSDT | 3 | 19484 | 6758 | 4.90e-15 |
| BTCUSDT | 3 | 19484 | 6758 | 4.33e-15 |
| ETHUSDT | 3 | 19484 | 6758 | 9.15e-15 |
| SOLUSDT | 3 | 19484 | 6758 | 5.56e-15 |
| XRPUSDT | 3 | 19484 | 6758 | 1.30e-14 |
| BNBUSDT | 4 | 26433 | 6951 | 5.33e-15 |
| BTCUSDT | 4 | 26433 | 6951 | 1.32e-14 |
| ETHUSDT | 4 | 26433 | 6951 | 1.68e-14 |
| SOLUSDT | 4 | 26433 | 6951 | 9.55e-15 |
| XRPUSDT | 4 | 26433 | 6951 | 1.45e-14 |
| BNBUSDT | 5 | 33575 | 6951 | 2.16e-14 |
| BTCUSDT | 5 | 33575 | 6951 | 7.10e-15 |
| ETHUSDT | 5 | 33575 | 6951 | 2.05e-14 |
| SOLUSDT | 5 | 33575 | 6951 | 2.31e-14 |
| XRPUSDT | 5 | 33575 | 6951 | 2.87e-14 |

Los 25 controles se superan. Esto verifica propiedades concretas del preprocesamiento, no una garantía universal de ausencia de fuga. [Verificación completa](../../outputs/tables/preprocessing_full_audit.csv).

### 2.9.8 Reproducibilidad

Ejecutar `python src/17_validate_pipeline.py`. El script comparte preparación, fronteras y clase de escalado con el modelo; no utiliza TEST. Los metadatos conservan las huellas de DEVELOPMENT y de los folds. La comprobación histórica de una única columna en `src/17_preprocessing_close.py` queda sustituida por esta verificación del protocolo vigente. [Notebook ejecutado](../../notebooks/17_preprocessing_close.ipynb).
