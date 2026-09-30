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

El procedimiento está en `src/11_bivariate_close.py`. Las tablas y trazabilidad se guardan en `outputs/tables/bivariate_*.csv` y `outputs/tables/bivariate_metadata.json`; las figuras se guardan en `book/_static/figures/bivariate_*.png`. Cada objetivo válido se verifica contra el cálculo directo de la desviación estándar de sus 24 retornos futuros. Se comprueban la alineación temporal y la conservación de DEVELOPMENT mediante SHA-256. Los datos originales y TEST permanecen intactos.
