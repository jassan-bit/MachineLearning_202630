---
title: Pronóstico de la Volatilidad Realizada a 24 Horas de BTC, ETH, BNB, XRP y SOL mediante SVR Lineal con Datos Horarios de Binance Spot (2020–2026)
short_title: 1. Base de Datos
---

**Estudiantes:** Jassan Arteta - Mateo Bernal  
**Profesor:** Lihki Rubio  
**Asignatura:** Machine Learning  
**Universidad del Norte**  
**2026**

## 1. Base de Datos

Esta sección documenta la selección, procedencia, estructura, tamaño, calidad y consideraciones éticas del dataset antes de iniciar el análisis exploratorio de datos (EDA). Las cifras de cobertura, faltantes, duplicados y partición corresponden a los resultados previamente confirmados del proyecto; aquí no se realizan nuevos análisis estadísticos ni se construye la variable objetivo.

### 1.1 Definición clara del problema de investigación

#### Pregunta de la guía

¿Cuál es el problema de investigación?

#### Respuesta

El proyecto busca pronosticar la volatilidad realizada futura a 24 horas de los siguientes pares, utilizando datos horarios históricos de Binance Spot:

- BTCUSDT
- ETHUSDT
- BNBUSDT
- XRPUSDT
- SOLUSDT

La variable objetivo es cuantitativa continua, por lo que el problema corresponde a una regresión. Dado que las observaciones tienen orden cronológico y el pronóstico se refiere a un horizonte futuro, se trata de una regresión temporal. Toda información utilizada para producir una predicción deberá estar disponible en el instante de predicción o antes.

En una etapa posterior se comparará el baseline Persistence con un modelo SVR lineal, sin presuponer la superioridad de ninguno. En esta sección no se ejecuta ni se entrena ninguno de estos modelos.

### 1.2 Justificación de la selección del dataset

#### Pregunta de la guía

¿Por qué se seleccionó este conjunto de datos?

#### Respuesta

La selección responde a los siguientes criterios:

1. Es pertinente para estudiar el riesgo y la volatilidad financiera.
2. Contiene información real de mercado procedente de Binance Spot.
3. Su frecuencia horaria permite plantear pronósticos a un horizonte de 24 horas.
4. Abarca un periodo de más de seis años.
5. Incluye cinco criptoactivos relevantes para el problema de investigación: BTC, ETH, BNB, XRP y SOL, negociados frente a USDT.
6. Permite estudiar la heterogeneidad entre activos.
7. Sus 267,710 observaciones superan ampliamente el mínimo de 20,000 señalado en la guía.
8. Su estructura temporal permite utilizar una metodología de validación cronológica.

Como limitación inicial, se utiliza exclusivamente Binance Spot y solo se incluyen cinco activos. Por tanto, el dataset no representa automáticamente todo el mercado mundial de criptomonedas.

### 1.3 Fuente de los datos y licencia de uso

#### Fuente

Binance Spot. Los datos fueron obtenidos mediante la API pública de Binance y las observaciones corresponden a velas horarias. El dataset maestro se encuentra en `data/processed/crypto_binance_master_1h.csv`.

#### Periodo

Del 11 de agosto de 2020, 06:00 UTC, al 20 de septiembre de 2026, 23:00 UTC. Los extremos del periodo corresponden a la fecha y hora de apertura de las velas (`open_time`).

#### Licencia o condiciones de uso

Los datos utilizados corresponden a información pública de mercado de Binance Spot, obtenida mediante las interfaces públicas proporcionadas por Binance. Binance ofrece acceso a datos históricos y de mercado mediante su API y mediante su portal oficial de datos públicos, data.binance.vision.

El repositorio oficial binance-public-data, empleado por Binance para documentar y facilitar la descarga de datos históricos públicos, se distribuye bajo licencia MIT. No obstante, el acceso y uso de los servicios y API de Binance se encuentra adicionalmente sujeto a los Binance Terms of Use y a los Product Terms aplicables al mercado Spot.

Por esta razón, en este proyecto los datos se utilizan exclusivamente con fines académicos y de investigación. No se interpreta la licencia MIT del repositorio como una licencia independiente sobre cualquier dato obtenido mediante la API, sino que se respetan también las condiciones de uso publicadas oficialmente por Binance.

**Fuentes oficiales:**

- Binance Developer Documentation — Spot Exchange Terms of Use:
  [https://developers.binance.com/en/docs/products/spot/PROD-TERMS-OF-USE](https://developers.binance.com/en/docs/products/spot/PROD-TERMS-OF-USE)

- Binance Public Data:
  [https://data.binance.vision/](https://data.binance.vision/)

- Repositorio oficial Binance Public Data:
  [https://github.com/binance/binance-public-data](https://github.com/binance/binance-public-data)

- Binance Terms of Use:
  [https://www.binance.com/en/terms](https://www.binance.com/en/terms)

### 1.4 Diccionario de variables

En el encabezado del CSV maestro se identifican 12 columnas. La siguiente tabla presenta su tipo, unidad, significado y rol inicial. Las fechas están expresadas en UTC. Los roles indicados aún no corresponden a una selección definitiva de predictores.

| Variable | Tipo | Unidad | Significado | Rol inicial |
|----------|------|--------|-------------|-------------|
| `open_time` | Temporal | Fecha y hora UTC, formato ISO 8601 | Fecha y hora de apertura de la vela. | Índice temporal; identifica la observación junto con `symbol`. |
| `open` | Numérica continua | USDT por unidad del activo base | Precio de apertura de la vela. | Información de precio; candidato a insumo histórico. |
| `high` | Numérica continua | USDT por unidad del activo base | Precio máximo durante la vela. | Información de precio; candidato a insumo histórico. |
| `low` | Numérica continua | USDT por unidad del activo base | Precio mínimo durante la vela. | Información de precio; candidato a insumo histórico. |
| `close` | Numérica continua | USDT por unidad del activo base | Precio de cierre de la vela. | Información de precio; posible insumo para construir retornos y volatilidad posteriormente. |
| `volume` | Numérica continua | Unidades del activo base: BTC, ETH, BNB, XRP o SOL | Volumen negociado del activo base durante la vela. | Información de actividad de mercado; candidato a insumo histórico. |
| `close_time` | Temporal | Fecha y hora UTC, formato ISO 8601 | Fecha y hora de cierre de la vela. | Referencia temporal para verificar la disponibilidad de la vela completa. |
| `quote_asset_volume` | Numérica continua | USDT | Volumen negociado expresado en el activo de cotización. | Información de actividad de mercado; candidato a insumo histórico. |
| `number_of_trades` | Numérica discreta | Número de operaciones | Cantidad de operaciones realizadas durante la vela. | Información de actividad de mercado; candidato a insumo histórico. |
| `taker_buy_base_asset_volume` | Numérica continua | Unidades del activo base: BTC, ETH, BNB, XRP o SOL | Volumen del activo base comprado mediante operaciones tomadoras de liquidez (taker). | Información de compras tomadoras de liquidez; candidato a insumo histórico. |
| `taker_buy_quote_asset_volume` | Numérica continua | USDT | Volumen de compras tomadoras de liquidez expresado en el activo de cotización. | Información de compras tomadoras de liquidez; candidato a insumo histórico. |
| `symbol` | Categórica nominal | No aplica | Símbolo del par de negociación. | Identificador de la entidad. |

Los campos que resumen la vela completa solo podrán utilizarse cuando dicha vela haya cerrado y la información esté disponible. La volatilidad realizada futura a 24 horas es una variable objetivo conceptual que todavía no forma parte de las columnas del dataset original.

### 1.5 Estructura de los datos

#### Unidad de observación

Una vela horaria correspondiente a un activo y a un intervalo temporal.

#### Nivel de agregación

1 hora.

#### Tipo de estructura

Panel o longitudinal temporal. Existen cinco entidades, cada una observada repetidamente en el tiempo, y las observaciones tienen orden cronológico.

#### ¿Es transversal?

No. Las mismas entidades se observan repetidamente a través del tiempo, en lugar de registrarse únicamente en un corte temporal.

#### ¿Es serie temporal?

Sí, existe un componente temporal: cada activo cuenta con una secuencia de observaciones horarias ordenadas cronológicamente.

#### ¿Es panel/longitudinal?

Sí. Hay múltiples activos observados repetidamente en el tiempo.

#### ¿Es espacial?

No. No existen variables de latitud ni longitud en el dataset.

#### ¿Es espacio-temporal?

No. Existe tiempo, pero no un componente geográfico.

### 1.6 Tamaño de la muestra

En las tablas se utiliza la coma como separador de miles y el punto como separador decimal.

| Concepto | Valor |
|----------|------:|
| Activos | 5 |
| Observaciones por activo | 53,542 |
| Observaciones totales | 267,710 |
| Horas esperadas por activo | 53,562 |
| DEVELOPMENT | 214,165 |
| TEST | 53,545 |

El tamaño total cumple $267{,}710 > 20{,}000$, por lo que supera el mínimo advertido por la guía. DEVELOPMENT contiene 42,833 timestamps por activo y TEST contiene 10,709 timestamps por activo. Estas cantidades describen la partición previamente establecida; TEST permanece reservado.

#### 1.6.1 Número de variables y relación $n/p$

Para el dataset original, $n = 267{,}710$ es el número total de filas previamente confirmado y $p = 12$ es el número total de columnas verificado mediante el encabezado del CSV maestro. Por tanto:

$$
\frac{n}{p} = \frac{267{,}710}{12} \approx 22{,}309.17.
$$

Esta relación corresponde al dataset original e incluye identificadores temporales y de entidad. Para la matriz efectiva del SVR, **p=168 rezagos de close por activo**. Los entrenamientos contienen entre 5,781 y 33,575 filas elegibles por activo, con n/p entre 34.41 y 199.85; el detalle por fold está en 2.4.1. Estos cocientes no miden tamaño muestral independiente: las ventanas se solapan y los activos pueden presentar dependencia transversal. TEST no interviene en el diagnóstico.

#### 1.6.2 Casos por clase o rango de la variable objetivo

##### Casos por clase

**NO APLICA.**

El proyecto es de regresión temporal y la variable objetivo es continua, por lo que no existen clases.

##### Rango de la variable objetivo

Calculado únicamente sobre DEVELOPMENT con la fórmula del profesor: desviación estándar centrada de 24 retornos horarios futuros, divisor 24 (`ddof=0`), expresada en porcentaje. La construcción y distribución se documentan en la sección 2.1.

| Activo | Mínimo (%) | Máximo (%) | Objetivos válidos |
| --- | --- | --- | --- |
| BTCUSDT | 0.0445 | 3.3765 | 42,539 |
| ETHUSDT | 0.0542 | 5.1053 | 42,539 |
| BNBUSDT | 0.0787 | 5.9566 | 42,539 |
| XRPUSDT | 0.1144 | 7.9232 | 42,564 |
| SOLUSDT | 0.1853 | 7.0645 | 42,564 |

#### 1.6.3 Número de entidades

Número de entidades: **5**.

- BTCUSDT
- ETHUSDT
- BNBUSDT
- XRPUSDT
- SOLUSDT

Aunque existen cinco entidades diferentes, no debe asumirse automáticamente independencia estadística entre ellas: los mercados de criptomonedas pueden presentar movimientos comunes y dependencia transversal.

### 1.7 Calidad de los datos

Las cifras del periodo completo corresponden a los hallazgos previamente confirmados. La matriz de faltantes y la evaluación del mecanismo de ausencia se calculan exclusivamente sobre DEVELOPMENT. Los aspectos aún no evaluados se identifican expresamente como pendientes.

#### 1.7.1 Valores faltantes

| Activo | Horas esperadas | Horas observadas | Timestamps faltantes | Porcentaje faltante aproximado |
|--------|---------------:|----------------:|---------------------:|-------------------------------:|
| BTCUSDT | 53,562 | 53,542 | 20 | 0.03734% |
| ETHUSDT | 53,562 | 53,542 | 20 | 0.03734% |
| BNBUSDT | 53,562 | 53,542 | 20 | 0.03734% |
| XRPUSDT | 53,562 | 53,542 | 20 | 0.03734% |
| SOLUSDT | 53,562 | 53,542 | 20 | 0.03734% |

Los mismos 20 timestamps faltan en los cinco activos. No se imputaron y los datos originales permanecen intactos. Estas cifras describen ausencias de velas en la cuadrícula horaria esperada; no equivalen a una verificación de valores nulos dentro de las filas presentes.

#### 1.7.2 Patrón de faltantes

En DEVELOPMENT, del 11 de agosto de 2020 a las 06:00 UTC al 1 de julio de 2025 a las 18:00 UTC, se esperan 42,853 horas por activo. Se observan 42,833 y faltan 20, equivalentes al 0.04667%. Este porcentaje utiliza únicamente las horas esperadas de DEVELOPMENT como denominador; el 0.03734% de la subsección anterior corresponde al periodo completo.

En las 214,165 filas presentes no se detectaron celdas vacías ni los marcadores `NA`, `NaN`, `null`, `None` o `NaT`, sin distinguir mayúsculas y minúsculas y descontando espacios exteriores. La comprobación abarcó las 12 columnas. Esto describe la ausencia de datos, no la validez de todos sus valores.

Las ausencias corresponden a velas completas y coinciden en los cinco activos. Para visualizarlas se reconstruye en memoria la cuadrícula horaria esperada, sin modificar ni imputar el CSV. En las horas ausentes, `open_time` y `symbol` identifican la posición esperada; los otros diez campos no están disponibles.

```{figure} ../_static/figures/development_missing_matrix.svg
:alt: Matriz de las veinte horas ausentes en DEVELOPMENT y los diez campos no disponibles en cada una. El patrón es idéntico en los cinco activos.

Matriz de faltantes de DEVELOPMENT, ampliada a las veinte horas con ausencias para que sean legibles. Cada fila es una hora y cada columna un campo de la vela. Las 42,833 horas observadas por activo, sin celdas nulas detectadas, se omiten de este detalle. El patrón mostrado se comprobó por separado en los cinco activos.
```

Las veinte horas forman diez intervalos de ausencia:

| Inicio (UTC) | Fin (UTC, inclusivo) | Horas por activo |
|--------------|---------------------|-----------------:|
| 2020-11-30 06:00 | 2020-11-30 06:00 | 1 |
| 2020-12-21 14:00 | 2020-12-21 17:00 | 4 |
| 2020-12-25 02:00 | 2020-12-25 02:00 | 1 |
| 2021-02-11 04:00 | 2021-02-11 04:00 | 1 |
| 2021-03-06 02:00 | 2021-03-06 02:00 | 1 |
| 2021-04-20 02:00 | 2021-04-20 03:00 | 2 |
| 2021-04-25 05:00 | 2021-04-25 07:00 | 3 |
| 2021-08-13 02:00 | 2021-08-13 05:00 | 4 |
| 2021-09-29 07:00 | 2021-09-29 08:00 | 2 |
| 2023-03-24 13:00 | 2023-03-24 13:00 | 1 |

#### 1.7.3 Mecanismo de ausencia: MCAR, MAR o MNAR

El mecanismo de ausencia no puede determinarse con la evidencia disponible. MCAR supone que la ausencia no depende de valores observados ni no observados; MAR permite que dependa de información observada y MNAR contempla dependencia de valores no observados, incluso después de considerar la información disponible. Distinguir MAR de MNAR requiere supuestos o información adicional: los datos observados por sí solos no resuelven esa distinción. Véase [Drawing Inferences from Incomplete Data, National Research Council](https://www.ncbi.nlm.nih.gov/books/NBK209900/).

Como evidencia descriptiva, se comparó el movimiento de precio anterior a los inicios de los huecos con el anterior a las horas observadas de DEVELOPMENT. Para cada hora $t$ se calculó $100\lvert\ln(C_{t-1}/C_{t-2})\rvert$, donde $C$ es el precio de cierre. Solo se incluyeron horas cuyos dos cierres previos estaban disponibles y separados exactamente por una hora. Esta medida usa información previa y no corresponde a la variable objetivo de volatilidad futura.

| Activo | Mediana antes de un hueco (%) | Mediana antes de una hora observada (%) |
|--------|-----------------------------:|---------------------------------------:|
| BTCUSDT | 0.1903 | 0.2491 |
| ETHUSDT | 0.3773 | 0.3284 |
| BNBUSDT | 0.3229 | 0.3170 |
| XRPUSDT | 0.5097 | 0.3923 |
| SOLUSDT | 0.6108 | 0.5681 |

La comparación incluye 10 inicios de hueco y 42,811 horas observadas elegibles por activo. Las horas interiores de cada hueco no tienen los cierres previos necesarios y se excluyen, sin imputación. Los diez eventos son comunes a los cinco activos: no constituyen cincuenta eventos independientes. Cuatro activos presentan una mediana previa mayor antes de los huecos y BTC presenta una menor. Estas diferencias son descriptivas; el reducido número de eventos, la dependencia temporal y los cambios de mercado entre periodos impiden interpretarlas como una prueba concluyente del mecanismo. No se calculan valores p ni se afirma significancia estadística.

No se aplica la prueba de Little a los campos de mercado: las filas presentes están completas y las velas ausentes carecen de todos esos campos. Por tanto, no hay valores de mercado observados dentro de las filas incompletas que permitan la comparación habitual entre patrones. Tampoco puede compararse directamente la distribución del precio o del volumen durante las horas ausentes, precisamente porque esos valores no están disponibles.

La coincidencia de las ausencias y su agrupación temporal son compatibles con interrupciones de adquisición o disponibilidad de Binance, pero no demuestran esa causa ni permiten afirmar MCAR, MAR o MNAR. Para avanzar se necesitarían registros de descarga, avisos oficiales de interrupciones o una comprobación independiente de las mismas horas. No se imputan datos ni se adopta un mecanismo como hecho comprobado.

El procedimiento reproducible está en `src/04_missingness_development.cjs` y sus resultados en `outputs/tables/development_missingness.json`. Solo se leyó `data/splits/development_80.csv`; TEST no intervino en esta evaluación.

#### 1.7.4 Duplicados exactos y casi-duplicados

| Verificación | Resultado confirmado |
|--------------|---------------------:|
| Duplicados exactos | 0 |
| Timestamps duplicados por activo | 0 |

La revisión de las 214,165 filas de DEVELOPMENT confirma también cero filas exactamente repetidas y cero repeticiones de la clave (`symbol`, `open_time`). Los duplicados exactos se cuentan como apariciones adicionales de una misma fila completa del CSV.

Para identificar candidatos a casi-duplicados se comparan velas del mismo activo separadas exactamente por una hora, sin cruzar huecos. La regla busca posibles repeticiones consecutivas de una vela con cambios numéricos mínimos. Se exige que `number_of_trades` sea igual y que la diferencia relativa de cada uno de los siguientes ocho campos sea menor o igual a $10^{-6}$ (0.0001%): `open`, `high`, `low`, `close`, `volume`, `quote_asset_volume`, `taker_buy_base_asset_volume` y `taker_buy_quote_asset_volume`.

Para dos valores $a$ y $b$, se define la diferencia relativa como:

$$
d(a,b)=
\begin{cases}
0, & a=b,\\
\dfrac{|a-b|}{\max(|a|,|b|)}, & a\ne b.
\end{cases}
$$

El umbral es una tolerancia operativa estricta, fijada antes de ejecutar la revisión; no es un límite universal ni se ajustó utilizando TEST. Las fechas se usan para identificar los pares consecutivos, no como variables de similitud.

| Activo | Pares horarios consecutivos evaluados | Candidatos a casi-duplicados |
|--------|-------------------------------------:|----------------------------:|
| BTCUSDT | 42,822 | 0 |
| ETHUSDT | 42,822 | 0 |
| BNBUSDT | 42,822 | 0 |
| XRPUSDT | 42,822 | 0 |
| SOLUSDT | 42,822 | 0 |
| **Total** | **214,110** | **0** |

No se detectaron candidatos bajo esta regla. El resultado se limita a velas consecutivas del mismo activo: no descarta similitudes entre horas no consecutivas ni bajo tolerancias más amplias. La similitud entre velas tampoco demostraría por sí sola un error de duplicación, pues podría reflejar actividad real del mercado. No se eliminó ningún registro.

La revisión es reproducible mediante `src/05_duplicates_development.cjs`; los resultados y la huella SHA-256 del archivo de entrada se guardan en `outputs/tables/development_duplicates.json`. Solo se utilizó DEVELOPMENT y se verificó que su contenido permaneciera intacto.

#### 1.7.5 Outliers

La detección se realiza exclusivamente sobre DEVELOPMENT, con 42,833 velas por activo. Se evalúan por separado las nueve variables numéricas originales: `open`, `high`, `low`, `close`, `volume`, `quote_asset_volume`, `number_of_trades`, `taker_buy_base_asset_volume` y `taker_buy_quote_asset_volume`. No se mezclan activos para calcular los umbrales.

Como diagnóstico adicional del movimiento de precios se utiliza el retorno logarítmico horario $r_t=100\ln(C_t/C_{t-1})$, expresado en porcentaje. Solo se calcula entre velas separadas exactamente por una hora: quedan 42,822 retornos por activo. La primera vela y las diez velas inmediatamente posteriores a los huecos no tienen un retorno horario válido. No se imputan ni se calculan retornos que atraviesen esos huecos. Esta medida no es la variable objetivo de volatilidad futura.

##### Criterio de detección

Para cada activo y variable se calculan los cuartiles con interpolación lineal y el rango intercuartílico $IQR=Q_3-Q_1$. Se señalan como candidatos a valores extremos las observaciones que cumplen:

$$
x<Q_1-1.5\,IQR \quad\text{o}\quad x>Q_3+1.5\,IQR.
$$

Esta es la regla convencional de las cercas del boxplot descrita por [NIST](https://itl.nist.gov/div898/handbook/eda/section3/boxplot.htm). Los límites se calculan en las unidades originales; las escalas logarítmicas de algunas figuras solo facilitan su visualización. Se trata de un diagnóstico global de DEVELOPMENT, no de un filtro causal listo para aplicar en validación temporal.

##### Resultados y distribuciones

La siguiente tabla resume los retornos horarios y el volumen negociado en USDT. Los conteos se refieren a cada variable por separado y no deben sumarse como si fueran filas distintas.

| Activo | Retornos señalados / 42,822 | Porcentaje | Volúmenes en USDT señalados / 42,833 | Porcentaje |
|--------|--------------------------:|-----------:|-----------------------------------:|-----------:|
| BTCUSDT | 3,866 | 9.03% | 2,818 | 6.58% |
| ETHUSDT | 3,838 | 8.96% | 2,844 | 6.64% |
| BNBUSDT | 3,901 | 9.11% | 4,356 | 10.17% |
| XRPUSDT | 3,836 | 8.96% | 4,469 | 10.43% |
| SOLUSDT | 3,244 | 7.58% | 3,116 | 7.27% |

Los cuartiles, límites y conteos de todas las variables evaluadas se encuentran en la [tabla completa de resultados IQR](../../outputs/tables/development_outliers_iqr.csv).

```{figure} ../_static/figures/development_outliers_boxplots.png
:alt: Boxplots por activo de retornos horarios y volumen en USDT en DEVELOPMENT, con puntos fuera de los bigotes.

Boxplots con bigotes hasta la última observación dentro de las cercas de 1.5 IQR. Los puntos fuera de los bigotes son candidatos a extremos. Los ejes horizontales usan escala logarítmica simétrica, lineal cerca de cero, para mostrar el rango completo.
```

```{figure} ../_static/figures/development_outliers_distributions.png
:alt: Histogramas de retornos y volumen por activo, con frecuencia logarítmica y límites IQR.

Distribuciones con 80 intervalos de igual anchura y frecuencia en escala logarítmica. Las líneas discontinuas muestran los límites IQR. Los límites negativos del volumen son umbrales matemáticos, no observaciones negativas.
```

Los gráficos muestran retornos extremos en ambas direcciones y una cola hacia volúmenes altos. El porcentaje señalado depende de la distribución de cada activo. Un conteo elevado bajo esta regla no demuestra contaminación: el criterio también puede señalar episodios reales de actividad intensa y cambios entre periodos de mercado.

##### Contexto temporal

```{figure} ../_static/figures/development_outliers_timeline.png
:alt: Series temporales de retornos y volumen por activo en DEVELOPMENT; los valores fuera de los límites IQR aparecen en naranja.

Ubicación temporal de los candidatos a extremos. El color naranja identifica los valores señalados por IQR. Los retornos no se conectan a través de huecos; el volumen utiliza escala logarítmica simétrica.
```

Los mayores retornos en valor absoluto no ocurren todos en la misma fecha:

| Activo | Apertura de la vela (UTC) | Retorno logarítmico horario (%) |
|--------|--------------------------|-------------------------------:|
| BTCUSDT | 2021-01-29 08:00 | 11.6145 |
| ETHUSDT | 2021-05-19 12:00 | -14.0175 |
| BNBUSDT | 2021-02-23 08:00 | -16.8726 |
| XRPUSDT | 2020-12-29 17:00 | 24.4873 |
| SOLUSDT | 2020-09-09 14:00 | 26.7351 |

Las fechas localizan los registros que requieren revisión; no identifican por sí solas la causa económica ni certifican la exactitud de la cotización. Los máximos de volumen y sus fechas también se incluyen en la [tabla de extremos observados](../../outputs/tables/development_outliers_extremes.csv).

##### Tratamiento

Se conservan todas las observaciones. No se eliminan, recortan, winsorizan ni sustituyen los valores señalados. En series financieras, un extremo puede representar un movimiento real del mercado y no necesariamente un error de medición. Antes de corregir un registro se requerirá contrastarlo con la verificación de consistencia y la fuente original.

Los umbrales globales en niveles de precio y volumen pueden reflejar cambios de régimen durante DEVELOPMENT; por ello no se interpretan automáticamente como límites de validez. Si más adelante se adopta algún tratamiento para modelar, sus parámetros deberán estimarse dentro de cada partición de entrenamiento, sin utilizar información futura ni TEST.

El procedimiento está en `src/06_outliers_development.py`. Los resultados se guardan en `outputs/tables/development_outliers_iqr.csv` y la trazabilidad en `outputs/tables/development_outliers_metadata.json`. Se verificó mediante SHA-256 que DEVELOPMENT permaneciera intacto; TEST no fue leído.

#### 1.7.6 Valores imposibles o inconsistentes

La verificación abarca las 214,165 filas de DEVELOPMENT. Se comprueban los valores numéricos, las relaciones entre campos y la estructura temporal, sin corregir ni eliminar registros. Los conteos siguientes corresponden a filas señaladas por cada regla; una misma fila podría incumplir más de una.

| Verificación | Filas evaluadas | Filas señaladas |
|--------------|----------------:|----------------:|
| Valores no numéricos o no finitos en las nueve columnas numéricas | 214,165 | 0 |
| Precios negativos en `open`, `high`, `low` o `close` | 214,165 | 0 |
| Precios iguales a cero | 214,165 | 0 |
| Volúmenes negativos en cualquiera de los cuatro campos de volumen | 214,165 | 0 |
| `high < low` | 214,165 | 0 |
| `open` fuera de [`low`, `high`] | 214,165 | 0 |
| `close` fuera de [`low`, `high`] | 214,165 | 0 |
| Número de operaciones negativo o no entero | 214,165 | 0 |
| `open_time` o `close_time` no interpretable como fecha ISO 8601 | 214,165 | 0 |
| Fecha sin zona UTC explícita | 214,165 | 0 |
| Apertura fuera del intervalo de DEVELOPMENT | 214,165 | 0 |
| Apertura no alineada a una hora exacta | 214,165 | 0 |
| Cierre fuera de la hora de la vela | 214,165 | 0 |
| Cierre distinto de `open_time + 1 hora − 1 milisegundo` | 214,165 | 28 |
| Símbolo distinto de los cinco pares previstos | 214,165 | 0 |
| Volumen comprador taker superior al volumen total, en activo base o USDT | 214,165 | 0 |

##### Cierres anteriores al final horario convencional

Se identifican 28 registros con un cierre anterior al último milisegundo de la hora, agrupados en seis horas de apertura. Ninguno tiene una fecha inválida ni un cierre fuera de su intervalo horario.

| Apertura (UTC) | Activos afectados | Registros |
|----------------|-------------------|----------:|
| 2020-12-21 13:00 | Los cinco activos | 5 |
| 2021-02-11 03:00 | Los cinco activos | 5 |
| 2021-04-25 04:00 | Los cinco activos | 5 |
| 2021-08-13 01:00 | Los cinco activos | 5 |
| 2021-12-24 04:00 | BNBUSDT, BTCUSDT y ETHUSDT | 3 |
| 2023-03-24 12:00 | Los cinco activos | 5 |

Por ejemplo, la vela BTCUSDT del 25 de abril de 2021 a las 04:00 UTC registra cierre a las 04:00:58.146 UTC. Las cinco velas del 13 de agosto de 2021 a las 01:00 UTC registran cierre a las 01:59:59 UTC, con una diferencia de 999 milisegundos frente al cierre convencional. La lectura admite fechas ISO 8601 con o sin fracción de segundo.

Cinco de estas seis horas preceden a intervalos de ausencia documentados en 1.7.2. Esa coincidencia justifica contrastar los registros con la fuente original, pero no demuestra una interrupción del mercado ni un error de adquisición. Las 28 velas se conservan y quedan identificadas en el [detalle de registros señalados](../../outputs/tables/development_consistency_flags.csv). Para construir retornos y ventanas analíticas, sus cierres se enmascaran sin modificar el archivo original; las ventanas afectadas quedan invalidadas según el protocolo común.

##### Coherencia de unidades y volúmenes

Los pares se cotizan frente a USDT. Como comprobación interna, se divide `quote_asset_volume` entre `volume`: cuando el volumen base es positivo, el cociente expresa un precio medio implícito y se contrasta con [`low`, `high`]. Se realiza la misma comprobación con los dos campos de volumen comprador taker.

| Comprobación | Filas evaluadas | Filas señaladas |
|--------------|----------------:|----------------:|
| Precio implícito del volumen total fuera de [`low`, `high`] | 214,155 | 0 |
| Precio implícito del volumen taker fuera de [`low`, `high`] | 214,155 | 0 |
| Volumen base cero con volumen cotizado no cero, total o taker | 10 | 0 |
| Cero operaciones con algún volumen no cero | 10 | 0 |

En las diez filas con volumen base cero no se calcula un cociente; se comprueba por separado la coherencia de los ceros. Las comparaciones de precios implícitos y volúmenes admiten una tolerancia de redondeo de $10^{-8}$ absoluta más $10^{-8}$ relativa a la magnitud comparada. Para comprobar ceros se utiliza una tolerancia absoluta de $10^{-8}$. Las relaciones OHLC y las comprobaciones de signo no utilizan esa tolerancia.

No se detectan contradicciones bajo estas reglas de unidades y volúmenes. Esto no certifica la escala ni la exactitud frente a una fuente externa: un cambio de escala aplicado de forma coherente a varios campos podría superar las comprobaciones internas. No se declaran los datos libres de cualquier error posible.

El procedimiento está en `src/07_consistency_development.py`. La [tabla completa de comprobaciones](../../outputs/tables/development_consistency_checks.csv) distingue las filas evaluadas de aquellas a las que no aplica cada condición. La huella SHA-256 confirma que DEVELOPMENT permaneció intacto. TEST no fue leído.

#### 1.7.7 Sesgos de muestreo y representatividad

Los datos provienen exclusivamente de Binance Spot y no incluyen otros exchanges. Solo se seleccionaron cinco activos, lo que introduce un posible sesgo de selección de activos. Además, el periodo observado está limitado a 2020–2026.

Los resultados no deben generalizarse automáticamente a todas las criptomonedas, a todos los exchanges ni a todos los periodos de mercado. La evaluación de estas limitaciones deberá profundizarse en etapas posteriores.

### 1.8 Consideraciones éticas

#### Datos personales

**NO APLICA.** El dataset contiene datos agregados de mercado y no información personal identificable de individuos.

#### Anonimización

**NO APLICA.** No existen personas identificables que deban anonimizarse.

#### Riesgo de reidentificación

**NO APLICA.** No se utilizan datos individuales, biométricos, médicos, demográficos ni coordenadas personales.

#### Consideraciones adicionales

Existe una responsabilidad en la interpretación y comunicación de los resultados: el modelo que se desarrolle será experimental y no constituirá asesoría financiera. Las predicciones contendrán incertidumbre y los resultados históricos no garantizan el comportamiento futuro.

```{admonition} Reserva metodológica del conjunto de prueba
:class: important

El conjunto TEST fue reservado cronológicamente antes de realizar el EDA o cualquier decisión de preprocesamiento.

DEVELOPMENT:
2020-08-11 06:00 UTC a 2025-07-01 18:00 UTC.

TEST:
2025-07-01 19:00 UTC a 2026-09-20 23:00 UTC.

TEST no será utilizado para selección de variables, transformaciones, rezagos, ventanas, escalado, hiperparámetros ni validación cruzada.
```
