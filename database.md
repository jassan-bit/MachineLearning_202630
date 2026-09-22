# 1. Base de Datos

## 1.1 Definición del problema de investigación

Este proyecto aborda un problema de **regresión temporal aplicado al mercado de criptomonedas**.

El objetivo consiste en utilizar información histórica de mercado para construir un modelo capaz de predecir una variable continua asociada al comportamiento futuro de los rendimientos de los criptoactivos.

Los criptoactivos considerados son:

- BTCUSDT
- ETHUSDT
- BNBUSDT
- XRPUSDT
- SOLUSDT

Debido a que las observaciones poseen fecha y hora, se encuentran ordenadas cronológicamente y presentan dependencia temporal, el problema corresponde a la **Ruta C: Regresión temporal** definida para el proyecto.

La variable objetivo se construirá a partir del **rendimiento logarítmico futuro del precio de cierre**, preservando estrictamente el orden temporal de las observaciones.

---

## 1.2 Justificación de la selección del dataset

El mercado de criptomonedas constituye un entorno apropiado para el estudio de modelos de Machine Learning aplicados a series temporales debido a características como:

- Alta frecuencia de observación.
- Elevada volatilidad.
- Comportamientos no lineales.
- Presencia de movimientos extremos.
- Cambios de régimen.
- Dependencia temporal.
- Operación continua durante las 24 horas del día.

Estas características permiten desarrollar un problema de regresión temporal con un número elevado de observaciones y evaluar rigurosamente aspectos como el orden cronológico, la volatilidad, la construcción de variables rezagadas y la prevención de fuga de información.

El dataset seleccionado supera ampliamente el número mínimo de observaciones requerido para el desarrollo del proyecto.

---

## 1.3 Fuente de los datos y condiciones de uso

Los datos fueron obtenidos directamente del mercado **Spot de Binance** mediante su interfaz pública de datos de mercado.

Se utilizó el endpoint de velas o *klines*:

**URL oficial:** <https://data-api.binance.vision/api/v3/klines>

Este endpoint público (GET /api/v3/klines) permite descargar directamente las velas históricas del mercado Spot de Binance.

Los pares analizados son:

- BTCUSDT
- ETHUSDT
- BNBUSDT
- XRPUSDT
- SOLUSDT

La frecuencia seleccionada es de **una hora (1h)** y todas las marcas temporales se manejan en **UTC**.

El periodo inicial de extracción fue establecido entre el **1 de enero de 2020** y el **20 de septiembre de 2026**.

Debido a que SOLUSDT comenzó a disponer de observaciones posteriormente, el análisis conjunto utiliza el periodo temporal común a los cinco criptoactivos.

Los datos corresponden a información pública de mercado. La fuente, el periodo, la frecuencia y el procedimiento de extracción se documentan con el propósito de garantizar la trazabilidad y reproducibilidad del análisis.

Las condiciones específicas de utilización y redistribución de la información estarán sujetas a los términos vigentes establecidos por el proveedor de los datos.

---

## 1.4 Diccionario de variables

| Variable | Tipo | Unidad | Significado |
|---|---|---|---|
| `symbol` | Categórica | - | Identificador del par negociado |
| `open_time` | Temporal | UTC | Fecha y hora de inicio de la vela |
| `open` | Numérica continua | USDT por unidad del activo base | Precio de apertura |
| `high` | Numérica continua | USDT por unidad del activo base | Precio máximo alcanzado durante la vela |
| `low` | Numérica continua | USDT por unidad del activo base | Precio mínimo alcanzado durante la vela |
| `close` | Numérica continua | USDT por unidad del activo base | Precio de cierre |
| `volume` | Numérica continua | Activo base | Volumen negociado del activo base |
| `close_time` | Temporal | UTC | Fecha y hora de cierre de la vela |
| `quote_asset_volume` | Numérica continua | USDT | Volumen negociado expresado en el activo cotizado |
| `number_of_trades` | Numérica discreta | Operaciones | Número de operaciones registradas durante la vela |
| `taker_buy_base_volume` | Numérica continua | Activo base | Volumen de compras *taker* expresado en el activo base |
| `taker_buy_quote_volume` | Numérica continua | USDT | Volumen de compras *taker* expresado en el activo cotizado |

---

## 1.5 Estructura de los datos

La unidad de observación está definida por la combinación:

**criptoactivo × hora**

El nivel de agregación temporal es de **una hora**.

El dataset presenta una estructura de **panel temporal multiactivo**, debido a que varios criptoactivos son observados repetidamente a través del tiempo.

Las características principales de la estructura son:

- **Número de criptoactivos:** 5.
- **Frecuencia temporal:** 1 hora.
- **Zona horaria:** UTC.
- **Mercado:** Binance Spot.
- **Tipo de problema:** regresión temporal.
- **Variable objetivo:** continua.
- **Componente espacial:** no aplica.

Para el análisis conjunto se utiliza un periodo común en el cual los cinco criptoactivos poseen exactamente las mismas marcas temporales.

La estructura cronológica de los datos obliga a mantener el orden temporal durante todas las etapas de partición, preprocesamiento, entrenamiento y evaluación.

---

## 1.6 Tamaño de la muestra

El dataset correspondiente al periodo temporal común contiene:

- **267.595 observaciones**.
- **5 criptoactivos**.
- **12 variables originales**.
- Aproximadamente **53.519 observaciones por criptoactivo** antes de construir la variable objetivo.

La relación entre el número de observaciones y el número de variables originales es:

$$
\frac{n}{p}
=
\frac{267\,595}{12}
\approx 22\,299.58
$$

donde:

- $n = 267\,595$ corresponde al número de observaciones.
- $p = 12$ corresponde al número de variables originales.

Por lo tanto, existe un número de observaciones considerablemente superior al número de variables.

El tamaño de la muestra supera ampliamente el mínimo de **20.000 observaciones** establecido para el entregable.

### Entidades independientes

El dataset contiene cinco entidades principales correspondientes a los pares:

1. BTCUSDT
2. ETHUSDT
3. BNBUSDT
4. XRPUSDT
5. SOLUSDT

Cada entidad es observada repetidamente a través del tiempo.

### Rango de la variable objetivo

La variable objetivo es continua y corresponde al rendimiento logarítmico futuro.

Su rango, tendencia central, dispersión, asimetría, curtosis y comportamiento en las colas serán estudiados detalladamente durante el Análisis Exploratorio de Datos.

---

## 1.7 Calidad de los datos

La calidad del dataset fue evaluada antes de iniciar el modelado.

Se verificaron valores faltantes, duplicados, consistencia de precios, volúmenes, continuidad temporal y presencia de valores extremos.

### 1.7.1 Valores faltantes

El control realizado sobre las variables originales no identificó valores faltantes.

Los valores ausentes encontrados posteriormente corresponden únicamente a variables auxiliares derivadas.

Por ejemplo, la variable utilizada para medir la diferencia temporal entre observaciones presenta naturalmente un valor ausente en la primera observación de cada criptoactivo, debido a que no existe una observación anterior con la cual realizar la comparación.

Estos valores no representan pérdida de información del dataset original.

Debido a que las variables originales no presentan valores faltantes, no resulta necesario clasificar un mecanismo de ausencia como:

- MCAR (*Missing Completely At Random*).
- MAR (*Missing At Random*).
- MNAR (*Missing Not At Random*).

Tampoco se requiere realizar imputación sobre las variables originales.

---

### 1.7.2 Duplicados exactos y casi duplicados

No se identificaron duplicados exactos.

Tampoco se identificaron registros duplicados para la combinación:

`symbol + open_time`

Esta combinación identifica de manera única cada observación dentro del panel temporal.

La revisión de registros casi duplicados se mantendrá como parte de la auditoría del dataset antes del modelado definitivo.

---

### 1.7.3 Outliers

El mercado de criptomonedas puede presentar movimientos de precios y rendimientos considerablemente elevados.

Por esta razón, un valor extremo no será considerado automáticamente un error.

El procedimiento utilizado será:

1. Detectar estadísticamente las observaciones extremas.
2. Compararlas con los precios originales.
3. Verificar su coherencia temporal.
4. Diferenciar errores de registro de movimientos reales del mercado.
5. Conservar los movimientos extremos legítimos.

Esta decisión es especialmente importante debido a que los eventos extremos pueden contener información relevante para la predicción financiera.

---

### 1.7.4 Valores imposibles o inconsistentes

Se realizaron controles sobre las variables de precios OHLC, volumen, número de operaciones y marcas temporales.

No se identificaron:

- Precios menores o iguales a cero.
- Volúmenes negativos.
- Número de operaciones negativo.
- Valores infinitos.
- Registros duplicados.
- Inconsistencias entre `Open`, `High`, `Low` y `Close`.
- Registros cuyo tiempo de cierre sea anterior o igual al tiempo de apertura.

Las relaciones utilizadas para comprobar la consistencia de las variables OHLC incluyen:

$$
High \geq Open
$$

$$
High \geq Close
$$

$$
High \geq Low
$$

$$
Low \leq Open
$$

$$
Low \leq Close
$$

No se detectaron violaciones de estas condiciones.

---

### 1.7.5 Continuidad temporal

Durante la auditoría temporal se detectaron **10 discontinuidades temporales** comunes a los cinco criptoactivos.

Las interrupciones aparecen simultáneamente en:

- BTCUSDT
- ETHUSDT
- BNBUSDT
- XRPUSDT
- SOLUSDT

Además, se verificó que, dentro del periodo temporal común, los cinco activos poseen exactamente el mismo conjunto de marcas temporales.

Por tanto, el panel se encuentra sincronizado entre activos.

No se realizará interpolación artificial de precios para completar las discontinuidades detectadas.

En su lugar, las discontinuidades serán identificadas explícitamente y la construcción de:

- rezagos,
- rendimientos,
- ventanas móviles,
- indicadores de volatilidad

se realizará respetando los segmentos temporales continuos.

Esto evita generar retornos artificiales a través de periodos sin observaciones.

---

### 1.7.6 Sesgos de muestreo y representatividad

Los datos proceden exclusivamente del mercado **Spot de Binance**.

Por esta razón, los resultados representan directamente el comportamiento de los pares seleccionados dentro de este mercado y no necesariamente el comportamiento completo del mercado global de criptomonedas.

También existe un criterio de selección asociado a utilizar cinco criptoactivos de alta relevancia y liquidez.

Los resultados deberán interpretarse teniendo en cuenta estas características y no generalizarse automáticamente a:

- Otros exchanges.
- Criptoactivos con menor liquidez.
- Mercados de derivados.
- Otros periodos históricos.
- La totalidad del mercado global de activos digitales.

---

## 1.8 Consideraciones éticas

El dataset contiene exclusivamente información pública y agregada del mercado.

No contiene:

- Nombres de personas.
- Identificadores personales.
- Información financiera individual.
- Direcciones de usuarios.
- Información médica.
- Coordenadas geográficas personales.
- Datos demográficos.
- Información privada de participantes del mercado.

Las marcas temporales corresponden a velas agregadas y no permiten identificar a participantes individuales.

Por esta razón, el riesgo de reidentificación de personas es mínimo.

Los datos serán utilizados exclusivamente con fines académicos y de investigación.

Asimismo, se mantendrá la trazabilidad de la fuente y se documentarán las transformaciones realizadas sobre los datos con el propósito de favorecer la transparencia y reproducibilidad del estudio.
