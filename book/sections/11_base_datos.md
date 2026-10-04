# 1. Base de datos

## Contexto del dataset

Se predice la volatilidad de BTCUSDT, ETHUSDT, BNBUSDT y XRPUSDT para los siguientes siete días. El periodo es **1 de enero de 2023 a 31 de diciembre de 2025, UTC**. La fuente es Binance, con cierres de **un minuto**. Se conservan 1.578.159 observaciones válidas por activo y 81 minutos faltantes; el día incompleto es el 24 de marzo de 2023. Se excluyen las ventanas afectadas, sin interpolación.

La frecuencia de adquisición es un minuto, la construcción de características resume información intradía y cada muestra supervisada se origina al cierre de un día UTC. Por tanto, millones de observaciones de minuto no equivalen a millones de ejemplos de entrenamiento. Las fechas se etiquetan por el día cuyo cierre ya se conoce.

(seleccion-dataset)=

## 1.1 Definición del problema de investigación

¿Puede un modelo clásico que utiliza información histórica diaria e intradía reducir el error de pronóstico de volatilidad a uno–siete días frente a repetir la volatilidad actual? La comparación se realiza por activo, ventana objetivo y horizonte, con selección temporal en 2024 y evaluación retrospectiva en 2025.

## 1.2 Justificación de la selección del dataset

Binance Vision permite reconstruir cierres y retornos con una fuente homogénea y archivos verificables mediante SHA-256. La frecuencia de minuto aporta medidas de variación intradía que los cierres diarios por sí solos no contienen. Se utiliza el mismo mercado spot y denominador USDT para mantener consistente la interpretación de precios. BTC, ETH, BNB y XRP constituyen la muestra disponible y comparable del proyecto; no se afirma que representen todo el mercado ni que su selección resulte de un ranking de liquidez. SOL queda fuera del comparativo porque no tiene datos y resultados equivalentes en los artefactos vigentes.

El periodo 2023–2025 permite separar un primer año de historia, un año de selección y un año de evaluación. Es una decisión de diseño del estudio y no demuestra estabilidad entre regímenes. Las ventanas de 7, 14, 21 y 28 días comparan definiciones semanales de volatilidad. Los resultados se limitan a estos activos, esta fuente y este periodo; la cotización USDT tampoco equivale automáticamente a dólares estadounidenses.

(fuente-licencia)=

## 1.3 Fuente de los datos y licencia de uso

Fuente: [Binance Vision](https://data.binance.vision/), archivos spot mensuales de klines de un minuto. La [documentación oficial](https://github.com/binance/binance-public-data) describe campos, intervalos, checksums y el cambio a timestamps en microsegundos desde enero de 2025. Los scripts convierten las fechas a UTC y conservan hashes para identificar la versión descargada.

Consulta documental: **3 de octubre de 2026**. Los [Binance Vision Dataset Terms](https://github.com/binance/binance-public-data/blob/master/TERMS_AND_CONDITIONS.md), versión 1.0, actualizados el 26 de agosto de 2026, establecen CC BY-NC-SA 4.0 salvo designación distinta. Incluyen investigación académica no comercial y requieren atribución y compartir bajo la misma licencia las obras derivadas redistribuidas. La mención MIT del repositorio no se utiliza como licencia del dataset.

Atribución para esta entrega: **Datos de mercado: Binance Vision; procesamiento y análisis: Jassan Arteta y Mateo Bernal.** Las transformaciones incluyen selección de cierres, agregación diaria, cálculo de retornos y construcción de características y objetivos. No se atribuye patrocinio de Binance. Queda pendiente conservar evidencia de los términos aplicables en la fecha de descarga: los términos actuales indican que sus cambios no alteran los derechos de descargas anteriores. Esta consulta documenta las condiciones publicadas actualmente y no certifica la licencia histórica de cada archivo.

(diccionario-variables)=

## 1.4 Diccionario de variables: tipo, unidad y significado

En las siguientes definiciones, los retornos diarios y de minuto se expresan como `100 × log(P_actual/P_anterior)`, y `b_t = max(volatilidad_actual, 1e-8)`. Las seis características se repiten con rezagos `k = 0,…,L−1`; todos se normalizan con la base del origen actual `b_t`.

| Variable | Tipo | Unidad | Significado y disponibilidad |
| --- | --- | --- | --- |
| `symbol` | Categórica | Identificador | BTCUSDT, ETHUSDT, BNBUSDT o XRPUSDT; identifica el par spot. |
| Timestamp de minuto | Fecha y hora | UTC | Inicio del intervalo de adquisición; el cierre se conoce al terminar ese minuto. |
| `date` / `origin` | Fecha | Día UTC | Día cuyo cierre ya está disponible; origen del pronóstico. |
| Cierre de minuto | Numérica continua | USDT por unidad del activo | Último precio del intervalo; arrays `.npy` por activo. |
| Cierre diario `P_t` | Numérica continua | USDT por unidad del activo | Último cierre del día completo; un día incompleto permanece faltante. |
| Retorno diario `r_t` | Numérica continua | Porcentaje logarítmico | Cambio entre cierres diarios consecutivos. |
| Volatilidad histórica `sigma_t` | Numérica continua | Puntos porcentuales | Desviación poblacional de los últimos `w` retornos diarios, sin anualizar. |
| `retorno_diario_relativo` | Numérica continua | Adimensional | Retorno del día rezagado dividido por `b_t`. |
| `retorno_cuadrado_relativo` | Numérica continua | Adimensional | Retorno diario al cuadrado dividido por `b_t²`. |
| `volatilidad_minuto_relativa` | Numérica continua | Adimensional | Raíz de la suma de cuadrados de retornos de minuto del día, dividida por `b_t`. |
| `retorno_absoluto_minuto_relativo` | Numérica continua | Adimensional | Suma de retornos de minuto absolutos dividida por `sqrt(1440)` y por `b_t`. |
| `volatilidad_negativa_relativa` | Numérica continua | Adimensional | Raíz de la suma de cuadrados de la parte negativa de retornos de minuto, dividida por `b_t`. |
| `max_retorno_minuto_relativo` | Numérica continua | Adimensional | Máximo retorno de minuto absoluto del día, dividido por `b_t`. |
| `decaimiento_conocido_h1` … `h7` | Numérica continua | Adimensional | Referencia de cambio relativo por salida de retornos antiguos; fórmula descrita en preprocesamiento. Solo usa historia conocida. |
| Objetivo `y_t,h` | Numérica continua | Puntos porcentuales | Volatilidad diaria móvil en `t+h`; disponible posteriormente, no como predictor. |
| Objetivo transformado `z_t,h` | Numérica continua | Adimensional | `y_t,h / b_t − 1`; se utiliza para ajustar el regresor. |
| `volatility_window` / `input_window` | Entera | Días | Ventana del objetivo `w` y cantidad de días de entrada `L`. |
| `horizon` | Entera | Días | Distancia futura de la salida, de 1 a 7. |

El diccionario cubre las entradas del SVR lineal, con dimensión `6L+7`. Los archivos originales contienen otros campos OHLCV; volumen y rango OHLC no forman parte del dataset procesado compartido.


## 1.5 Estructura de los datos

La estructura es un panel temporal multiactivo: la observación original es un activo-minuto UTC y la muestra supervisada es un activo-día de origen. La clave conceptual es `(symbol, timestamp)` en minuto y `(symbol, origin)` en el modelo. El archivo diario usa formato ancho: una fila por fecha y una columna de cierre por activo. Los modelos se ajustan por activo y configuración; no se presume independencia entre activos, días u objetivos superpuestos. No hay coordenadas ni componente espacial geográfico.

## 1.6 Tamaño de la muestra

Se dispone de cuatro activos y 1.096 fechas diarias por activo. La adquisición
supera 20.000 observaciones por activo, pero el tamaño efectivo del modelado
es diario: no se cuentan los minutos como muestras supervisadas independientes.
La dimensión del SVR es 49, 91, 133 o 175 características según la entrada.
Los tamaños de entrenamiento y validación se detallan en el capítulo del modelo.

### 1.6.1 Número de variables y relación n/p

El número de predictores es `p = 6L+7`: 49, 91, 133 o 175. Hay siete salidas,
una por horizonte, y se ajusta un regresor por salida. Para el primer corte
publicado, `n = 274` orígenes de entrenamiento por activo: `n/p` es 5,59;
3,01; 2,06 o 1,57, respectivamente. Estos cocientes describen filas disponibles,
no observaciones independientes; cambian en los cortes posteriores. El
[capítulo del modelo](13_modelo_base_svr.md) informa los tamaños de cada corte.

### 1.6.2 Rango de la variable objetivo

Es un problema de regresión; los casos por clase no aplican. El objetivo es
volatilidad no anualizada, en puntos porcentuales, con ventanas de 7, 14, 21
y 28 días y horizontes de 1 a 7 días. **Pendiente:** reportar mínimo y máximo
observados del objetivo por activo y ventana, indicando el periodo calculado.
Los extremos de retornos del EDA no sustituyen el rango de volatilidad.

### 1.6.3 Entidades y tamaño efectivo

Se estudian cuatro activos, con 1.096 fechas diarias por activo. No se acredita
que sean cuatro entidades estadísticamente independientes: comparten mercado
y pueden estar correlacionados. También hay dependencia temporal y objetivos
superpuestos. **El mínimo de 20.000 se supera en registros de adquisición,
pero no en muestras supervisadas por activo.** No se acredita el cumplimiento
del mínimo para el modelado ni se ha estimado el tamaño efectivo independiente.

## 1.7 Calidad de los datos

### 1.7.1 Valores faltantes: patrón y mecanismo

Se reportan 81 minutos faltantes por activo y un día incompleto, el 24 de marzo
de 2023. Se mantiene el calendario y se excluyen ventanas afectadas, sin
interpolación. **Pendiente:** matriz de nulos del dataset vigente y evidencia
para discutir MCAR, MAR o MNAR. La coincidencia temporal de los huecos no
permite asignar por sí sola un mecanismo de ausencia.

### 1.7.2 Duplicados exactos y casi duplicados

La descarga implementa controles de duplicados y orden temporal.
**Pendiente:** consolidar conteos de duplicados exactos y casi duplicados por
activo y archivo, definir el criterio de casi duplicado e informar tratamiento.
Un precio repetido en minutos distintos no demuestra un registro duplicado.

### 1.7.3 Outliers: detección y tratamiento

El [EDA](12_eda.md) documenta extremos de retornos por activo. Se conservan
movimientos de mercado y se reportan MAE y RMSE. **Pendiente:** documentar un
criterio de detección, conteos y análisis de sensibilidad sobre el dataset
vigente; los extremos descriptivos no completan este requisito.

### 1.7.4 Valores imposibles o inconsistentes

La descarga incluye controles de precios finitos positivos, alineación de
timestamps y días completos. **Pendiente:** informe de ejecución con conteos
antes y después de exclusiones por activo y archivo, incluyendo unidades y
conversión de timestamps. La implementación de controles no acredita sus
resultados en todos los archivos utilizados.

### 1.7.5 Sesgos de muestreo y representatividad

La muestra contiene cuatro mercados spot de un único proveedor. No representa
todos los activos, plataformas ni regímenes de mercado. La selección responde
a disponibilidad comparable, no a muestreo probabilístico; las conclusiones
se limitan a estos activos y al periodo 2023-2025.

## 1.8 Consideraciones éticas

Los datos son precios agregados públicos; no contienen identificadores
personales ni coordenadas geográficas. Las marcas temporales corresponden a
intervalos de mercado y no a transacciones de personas identificadas. No se
requiere anonimización de personas en las variables utilizadas; esta revisión
no cubre datos individuales externos que pudieran vincularse posteriormente.
La publicación debe respetar las condiciones de uso y atribución de la fuente.

## Revisión numerada de cumplimiento de la guía

La revisión evalúa la evidencia documental disponible en los capítulos
vigentes; no sustituye una nueva auditoría de los archivos de datos.
**Cumple** indica cobertura documental, **parcial** indica evidencia incompleta
y **pendiente** indica que falta el resultado solicitado.

| Punto | Requisito | Estado | Evidencia o ajuste necesario |
| --- | --- | --- | --- |
| 1.1 | Problema de investigación | Cumple | Pregunta, activos, objetivo y horizontes definidos. |
| 1.2 | Justificación del dataset | Cumple | Fuente homogénea, frecuencia intradía y periodo justificados. |
| 1.3 | Fuente y licencia | Parcial | Enlaces y atribución documentados; falta evidencia de términos a la fecha de descarga. |
| 1.4 | Diccionario: tipo, unidad y significado | Cumple | Tabla de variables utilizadas y disponibilidad temporal. |
| 1.5 | Unidad, agregación y tipo de estructura | Cumple | Activo-minuto, activo-día y panel temporal descritos. |
| 1.6 | Mínimo de 20.000 observaciones | Parcial | Se alcanza en adquisición; no en muestras supervisadas por activo. |
| 1.6.1 | Número de variables y n/p | Cumple | Dimensión y cocientes del primer corte; tamaños posteriores en el capítulo 3. |
| 1.6.2 | Rango del objetivo en regresión | Pendiente | Falta mínimo y máximo de volatilidad por activo y ventana. |
| 1.6.3 | Entidades independientes y tamaño efectivo | Parcial | Cuatro activos identificados; dependencia reconocida, tamaño efectivo sin estimar. |
| 1.7.1 | Faltantes: patrón, visualización y mecanismo | Parcial | Conteos y fecha disponibles; faltan matriz y evidencia del mecanismo. |
| 1.7.2 | Duplicados exactos y casi duplicados | Pendiente | Faltan conteos, criterio de casi duplicado y reporte de tratamiento. |
| 1.7.3 | Outliers: detección y tratamiento | Parcial | Extremos y conservación descritos; faltan criterio, conteos y sensibilidad. |
| 1.7.4 | Valores imposibles o inconsistentes | Parcial | Controles implementados; falta informe consolidado de ejecución. |
| 1.7.5 | Sesgos y representatividad | Cumple | Límites de selección, proveedor y periodo explícitos. |
| 1.8 | Datos personales, anonimización y reidentificación | Cumple | Alcance agregado y ausencia de identificadores documentados. |
