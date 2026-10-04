# 1. Base de datos

## Objetivo y datos

Se predice la volatilidad de BTCUSDT, ETHUSDT, BNBUSDT y XRPUSDT para los siguientes siete días. El periodo es **1 de enero de 2023 a 31 de diciembre de 2025, UTC**. La fuente es Binance, con cierres de **un minuto**. Se conservan 1.578.159 observaciones válidas por activo y 81 minutos faltantes; el día incompleto es el 24 de marzo de 2023. Se excluyen las ventanas afectadas, sin interpolación.

La frecuencia de adquisición es un minuto, la construcción de características resume información intradía y cada muestra supervisada se origina al cierre de un día UTC. Por tanto, millones de observaciones de minuto no equivalen a millones de ejemplos de entrenamiento. Las fechas se etiquetan por el día cuyo cierre ya se conoce.

(seleccion-dataset)=

## Pregunta de investigación y selección del dataset

¿Puede un modelo clásico que utiliza información histórica diaria e intradía reducir el error de pronóstico de volatilidad a uno–siete días frente a repetir la volatilidad actual? La comparación se realiza por activo, ventana objetivo y horizonte, con selección temporal en 2024 y evaluación retrospectiva en 2025.

Binance Vision permite reconstruir cierres y retornos con una fuente homogénea y archivos verificables mediante SHA-256. La frecuencia de minuto aporta medidas de variación intradía que los cierres diarios por sí solos no contienen. Se utiliza el mismo mercado spot y denominador USDT para mantener consistente la interpretación de precios. BTC, ETH, BNB y XRP constituyen la muestra disponible y comparable del proyecto; no se afirma que representen todo el mercado ni que su selección resulte de un ranking de liquidez. SOL queda fuera del comparativo porque no tiene datos y resultados equivalentes en los artefactos vigentes.

El periodo 2023–2025 permite separar un primer año de historia, un año de selección y un año de evaluación. Es una decisión de diseño del estudio y no demuestra estabilidad entre regímenes. Las ventanas de 7, 14, 21 y 28 días comparan definiciones semanales de volatilidad. Los resultados se limitan a estos activos, esta fuente y este periodo; la cotización USDT tampoco equivale automáticamente a dólares estadounidenses.

(fuente-licencia)=

## Fuente, condiciones de uso y atribución

Fuente: [Binance Vision](https://data.binance.vision/), archivos spot mensuales de klines de un minuto. La [documentación oficial](https://github.com/binance/binance-public-data) describe campos, intervalos, checksums y el cambio a timestamps en microsegundos desde enero de 2025. Los scripts convierten las fechas a UTC y conservan hashes para identificar la versión descargada.

Consulta documental: **3 de octubre de 2026**. Los [Binance Vision Dataset Terms](https://github.com/binance/binance-public-data/blob/master/TERMS_AND_CONDITIONS.md), versión 1.0, actualizados el 26 de agosto de 2026, establecen CC BY-NC-SA 4.0 salvo designación distinta. Incluyen investigación académica no comercial y requieren atribución y compartir bajo la misma licencia las obras derivadas redistribuidas. La mención MIT del repositorio no se utiliza como licencia del dataset.

Atribución para esta entrega: **Datos de mercado: Binance Vision; procesamiento y análisis: Jassan Arteta y Mateo Bernal.** Las transformaciones incluyen selección de cierres, agregación diaria, cálculo de retornos y construcción de características y objetivos. No se atribuye patrocinio de Binance. Queda pendiente conservar evidencia de los términos aplicables en la fecha de descarga: los términos actuales indican que sus cambios no alteran los derechos de descargas anteriores. Esta consulta documenta las condiciones publicadas actualmente y no certifica la licencia histórica de cada archivo.

(diccionario-variables)=

## Estructura y diccionario de variables

La estructura es un panel temporal multiactivo: la observación original es un activo–minuto UTC y la muestra supervisada es un activo–día de origen. La clave conceptual es `(symbol, timestamp)` en minuto y `(symbol, origin)` en el modelo. El archivo diario usa formato ancho: una fila por fecha y una columna de cierre por activo. Los modelos se ajustan por activo y configuración; no se presume independencia entre activos, días u objetivos superpuestos. No hay coordenadas ni componente espacial geográfico.

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


## Tamaño, representatividad y ética

Se dispone de cuatro activos y 1.096 fechas diarias por activo. La adquisición
supera 20.000 observaciones por activo, pero el tamaño efectivo del modelado
es diario: no se cuentan los minutos como muestras supervisadas independientes.
La dimensión del SVR es 49, 91, 133 o 175 características según la entrada.
Los tamaños de entrenamiento y validación se detallan en el capítulo del modelo.

La muestra contiene cuatro mercados spot de un único proveedor. No representa
todos los activos, plataformas ni regímenes de mercado. Los datos son precios
agregados públicos; no contienen identificadores personales. La publicación
debe respetar las condiciones de uso y atribución de la fuente.
