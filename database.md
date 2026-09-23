# 1. Base de Datos

## Fuente, frecuencia y período

La fuente es **Binance Spot**, mediante su endpoint público de velas históricas:

[https://data-api.binance.vision/api/v3/klines](https://data-api.binance.vision/api/v3/klines)

La frecuencia es **1h** y el período solicitado es **2020-01-01 a 2026-09-20**, con la fecha final interpretada como día completo en UTC. Se consideran cinco activos: **BTCUSDT, ETHUSDT, BNBUSDT, XRPUSDT y SOLUSDT**. La disponibilidad efectiva de cada serie puede comenzar después del inicio solicitado.

El flujo de adquisición descarga automáticamente los datos mediante paginación cuando no dispone de la caché local, o cuando se fuerza explícitamente una descarga. El archivo `data/processed/crypto_binance_master_1h.csv` funciona como **caché reproducible**: con `FORCE_DOWNLOAD = False` y el archivo existente, se carga el CSV sin consultar de nuevo Binance. Este libro utiliza la caché y las salidas existentes; no descarga ni regenera datos.

## Variables originales y símbolo

Las velas proporcionan las siguientes variables:

| Variable | Descripción |
|---|---|
| `open_time` | Fecha y hora de apertura de la vela. |
| `open` | Precio de apertura. |
| `high` | Precio máximo. |
| `low` | Precio mínimo. |
| `close` | Precio de cierre. |
| `volume` | Volumen negociado en el activo base. |
| `close_time` | Fecha y hora de cierre de la vela. |
| `quote_asset_volume` | Volumen negociado en el activo de cotización. |
| `number_of_trades` | Número de operaciones. |
| `taker_buy_base_asset_volume` | Volumen comprador taker en el activo base. |
| `taker_buy_quote_asset_volume` | Volumen comprador taker en el activo de cotización. |
| `ignore` | Campo auxiliar del endpoint, descartado durante el procesamiento. |

Se añade `symbol` para identificar cada criptomoneda. Se conservan las variables OHLCV y de actividad suministradas por la fuente, aunque la entrada del MLP utiliza únicamente la historia de `volatility`.

## Procesamiento temporal y período común

Los timestamps numéricos recibidos del endpoint se convierten desde milisegundos a UTC. Al leer el CSV, las fechas ISO con formatos mixtos se convierten mediante `pd.to_datetime(..., format="mixed", utc=True)` para `open_time` y `close_time`. Las variables numéricas se convierten a sus tipos correspondientes.

Las observaciones se ordenan por `symbol` y `open_time` y se eliminan duplicados de esa pareja. El período común se obtiene tomando la **mayor fecha inicial** y la **menor fecha final** disponibles entre los cinco activos. A continuación, cada serie se restringe a ese intervalo compartido.

La diferencia entre aperturas consecutivas se calcula por símbolo en `delta_h`. Cuando no es exactamente una hora, se marca `new_segment` y se inicia un nuevo `segment_id`. Los identificadores de segmento se interpretan junto con el símbolo. No se rellenan huecos ni se construyen retornos o ventanas supervisadas atravesando discontinuidades.

## Retorno logarítmico y variable objetivo

Dentro de cada símbolo y segmento continuo se calcula:

$$r_t = \ln\left(\frac{P_t}{P_{t-1}}\right),$$

donde $P_t$ es `close`. Esta variable se almacena como `log_return`.

El target único es `volatility`, la desviación estándar móvil muestral de los últimos **30 retornos horarios**:

$$\sigma_t = \operatorname{std}(r_{t-29},\ldots,r_t).$$

Se utiliza `window=30`, `min_periods=30` y `ddof=1`, sin anualizar. El primer retorno de cada segmento no está definido y se requieren 31 cierres consecutivos para obtener 30 retornos válidos. Solo se excluyen las filas no utilizables por `log_return` o `volatility`, y se comprueba que no existan infinitos.

La serie limpia alimenta el EDA y las ventanas supervisadas del modelo. Los valores **7, 14, 21 y 28** representan longitudes de entrada del MLP; no cambian la definición del target de 30 horas. Todos los experimentos predicen las siguientes **7 horas** de la misma variable objetivo.

## De la serie preparada a las muestras supervisadas

Para cada activo y `segment_id`, la entrada contiene las últimas `w` observaciones de `volatility`, con `w` en {7, 14, 21, 28}; la salida contiene las siguientes siete observaciones. El salto entre muestras es una hora y la ventana completa de entrada y salida debe pertenecer a un mismo segmento continuo.

La separación cronológica se realiza antes del escalamiento. Se usan cinco folds expansivos a partir de siete bloques consecutivos y se purgan las muestras que se solapan en las fronteras entre train, validation y test. En cada fold, los dos `StandardScaler`, para entradas y objetivos, se ajustan solo con train. El detalle del diagnóstico de timeseries-cv y de la adaptación se conserva en [el capítulo de modelado](model_base.ipynb); las tablas y figuras de preparación se presentan en [el EDA](eda.ipynb).


## 1.6 Definición del problema de investigación

El problema consiste en pronosticar la volatilidad futura de cinco criptomonedas usando exclusivamente información histórica disponible hasta el tiempo t, entendido como el cierre de la última vela observada. La salida multistep es `volatility_(t+1), ..., volatility_(t+7)`. El target permanece definido como la desviación estándar muestral móvil de 30 retornos horarios, sin anualización.

## 1.7 Justificación de la selección del dataset

Binance Spot proporciona información financiera real, disponibilidad histórica y frecuencia horaria mediante una API pública. El alto número de observaciones permite estudiar un problema de forecasting temporal con varios períodos de evaluación. La caché CSV y la configuración explícita de fechas, activos y transformaciones facilitan la reproducibilidad del estudio.

## 1.8 Tamaño de la muestra

Las cifras siguientes proceden de las salidas guardadas de adquisición, período común y construcción del target en el maestro (celdas 4, 6 y 8, numeración desde 1).

| Activo | Filas originales en la caché | Filas del período común | Observaciones utilizables |
|---|---:|---:|---:|
| BTCUSDT | 58.888 | 53.542 | 53.212 |
| ETHUSDT | 58.888 | 53.542 | 53.212 |
| BNBUSDT | 58.888 | 53.542 | 53.212 |
| XRPUSDT | 58.888 | 53.542 | 53.212 |
| SOLUSDT | 53.542 | 53.542 | 53.212 |
| Total | 289.094 | 267.710 | 266.060 |

Son cinco activos. La caché tiene 12 columnas: 11 campos conservados de Binance y `symbol`; `ignore` ya fue descartado. La preparación añade `delta_h`, `new_segment`, `segment_id`, `log_return` y `volatility`, para 17 columnas. Esto no equivale a 17 predictores: el MLP usa una única variable, `volatility`, representada mediante 7, 14, 21 o 28 rezagos. El tamaño supera ampliamente las 20.000 observaciones indicadas por la guía, incluso por activo. La dependencia temporal impide interpretar cada fila como una réplica estadística independiente.

## 1.9 Calidad de los datos

Las salidas existentes registran cero NaN en las 12 columnas originales del período común y cero duplicados de la pareja `symbol/open_time` tras ordenar y deduplicar. Una pareja única implica también ausencia de filas completas duplicadas en ese conjunto; fechas iguales entre activos distintos son esperadas y no son duplicados de una serie.

Se registraron 10 saltos horarios por activo, 50 eventos activo-fecha en total, y 11 segmentos por activo. No son 50 horas ausentes: cada salto puede cubrir más de una hora. `delta_h` mide la separación horaria y tiene cinco NaN iniciales, uno por activo, porque no existe una observación previa. `segment_id` reinicia la construcción de retornos y ventanas ante discontinuidades.

El primer retorno de cada segmento no puede calcularse: hay 55 NaN de `log_return` (11 por activo). La volatilidad requiere 30 retornos válidos, equivalentes a 31 cierres consecutivos; aparecen 1.650 NaN de `volatility` (330 por activo). Los NaN de retorno están incluidos en esas filas, por lo que se eliminan determinísticamente 1.650 filas, no la suma de ambas cuentas. La auditoría guardada informa cero infinitos en todas las columnas numéricas. Las observaciones útiles tienen retornos y volatilidad finitos; no se rellenan huecos ni se unen extremos de segmentos.

Los valores extremos no se eliminan automáticamente, ya que en series financieras los retornos extremos y los episodios de alta volatilidad contienen información relevante sobre riesgo y comportamiento del mercado.

## 1.10 Sesgos y representatividad

Se estudian cinco criptomonedas de Binance Spot durante un período específico: solicitado desde 2020-01-01 hasta 2026-09-20, con período común desde 2020-08-11 06:00 UTC hasta 2026-09-20 23:00 UTC. La selección de activos, exchange y regímenes observados limita la representatividad. Los resultados no necesariamente representan otros exchanges ni todos los criptoactivos.

## 1.11 Consideraciones éticas

Se utilizan datos de mercado agregados: no existen datos personales, identificadores humanos ni información médica o privada en las variables del estudio. No se requieren procedimientos de anonimización. Los resultados tienen fines académicos y no constituyen recomendación financiera.
