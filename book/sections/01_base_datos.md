# 1. Datos y problema de investigación

## Pregunta y alcance

¿Puede un SVR lineal, usando únicamente cierres diarios pasados, mejorar la persistencia al pronosticar
la volatilidad móvil de BTC, ETH, BNB y XRP durante los siguientes siete días?
Los cuatro activos se modelan por separado; no se incorporan noticias, volumen ni variables externas como predictores.

## Fuente, frecuencia y fechas

Binance Spot, pares contra USDT, del 01/01/2020 al 31/12/2025 inclusive (fin exclusivo 01/01/2026).
Los archivos mensuales oficiales contienen velas de **un minuto** procedentes de `/api/v3/klines`.
Esta descarga conserva la granularidad exigida; el modelado utiliza el último cierre diario UTC para limitar
la dimensión de entrada. La frecuencia de adquisición y la de modelado se documentan por separado.
La [documentación oficial](https://github.com/binance/binance-public-data) describe los archivos,
checksums y el cambio a timestamps en microsegundos desde enero de 2025; el lector normaliza ambos formatos a UTC.

## Calidad y cobertura ejecutada

| symbol | calendar_days | complete_days | minute_rows | invalid_closes | excluded_days |
| --- | --- | --- | --- | --- | --- |
| BNBUSDT | 2192 | 2176 | 3154154 | 8 | 16 |
| BTCUSDT | 2192 | 2176 | 3154155 | 8 | 16 |
| ETHUSDT | 2192 | 2176 | 3154154 | 8 | 16 |
| XRPUSDT | 2192 | 2177 | 3154155 | 7 | 15 |

Un día es elegible si contiene exactamente 1.440 minutos únicos y todas las velas tienen tiempos de cierre válidos.
Los días incompletos quedan como faltantes en el calendario, sin interpolación ni compresión del tiempo.
Se excluyen las muestras que requieren cualquiera de esos días. Es una política conservadora: también descarta
días cuyo cierre final podría existir, pero cuya cobertura intradiaria no es completa. No se atribuyen las anomalías
a una causa sin evidencia. La selección puede cambiar la representatividad del conjunto.

## Diccionario

| Campo | Uso y unidad |
| --- | --- |
| `symbol` | Identificador del par; cuatro modelos separados por objetivo |
| `date` | Día UTC de la vela agregada; disponible al finalizar ese día |
| `close` | Último precio de cierre del día, en USDT; NaN si día inelegible |
| `minutes` | Número observado de velas de un minuto |
| `invalid_close_times` | Velas con final temporal anómalo |
| `complete` | Indicador de día elegible |
| `log_return` | Logaritmo del cociente de dos cierres diarios consecutivos |
| `volatility_w` | 100 por desviación estándar poblacional de w retornos diarios |

El precio es una magnitud positiva; los retornos pueden ser negativos; la volatilidad observada es no negativa.
Los archivos originales incluyen OHLC, volúmenes y operaciones, pero estos campos no entran en el SVR.
`results/minute_sources.csv` registra la URL, hash, cobertura y unidad temporal de cada uno de los 288 archivos.

## Representatividad y uso

La muestra representa cuatro pares de un único exchange, con selección retrospectiva de activos.
No representa todos los mercados ni garantiza generalización a otros periodos o monedas. Se respetan los
[términos del proveedor](https://github.com/binance/binance-public-data/blob/master/TERMS_AND_CONDITIONS.md).
El estudio no contiene datos personales y sus errores de pronóstico no se interpretan como rentabilidad de una estrategia.
