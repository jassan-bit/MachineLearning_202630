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
