# Dashboard comparativo de volatilidad

El dashboard local resume siete modelos sobre las mismas observaciones de test. Se ejecuta desde la raíz del repositorio:

    .\.venv-repro\Scripts\python.exe -m volatility_dashboard.app

Abrir <http://127.0.0.1:8050/>. Contiene exactamente tres pestañas: contexto, EDA y comparación de modelos. Los dashboards anteriores conservan sus archivos y comandos.

## Datos y procedencia

La fuente principal es data/processed/minute_2023_2025/daily_target_closes.csv: 1.096 fechas UTC de 2023–2025 y cierres diarios de BTCUSDT, ETHUSDT, BNBUSDT y XRPUSDT. Los archivos .npy de la misma carpeta conservan cierres por minuto. Los hashes se verifican contra results/minute_2023_2025/data_manifest.json.

SOLUSDT no forma parte de este dataset ni del test comparable. Se muestra como no disponible. El dataset procesado no contiene volumen relativo ni rangos OHLC; esas variables no se incluyen en las características ni en el EDA.

## Preprocesamiento y objetivo

Se conservan los huecos. No se imputan cierres ni se ajustan transformaciones sobre el test. El cierre diario corresponde al último cierre de un día completo UTC.

Con \(P_t\) como cierre diario:

$$
r_t=100\ln(P_t/P_{t-1}),\qquad
\sigma_t^{(w)}=\sqrt{\frac1w\sum_{i=0}^{w-1}(r_{t-i}-\bar r_t)^2}.
$$

La ventana incluye el retorno actual y utiliza ddof=0. El objetivo es \(y_{t,h}=\sigma_{t+h}^{(w)}\), con horizontes de 1 a 7 días y ventanas de volatilidad de 7, 14, 21 y 28 días. Las unidades son puntos porcentuales de volatilidad no anualizada.

Para cada uno de los L días de entrada se incorporan seis características: retorno diario, retorno diario al cuadrado y cuatro resúmenes de retornos por minuto (raíz de suma de cuadrados, suma de absolutos dividida por raíz de 1.440, semivolatilidad negativa y máximo absoluto). Se normalizan por la volatilidad actual o su cuadrado. Siete características adicionales representan la salida conocida de retornos antiguos de la ventana.

La dimensión es 6L+7. Estas son características derivadas de cierres, no ventanas crudas de precios. El dashboard no altera esta metodología.

## Protocolo temporal y entrenamiento

Se reutilizan seis bloques crecientes de validación de 2024, con inicio en enero, marzo, mayo, julio, septiembre y noviembre. Las etiquetas de entrenamiento terminan antes del primer origen de validación. Las etiquetas de validación terminan antes de 2025.

El ajuste final utiliza orígenes elegibles con todas sus etiquetas anteriores al 1 de enero de 2025. Los modelos permanecen fijos durante el test. El test común comprende 358 orígenes, del 1 de enero al 24 de diciembre de 2025. Los últimos objetivos terminan el 31 de diciembre. Cada método aporta 40.096 predicciones: cuatro activos, cuatro ventanas, siete horizontes y 358 orígenes.

El escalador se ajusta exclusivamente con train por fold y con train final para los artefactos de test. Random Forest y XGBoost no utilizan un escalador ajustado. La aplicación comprueba las medias del escalador contra las características de entrenamiento.

Los modelos aprenden correcciones relativas a la volatilidad actual. La predicción se reconstruye en unidades originales y se limita inferiormente a cero. El MLP seleccionado incorpora ponderación por volatilidad y un promedio de inicializaciones, cuando fue seleccionado, con mezcla con persistencia o decaimiento conocido. Sus etapas adicionales de búsqueda usan la misma validación de 2024; el presupuesto de selección no es igual al de los otros métodos.

## Archivos de resultados utilizados

| Modelo | Carpeta en results |
|---|---|
| k-NN | optimized_minute_knn_2023_2025 |
| Ridge | optimized_minute_ridge_2023_2025 |
| Lasso | optimized_minute_lasso_2023_2025 |
| Random Forest | optimized_minute_randomforest_2023_2025 |
| XGBoost | optimized_minute_xgboost_2023_2025 |
| SVR Lineal | optimized_minute_2023_2025 |
| MLP | ensemble_tuned_minute_mlp_2023_2025 |

Se leen predicciones, selecciones, configuraciones, auditorías temporales, verificaciones y artefactos ya entrenados. Los nombres knn y svr se normalizan a forecast exclusivamente en memoria. Los archivos fuente permanecen intactos.

La auditoría valida las claves exactas activo/ventana/origen/horizonte, los objetivos contra el dataset, ausencia de duplicados, valores finitos, hashes de selección, calendario de validación, fechas del ajuste, escaladores y predicciones de los modelos serializados. Un modelo inconsistente se excluye y se muestra el motivo.

Las evaluaciones históricas de minute_2023_2025 y cv_comparison se conservan como referencias, fuera del ranking principal: sus protocolos o ajustes no coinciden con la comparación actual.

## EDA

Se muestran series de cierre, retornos y volatilidad; histogramas; estadísticas descriptivas; correlaciones Pearson o Spearman; relaciones feature/target; ACF de retornos, cuadrados y volatilidad; y distribuciones entre activos.

Las características y objetivos se construyen antes del filtro visual. Para una selección de fechas, las etiquetas futuras deben terminar dentro de ese periodo. Los huecos se conservan en series y ACF; los pares completos se usan en correlación y scatter. La vista inicial corresponde a development.

Explorar 2025 es retrospectivo y no genera nuevas variables, selecciones o modelos. No se atribuyen pruebas Jarque–Bera guardadas para otras muestras a una selección interactiva distinta.

## Métricas e interpretación

Se recalculan MAE, RMSE, MSE y R² a partir de todas las predicciones guardadas y alineadas de cada activo, ventana y horizonte. No se aplica muestreo a las métricas.

Cuando se seleccionan todos los horizontes, se promedian las métricas de los siete horizontes. Cuando se seleccionan todos los activos, se promedian las métricas por activo. Por ello, RMSE macro no es la raíz de MSE macro; R² macro tampoco es R² de todos los objetivos concatenados.

La ventana de volatilidad permanece fija para comparar modelos. Cambiar la ventana cambia el objetivo: un RMSE menor entre ventanas distintas no identifica un mejor algoritmo para el mismo problema.

Los rankings se construyen para una sola métrica: MAE y RMSE ascendentes, R² descendente. No hay score artificial ni ganador combinado. MAPE se omite de la comparación principal; no se presupone su estabilidad ante objetivos próximos a cero. No se utiliza R² ajustado.

## Residuos e interpretabilidad

El residuo es real menos predicho. Se muestran serie temporal, histograma, scatter contra predicción, media, desviación y ACF. Los objetivos móviles se superponen, por lo que la dependencia residual debe interpretarse con ese contexto.

No se inventan pruebas Jarque–Bera, Breusch–Pagan o BDS faltantes para estos modelos y observaciones. Las pruebas de otros experimentos no se transfieren al modelo actual.

Ridge y Lasso muestran coeficientes por característica estandarizada y salida de corrección relativa. Se cuentan coeficientes exactamente cero y cercanos a cero con umbral absoluto explícito de 10⁻⁶. El SVR lineal deshace el escalado de la salida para interpretar coeficientes sobre las entradas estandarizadas.

Random Forest muestra reducción de impureza y XGBoost ganancia normalizada, agregadas sobre las salidas. k-NN y MLP muestran permutation importance por grupos de lags sobre hasta 120 fechas espaciadas de test, con tres repeticiones y semilla 42. Es un diagnóstico descriptivo sobre modelos fijos, no un criterio de selección, prueba causal ni una escala comparable a coeficientes o ganancia.

Los parámetros y dimensiones se leen de los artefactos finales. Las curvas de loss del MLP provienen de loss_curve_. El diagnóstico guardado del MLP ajustado previo se presenta separado y con otra escala: no se atribuye al ensemble final.

## Limitaciones y conclusiones

La validación de 2024 se reutilizó durante varias mejoras y 2025 ya se examinó. Aunque las selecciones guardadas se realizaron con development y los ajustes finales excluyen test, el resultado es retrospectivo y requiere confirmación cronológica nueva.

La igualdad del test permite una comparación descriptiva sobre las mismas observaciones. No iguala los presupuestos de tuning ni demuestra diferencias estadísticamente significativas. Las interpretaciones automáticas describen cifras observadas y evitan conclusiones causales.

No hay SOL, volumen procesado, pruebas residuales alineadas completas ni tiempos separados de entrenamiento/predicción para todas las variantes. Los tiempos globales existentes se etiquetan con su alcance y no se usan como ranking de velocidad.

La aplicación reutiliza los resultados y mantiene intactos los experimentos. El informe detallado previo se conserva en el Book; esta sección documenta la integración comparativa.

## Referencia de interfaz

La implementación utiliza [Tabs de Dash](https://dash.plotly.com/dash-core-components/tabs), callbacks y DataTable. La arquitectura está dividida entre cargador, métricas, utilidades, tres módulos de pestañas y CSS local.


<!-- dashboard-results:start -->

## Resultados verificados por ventana del objetivo

Cada tabla mantiene fija la definicion de volatilidad. Los valores promedian las metricas por activo y horizonte sobre el mismo test; no son metricas de objetivos concatenados. No se mezclan ventanas para elegir un ganador.

### Volatilidad de 7 dias

| model | rmse | mae | mse | r2 |
| --- | --- | --- | --- | --- |
| Lasso | 1.063671 | 0.749403 | 1.395908 | 0.382572 |
| MLP | 1.036057 | 0.715246 | 1.309136 | 0.420695 |
| Random Forest | 1.030813 | 0.732621 | 1.263555 | 0.425665 |
| Ridge | 1.049503 | 0.743878 | 1.338345 | 0.409274 |
| SVR Lineal | 1.048824 | 0.739312 | 1.338715 | 0.406163 |
| XGBoost | 1.020043 | 0.711507 | 1.253739 | 0.432175 |
| k-NN | 1.044920 | 0.719115 | 1.323149 | 0.415625 |

### Volatilidad de 14 dias

| model | rmse | mae | mse | r2 |
| --- | --- | --- | --- | --- |
| Lasso | 0.609228 | 0.420881 | 0.474398 | 0.703689 |
| MLP | 0.566756 | 0.371767 | 0.396521 | 0.748121 |
| Random Forest | 0.596096 | 0.408132 | 0.434713 | 0.719644 |
| Ridge | 0.599735 | 0.407058 | 0.452337 | 0.721810 |
| SVR Lineal | 0.591042 | 0.404936 | 0.434912 | 0.728488 |
| XGBoost | 0.565026 | 0.372775 | 0.388671 | 0.744392 |
| k-NN | 0.584656 | 0.382607 | 0.413330 | 0.727090 |

### Volatilidad de 21 dias

| model | rmse | mae | mse | r2 |
| --- | --- | --- | --- | --- |
| Lasso | 0.439136 | 0.303533 | 0.254732 | 0.805922 |
| MLP | 0.394816 | 0.255497 | 0.196161 | 0.836286 |
| Random Forest | 0.435950 | 0.299308 | 0.239755 | 0.804543 |
| Ridge | 0.414506 | 0.286192 | 0.219793 | 0.820120 |
| SVR Lineal | 0.426074 | 0.291702 | 0.235996 | 0.813140 |
| XGBoost | 0.410598 | 0.266492 | 0.208017 | 0.823010 |
| k-NN | 0.418602 | 0.272643 | 0.212005 | 0.814415 |

### Volatilidad de 28 dias

| model | rmse | mae | mse | r2 |
| --- | --- | --- | --- | --- |
| Lasso | 0.314492 | 0.211144 | 0.129288 | 0.868438 |
| MLP | 0.298749 | 0.187235 | 0.113147 | 0.877535 |
| Random Forest | 0.347242 | 0.228007 | 0.160258 | 0.838403 |
| Ridge | 0.324674 | 0.218112 | 0.137620 | 0.857189 |
| SVR Lineal | 0.326377 | 0.220179 | 0.139066 | 0.855149 |
| XGBoost | 0.323392 | 0.203785 | 0.135729 | 0.862336 |
| k-NN | 0.326547 | 0.213070 | 0.133938 | 0.858639 |

## Auditoria de comparabilidad

| model | n_test | fecha_inicio_test | fecha_fin_test | estado_comparable |
| --- | --- | --- | --- | --- |
| k-NN | 358.000000 | 2025-01-01 | 2025-12-24 | Comparable |
| Ridge | 358.000000 | 2025-01-01 | 2025-12-24 | Comparable |
| Lasso | 358.000000 | 2025-01-01 | 2025-12-24 | Comparable |
| Random Forest | 358.000000 | 2025-01-01 | 2025-12-24 | Comparable |
| XGBoost | 358.000000 | 2025-01-01 | 2025-12-24 | Comparable |
| SVR Lineal | 358.000000 | 2025-01-01 | 2025-12-24 | Comparable |
| MLP | 358.000000 | 2025-01-01 | 2025-12-24 | Comparable |
| Referencia historica: minute_2023_2025 | No aplica |  |  | Excluido |
| Referencia historica: cv_comparison | No aplica |  |  | Excluido |

Las siete variantes actuales son comparables en observaciones y objetivos; las referencias historicas quedan fuera. Los archivos originales de resultados y modelos no se modificaron.
<!-- dashboard-results:end -->
