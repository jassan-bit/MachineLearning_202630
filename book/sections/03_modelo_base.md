# 3. Validación temporal y modelo

## Adaptación de la guía

Se conserva Binance y se sustituye el MLP de la guía por **SVR lineal**, según el alcance acordado.
Se instala `timeseries-cv==0.1.5` y se importa `tsxv`. La firma real de la versión instalada es
`split_train_val_test_groupKFold(sequence, numInputs, numOutputs, numJumps)`;
los nombres de argumentos de algunos ejemplos de la guía no coinciden con esta versión.

## Particiones y prevención de fuga

Se invoca la función exigida sobre índices del calendario con `numInputs=28`, `numOutputs=7`, `numJumps=1`.
Las ventanas de índices se traducen después a los cierres (recortados a L) y a la volatilidad del objetivo w.
El calendario de 28 días se fija para todas las comparaciones; no se vuelve a generar un calendario distinto para cada L.
Es una **adaptación de Group K-Fold de tsxv con filtros cronológicos**, no el método nativo sin cambios.

La implementación nativa intercala entrenamiento, validación y test; puede entrenar con fechas posteriores
a las de evaluación y no garantiza por sí sola prevención de fuga. Se conservan únicamente:

- TRAIN: etiquetas completamente anteriores al 01/01/2023.
- Validación: origen desde 01/01/2023 y etiquetas anteriores al 01/01/2024.
- TEST: origen desde 01/01/2024 y etiquetas hasta 31/12/2025.

Se comprueba que la última etiqueta de TRAIN preceda al primer origen de validación y que la última etiqueta
de validación preceda al primer origen de test. El historial conocido puede cruzar hacia un bloque anterior;
eso representa información disponible en el origen. Los modelos de cada fold se ajustan solo con TRAIN.

| fold | test | train | val |
| --- | --- | --- | --- |
| 1 | 9 | 23 | 3 |
| 2 | 9 | 23 | 3 |
| 3 | 8 | 23 | 3 |
| 4 | 8 | 24 | 3 |
| 5 | 8 | 24 | 4 |

**Limitación:** numJumps=1 no significa un origen de evaluación diario en esta librería.
El muestreo nativo deja pocas muestras por fold; los cinco folds se solapan y no son cinco réplicas independientes.
Las desviaciones estándar entre folds son descriptivas, no errores estándar ni intervalos de confianza.
La auditoría exacta de fechas se entrega en `results/fold_calendar.csv`.

## SVR lineal multisalida

`StandardScaler` de X seguido de `TransformedTargetRegressor`, con escalado de y y
`MultiOutputRegressor(LinearSVR(...))`: siete regresores independientes para h=1,…,7.
Se usa pérdida `squared_epsilon_insensitive`, `dual=False`, tolerancia 1e-6, máximo 50.000 iteraciones y semilla 42.
Un aviso de no convergencia detiene la ejecución. Las predicciones negativas se conservan y se contabilizan,
pues el SVR no impone la restricción física de volatilidad no negativa.

Se prueba C en {0.01,0.1,1,10} y epsilon en {0.01,0.1}, en unidades estandarizadas del objetivo.
Por activo, w y L, se selecciona la pareja que minimiza el RMSE de validación promediado sobre horizontes y folds.
Después se elige L por la misma métrica de validación, manteniendo fijo w. TEST no participa en estas selecciones.
Persistence repite la volatilidad conocida en t en los siete horizontes.

## Evaluación y residuos

Se guardan todas las predicciones TRAIN/VAL/TEST con origen y fecha objetivo. MAPE se expresa en porcentaje;
MAE y RMSE en puntos porcentuales de volatilidad; MSE en puntos porcentuales al cuadrado.
La columna horizonte=0 es el promedio de las siete métricas por horizonte; el promedio de RMSE no es la raíz
del MSE global. Los objetivos cero se excluyen solo de MAPE y se cuentan explícitamente.

BDS usa `statsmodels.tsa.stattools.bds`, dimensión 2 y distancia por defecto (1.5 desviaciones estándar).
Se calcula sobre residuos h=1 ordenados temporalmente de cada fold test. Se marca no estimable si hay menos
de seis residuos, si son constantes o si el cálculo no es finito. Seis es un control operativo, no garantía asintótica.
El tamaño reducido y el espaciado entre orígenes limitan fuertemente la inferencia. p>0.05 no demuestra iid;
solo indica no rechazo. El promedio de valores p exigido por la guía es descriptivo, no un contraste combinado.

## Fuentes metodológicas

- [timeseries-cv y atribución a Filipe Roberto Ramos](https://pypi.org/project/timeseries-cv/).
- [LinearSVR](https://scikit-learn.org/stable/modules/generated/sklearn.svm.LinearSVR.html).
- [Regresión multisalida](https://scikit-learn.org/stable/modules/generated/sklearn.multioutput.MultiOutputRegressor.html).
- [Test BDS](https://www.statsmodels.org/stable/generated/statsmodels.tsa.stattools.bds.html).
