# Entregable 1: volatilidad con SVR lineal

**Restricción vigente del curso:** solo modelos clásicos sin redes neuronales.
El dashboard compara k-NN, Ridge, Lasso, Random Forest, XGBoost y SVR Lineal;
MLP está excluido. La mejora HAR-Ridge + XGBoost tampoco utiliza neuronas.
Consultar [DASHBOARD.md](DASHBOARD.md) para ejecución y resultados actuales.

Los experimentos MLP descritos a continuación son historial de investigación
y no forman parte del comparativo vigente ni deben incluirse en la entrega
del curso.

**MLP ajustado con pérdida ponderada:** selecciona en validación temporal de 2024 una potencia de ponderación de 0, 1 o 2 sobre la volatilidad actual. Conserva el original cuando gana en validación. En el test retrospectivo de 2025, R² macro pasa de 0,6753 a **0,7081**, RMSE de 0,6255 a **0,5925**, MAE de 0,4336 a **0,3965**, MSE de 0,5841 a **0,5207** y MAPE de 16,60 % a **14,65 %**. Modelos y predicciones verificados; la validación se ha reutilizado y los resultados necesitan confirmación en datos nuevos. [Informe](results/weighted_minute_mlp_2023_2025/index.html) y [notebook](notebooks/MLP_Ponderado_Minuto.ipynb).

```bash
python src/improve_weighted_mlp.py
python src/verify_weighted_mlp.py
python src/report_improved_mlp.py --weighted
```

Experimento con **MLP (Multi-Layer Perceptron)**: mismas variables, objetivos y cortes temporales. Busca 256 combinaciones de ventana de entrada, arquitectura y regularizacion L2 mediante seis bloques de validacion de 2024. Usa escalado ajustado dentro de cada entrenamiento, activacion tanh y siete salidas; evalua retrospectivamente en 2025. Resultados verificados en 358 fechas de prueba: **R? macro 0,6753**, **RMSE 0,6255** y mejora frente a persistencia en **16/16 configuraciones**. El [informe](results/optimized_minute_mlp_2023_2025/index.html) y el [notebook](notebooks/Optimizacion_MLP_Minuto.ipynb) incluyen resultados verificados.

```bash
python src/optimize_minute_mlp.py
python src/report_optimized_mlp.py
python src/compare_optimized_models.py
```

Experimento con **XGBoost**: mismas variables, objetivos y cortes temporales; seleccion en 2024 y evaluacion retrospectiva en 2025. El [informe](results/optimized_minute_xgboost_2023_2025/index.html) y el [notebook](notebooks/Optimizacion_XGBoost_Minuto.ipynb) incluyen resultados verificados, importancias por ganancia y comparacion con los otros modelos.

```bash
python -m pip install -r requirements-xgboost.txt
python src/optimize_minute_xgboost.py
python src/report_optimized_xgboost.py
python src/compare_optimized_models.py
```

Experimento con **Random Forest**: replica las variables, objetivos y validacion temporal de los modelos anteriores. Busca 256 combinaciones con 100 arboles por bosque, selecciona en 2024 y evalua retrospectivamente en 2025. El [informe](results/optimized_minute_randomforest_2023_2025/index.html) y el [notebook](notebooks/Optimizacion_RandomForest_Minuto.ipynb) incluyen los resultados verificados y la comparacion con SVR, k-NN, Ridge, Lasso y persistencia.

```bash
python src/optimize_minute_randomforest.py
python src/report_optimized_randomforest.py
python src/compare_optimized_models.py
```

Experimentos adicionales con **Ridge y Lasso**: usan las mismas características, objetivos y fechas de prueba de SVR y k-NN. La ventana de entrada y la regularización se seleccionan con validación temporal de 2024. Consultar el [informe Ridge](results/optimized_minute_ridge_2023_2025/index.html), el [informe Lasso](results/optimized_minute_lasso_2023_2025/index.html) y la [comparación de modelos](results/optimized_model_comparison_2023_2025/index.html). Los notebooks de resultados están en `notebooks/Optimizacion_Ridge_Minuto.ipynb` y `notebooks/Optimizacion_Lasso_Minuto.ipynb`.

Para reproducirlos:

```bash
python src/optimize_minute_linear.py --model ridge
python src/optimize_minute_linear.py --model lasso
python src/report_optimized_linear.py --model ridge
python src/report_optimized_linear.py --model lasso
python src/compare_optimized_models.py
```

Experimento adicional con **k-NN optimizado**: R² macro **0,7039**, RMSE **0,5937**, y mejora frente a persistencia en **16/16 configuraciones**. Usa las mismas características y las mismas 358 fechas de prueba que el SVR, con selección temporal en 2024. La evaluación de 2025 es retrospectiva. Consultar el [informe k-NN](results/optimized_minute_knn_2023_2025/index.html), el [notebook de resultados](notebooks/Optimizacion_kNN_Minuto.ipynb) y la [verificación](results/optimized_minute_knn_2023_2025/verification.json).

Para reproducir k-NN con el entorno del proyecto:

```bash
python src/optimize_minute_knn.py
python src/report_optimized_knn.py
```

**Jassan Arteta y Mateo Bernal**

Estudio Binance de BTC, ETH, BNB y XRP durante **2023–2025**, con datos de **un minuto**, características derivadas en ventanas de **7, 14, 21 y 28 días** y siete salidas diarias de volatilidad. La variante vigente usa correcciones relativas a persistencia, seis cortes temporales crecientes para selección en 2024 y ajuste final con etiquetas anteriores a 2025.

En 358 fechas de prueba de 2025: **R² macro 0,7007**, RMSE **0,5981** y mejora de RMSE frente a persistencia en **16/16 configuraciones**. La evaluación es retrospectiva; 2025 ya había sido examinado.

Consultar el [informe completo](book/sections/07_estudio_minuto.md), el [notebook ejecutado](notebooks/Entregable_1_Minuto.ipynb) y la [entrega reproducible](delivery/Entregable1_optimizado_2023_2025.zip).

El [modelo original con precios por minuto](book/sections/09_referencia_minuto_original.md) y el [estudio diario anterior](book/sections/08_referencia_diaria.md) quedan como referencias históricas. Los resultados vigentes están en `results/optimized_minute_2023_2025`.

## Reproducir

```bash
python -m pip install -r requirements.txt -r requirements-dashboard.txt
python src/optimize_minute_svr.py
python src/report_optimized_svr.py
python src/apply_optimized_delivery.py
```

Para consultar el informe sin entrenar:

```bash
python -m http.server 8051 --bind 127.0.0.1 --directory results/optimized_minute_2023_2025
```

Abrir http://localhost:8051. Los datos procesados están incluidos. Su procedencia y hashes se conservan en `results/minute_2023_2025/sources.csv` y `data_manifest.json`. `src/download_three_years.py` y `src/minute_experiment.py` permiten reconstruirlos desde los archivos mensuales de Binance.

`experiment_active.json` identifica el entregable vigente; `experiment_minute.json` conserva la configuración histórica y del periodo de adquisición. El dashboard de puerto 8050 y `app/api.py` todavía sirven el modelo original. Para las métricas optimizadas usar el informe de puerto 8051. Ningún script de esta reproducción publica en GitHub.
