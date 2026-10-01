# Entregable 2: dashboard e interpretación de resultados

El dashboard organiza el proyecto en tres pestañas: contexto, EDA y modelos base. Lee cinco tablas calculadas por el pipeline del Entregable 1; no entrena modelos ni consulta TEST al recibir visitas. El selector de criptomoneda actualiza las seis figuras. Los filtros de fold y métrica se aplican a la comparación de modelos; la figura de evolución por fold conserva los cinco bloques para facilitar su comparación.

## Contexto: cobertura mensual

La altura de cada barra corresponde al número de registros horarios disponibles para el activo y el mes. Diferencias entre meses pueden deberse a su duración, a límites parciales del periodo o a huecos. La figura no identifica mecanismos de ausencia. El conteo tampoco equivale a un tamaño muestral independiente: hay dependencia temporal y entre activos. Fuente: `dashboard_data/univariate_monthly_coverage.csv`.

La definición del objetivo, la procedencia y los controles de calidad se detallan en [Base de datos](01_base_datos.md). El objetivo es la desviación estándar poblacional centrada de los próximos 24 retornos logarítmicos horarios, multiplicada por 100. No representa el retorno acumulado de un día ni una medida anualizada.

## EDA: cuantiles, evolución y autocorrelación

**Cuantiles del objetivo.** Las barras muestran mínimo, Q1, mediana, Q3, P95, P99 y máximo de todas las etiquetas válidas de DEVELOPMENT. Son estadísticos ordenados, no frecuencias de un histograma. Para BTC, la mediana es aproximadamente 0,485 % y P99 es 1,775 %; para SOL, aproximadamente 0,961 % y 3,924 %. Esto ilustra diferencias de nivel y una cola superior extensa. Un extremo no se elimina automáticamente: puede ser una observación válida de mercado. Fuente: `eda_target_summary.csv`.

**Media y mediana trimestrales.** Las curvas permiten comparar periodos dentro del mismo activo. La separación entre media y mediana es compatible con asimetría y observaciones altas, pero no identifica una causa. Cada trimestre usa sus etiquetas válidas; no se asume igual número de observaciones. Los cambios no prueban que el patrón continúe fuera del periodo estudiado. Fuente: `temporal_target_quarters.csv`.

**Autocorrelación del objetivo.** La curva muestra ACF por rezago horario sobre el segmento documentado por el análisis temporal. En el primer rezago los valores son altos; parte de esa persistencia procede del solapamiento de las ventanas de 24 retornos. No se debe interpretar como prueba suficiente de pronosticabilidad ni como 24 observaciones independientes. El archivo conserva el comienzo y tamaño del segmento utilizado. Fuente: `temporal_target_correlations.csv`.

Los diagnósticos de distribución, huecos, dependencia y preprocesamiento se amplían en [EDA](02_eda.md). Los resúmenes descriptivos usan DEVELOPMENT; el escalado del modelo se ajusta exclusivamente en TRAIN dentro de cada fold.

## Modelos: comparación y variación entre folds

**Comparación filtrada.** Para el activo elegido, las barras contrastan SVR lineal y Persistence en un fold o mediante media aritmética de los cinco folds. Las métricas se leen del archivo `base_metrics_by_fold.csv`: no se recalculan sobre una muestra visual. Menor RMSE, MAE o MAPE indica menor error; mayor R² indica mejor desempeño. RMSE y MAE se expresan en puntos porcentuales de volatilidad; MAPE se expresa en porcentaje y es sensible a etiquetas cercanas a cero. R² negativo indica un error cuadrático superior al de la media observada del bloque, referencia matemática que no es un pronóstico disponible anticipadamente.

**Evolución por fold.** La segunda gráfica conserva los cinco folds para el activo y métrica seleccionados. Permite detectar heterogeneidad temporal que una media oculta. La línea une bloques cronológicos, no observaciones horarias. No debe confundirse con una curva de aprendizaje ni con una serie de predicciones individuales.

**Conclusión global.** La media de los 25 bloques activo-fold es 0,389006 pp de RMSE para Persistence y 0,623259 pp para SVR. El incremento es 0,234253 pp: el SVR no mejora la referencia en el protocolo evaluado. El intervalo bootstrap por bloques del 95 % de la diferencia es [0,204497; 0,269379] pp; está condicionado a la configuración elegida y no corrige por completo la selección sobre los mismos folds. Los resultados de un filtro pueden diferir del agregado global.

El SVR usa 168 cierres horarios consecutivos. Se exige que historia, referencia y etiqueta estén contenidas en cada bloque; los escaladores se ajustan en entrenamiento. La búsqueda, los residuos, los intervalos y las limitaciones están en [Modelo base](03_modelo_base.md). TEST permanece reservado y no sustenta las conclusiones del dashboard.

## Ejecución y despliegue

Desde la raíz del proyecto: instalar `requirements-dashboard.txt` y ejecutar `python app.py`. Abrir `http://127.0.0.1:8050`. La aplicación necesita `app.py`, `assets/dashboard.css` y las cinco tablas de `dashboard_data/`.

En Render se configura Python 3, raíz vacía, instalación con `pip install -r requirements-dashboard.txt` y arranque con `gunicorn app:server --bind 0.0.0.0:$PORT --workers 1`. `render.yaml` conserva la alternativa Blueprint con plan Free. El dashboard enlaza el Book mediante `BOOK_URL`, con la dirección del sitio existente como valor predeterminado. El enlace público del dashboard se documentará cuando el servicio esté desplegado y verificado.

Las tablas son copias del pipeline: si se recalcula el análisis, se deben actualizar conjuntamente las cinco tablas y sus interpretaciones. El dashboard es una interfaz de consulta de resultados guardados; la reproducción del entrenamiento continúa documentada en el Entregable 1.
