# 4. Conclusiones

## Respuesta al problema de investigación

En las condiciones evaluadas, el SVR lineal alimentado con 168 cierres horarios no mejora el pronóstico de Persistence de la volatilidad futura. La variable objetivo es la desviación estándar poblacional centrada de los próximos 24 retornos logarítmicos horarios, multiplicada por 100. Es una medida de dispersión horaria estimada en una ventana futura de 24 horas; no representa la volatilidad acumulada o anualizada del retorno de un día.

Se evaluaron cinco activos en cinco folds cronológicos dentro de DEVELOPMENT. Se utilizaron las mismas observaciones para ambos modelos y se exigió que historia, referencia y etiqueta estuvieran contenidas en cada bloque. Los escaladores se ajustaron únicamente en TRAIN. TEST permanece reservado y no sustenta ninguna conclusión de este entregable.

| Modelo | RMSE medio (puntos porcentuales) | MAPE medio (%) | R² medio |
|---|---:|---:|---:|
| Persistence | 0,389006 | 40,555962 | −0,044738 |
| SVR lineal | 0,623259 | 109,417896 | −1,905431 |

Son medias con igual ponderación de los 25 bloques activo-fold, no métricas calculadas sobre una única concatenación. [Resultados por fold](../../outputs/tables/base_metrics_by_fold.csv). Un R² negativo indica que el error cuadrático supera el de la media observada de la etiqueta en el bloque evaluado; esa media de validación es la referencia matemática de R², no un predictor disponible anticipadamente. MAPE es sensible a objetivos próximos a cero y no se interpreta como porcentaje de exactitud.

El incremento de RMSE del SVR respecto de Persistence fue **0,234253 puntos porcentuales**, con intervalo bootstrap del 95 % **[0,204497; 0,269379]**, usando bloques de 168 horas, 499 réplicas y semilla 42. La sensibilidad con bloques de 24 y 336 horas conserva el signo positivo. El remuestreo respeta el emparejamiento de modelos y activos dentro de cada fold. [Intervalos](../../outputs/tables/base_confidence_intervals.csv) y [sensibilidad](../../outputs/tables/base_bootstrap_sensitivity.csv). Estos intervalos están condicionados a la configuración seleccionada en los mismos folds; no corrigen por completo el optimismo de la selección.

## Implicaciones del análisis exploratorio

La dependencia temporal, la redundancia de los cierres y los cambios de distribución justifican la partición cronológica, el escalado por fold y el análisis de residuos con distancias horarias reales. El número de filas no equivale a observaciones independientes: las etiquetas horarias consecutivas comparten retornos y los activos pueden moverse conjuntamente. Por ello se reporta incertidumbre por bloques y se evitan conclusiones apoyadas únicamente en valores p calculados bajo independencia.

El diagnóstico de cambios y las ventanas alrededor de eventos son exploratorios. No atribuyen causalidad a los eventos. Los cambios marginales de volatilidad tampoco demuestran por sí solos un cambio en la relación condicional entre predictores y objetivo. La sensibilidad al tamaño del bloque y la incertidumbre en las fechas de cambio se presentan en la sección 2.6.11.

## Limitaciones

- Los registros disponibles no identifican el mecanismo MCAR, MAR o MNAR de los huecos. No se atribuye una causa específica ni se interpolan retornos a través de ellos.
- La verificación conservada de la partición acredita cortes y claves temporales disjuntas; no certifica la ausencia de vectores casi idénticos entre DEVELOPMENT y TEST al omitir fechas.
- El EDA y la selección se realizaron sobre DEVELOPMENT. La validación interna no reemplaza una evaluación final independiente y no permite garantizar ausencia universal de sobreajuste.
- La disponibilidad de las velas es nominal tras su cierre; no se dispone de registros de latencia operativa.
- La evidencia corresponde a cinco activos, un proveedor y los periodos observados. No demuestra generalización a otros mercados ni una estrategia rentable de inversión.

## Reproducibilidad y siguiente etapa

**Notebook completo:** [Entregable_1_Completo.ipynb](../../notebooks/Entregable_1_Completo.ipynb). Incluye el informe, tablas, figuras incorporadas, interpretaciones y el código ejecutable de todas las etapas. Al ejecutar todas las celdas se repiten los cálculos y los 170 ajustes en una carpeta nueva, usando exclusivamente DEVELOPMENT. Para repetir la ejecución se necesita el repositorio completo y el entorno indicado en el notebook.

La reproducción en un entorno limpio completó 17 etapas, incluidos 170 ajustes y 12 pruebas; se contrastaron 13 tablas numéricas con las tolerancias documentadas. Se ejecutaron además seis notebooks en el kernel del entorno nuevo. La reproducción comienza en el snapshot de DEVELOPMENT y no repite la descarga ni la partición original. [Registro de reproducción](../../outputs/tables/clean_reproduction_metadata.json).

**Materiales de la entrega:** [snapshot de DEVELOPMENT](../../data/splits/development_80.csv), [dependencias](../../requirements-lock-windows-py310.txt), [notebook del modelo](../../notebooks/18_base_model.ipynb) y [repositorio de código y notebooks](https://github.com/jassan-bit/MachineLearning_202630). TEST y el archivo maestro no forman parte del paquete reproducible. El snapshot conserva la huella SHA-256 indicada en el registro; su distribución no implica una licencia abierta adicional sobre los datos del proveedor.

El entregable documenta un modelo base reproducible y una referencia temporal más competitiva. Cualquier nueva transformación, predictor o algoritmo deberá decidirse y validarse dentro de DEVELOPMENT. Antes de abrir TEST se congelarán el protocolo, las transformaciones y la configuración final; sus resultados se reportarán como evaluación separada y no se utilizarán para reajustar el modelo.
