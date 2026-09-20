# Selección de Base de Datos, Análisis Exploratorio e Implementación de Modelo Base

**Asignatura:** Machine Learning  
**Autor:** Jassan Alberto Arteta Chavarro  
**Universidad:** Universidad del Norte  
**Periodo:** 2026-30  

---

## Objetivo del Entregable

Establecer las bases del proyecto de investigación mediante:

1. La selección de una base de datos pertinente.
2. Un análisis exploratorio de datos (EDA) riguroso y exhaustivo.
3. La implementación de un modelo base como referencia inicial, comparado contra una línea base trivial.

---

## Ruta de Trabajo según el Tipo de Dataset

El proyecto corresponde a la **Ruta C: Regresión temporal**, debido a que la variable objetivo es continua y los datos presentan una estructura cronológica con fecha y hora.

La ruta metodológica correspondiente contempla:

- Análisis exploratorio general y temporal.
- Auditoría de fuga de información.
- Partición cronológica de los datos.
- Validación mediante `TimeSeriesSplit` con separación temporal.
- Construcción de rezagos y ventanas móviles utilizando exclusivamente información pasada.
- Comparación contra una línea base temporal.
- Evaluación mediante RMSE, MAPE, \(R^2\) y análisis de autocorrelación de residuos.
