# Primer Entregable del Proyecto de Investigación

**Jassan Arteta y Mateo Bernal**  
Universidad del Norte · Machine Learning · 2026-30

## Selección de base de datos, EDA e implementación del modelo base

:::{div}
:class: correccion-profesor

El estudio pronostica volatilidad de BTC, ETH, BNB y XRP con datos de Binance
de 2023–2025. **El modelo inicial es únicamente el SVR lineal**;
persistencia se conserva como referencia trivial sin entrenamiento.
Los modelos HAR y la corrección AR pertenecen a análisis posteriores de
comparación y no forman parte del modelo inicial. Sus resultados se
presentan como ampliación retrospectiva.

:::

(revision-profesor)=
## Correcciones a la retroalimentación del profesor

:::{div}
:class: correccion-profesor

**Los bloques con fondo amarillo identifican las aclaraciones y adiciones de
esta revisión del 21 de octubre de 2026.** Los cálculos complementarios
citados se ejecutaron el 18 de octubre de 2026. Se verificaron sus tablas,
protocolos y registros de integridad antes de incorporarlos al libro.

La calificación comunicada fue **4,5/5,0**: base de datos **4,5**, EDA **4,5**
y modelo base **4,5**. Esta revisión responde a las observaciones recibidas;
no constituye una nueva calificación.

| Observación | Verificación y cambio incorporado | Estado y alcance |
| --- | --- | --- |
| EDA elaborado después del modelo y 2025 ya explorado | [Cronología y alcance](sections/12_eda.md#cronologia-retroalimentacion): se conserva la declaración retrospectiva y se distingue la evaluación adicional congelada de la ampliación posterior. | **Limitación histórica que permanece**; recalcular con desarrollo no convierte el EDA en previo al modelo. |
| Falta información mutua | [Información mutua](sections/12_eda.md#informacion-mutua-profesor): 22.400 estimaciones marginales, objetivos absolutos y relativos, solo desarrollo. | **Análisis ejecutado e incorporado** en HAR44 y daily6; no cubre todos los rezagos del SVR ni demuestra importancia condicional o mejora predictiva. |
| Falta VIF | [Redundancia y VIF](sections/12_eda.md#vif-profesor): 1.600 registros, columnas constantes y dependencias lineales identificadas. | **Análisis ejecutado e incorporado** para los diseños HAR44 y daily6; no equivale al VIF completo de todos los rezagos del SVR. |
| Falta estacionalidad intradía | [Perfiles intradía](sections/12_eda.md#intradia-profesor): perfiles por minuto y hora UTC, comparación 2023/2024 e intervalos por bloques. | **Diagnóstico descriptivo ejecutado e incorporado**; no es una prueba formal de estacionalidad estable. |
| Única referencia: persistencia | [Referencias HAR](sections/13_modelo_base_svr.md#har-profesor) y [evaluación ampliada](sections/14_evaluacion.md#evaluacion-retroalimentacion): HAR diario OLS y referencias HAR de variación intradía sobre las mismas fechas y objetivos. | **Comparación ampliada**; los resultados originales del SVR se conservan. |
| Residuos autocorrelacionados sin modelar | [Corrección con errores observables](sections/13_modelo_base_svr.md#residuos-profesor): experimento AR causal sobre HAR-Ridge + XGBoost y diagnóstico posterior. | **Atención parcial**: no se corrigieron dinámicamente los residuos del SVR; persiste dependencia y no se satisface la regla de sustitución del HAR-XG. |

Las fortalezas reconocidas —auditoría de fuentes con hashes, validación
cronológica, bootstrap por bloques, diagnóstico de residuos y evaluación
adicional con protocolo congelado— mantienen su evidencia en los capítulos
originales. También se conserva el límite del número de muestras
supervisadas y su dependencia temporal; esta ampliación no aumenta ese tamaño.

[Métricas complementarias](../results/professor_feedback_20261018/macro_metrics.csv),
[resumen de evaluación](../results/professor_feedback_20261018/summary.json) y
[reproducción de la ampliación](sections/15_reproducibilidad.md#reproduccion-retroalimentacion).

:::

## Ruta de lectura

1. [Base de datos](sections/11_base_datos.md): problema, justificación, fuente,
   condiciones de uso, diccionario, estructura, tamaño y representatividad.
2. [EDA](sections/12_eda.md): objetivo, análisis unidimensional, bidimensional,
   multivariado, fuga, componente temporal y preprocesamiento.
3. [Modelo base y SVR lineal](sections/13_modelo_base_svr.md): persistencia,
   características, entrenamiento y validación cronológica.
4. [Evaluación](sections/14_evaluacion.md): comparación en las mismas fechas,
   interpretación de métricas y limitaciones.
5. [Reproducibilidad](sections/15_reproducibilidad.md): archivos, dependencias,
   semilla y comandos.

La ruta corresponde a **regresión temporal**. Los componentes espaciales no
aplican porque no hay coordenadas. La estructura sigue la guía del entregable,
que asigna 10 % a base de datos, 60 % a EDA y 30 % a modelo base.

## Resultados y alcance de la evidencia

El SVR obtiene R² macro **0,7007** y RMSE **0,5981**, y mejora el RMSE frente
a persistencia en las **16 configuraciones** de activo y ventana.
La evaluación de 2025 es retrospectiva: ese periodo ya había sido explorado.
La auditoría ejecutada el **4 de octubre de 2026** incorpora controles de
calidad, rangos del objetivo, EDA de desarrollo, intervalos de confianza,
diagnósticos de residuos, curva de aprendizaje y coeficientes del SVR.
Los capítulos incluyen resultados y evidencias descargables.

Además, con los modelos guardados y un protocolo fijado antes de descargar
enero–agosto de **2026**, se evaluaron 236 orígenes nuevos respecto a los
artefactos del proyecto: R² macro **0,7239**, RMSE **0,5976**, frente a
**0,768** de persistencia, una reducción del **22,16 %**. La
[evaluación](sections/14_evaluacion.md) explica el alcance y los intervalos.

El estudio conserva límites explícitos: no alcanza 20.000 ejemplos
supervisados diarios por activo, no puede identificar MCAR/MAR/MNAR
con una única interrupción y no convierte el año 2025 ya explorado
en una reserva inicial intacta. La revisión distingue requisitos
analizados de propiedades que estos datos no permiten certificar.
