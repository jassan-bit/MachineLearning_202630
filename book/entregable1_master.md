# Primer Entregable del Proyecto de Investigación

**Jassan Arteta y Mateo Bernal**  
Universidad del Norte · Machine Learning · 2026-30

## Selección de base de datos, EDA e implementación del modelo base

El estudio pronostica volatilidad de BTC, ETH, BNB y XRP con datos de Binance
de 2023–2025. La entrega contiene **persistencia como línea base trivial** y
**SVR lineal como único modelo entrenado**.

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
El informe identifica los controles verificados y los requisitos pendientes;
no presenta esos pendientes como resultados realizados.
