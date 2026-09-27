# Pronóstico de volatilidad: SVR lineal

Jupyter Book actualizado con la base de datos, EDA 2.1–2.9 y modelo base SVR lineal frente a persistencia.

- Libro: https://jassan-bit.github.io/MachineLearning_202630/
- Fuentes: `book/sections/` y `book/entregable1_master.md`.
- Notebooks por sección: `notebooks/06_*.ipynb` a `notebooks/18_*.ipynb`.
- Código: `src/`; tablas y trazabilidad: `outputs/tables/`.
- Entorno: instalar `requirements.txt` con Python 3.10.

## Publicación

El workflow existente publica GitHub Pages desde `main` con `myst build --html`, sin ejecutar notebooks ni entrenar modelos. El `myst.yml` de la raíz define la navegación publicada. No se versionan datasets, TEST ni los modelos serializados.

## Reproducción

Los notebooks requieren la estructura de este repositorio y los CSV locales en `data/processed/` y `data/splits/`, que no se distribuyen en esta actualización. Los scripts 01–03 contienen la adquisición y partición originales; no es necesario ejecutarlos para compilar el libro. No regenerar datos para consultar los resultados guardados.

El notebook `notebooks/18_base_model.ipynb` presenta los resultados de los 170 ajustes ya realizados. Incluye el código del experimento; `REENTRENAR = True` permite repetirlo con los datos locales. Por defecto presenta resultados guardados. TEST permanece reservado. Las métricas actuales son de validación cronológica dentro de DEVELOPMENT.

## Versiones anteriores

Se conservan los archivos históricos de MLP y sus resultados sin borrarlos. La navegación principal muestra la versión actual del proyecto SVR. La documentación y dependencias del trabajo previo están en `README_MLP_legacy.md` y `requirements_mlp_legacy.txt`; sus metodologías y resultados no deben mezclarse con los del experimento actual.

La sección 2.1 conserva resultados anteriores según la limitación metodológica indicada en la portada; no deben confundirse con el objetivo recalculado en las secciones posteriores. Esta publicación no recalcula análisis.
