> **Entregable vigente:** las métricas optimizadas están en http://localhost:8051, servidas desde `results/optimized_minute_2023_2025`. El dashboard/API descritos a continuación corresponden al modelo original y permanecen como referencia histórica.

## Alcance del sitio publicado

**Auditoría del 4 de octubre de 2026:** el sitio incorpora los resultados
de calidad, EDA, bootstrap temporal, diagnósticos de residuos, curva de
aprendizaje e interpretación de coeficientes de los modelos vigentes.
Las evidencias están en `results/current_delivery_audit/` y el suplemento
descargable es `delivery/Entregable1_auditoria_actualizada.zip`.
La evaluación adicional de enero–agosto de 2026 utiliza los 16 SVR guardados:
236 orígenes, R² macro 0,7239 y RMSE 0,5976, un 22,16 % menor que persistencia.
La revisión conserva explícitos los límites del tamaño supervisado, la
identificación del mecanismo de ausencia y la exploración previa de 2025.

El sitio https://jassan-bit.github.io/MachineLearning_202630/ contiene únicamente
el modelo base de persistencia y el SVR lineal. El dashboard comparativo y los
experimentos con otros modelos se conservan localmente y quedan fuera del sitio.

La navegación del entregable sigue la guía: Base de datos, EDA, Modelo base y
SVR lineal, Evaluación y Reproducibilidad. Los capítulos publicados son
`book/sections/11_base_datos.md` a `book/sections/15_reproducibilidad.md`;
las versiones históricas no forman parte del menú. Los requisitos sin
evidencia del dataset vigente se identifican como pendientes.

## Documentación de la auditoría del entregable

El [informe del estudio vigente](book/sections/07_estudio_minuto.md) incluye la pregunta de investigación, justificación del dataset, fuente y condiciones de uso consultadas el 3 de octubre de 2026, estructura temporal multiactivo y diccionario de variables con tipos, unidades y disponibilidad. También relaciona los hallazgos del EDA con las decisiones metodológicas.

El SVR es el modelo de referencia frente a persistencia; el comparativo conserva seis modelos clásicos y añade HAR-Ridge + XGBoost. La evaluación de 2025 es retrospectiva. Quedan abiertos la evidencia de condiciones aplicables en la fecha de descarga, la consolidación de controles de calidad y sensibilidad a extremos del dataset vigente, y una evaluación en un periodo nunca explorado con el procedimiento fijado previamente. Estos pendientes no se presentan como verificaciones completadas.

# Dashboard y API del estudio 2023–2025

Ejecutar `python dashboard.py` y abrir http://localhost:8050.
Lee `results/minute_2023_2025` cuando status.json indica complete; conserva compatibilidad con el estudio diario anterior.
Permite filtrar criptomoneda, método, ventana de entrada, ventana de volatilidad y fechas comunes/todas.
Las métricas incluyen R², RMSE y MAE, junto con el número de orígenes y comparación con persistencia.

Para la API: `uvicorn app.api:app --host 127.0.0.1 --port 8000`.
Consultar `/models` antes de enviar `/predict`. `n_features` indica 10.080, 20.160, 30.240 o 40.320 cierres
de un minuto consecutivos. `input_window` se expresa en días. Son siete salidas diarias de volatilidad.
El cliente debe proporcionar una ventana completa que termine en el cierre diario UTC.

Los modelos conservan el entrenamiento de 2023 para reproducir las predicciones reportadas.
El Dockerfile incluye los modelos y metadatos; no descarga datos ni entrena al arrancar.
Un Dockerfile preparado no equivale a un despliegue remoto verificado.


## Nuevo modelo propuesto: HAR-Ridge + XGBoost

Se conservan los modelos anteriores, sus artefactos y sus resultados para la
comparación. **XGBoost se presenta como modelo individual y HAR-Ridge + XGBoost
como mejora propuesta**. La combinación se incorpora como un nuevo modelo
del estudio, sin sustituir los modelos existentes, y cumple la restricción
de no usar redes neuronales.

Se añadió `src/improve_classical_forecast.py`: combina XGBoost con HAR-Ridge,
una regresión regularizada con resúmenes de volatilidad de 1, 3, 7, 14 y 28 días,
riesgo negativo y los retornos conocidos que abandonan la ventana de volatilidad.
Predice **volatilidad**, no precio ni rentabilidad, a siete horizontes diarios
para BTC, ETH, BNB y XRP. Usa los datos por minuto existentes.

Los hiperparámetros de XGBoost proceden de su selección anterior en 2024.
Cinco penalizaciones Ridge y cinco pesos de combinación se comparan en seis
bloques expansivos de 2024. Las etiquetas de entrenamiento terminan antes del
primer origen de validación; cada scaler se ajusta solo con su entrenamiento.
El peso cero permite conservar XGBoost. Todas las decisiones se fijan antes
de calcular métricas de 2025. Los modelos finales se entrenan con etiquetas
que terminan antes de 2025 y permanecen fijos durante la evaluación.

Resultados macro de 2025, promediados sobre cuatro monedas y cuatro ventanas:

| Modelo | R² | RMSE | MAE |
| --- | ---: | ---: | ---: |
| Persistencia | 0.5377 | 0.74770 | 0.49719 |
| XGBoost (modelo individual) | 0.7155 | 0.57976 | 0.38864 |
| HAR-Ridge + XGBoost (mejora propuesta) | **0.7380** | **0.55817** | **0.37557** |

El RMSE macro baja un **3.73 %** frente a XGBoost y un **25.35 %** frente
a persistencia. El RMSE promedio mejora para las cuatro monedas; XRP presenta
un MAE ligeramente mayor (0.50280 frente a 0.49501). No todas las métricas ni
configuraciones individuales tienen que mejorar. El RMSE usa la media del
RMSE de cada horizonte, conforme a los experimentos existentes.

2025 ya se había examinado en estudios anteriores: esta comparación es
retrospectiva, no un test nuevo independiente. No se han calculado intervalos
predictivos. La prueba de Diebold–Mariano descrita abajo evalúa diferencias
de pérdida cuadrática en estas predicciones retrospectivas.

Ejecutar desde la raíz:

```powershell
.\.venv-repro\Scripts\python.exe src/improve_classical_forecast.py
.\.venv-repro\Scripts\python.exe src/verify_improved_classical.py
.\.venv-repro\Scripts\python.exe -m unittest discover -s tests -p test_improved_classical.py
```

Artefactos en `results/improved_classical_2023_2025/`: modelos `.joblib`,
predicciones, métricas por moneda/ventana, métricas macro, búsqueda, selección,
calendario y configuración. El verificador comprueba hashes del dataset,
separación temporal, selección, coincidencia con el XGBoost anterior y
reproducción de todas las predicciones desde los modelos guardados.
`predict_bundle` reconstruye las variables causales para inferencia; necesita
el historial de cierres diarios y resúmenes por minuto hasta el origen.
El dashboard conserva los seis modelos clásicos e incluye HAR-Ridge + XGBoost
en las tablas, filtros, rankings y diagnósticos de Comparación de modelos.
La API conserva su configuración.

## Dashboard comparativo: seis modelos clásicos y mejora propuesta

### Prueba de Diebold–Mariano

La pestaña Comparación de modelos incluye una tabla exportable de pruebas
bilaterales entre todos los pares de modelos seleccionados. H₀ establece
igual pérdida cuadrática esperada; DM negativo favorece al modelo A.
Se usa la corrección Harvey–Leybourne–Newbold, distribución t con n−1 grados
de libertad, varianza HAC Bartlett y ajuste Holm de los valores p entre los
pares de cada selección. La conclusión usa p ajustado < 0.05.

La prueba usa errores diarios alineados, no las métricas del ranking.
Cuando activo u horizonte es TODOS, se promedian los errores cuadrados
dentro de cada origen; n cuenta días y conserva la dependencia entre activos
y horizontes. Los rezagos HAC son el máximo de ventana + horizonte − 2 y
floor(4(n/100)^(2/9)), para cubrir la superposición del objetivo móvil.
La corrección HLN usa el horizonte elegido, o 7 al promediar horizontes.
Este ancho de banda es una decisión metodológica: no asegura capturar toda
la dependencia y la conclusión puede variar con otros anchos de banda.

Ejecutar `.\.venv-repro\Scripts\python.exe src/run_diebold_mariano.py` para
guardar todas las selecciones por activo, ventana y horizonte en
`results/diebold_mariano_2025/comparisons.csv`, junto con la auditoría.
Holm se aplica dentro de cada selección, no a toda la colección de filtros.
La prueba compara MSE aunque el ranking se ordene por otra métrica.
No rechazar H₀ no demuestra equivalencia. La evaluación sigue siendo
retrospectiva y no corrige la selección previa usando resultados de 2025.

Referencia: [forecast: DM modificado y estimador Bartlett](https://pkg.robjhyndman.com/forecast/reference/dm.test.html).

Por requisito del curso, el comparativo vigente excluye MLP y cualquier red
neuronal. Incluye k-NN, Ridge, Lasso, Random Forest, XGBoost y SVR Lineal.
MLP pertenece a las redes neuronales; su clasificación como deep learning
depende de su profundidad, pero queda excluido por la restricción del profesor.
Los experimentos anteriores se conservan como historial y no se cargan en este
dashboard. El nuevo modelo propuesto HAR-Ridge + XGBoost también cumple la restricción
y se incorpora al comparativo, conservando los seis modelos anteriores.

### Publicar el comparativo en Render

`render.yaml` apunta a `volatility_dashboard.app:server`. Instala el runtime
con `requirements-render.txt` y sirve la aplicación con Gunicorn en `$PORT`.

1. Publicar los archivos actualizados en el repositorio conectado a Render.
2. En Render, crear un **Web Service** conectado a ese repositorio y seleccionar
   **Python 3**. Dejar **Root Directory** vacío si estos archivos están en la raíz.
3. Usar **Build Command**: `pip install -r requirements-render.txt`.
4. Usar **Start Command**: `gunicorn volatility_dashboard.app:server --bind 0.0.0.0:$PORT --workers 1 --threads 2 --timeout 180 --access-logfile - --error-logfile -`.
5. Configurar `PYTHON_VERSION=3.11.11` y `OMP_NUM_THREADS=1`,
   `OPENBLAS_NUM_THREADS=1`, `MKL_NUM_THREADS=1`. Health check: `/healthz`.
6. Crear el servicio y revisar los logs. Cuando termine el despliegue, abrir
   la URL pública y comprobar Contexto, EDA y Comparación de modelos.

También se puede crear un Blueprint a partir de `render.yaml`.
No cambiar el servicio a sitio estático: los filtros requieren un servidor Python.

El repositorio remoto debe incluir `volatility_dashboard/`, `src/`,
`data/processed/minute_2023_2025/`, `results/minute_2023_2025/data_manifest.json`,
las siete carpetas de resultados enumeradas en `volatility_dashboard/data_loader.py`
(incluidos modelos, predicciones y verificaciones), y
`book/sections/10_dashboard_comparativo.md`. Mantener sus rutas relativas.
Si los artefactos usan Git LFS, verificar que se descarguen los archivos reales.
El dashboard audita y carga modelos guardados; no entrena al arrancar.
Comprobar memoria y tiempos con los datos completos antes de elegir el plan.
Una validación local no confirma que el servicio remoto esté desplegado.

Documentación: https://render.com/docs/web-services

### Ajuste para errores 502 durante la carga inicial

La auditoría inicial se ejecuta una sola vez a la vez entre los threads del
servidor. Los hashes se leen en bloques; los resúmenes diarios de los datos
por minuto se reutilizan entre modelos. Las cachés de modelos y matrices
tienen límites pequeños para reducir la memoria retenida. Se mantienen las
verificaciones de fechas, hashes y predicciones, sin entrenar modelos.

Después de publicar estos cambios, configurar **Health Check Path** como
`/healthz` en el servicio existente y volver a desplegar. Este endpoint verifica
que el servidor responde; comprobar también las tres pestañas para validar los datos.
Si reaparece el 502, revisar los logs de ejecución y los eventos de memoria:
esta optimización no confirma por sí sola la causa del fallo remoto.

Implementacion modular en volatility_dashboard/, con exactamente tres pestanas: contexto, EDA y comparacion. Reutiliza el dataset y los modelos existentes; no entrena ni modifica resultados.

Desde la raiz:

    .\.venv-repro\Scripts\python.exe -m volatility_dashboard.app

Abrir http://127.0.0.1:8050/. Si otro dashboard ocupa 8050, detener esa instancia antes de iniciar este comando. Las dependencias estan en requirements.txt, requirements-dashboard.txt y requirements-xgboost.txt.

Si aparecen los textos de EDA pero faltan las graficas o tablas, reiniciar el
servidor con el codigo actualizado y recargar con Ctrl+F5. El dashboard carga
los recursos JavaScript de Plotly y las tablas desde la pagina inicial, y
reserva altura para las graficas. En Render, volver a desplegar estos cambios.

La auditoría previa comprueba hashes, fechas, objetivos, calendario, medias del scaler y predicciones serializadas. El dashboard verifica los archivos y las predicciones contra esa auditoría y mantiene los controles de comparabilidad. BTC, ETH, BNB y XRP estan disponibles; SOL no tiene datos comparables.

Informe detallado: book/sections/10_dashboard_comparativo.md. No hay volumen procesado, pruebas residuales alineadas ni tiempos separados completos. MAPE no se incluye automaticamente. Las importancias de k-NN son diagnosticos descriptivos por grupos de lags, sin tuning.

Validar:

    .\.venv-repro\Scripts\python.exe -m unittest discover -s tests -p test_comparative_dashboard.py


## Carga de Render con los modelos originales

Antes de publicar, ejecutar `python src/build_dashboard_audit.py`. Este comando
reproduce los 112 artefactos originales, sin entrenar ni modificar predicciones,
y escribe `dashboard_data/comparative_audit.json`. Publicar este archivo junto
con el código y los resultados. Comprobarlo con
`python src/build_dashboard_audit.py --check` desde el directorio que se publica.

La web verifica los hashes de datos, modelos, fuentes y resultados, además del
contenido numérico de las predicciones, las fechas y los objetivos. Reutiliza
los seis cortes temporales guardados y evita reconstruir cientos de matrices
nativas o volver a predecir con todos los modelos durante la primera petición.
La auditoría completa sigue disponible mediante el comando de publicación.
Si cambian sus archivos fuente, regenerar el manifiesto antes de desplegar.

HAR-Ridge + XGBoost conserva `results/improved_classical_2023_2025/`, sus
predicciones y la prueba Diebold–Mariano original: MSE 1.186401 en
TODOS / ventana 7 / TODOS. Las pruebas y el ajuste Holm conservan su protocolo.
